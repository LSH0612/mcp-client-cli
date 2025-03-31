#!/usr/bin/env python3

"""
API Server for MCP Client CLI that streams responses using SSE
"""

import asyncio
import json
import logging
import sys
import io
import argparse
from typing import Dict, Any, List, Optional, AsyncGenerator
from contextlib import redirect_stdout, asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

# Import CLI functionality directly to avoid circular imports
from mcp_client_cli.cli import handle_conversation, SqliteStore
from mcp_client_cli.cli import HumanMessage  # Import HumanMessage class
from mcp_client_cli.const import SQLITE_DB
from mcp_client_cli.config import AppConfig
from mcp_client_cli.output import OutputHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mcp-api-server")

# Replace the entire StreamingOutputHandler class with this implementation

class StreamingOutputHandler(OutputHandler):
    def __init__(self, text_only=True):
        super().__init__(text_only=text_only, only_last_message=False)
        self.text_queue = asyncio.Queue()
        self.is_done = False
        self.buffer = ""  # Buffer for plain text responses
        self._initial_message_sent = False
    
    def update(self, chunk: any):
        """Capture output and put it on the queue."""
        # Send an initial message to let the client know processing has started
        if not self._initial_message_sent:
            self.text_queue.put_nowait("event: start\ndata: Processing request\n\n")
            self._initial_message_sent = True
        
        # Direct handling of chunk to extract text content
        extracted_text = self._extract_text_from_chunk(chunk)
        if extracted_text:
            logger.info(f"Extracted text from chunk: {extracted_text[:50]}...")
            self.text_queue.put_nowait(extracted_text)
            self.buffer += extracted_text
        
        # Still call parent update for other processing
        super().update(chunk)
    
    def _extract_text_from_chunk(self, chunk: any) -> str:
        """Extract text content directly from various chunk formats."""
        # Handle various chunk formats directly
        if isinstance(chunk, tuple) and len(chunk) >= 2:
            # Handle message chunks
            if chunk[0] == "messages" and len(chunk[1]) > 0:
                message = chunk[1][0]
                if hasattr(message, "content"):
                    if isinstance(message.content, str):
                        return message.content
                    elif isinstance(message.content, list):
                        # Extract text from content list (multimodal format)
                        text_parts = []
                        for item in message.content:
                            if isinstance(item, dict) and "text" in item:
                                text_parts.append(item["text"])
                        return "".join(text_parts)
            
            # Handle values chunks
            elif chunk[0] == "values" and isinstance(chunk[1], dict):
                if "messages" in chunk[1] and len(chunk[1]["messages"]) > 0:
                    message = chunk[1]["messages"][-1]
                    if hasattr(message, "content"):
                        if isinstance(message.content, str):
                            return message.content
                        elif isinstance(message.content, list):
                            # Extract text from content list (multimodal format)
                            text_parts = []
                            for item in message.content:
                                if isinstance(item, dict) and "text" in item:
                                    text_parts.append(item["text"])
                            return "".join(text_parts)
        
        # For direct string chunks
        elif isinstance(chunk, str):
            return chunk
        
        return ""
    
    def update_error(self, error: Exception):
        """Capture error and put it on the queue."""
        error_msg = f"Error: {str(error)}"
        self.text_queue.put_nowait(f"event: error\ndata: {error_msg}\n\n")
        self.buffer += f"\nERROR: {error_msg}"  # Add to buffer for plain text
        super().update_error(error)
        self.is_done = True
    
    def confirm_tool_call(self, config: dict, chunk: any) -> bool:
        """Auto-confirm tool calls in API mode."""
        # For API server, we auto-confirm all tool calls
        if self._is_tool_call_requested(chunk, config):
            tool_info = json.dumps({"tool_call": True, "auto_confirmed": True})
            self.text_queue.put_nowait(f"event: tool_call\ndata: {tool_info}\n\n")
            self.buffer += f"\n[Tool call auto-confirmed]"  # Add to buffer for plain text
            return True
        return True
    
    def finish(self):
        """Mark streaming as complete."""
        if not self.is_done:  # Only send close event if not already done
            self.text_queue.put_nowait(f"event: close\ndata: Stream closed\n\n")
            self.buffer += "\n[Response complete]"  # Add to buffer for plain text
            self.is_done = True
        super().finish()

# API Models
class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., description="The message to send to the LLM")
    stream: bool = Field(True, description="Whether to stream the response")
    continue_conversation: bool = Field(False, description="Whether to continue the previous conversation")
    no_tools: bool = Field(False, description="Whether to disable tools")
    model: Optional[str] = Field(None, description="Override the model specified in config")

class ChatResponse(BaseModel):
    """Non-streaming chat response."""
    response: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Load app configuration on startup
    app.state.config = AppConfig.load()
    yield
    # Clean up on shutdown
    logger.info("Shutting down API server")

# Initialize FastAPI app
app = FastAPI(
    title="MCP Client API",
    description="API Server for MCP Client CLI",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware with proper configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
    expose_headers=["Content-Type", "Content-Length"],  # Expose these headers
    max_age=86400,  # Cache preflight requests for 1 day
)

async def stream_response(output_handler: StreamingOutputHandler) -> AsyncGenerator[str, None]:
    """Generate SSE response from output handler."""
    # Send initial message immediately to establish connection
    yield "event: start\ndata: Connection established\n\n"
    
    while True:
        try:
            # Try to get a message from the queue with a short timeout
            message = await asyncio.wait_for(output_handler.text_queue.get(), timeout=0.1)
            
            # Format and send the message
            if message.startswith("event:"):
                # Already formatted as SSE
                yield message
            else:
                # Format as SSE message
                yield f"data: {message}\n\n"
                
            # If we got the close event, exit the loop
            if message.startswith("event: close"):
                break
                
        except asyncio.TimeoutError:
            # No message available - if handler is done and queue is empty, break
            if output_handler.is_done and output_handler.text_queue.empty():
                yield "event: close\ndata: Stream complete\n\n"
                break
                
            # Otherwise, send a heartbeat and continue
            yield ":heartbeat\n\n"
            await asyncio.sleep(0.5)  # Send heartbeats every 0.5 seconds
            
        except Exception as e:
            logger.exception(f"Error in stream_response: {str(e)}")
            yield f"event: error\ndata: {str(e)}\n\n"
            yield "event: close\ndata: Stream closed due to error\n\n"
            break

async def stream_text(output_handler: StreamingOutputHandler) -> AsyncGenerator[str, None]:
    """Generate plain text stream response for curl."""
    # Immediately send an empty line to establish connection
    yield "\n"
    
    while True:
        try:
            # Try to get a message with a short timeout
            message = await asyncio.wait_for(output_handler.text_queue.get(), timeout=0.1)
            
            # For plain text, filter out SSE formatting events and just yield content
            if not message.startswith("event:"):
                yield message
                
            # If we received close event, exit after sending any remaining content
            if message.startswith("event: close"):
                break
                
        except asyncio.TimeoutError:
            # If done and no more messages, break
            if output_handler.is_done and output_handler.text_queue.empty():
                break
                
            # Send a small heartbeat to keep the connection alive
            yield ""
            await asyncio.sleep(0.5)
            
        except Exception as e:
            logger.exception(f"Error in stream_text: {str(e)}")
            yield f"\nError: {str(e)}\n"
            break

def prepare_query(message: str, continue_conversation: bool = False) -> tuple[HumanMessage, bool]:
    """Prepare query for the CLI handler without using parse_query."""
    is_continuation = continue_conversation
    
    if continue_conversation:
        query_text = f"c {message}"
    else:
        query_text = message
    
    # Create a HumanMessage directly without going through parse_query
    return HumanMessage(content=query_text), is_continuation

async def handle_chat(
    request_message: str,
    continue_conversation: bool = False,
    no_tools: bool = False,
    model: Optional[str] = None,
    app_config: AppConfig = None
) -> StreamingOutputHandler:
    """Handle a chat request."""
    # Create custom args similar to CLI
    args = argparse.Namespace()
    args.no_tools = no_tools
    args.model = model
    args.force_refresh = False
    args.no_confirmations = True  # Auto-confirm tools in API mode
    args.text_only = True
    args.no_intermediates = False
    args.list_tools = False
    args.list_prompts = False
    args.show_memories = False
    
    # Prepare the query directly without using parse_query
    query, is_conversation_continuation = prepare_query(
        request_message, continue_conversation
    )
    
    # Create a custom output handler for streaming
    output_handler = StreamingOutputHandler(text_only=True)
    
    # Replace the output handler in conversation handling
    from mcp_client_cli import output
    original_handler = output.OutputHandler
    output.OutputHandler = lambda *args, **kwargs: output_handler
    
    # Start a separate task to handle the conversation
    conversation_task = asyncio.create_task(
        handle_conversation(args, query, is_conversation_continuation, app_config)
    )
    
    # Return the output handler immediately so we can start streaming responses
    # The conversation will continue in the background
    return output_handler, conversation_task

@app.post("/api/chat", response_model=None)
async def chat_post(
    request: ChatRequest,
    app_config: AppConfig = Depends(lambda: app.state.config)
):
    """Chat with the LLM - POST endpoint."""
    logger.info(f"POST /api/chat - message: '{request.message[:50]}...' (stream: {request.stream})")
    
    if request.stream:
        # Create a background task for the conversation
        output_handler, conversation_task = await handle_chat(
            request.message, 
            request.continue_conversation, 
            request.no_tools, 
            request.model, 
            app_config
        )
        
        # Set up streaming response with background task cleanup
        async def event_generator():
            try:
                async for chunk in stream_response(output_handler):
                    yield chunk
            except asyncio.CancelledError:
                # If the client disconnects, cancel the conversation task
                logger.info("Client disconnected, canceling conversation task")
                conversation_task.cancel()
                raise
            except Exception as e:
                logger.exception(f"Error in event generator: {str(e)}")
                yield f"event: error\ndata: {str(e)}\n\n"
                yield f"event: close\ndata: Stream closed due to error\n\n"
            finally:
                # Make sure we mark the output handler as done when the stream ends
                if not output_handler.is_done:
                    output_handler.finish()
        
        # Configure EventSourceResponse with the right parameters
        return EventSourceResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Prevents Nginx from buffering the response
            }
        )
    else:
        # For non-streaming, collect the full response
        try:
            output_handler, conversation_task = await handle_chat(
                request.message, request.continue_conversation,
                request.no_tools, request.model, app_config
            )
            
            # Wait for conversation to complete
            await conversation_task
            
            # Collect all text from the queue
            full_response = ""
            while not output_handler.is_done or not output_handler.text_queue.empty():
                try:
                    msg = await asyncio.wait_for(output_handler.text_queue.get(), timeout=0.1)
                    if not msg.startswith("event:"):  # Skip SSE formatting
                        full_response += msg
                except asyncio.TimeoutError:
                    if output_handler.is_done and output_handler.text_queue.empty():
                        break
                    await asyncio.sleep(0.01)
            
            return {"response": full_response}
        except Exception as e:
            logger.exception("Error processing non-streaming request")
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chat", response_model=None)
async def chat_get(
    message: str,
    continue_conversation: bool = False,
    no_tools: bool = False,
    model: Optional[str] = None,
    stream: bool = True,
    app_config: AppConfig = Depends(lambda: app.state.config)
):
    """Chat with the LLM - GET endpoint."""
    logger.info(f"GET /api/chat - message: '{message[:50]}...' (stream: {stream})")
    
    if stream:
        # Process the conversation
        output_handler, conversation_task = await handle_chat(
            message, 
            continue_conversation, 
            no_tools, 
            model, 
            app_config
        )
        
        # Set up streaming response with background task cleanup
        async def event_generator():
            try:
                async for chunk in stream_response(output_handler):
                    yield chunk
            except asyncio.CancelledError:
                # If the client disconnects, cancel the conversation task
                conversation_task.cancel()
                raise
            except Exception as e:
                logger.exception("Error in event generator")
                yield f"event: error\ndata: {str(e)}\n\n"
                yield f"event: close\ndata: Stream closed due to error\n\n"
            finally:
                # Make sure we mark the output handler as done when the stream ends
                if not output_handler.is_done:
                    output_handler.finish()
        
        return EventSourceResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
    else:
        # For non-streaming, collect the full response
        try:
            output_handler, conversation_task = await handle_chat(
                message, continue_conversation, no_tools, model, app_config
            )
            
            # Wait for conversation to complete
            await conversation_task
            
            # Collect all text from the queue
            full_response = ""
            while not output_handler.is_done or not output_handler.text_queue.empty():
                try:
                    msg = await asyncio.wait_for(output_handler.text_queue.get(), timeout=0.1)
                    if not msg.startswith("event:"):  # Skip SSE formatting
                        full_response += msg
                except asyncio.TimeoutError:
                    if output_handler.is_done and output_handler.text_queue.empty():
                        break
                    await asyncio.sleep(0.01)
            
            return {"response": full_response}
        except Exception as e:
            logger.exception("Error processing non-streaming request")
            raise HTTPException(status_code=500, detail=str(e))

# Add a curl-friendly text endpoint
@app.get("/api/text", response_model=None)
async def text_chat(
    message: str,
    continue_conversation: bool = False,
    no_tools: bool = False,
    model: Optional[str] = None,
    app_config: AppConfig = Depends(lambda: app.state.config)
):
    """
    Chat with the LLM and return a plaintext response (curl-friendly).
    """
    logger.info(f"Processing text request: '{message}' (continuation: {continue_conversation})")
    
    # Process the conversation
    output_handler, conversation_task = await handle_chat(
        message, 
        continue_conversation, 
        no_tools, 
        model, 
        app_config
    )
    
    # Set up chunked streaming text response with background task cleanup
    async def text_generator_with_cleanup():
        try:
            async for chunk in stream_text(output_handler):
                yield chunk
        except asyncio.CancelledError:
            # If the client disconnects, cancel the conversation task
            conversation_task.cancel()
            raise
        except Exception as e:
            logger.exception("Error in text generator")
            yield f"\nError: {str(e)}\n"
        finally:
            # Make sure we mark the output handler as done when the stream ends
            if not output_handler.is_done:
                output_handler.finish()
    
    return StreamingResponse(
        text_generator_with_cleanup(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

# Add a simple plaintext endpoint that returns non-streaming response
@app.get("/api/plaintext")
async def plaintext_chat(
    message: str,
    continue_conversation: bool = False,
    no_tools: bool = False,
    model: Optional[str] = None,
    app_config: AppConfig = Depends(lambda: app.state.config)
):
    """
    Chat with the LLM and return a simple plaintext response (not streaming).
    """
    try:
        # Process the conversation
        output_handler, conversation_task = await handle_chat(
            message, continue_conversation, no_tools, model, app_config
        )
        
        # Wait for the conversation task to complete
        await conversation_task
        
        # Ensure all messages are processed
        while not output_handler.text_queue.empty():
            await asyncio.sleep(0.01)
        
        # Finish the handler if not already done
        if not output_handler.is_done:
            output_handler.finish()
        
        # Return the complete buffer content
        return StreamingResponse(
            content=iter([output_handler.buffer]),
            media_type="text/plain"
        )
    except Exception as e:
        logger.exception("Error processing plaintext request")
        return StreamingResponse(
            content=iter([f"Error: {str(e)}"]),
            media_type="text/plain",
            status_code=500
        )

@app.post("/api/json-chat")
async def json_chat(
    request: ChatRequest,
    app_config: AppConfig = Depends(lambda: app.state.config)
):
    """Chat with the LLM with JSON response format."""
    logger.info(f"POST /api/json-chat - message: '{request.message[:50]}...'")
    
    # Process the conversation
    output_handler, conversation_task = await handle_chat(
        request.message, 
        request.continue_conversation, 
        request.no_tools, 
        request.model, 
        app_config
    )
    
    # For JSON streaming, we'll collect all chunks and construct a proper response
    if request.stream:
        async def json_stream():
            try:
                # Initiate the stream with an opening event
                yield json.dumps({"event": "start", "data": "Connection established"}) + "\n"
                
                while True:
                    try:
                        # Get a message with short timeout
                        message = await asyncio.wait_for(output_handler.text_queue.get(), timeout=0.1)
                        
                        # Skip events other than normal text
                        if message.startswith("event:"):
                            if "close" in message:
                                # Send final message and exit
                                yield json.dumps({"event": "close", "data": "Stream complete"}) + "\n"
                                break
                            continue
                            
                        # Send the actual content
                        yield json.dumps({"event": "message", "data": message}) + "\n"
                        
                    except asyncio.TimeoutError:
                        # If handler is done and queue is empty, end the stream
                        if output_handler.is_done and output_handler.text_queue.empty():
                            yield json.dumps({"event": "close", "data": "Stream complete"}) + "\n"
                            break
                            
                        # Otherwise, send a heartbeat
                        yield json.dumps({"event": "heartbeat"}) + "\n"
                        await asyncio.sleep(0.5)
                        
            except Exception as e:
                logger.exception(f"Error in JSON stream: {str(e)}")
                yield json.dumps({"event": "error", "data": str(e)}) + "\n"
                
            finally:
                # Make sure conversation task is handled properly
                if not output_handler.is_done:
                    output_handler.finish()
                    
        return StreamingResponse(
            json_stream(),
            media_type="application/x-ndjson",  # Use newline-delimited JSON format
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
    else:
        # Wait for conversation to complete
        await conversation_task
        
        # Collect all text from the queue
        full_response = ""
        while not output_handler.is_done or not output_handler.text_queue.empty():
            try:
                msg = await asyncio.wait_for(output_handler.text_queue.get(), timeout=0.1)
                if not msg.startswith("event:"):  # Skip SSE formatting
                    full_response += msg
            except asyncio.TimeoutError:
                if output_handler.is_done and output_handler.text_queue.empty():
                    break
                await asyncio.sleep(0.01)
        
        return {"response": full_response}

@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}

def main():
    """Start the API server."""
    import uvicorn
    
    logger.info("Starting MCP API Server on port 5000")
    uvicorn.run(app, host="0.0.0.0", port=5000)

if __name__ == "__main__":
    main()