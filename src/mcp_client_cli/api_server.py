#!/usr/bin/env python3

"""
Simplified API Server for MCP Client CLI that directly executes the CLI in a subprocess
"""

import asyncio
import json
import logging
import sys
import os
import io
import subprocess
import tempfile
from typing import Dict, Any, List, Optional, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mcp-api-server")

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
    # Import config here to avoid circular imports
    from mcp_client_cli.config import AppConfig
    try:
        app.state.config = AppConfig.load()
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        app.state.config = None
    
    yield
    
    logger.info("API server shutting down")

# Initialize FastAPI app
app = FastAPI(
    title="MCP Client API",
    description="Simplified API Server for MCP Client CLI",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "Content-Length"],
)

async def run_cli_process(message: str, continue_conversation: bool = False,
                         no_tools: bool = False, model: Optional[str] = None):
    """Run the CLI in a subprocess and capture its output."""
    # Prepare command arguments
    cmd = [sys.executable, "-m", "mcp_client_cli.cli"]
    
    # Add options
    if no_tools:
        cmd.append("--no-tools")
    
    if model:
        cmd.extend(["--model", model])
    
    # Add the message
    if continue_conversation:
        cmd.extend(["c", message])
    else:
        cmd.append(message)
    
    # Create a queue for output
    queue = asyncio.Queue()
    
    # Function to process output
    async def process_output(stream, event_type):
        try:
            while True:
                line = await stream.readline()
                if not line:
                    break
                
                # Decode the line
                try:
                    text = line.decode('utf-8').strip()
                    if text:
                        # Skip initial connection/loading messages
                        if "INFO" in text and ("Starting" in text or "Loading" in text):
                            continue
                            
                        # Log the output for debugging
                        logger.info(f"CLI {event_type}: {text[:100]}")
                        
                        # Send to client
                        await queue.put({"event": "message", "data": text})
                except Exception as e:
                    logger.error(f"Error processing output: {e}")
        except Exception as e:
            logger.exception(f"Error in output processing: {e}")
    
    # Send initial message
    await queue.put({"event": "start", "data": "Processing request..."})
    
    try:
        # Start the process
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Process stdout and stderr concurrently
        stdout_task = asyncio.create_task(process_output(process.stdout, "stdout"))
        stderr_task = asyncio.create_task(process_output(process.stderr, "stderr"))
        
        # Wait for process to complete
        exit_code = await process.wait()
        
        # Wait for output processing to complete
        await stdout_task
        await stderr_task
        
        # Send close event
        await queue.put({"event": "close", "data": f"CLI process completed with exit code {exit_code}"})
    except Exception as e:
        logger.exception(f"Error running CLI process: {e}")
        await queue.put({"event": "error", "data": str(e)})
        await queue.put({"event": "close", "data": "Error running CLI process"})
    
    return queue

# Alternative approach: Write to a temporary file and read from it
async def run_with_temp_file(message: str, continue_conversation: bool = False,
                            no_tools: bool = False, model: Optional[str] = None):
    """Run the CLI and capture output to a temporary file."""
    # Create a temporary file for output
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
        temp_path = temp_file.name
    
    # Prepare command arguments
    cmd = [sys.executable, "-m", "mcp_client_cli.cli"]
    
    # Add options
    if no_tools:
        cmd.append("--no-tools")
    
    if model:
        cmd.extend(["--model", model])
    
    # Add the message
    if continue_conversation:
        cmd.extend(["c", message])
    else:
        cmd.append(message)
    
    # Create a queue for output
    queue = asyncio.Queue()
    
    # Send initial message
    await queue.put({"event": "start", "data": "Processing request..."})
    
    try:
        # Start the process with output to the temp file
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=open(temp_path, 'w'),
            stderr=subprocess.STDOUT
        )
        
        # Function to check for new output in the file
        async def check_output():
            last_size = 0
            
            while True:
                # Check if process is still running
                if process.returncode is not None:
                    # Process has completed
                    break
                
                # Check current file size
                current_size = os.path.getsize(temp_path)
                
                if current_size > last_size:
                    # File has grown, read the new content
                    with open(temp_path, 'r') as f:
                        f.seek(last_size)
                        new_content = f.read()
                        
                        if new_content:
                            logger.info(f"New content: {new_content[:100]}")
                            await queue.put({"event": "message", "data": new_content})
                    
                    last_size = current_size
                
                # Sleep before checking again
                await asyncio.sleep(0.1)
        
        # Start checking for output
        output_task = asyncio.create_task(check_output())
        
        # Wait for process to complete
        exit_code = await process.wait()
        
        # Wait for output checking to complete
        await output_task
        
        # Read any remaining output
        with open(temp_path, 'r') as f:
            remaining = f.read()
            if remaining:
                await queue.put({"event": "message", "data": remaining})
        
        # Send close event
        await queue.put({"event": "close", "data": f"CLI process completed with exit code {exit_code}"})
    except Exception as e:
        logger.exception(f"Error running CLI process: {e}")
        await queue.put({"event": "error", "data": str(e)})
        await queue.put({"event": "close", "data": "Error running CLI process"})
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except:
            pass
    
    return queue

# Direct integration approach
async def direct_integrate(message: str, continue_conversation: bool = False,
                         no_tools: bool = False, model: Optional[str] = None):
    """Directly integrate with the CLI module."""
    # Import CLI modules
    from mcp_client_cli.cli import handle_conversation, parse_query
    from mcp_client_cli.output import OutputHandler
    
    # Create a queue for output
    queue = asyncio.Queue()
    
    # Create a custom output handler
    class ApiOutputHandler(OutputHandler):
        def __init__(self, queue, text_only=True):
            super().__init__(text_only=text_only, only_last_message=False)
            self.output_queue = queue
            self.buffer = ""
        
        def update(self, chunk):
            # Call parent implementation first
            super().update(chunk)
            
            # Extract text content
            content = self._extract_text(chunk)
            if content and content.strip():
                logger.info(f"Extracted content: {content[:100]}")
                self.buffer += content
                asyncio.create_task(self.output_queue.put({"event": "message", "data": content}))
        
        def _extract_text(self, chunk):
            text = ""
            
            # Handle different chunk types
            if isinstance(chunk, tuple) and len(chunk) >= 2:
                chunk_type, chunk_data = chunk
                
                # Handle messages
                if chunk_type == "messages" and chunk_data:
                    message = chunk_data[0]
                    if hasattr(message, "content"):
                        if isinstance(message.content, str):
                            text = message.content
                        elif isinstance(message.content, list):
                            parts = []
                            for item in message.content:
                                if isinstance(item, dict) and "text" in item:
                                    parts.append(item["text"])
                            text = "".join(parts)
                
                # Handle values
                elif chunk_type == "values" and isinstance(chunk_data, dict):
                    if "messages" in chunk_data and chunk_data["messages"]:
                        message = chunk_data["messages"][-1]
                        if hasattr(message, "content"):
                            if isinstance(message.content, str):
                                text = message.content
            
            # Handle direct string chunks
            elif isinstance(chunk, str):
                text = chunk
            
            return text
        
        def update_error(self, error):
            super().update_error(error)
            
            # Send error to client
            error_text = f"Error: {str(error)}"
            asyncio.create_task(self.output_queue.put({"event": "error", "data": error_text}))
        
        def confirm_tool_call(self, config, chunk):
            # Auto-confirm for API
            return True
        
        def finish(self):
            super().finish()
            
            # Send close event
            asyncio.create_task(self.output_queue.put({"event": "close", "data": "Conversation complete"}))
    
    # Send initial message
    await queue.put({"event": "start", "data": "Processing request..."})
    
    try:
        # Create args
        import argparse
        args = argparse.Namespace()
        args.no_tools = no_tools
        args.model = model
        args.force_refresh = False
        args.no_confirmations = True
        args.text_only = True
        args.no_intermediates = False
        args.list_tools = False
        args.list_prompts = False
        args.show_memories = False
        
        # Set up query
        if continue_conversation:
            args.query = ["c", message]
        else:
            args.query = [message]
        
        # Parse query
        query, is_conversation_continuation = parse_query(args)
        
        # Prepare output handler
        output_handler = ApiOutputHandler(queue, text_only=True)
        
        # Custom redirect to capture output
        class TeeOutput:
            def __init__(self, original, queue):
                self.original = original
                self.queue = queue
            
            def write(self, text):
                # Write to original
                self.original.write(text)
                
                # Send to client if meaningful
                if text and len(text.strip()) > 5:
                    asyncio.create_task(self.queue.put({"event": "message", "data": text}))
                
                return len(text)
            
            def flush(self):
                self.original.flush()
        
        # Set up output redirection
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = TeeOutput(original_stdout, queue)
        sys.stderr = TeeOutput(original_stderr, queue)
        
        # Override OutputHandler
        import mcp_client_cli.output
        original_handler_class = mcp_client_cli.output.OutputHandler
        mcp_client_cli.output.OutputHandler = lambda *args, **kwargs: output_handler
        
        try:
            # Run the conversation
            conversation_task = asyncio.create_task(
                handle_conversation(args, query, is_conversation_continuation, None)
            )
            
            # Wait for conversation to complete
            await conversation_task
            
            # Send close event
            await queue.put({"event": "close", "data": "Conversation complete"})
        finally:
            # Restore original stdout/stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            
            # Restore original OutputHandler
            mcp_client_cli.output.OutputHandler = original_handler_class
    except Exception as e:
        logger.exception(f"Error in direct integration: {e}")
        await queue.put({"event": "error", "data": str(e)})
        await queue.put({"event": "close", "data": "Error in direct integration"})
    
    return queue

# Use with Web UI
class WebAPIHandler:
    """Handler for web API that returns direct responses."""
    
    def __init__(self):
        pass
    
    async def handle_request(self, message, continue_conversation=False, no_tools=False, model=None):
        """Handle a request and return the response."""
        # Generate and run the CLI command
        cmd = [sys.executable, "-m", "mcp_client_cli.cli"]
        
        if no_tools:
            cmd.append("--no-tools")
        
        if model:
            cmd.extend(["--model", model])
        
        # Set text-only flag for cleaner output
        cmd.append("--text-only")
        
        # Add message
        if continue_conversation:
            cmd.extend(["c", message])
        else:
            cmd.append(message)
        
        try:
            # Run process and capture output
            logger.info(f"Running command: {' '.join(cmd)}")
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Collect stdout
            stdout_data = await process.stdout.read()
            stderr_data = await process.stderr.read()
            
            # Wait for process to complete
            await process.wait()
            
            # Decode output
            stdout_text = stdout_data.decode('utf-8')
            stderr_text = stderr_data.decode('utf-8')
            
            logger.info(f"CLI stdout: {stdout_text[:100]}")
            logger.info(f"CLI stderr: {stderr_text[:100]}")
            
            # Return the output
            return stdout_text or stderr_text
        except Exception as e:
            logger.exception(f"Error running CLI: {e}")
            return f"Error: {str(e)}"

async def stream_response(queue):
    """Stream SSE events from a queue."""
    while True:
        try:
            msg = await asyncio.wait_for(queue.get(), timeout=1.0)
            
            event_type = msg.get('event', '')
            event_data = msg.get('data', '')
            
            if event_type == 'heartbeat':
                yield ":heartbeat\n\n"
            else:
                yield f"event: {event_type}\ndata: {event_data}\n\n"
            
            if event_type == 'close':
                break
        except asyncio.TimeoutError:
            # Send heartbeat
            yield ":heartbeat\n\n"
            await asyncio.sleep(0.5)
        except Exception as e:
            logger.exception(f"Error in stream_response: {e}")
            yield f"event: error\ndata: {str(e)}\n\n"
            yield f"event: close\ndata: Stream closed due to error\n\n"
            break

@app.post("/api/chat")
async def chat_post(request: ChatRequest):
    """Chat with the LLM - POST endpoint."""
    logger.info(f"POST /api/chat - message: '{request.message[:50]}...' (stream: {request.stream})")
    
    # Use direct web API handler
    handler = WebAPIHandler()
    
    if request.stream:
        # Run CLI process
        queue = await run_cli_process(
            request.message,
            request.continue_conversation,
            request.no_tools,
            request.model
        )
        
        # Stream the response
        return EventSourceResponse(
            stream_response(queue),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
    else:
        # For non-streaming, get the complete response directly
        response = await handler.handle_request(
            request.message,
            request.continue_conversation,
            request.no_tools,
            request.model
        )
        
        return {"response": response}

@app.get("/api/chat")
async def chat_get(
    message: str,
    continue_conversation: bool = False,
    no_tools: bool = False,
    model: Optional[str] = None,
    stream: bool = True
):
    """Chat with the LLM - GET endpoint."""
    logger.info(f"GET /api/chat - message: '{message[:50]}...' (stream: {stream})")
    
    # Use direct web API handler
    handler = WebAPIHandler()
    
    if stream:
        # Run CLI process
        queue = await run_cli_process(
            message,
            continue_conversation,
            no_tools,
            model
        )
        
        # Stream the response
        return EventSourceResponse(
            stream_response(queue),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
    else:
        # For non-streaming, get the complete response directly
        response = await handler.handle_request(
            message,
            continue_conversation,
            no_tools,
            model
        )
        
        return {"response": response}

@app.get("/api/text")
async def text_chat(
    message: str,
    continue_conversation: bool = False,
    no_tools: bool = False,
    model: Optional[str] = None
):
    """Chat with the LLM and return a plaintext streaming response."""
    logger.info(f"GET /api/text - message: '{message[:50]}...'")
    
    # Run CLI process
    queue = await run_cli_process(
        message,
        continue_conversation,
        no_tools,
        model
    )
    
    # Stream text
    async def text_generator():
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=1.0)
                    
                    if msg.get("event") == "message":
                        yield msg.get("data", "")
                    
                    if msg.get("event") == "close":
                        break
                except asyncio.TimeoutError:
                    # Keep connection alive
                    yield ""
                    await asyncio.sleep(0.5)
        except Exception as e:
            logger.exception(f"Error in text generator: {e}")
            yield f"\nError: {str(e)}\n"
    
    return StreamingResponse(
        text_generator(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.get("/api/plaintext")
async def plaintext_chat(
    message: str,
    continue_conversation: bool = False,
    no_tools: bool = False,
    model: Optional[str] = None
):
    """Chat with the LLM and return a non-streaming plaintext response."""
    logger.info(f"GET /api/plaintext - message: '{message[:50]}...'")
    
    # Use direct web API handler for simplicity
    handler = WebAPIHandler()
    response = await handler.handle_request(
        message,
        continue_conversation,
        no_tools,
        model
    )
    
    return StreamingResponse(
        content=iter([response]),
        media_type="text/plain"
    )

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "version": "1.0.0"}

@app.get("/api/direct")
async def direct_response(
    message: str,
    continue_conversation: bool = False
):
    """Get a direct response without streaming."""
    # Create a command line for the llm tool
    cmd = [sys.executable, "-m", "mcp_client_cli.cli"]
    
    # Add text-only for cleaner output
    cmd.append("--text-only")
    
    # Add message
    if continue_conversation:
        cmd.extend(["c", message])
    else:
        cmd.append(message)
    
    try:
        # Run the command and capture output
        process = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=False
        )
        
        # Return the output
        return {"response": process.stdout or process.stderr}
    except Exception as e:
        logger.exception(f"Error in direct response: {e}")
        return {"error": str(e)}

def main():
    """Start the API server."""
    import uvicorn
    
    logger.info("Starting MCP API Server on port 5000")
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")

if __name__ == "__main__":
    main()
