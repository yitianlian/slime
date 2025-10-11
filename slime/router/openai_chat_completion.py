"""
Simplified OpenAI Chat Completion API implementation for Slime Router.

This module provides 100% OpenAI-compatible Chat Completion API while leveraging
Slime Router's Radix Cache for optimal performance in multi-turn conversations.

Key Features:
- Full OpenAI API compatibility (text in/out)
- Unified flow: messages → generate → OpenAI format
- Radix Tree Middleware integration for automatic caching
- Streaming and non-streaming support
- Simplified architecture with minimal abstraction

Architecture:
- Detect RadixTreeMiddleware presence
- Use query_cache_by_messages_template for semantic caching
- Forward to /generate endpoint for consistent processing
- Convert responses to OpenAI format
"""

import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.responses import Response


class ChatCompletionHandler:
    """
    Simplified Chat Completion handler with unified processing flow.

    This handler automatically detects cache capability by testing the
    /retrieve_from_messages_template endpoint and uses the appropriate
    processing path:
    - With cache support: Use messages template caching
    - Without cache support: Direct proxy to SGLang
    """

    def __init__(self, router):
        """
        Initialize Chat Completion handler.

        Args:
            router: SlimeRouter instance for accessing middleware and workers
        """
        self.router = router
        self.args = router.args
        self._cache_available = None  # Cache the availability check result

    async def _check_cache_availability(self):
        """
        Test if cache support is available by checking router's component registry.

        Returns:
            bool: True if cache is available, False otherwise
        """
        if self._cache_available is not None:
            return self._cache_available

        try:
            # Check if router has the required components
            if (hasattr(self.router, 'component_registry') and
                self.router.component_registry.has("radix_tree") and
                self.router.component_registry.has("tokenizer")):

                # Cache is available if both components exist
                self._cache_available = True

                if getattr(self.args, 'verbose', False):
                    print(f"[slime-router] Cache available through component registry")

                return self._cache_available

            else:
                # Cache not available
                self._cache_available = False

                if getattr(self.args, 'verbose', False):
                    print(f"[slime-router] Cache not available: missing components")

                return self._cache_available

        except Exception as e:
            # Any error means cache is not available
            if getattr(self.args, 'verbose', False):
                print(f"[slime-router] Cache availability check error: {e}")
            self._cache_available = False
            return False

    async def handle_request(self, request: Request):
        """
        Handle Chat Completion request with unified flow.

        Args:
            request: FastAPI Request object

        Returns:
            Either JSON response (non-streaming) or StreamingResponse
        """
        try:
            request_data = await request.json()
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid JSON in request body: {str(e)}"
            )

        # Validate request structure
        self._validate_chat_completion_request(request_data)

        stream = request_data.get("stream", False)

        # Check if cache support is available
        cache_available = await self._check_cache_availability()

        if not cache_available:
            # Direct mode: Proxy to SGLang Chat Completion API
            return await self._proxy_to_sglang_chat(request)

        # Cached mode: Use unified generate flow
        return await self._handle_with_generate_flow(request_data, stream)

    def _validate_chat_completion_request(self, request_data: dict):
        """
        Minimal validation for Chat Completion request.

        Only validate absolutely required fields. Let SGLang handle
        detailed parameter validation and return appropriate errors.

        Args:
            request_data: Parsed request data

        Raises:
            HTTPException: If basic validation fails
        """
        # Only check the absolute minimum required for OpenAI API compatibility
        if "messages" not in request_data:
            raise HTTPException(
                status_code=400,
                detail="Invalid request: 'messages' field is required"
            )

        messages = request_data["messages"]
        if not isinstance(messages, list) or len(messages) == 0:
            raise HTTPException(
                status_code=400,
                detail="Invalid request: 'messages' must be a non-empty list"
            )

        # Basic message structure check - let SGLang handle detailed validation
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid request: message at index {i} must be a dictionary"
                )

            if "role" not in message or "content" not in message:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid request: message at index {i} must have 'role' and 'content' fields"
                )

    async def _proxy_to_sglang_chat(self, request: Request):
        """
        Direct proxy mode: Forward request to SGLang Chat Completion API.

        Args:
            request: FastAPI Request object

        Returns:
            Direct response from SGLang
        """
        worker_url = await self.router._use_url()
        sglang_url = f"{worker_url}/v1/chat/completions"

        body = await request.body()
        headers = dict(request.headers)

        try:
            # Check if streaming request
            try:
                request_data = json.loads(body) if body else {}
                is_streaming = request_data.get("stream", False)
            except (json.JSONDecodeError, TypeError):
                is_streaming = False

            if is_streaming:
                # Streaming proxy
                async with self.router.client.stream(
                    "POST",
                    sglang_url,
                    content=body,
                    headers=headers,
                    timeout=None
                ) as response:
                    async def generate_chunks():
                        async for chunk in response.aiter_bytes():
                            yield chunk

                    return StreamingResponse(
                        generate_chunks(),
                        status_code=response.status_code,
                        headers=dict(response.headers)
                    )
            else:
                # Non-streaming proxy
                response = await self.router.client.request("POST", sglang_url, content=body, headers=headers)
                content = await response.aread()
                return Response(
                    content=content,
                    status_code=response.status_code,
                    headers=dict(response.headers)
                )
        finally:
            await self.router._finish_url(worker_url)

    async def _proxy_to_sglang_chat_from_data(self, request_data: dict):
        """
        Direct proxy mode: Forward request data to SGLang Chat Completion API.

        This is a helper method for when we need to proxy from parsed data instead of a Request object.

        Args:
            request_data: Parsed request data

        Returns:
            Direct response from SGLang
        """
        worker_url = await self.router._use_url()
        sglang_url = f"{worker_url}/v1/chat/completions"

        try:
            # Check if streaming request
            is_streaming = request_data.get("stream", False)

            if is_streaming:
                # Streaming proxy
                async with self.router.client.stream(
                    "POST",
                    sglang_url,
                    json=request_data,
                    timeout=None
                ) as response:
                    async def generate_chunks():
                        async for chunk in response.aiter_bytes():
                            yield chunk

                    return StreamingResponse(
                        generate_chunks(),
                        status_code=response.status_code,
                        headers=dict(response.headers)
                    )
            else:
                # Non-streaming proxy with error mapping
                try:
                    response = await self.router.client.request("POST", sglang_url, json=request_data)
                    content = await response.aread()

                    # Check for SGLang errors and map to OpenAI format
                    if response.status_code >= 400:
                        await self._handle_sglang_error(response, content)

                    return Response(
                        content=content,
                        status_code=response.status_code,
                        headers=dict(response.headers)
                    )
                except httpx.HTTPStatusError as e:
                    # Handle HTTP errors from SGLang
                    await self._handle_sglang_error(e.response, await e.response.aread())
                    raise
                except httpx.RequestError as e:
                    # Handle connection/network errors
                    raise HTTPException(
                        status_code=503,
                        detail="Service temporarily unavailable: Unable to reach inference backend"
                    )
        finally:
            await self.router._finish_url(worker_url)

    async def _handle_sglang_error(self, response, content):
        """
        Map SGLang errors to OpenAI-compatible error format.

        Args:
            response: HTTP response from SGLang
            content: Response content bytes
        """
        try:
            error_data = json.loads(content.decode('utf-8')) if content else {}

            # Map common SGLang errors to OpenAI format
            if response.status_code == 400:
                # Validation errors - pass through SGLang's message
                detail = error_data.get('error', error_data.get('detail', 'Invalid request parameters'))
                raise HTTPException(
                    status_code=400,
                    detail=detail
                )
            elif response.status_code == 429:
                # Rate limiting
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Please try again later."
                )
            elif response.status_code >= 500:
                # Server errors
                raise HTTPException(
                    status_code=response.status_code,
                    detail="Inference service error. Please try again later."
                )
            else:
                # Other errors
                raise HTTPException(
                    status_code=response.status_code,
                    detail=error_data.get('error', error_data.get('detail', 'Unknown error'))
                )
        except (json.JSONDecodeError, UnicodeDecodeError):
            # If we can't parse the error, return a generic message
            raise HTTPException(
                status_code=response.status_code,
                detail="Service error: Unable to process request"
            )

    async def _handle_with_generate_flow(self, request_data: dict, stream: bool):
        """
        Cached mode: Use unified generate flow with cache support.

        Args:
            request_data: Parsed request data
            stream: Whether streaming is requested

        Returns:
            OpenAI-formatted response
        """
        messages = request_data.get("messages", [])
        tools = request_data.get("tools", None)

        # Step 1: Get tokens directly from router's component registry
        try:
            # Get radix tree and tokenizer from router's component registry
            radix_tree = None
            tokenizer = None

            if hasattr(self.router, 'component_registry'):
                if self.router.component_registry.has("radix_tree"):
                    radix_tree = self.router.component_registry.get("radix_tree")
                if self.router.component_registry.has("tokenizer"):
                    tokenizer = self.router.component_registry.get("tokenizer")

            # Fallback to legacy router.radix_tree if available
            if not radix_tree and hasattr(self.router, 'radix_tree'):
                radix_tree = self.router.radix_tree
                if hasattr(self.router, 'component_registry') and self.router.component_registry.has("tokenizer"):
                    tokenizer = self.router.component_registry.get("tokenizer")

            if not radix_tree or not tokenizer:
                raise RuntimeError("Radix tree or tokenizer not available")

            # Convert messages to text using tokenizer.apply_chat_template
            text = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=False
            )

            if not text or not text.strip():
                raise RuntimeError("Messages template resulted in empty text")

            # Get tokenization from radix tree
            generation_tokens, _, _, _ = await radix_tree.get_or_create_tokenization_async(text)

            if not generation_tokens:
                raise RuntimeError("Failed to get tokens from radix tree")

        except Exception as e:
            if getattr(self.args, 'verbose', False):
                print(f"[slime-router] Warning: Failed to get cached tokens, falling back to direct mode: {e}")
            # Fallback to direct proxy
            return await self._proxy_to_sglang_chat_from_data(request_data)

        # Step 2: Construct generate request for SGLang compatibility
        # This request will be processed by RadixTreeMiddleware before reaching SGLang workers
        generate_request = {
            "input_tokens": generation_tokens,  # Tokens from cache (prefix) + remaining text
            "sampling_params": self._build_sampling_params(request_data, stream)
        }

        # Step 3: Forward to router's /generate endpoint
        # The processing pipeline is:
        # 1. RadixTreeMiddleware.query_cache_by_text() - cache lookup
        # 2. Forward to SGLang worker with cache context
        # 3. RadixTreeMiddleware.dispatch() - cache insertion
        # 4. Return response with accurate token counts
        if stream:
            return await self._stream_generate_response(generate_request)
        else:
            return await self._non_stream_generate_response(generate_request)

    def _build_sampling_params(self, request_data: dict, stream: bool) -> dict:
        """
        Build sampling parameters for SGLang generation request.

        Args:
            request_data: Parsed request data from Chat Completion API
            stream: Whether streaming is requested

        Returns:
            Dictionary of sampling parameters compatible with SGLang
        """
        sampling_params = {
            # Core generation parameters
            "max_new_tokens": request_data.get("max_tokens", 1024),
            "temperature": request_data.get("temperature", 1.0),
            "top_p": request_data.get("top_p", 1.0),
            "top_k": request_data.get("top_k", -1),
            "min_p": request_data.get("min_p", 0.0),

            # Penalty parameters
            "frequency_penalty": request_data.get("frequency_penalty", 0.0),
            "presence_penalty": request_data.get("presence_penalty", 0.0),

            # Stop conditions
            "stop": request_data.get("stop"),
            "stop_token_ids": request_data.get("stop_token_ids"),
            "ignore_eos": request_data.get("ignore_eos"),

            # Special token handling
            "skip_special_tokens": request_data.get("skip_special_tokens"),
            "spaces_between_special_tokens": request_data.get("spaces_between_special_tokens"),
            "no_stop_trim": request_data.get("no_stop_trim"),

            # Streaming flag
            "stream": stream
        }

        # Remove None values to keep request clean and avoid SGLang errors
        return {k: v for k, v in sampling_params.items() if v is not None}

    async def _stream_generate_response(self, generate_request: dict):
        """
        Handle streaming generate response and convert to OpenAI Server-Sent Events format.

        This method:
        1. Sends request to router's /generate endpoint (processed by RadixTreeMiddleware)
        2. Processes SGLang's SSE response format
        3. Converts to OpenAI-compatible chat.completion.chunk format
        4. Handles errors gracefully with proper resource cleanup

        Args:
            generate_request: Generate API request with input_tokens and sampling_params

        Returns:
            StreamingResponse: OpenAI-formatted SSE stream with chat.completion.chunk objects
        """
        # Create streaming request to /generate endpoint
        port = getattr(self.args, 'sglang_router_port', None) or getattr(self.args, 'port', 30000)
        router_url = f"http://localhost:{port}/generate"

        async def generate_openai_chunks():
            """Generate OpenAI-formatted SSE chunks."""
            request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            created_time = int(time.time())

            response = None
            try:
                # Send to router's /generate endpoint
                response = await self.router.client.stream(
                    "POST",
                    router_url,
                    json=generate_request,
                    timeout=None
                )

                async with response:
                    accumulated_text = ""

                    async for line in response.aiter_lines():
                        if line.strip() and line.startswith("data: "):
                            chunk_data = line[6:]  # Remove "data: " prefix
                            if chunk_data == "[DONE]":
                                break

                            try:
                                chunk = json.loads(chunk_data)
                                if chunk.get("text"):
                                    accumulated_text += chunk.get("text", "")

                                    # OpenAI format chunk
                                    openai_chunk = {
                                        "id": request_id,
                                        "object": "chat.completion.chunk",
                                        "created": created_time,
                                        "model": getattr(self.args, 'model_name', 'slime-model'),
                                        "choices": [{
                                            "index": 0,
                                            "delta": {
                                                "content": chunk.get("text", "")
                                            },
                                            "finish_reason": None
                                        }]
                                    }
                                    yield f"data: {json.dumps(openai_chunk)}\n\n"
                            except json.JSONDecodeError:
                                continue

                    # Final chunk
                    final_chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": getattr(self.args, 'model_name', 'slime-model'),
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

            except Exception as e:
                # Log error but ensure connection is closed
                if getattr(self.args, 'verbose', False):
                    print(f"[slime-router] Streaming error: {e}")

                # Send error chunk to client
                error_chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": getattr(self.args, 'model_name', 'slime-model'),
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "error"
                    }],
                    "error": {
                        "message": "Streaming interrupted",
                        "type": "internal_error"
                    }
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
                yield "data: [DONE]\n\n"

                # Re-raise to let FastAPI handle the error appropriately
                raise
            finally:
                # Ensure response is closed if it was opened
                if response is not None:
                    try:
                        await response.aclose()
                    except Exception:
                        pass  # Ignore cleanup errors

        return StreamingResponse(
            generate_openai_chunks(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache"}
        )

    async def _non_stream_generate_response(self, generate_request: dict):
        """
        Handle non-streaming generate response and convert to OpenAI chat.completion format.

        This method:
        1. Sends request to router's /generate endpoint (processed by RadixTreeMiddleware)
        2. Processes SGLang's JSON response
        3. Extracts accurate token counts from SGLang response
        4. Converts to OpenAI-compatible chat.completion format
        5. Provides proper error handling and timeout management

        Args:
            generate_request: Generate API request with input_tokens and sampling_params

        Returns:
            JSONResponse: OpenAI-formatted chat.completion response with accurate token usage

        Raises:
            HTTPException: Various error conditions (timeout, connection issues, etc.)
        """
        # Send to router's /generate endpoint
        port = getattr(self.args, 'sglang_router_port', None) or getattr(self.args, 'port', 30000)
        router_url = f"http://localhost:{port}/generate"

        try:
            response = await self.router.client.post(
                router_url,
                json=generate_request,
                timeout=httpx.Timeout(30.0)  # 30 second timeout
            )
            response.raise_for_status()
        except httpx.TimeoutException:
            raise HTTPException(
                status_code=504,
                detail="Request timeout while calling generate endpoint"
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Generate endpoint error: {e.response.text}"
            )
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to connect to generate endpoint: {str(e)}"
            )

        try:
            generate_data = response.json()
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=502,
                detail=f"Invalid JSON response from generate endpoint: {str(e)}"
            )
        generated_text = generate_data.get("text", "")

        # Calculate accurate token counts from SGLang response
        prompt_tokens = len(generate_request["input_tokens"]) if generate_request.get("input_tokens") else 0

        # Use actual token counts from SGLang response if available
        if "output_ids" in generate_data:
            completion_tokens = len(generate_data["output_ids"])
        elif "meta_info" in generate_data and "output_token_logprobs" in generate_data["meta_info"]:
            completion_tokens = len(generate_data["meta_info"]["output_token_logprobs"])
        else:
            # Fallback to text-based estimation (less accurate)
            completion_tokens = len(generated_text.split()) if generated_text else 0

        total_tokens = prompt_tokens + completion_tokens

        # Convert to OpenAI format
        openai_response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": getattr(self.args, 'model_name', 'slime-model'),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        }

        return JSONResponse(content=openai_response)


# Factory function for creating ChatCompletion handlers
def create_chat_completion_handler(router) -> ChatCompletionHandler:
    """
    Factory function to create Chat Completion handler.

    Args:
        router: SlimeRouter instance

    Returns:
        Configured Chat Completion handler
    """
    return ChatCompletionHandler(router)