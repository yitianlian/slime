import argparse
import asyncio
import json
import threading

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.responses import Response

from slime.router.component_registry import ComponentRegistry
from slime.router.middleware_hub.radix_tree_middleware import _filter_headers
from slime.utils.misc import load_function


def run_router(args):
    """
    Run the Slime router with the specified configuration.
    """
    # Initialize the router with tokenizer and lazy worker initialization
    slime_router = SlimeRouter(args, verbose=args.verbose)

    # Start the server
    uvicorn.run(slime_router.app, host=args.sglang_router_ip, port=args.sglang_router_port, log_level="info")


class SlimeRouter:
    def __init__(self, args, verbose=False):
        """Initialize the slime-router with SGLang router address"""
        self.args = args
        self.verbose = verbose

        self.app = FastAPI()

        # Initialize lazy component registry for dependency injection
        self._component_registry = None
        self._registry_lock = threading.Lock()

        # Worker information
        self.worker_urls: dict[str, int] = {}
        self.max_weight_version = None

        # Concurrency control for worker URL selection
        self._url_lock = asyncio.Lock()

        # TODO: remove this hardcode
        self.client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=args.sglang_server_concurrency
                * args.rollout_num_gpus
                // args.rollout_num_gpus_per_engine
            ),
            timeout=httpx.Timeout(None),
        )

        self._setup_routes()

        for middleware_path in args.slime_router_middleware_paths or []:
            if self.verbose:
                print(f"[slime-router] Loading middleware from: {middleware_path}")
            middleware = load_function(middleware_path)
            self.app.add_middleware(middleware, router=self)

    def _setup_routes(self):
        """Setup all the HTTP routes"""
        # sglang-router api
        self.app.post("/add_worker")(self.add_worker)
        self.app.get("/list_workers")(self.list_workers)
        self.app.post("/retrieve_from_text")(self.retrieve_from_text)
        self.app.post("/retrieve_from_messages_template")(self.retrieve_from_messages_template)
        self.app.get("/metrics")(self.get_metrics)

        # Generate API endpoint - this MUST be registered before the catch-all route
        # to ensure it's handled by middleware chain, not by proxy
        self.app.post("/generate")(self.generate)
        self.app.get("/generate")(self.generate)

        # OpenAI Chat Completion API (if enabled)
        if (hasattr(self.args, 'enable_openai_chat_completion') and
            self.args.enable_openai_chat_completion):
            self.app.post("/v1/chat/completions")(self.chat_completions)

        # Catch-all route for proxying to SGLang - must be registered LAST
        self.app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])(self.proxy)

    async def generate(self, request: Request):
        """
        Generate API endpoint that forwards to SGLang workers.

        This endpoint is processed by the middleware chain (including RadixTreeMiddleware)
        before being forwarded to SGLang workers. This enables caching for all
        generate requests, including those from Chat Completion API.
        """
        # Forward to SGLang worker using existing proxy logic
        return await self.proxy(request, "generate")

    async def health_check(self, request: Request):
        # TODO: do health check in background
        pass

    async def proxy(self, request: Request, path: str):
        """Proxy all other requests to the SGLang router with streaming support"""
        # Forward all other paths to SGLang router
        worker_url = await self._use_url()
        url = f"{worker_url}/{path}"

        # Get request body and headers
        body = await request.body()
        headers = dict(request.headers)

        try:
            # Check if this is a streaming request
            try:
                request_data = json.loads(body) if body else {}
                is_streaming = request_data.get("stream", False)
            except (json.JSONDecodeError, TypeError):
                # If JSON parsing fails, treat as non-streaming
                is_streaming = False

            if is_streaming:
                # Streaming proxy: forward SSE stream directly
                response = None
                try:
                    response = await self.client.stream(
                        request.method,
                        url,
                        content=body,
                        headers=headers,
                        timeout=httpx.Timeout(None)  # No timeout for streaming
                    )

                    async with response:
                        async def generate_chunks():
                            try:
                                async for chunk in response.aiter_bytes():
                                    yield chunk
                            except Exception:
                                # If iteration fails, try to close connection
                                pass

                        # Filter headers for streaming response
                        filtered_headers = _filter_headers(response.headers)
                        return StreamingResponse(
                            generate_chunks(),
                            status_code=response.status_code,
                            headers=filtered_headers
                        )
                except Exception as e:
                    # Log streaming errors if verbose
                    if self.verbose:
                        print(f"[slime-router] Streaming proxy error: {e}")
                    # Ensure response is closed if it was opened
                    if response is not None:
                        try:
                            await response.aclose()
                        except Exception:
                            pass
                    # Re-raise to let FastAPI handle the error
                    raise
            else:
                # Non-streaming: original logic
                response = await self.client.request(request.method, url, content=body, headers=headers)
                # Eagerly read content so we can return JSON (not streaming)
                content = await response.aread()
                content_type = response.headers.get("content-type", "")
                try:
                    # Prefer parsing JSON if possible
                    data = json.loads(content)
                    return JSONResponse(
                        content=data,
                        status_code=response.status_code,
                        headers=dict(response.headers),
                    )
                except Exception as e:
                    # Phase 2 TODO: Implement secure exception handling with proper error categorization
                    # Currently falls back to raw body, should implement:
                    # - Malformed JSON detection and specific error messages
                    # - Content-type validation with security checks
                    # - Structured error responses with error codes
                    # - Rate limiting for error responses
                    # - Security: prevent error information leakage
                    return Response(
                        content=content,
                        status_code=response.status_code,
                        headers=dict(response.headers),
                        media_type=content_type or None,
                    )

        finally:
            await self._finish_url(worker_url)

    async def add_worker(self, request: Request):
        """Add a new worker to the router.
        Supports providing the URL via query string or JSON body.
        Examples:
        - POST /add_worker?url=http://127.0.0.1:10090
        - POST /add_worker  with body {"url": "http://127.0.0.1:10090"}
        """
        # 1) Prefer query param
        worker_url = request.query_params.get("url") or request.query_params.get("worker_url")

        # 2) Fallback to JSON body
        if not worker_url:
            body = await request.body()
            payload = json.loads(body) if body else {}
            worker_url = payload.get("url") or payload.get("worker_url")

        if not worker_url:
            return JSONResponse(
                status_code=400, content={"error": "worker_url is required (use query ?url=... or JSON body)"}
            )

        # Add if new, keep a simple request count per worker
        if worker_url not in self.worker_urls:
            self.worker_urls[worker_url] = 0
            if self.verbose:
                print(f"[slime-router] Added new worker: {worker_url}")

        return {"status": "success", "worker_urls": self.worker_urls}

    async def list_workers(self, request: Request):
        """List all registered workers"""
        return {"urls": list(self.worker_urls.keys())}

    async def get_metrics(self, request: Request):
        """GET /metrics - Return router and cache metrics."""
        metrics = {
            "router": {
                "active_workers": len(self.worker_urls),
                "worker_loads": dict(self.worker_urls),
                "total_in_flight": sum(self.worker_urls.values()),
            }
        }

        # Get radix tree from component registry (preferred) or fallback to attribute
        if hasattr(self, 'component_registry') and self.component_registry.has("radix_tree"):
            radix_tree = self.component_registry.get("radix_tree")
        elif hasattr(self, 'radix_tree'):
            radix_tree = self.radix_tree
        else:
            # No radix tree available, skip cache metrics
            return JSONResponse(content=metrics)

        # Phase 1 TODO: migrate to async get_stats() when radix_tree is fully async
        # For now, use sync version but this blocks the event loop briefly
        cache_stats = radix_tree.get_stats()
        # Estimate memory usage (16 bytes per token ID)
        cache_stats["cache_size_mb"] = (
            cache_stats["cur_cache_size"] * 16 / 1024 / 1024
        )
        metrics["cache"] = cache_stats

        return JSONResponse(content=metrics)

    async def retrieve_from_text(self, request: Request):
        """Get token information from text input"""
        body = await request.body()
        payload = json.loads(body) if body else {}

        text = payload.get("text", "")

        # Get radix tree from component registry (preferred) or fallback to attribute
        if hasattr(self, 'component_registry') and self.component_registry.has("radix_tree"):
            radix_tree = self.component_registry.get("radix_tree")
        elif hasattr(self, 'radix_tree'):
            radix_tree = self.radix_tree
        else:
            raise RuntimeError(
                "Radix tree not available. Please ensure RadixTreeMiddleware is properly initialized."
            )

        # Use radix tree's retrieve_from_text method (no need to fetch weight version here)
        token_ids, logp, loss_mask = radix_tree.retrieve_from_text(text, return_logprob=True)

        # Handle the result based on whether logp was requested
        result = {
            "tokens": token_ids,  # token IDs
            "response": text,  # The input text
            "loss_mask": loss_mask,  # Loss mask for the tokens
            "token_length": len(token_ids),
            "loss_mask_length": len(loss_mask),
            "rollout_logp": logp,
        }

        # # Add logp to response if requested
        # if sum(logp) > 0:
        #     result["rollout_logp"] = logp

        return result

    async def retrieve_from_messages_template(self, request: Request):
        """Get token information from messages template using apply_chat_template"""
        body = await request.body()
        payload = json.loads(body) if body else {}

        messages = payload.get("messages", [])
        tools = payload.get("tools", None)
        add_generation_prompt = payload.get("add_generation_prompt", True)

        # Get radix tree and tokenizer using existing pattern (similar to retrieve_from_text)
        radix_tree = None
        tokenizer = None

        # Try component registry first
        if hasattr(self, 'component_registry') and self.component_registry.has("radix_tree"):
            radix_tree = self.component_registry.get("radix_tree")
            if self.component_registry.has("tokenizer"):
                tokenizer = self.component_registry.get("tokenizer")
        elif hasattr(self, 'radix_tree'):
            radix_tree = self.radix_tree
            # Try to get tokenizer from registry as fallback
            if hasattr(self, 'component_registry') and self.component_registry.has("tokenizer"):
                tokenizer = self.component_registry.get("tokenizer")

        if not radix_tree or not tokenizer:
            raise RuntimeError(
                "Radix tree and tokenizer not available. Please ensure RadixTreeMiddleware is properly initialized."
            )

        # Convert messages to text using tokenizer.apply_chat_template
        # This logic is copied from RadixTreeMiddleware.query_cache_by_messages_template
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=add_generation_prompt,
                tokenize=False  # Return string, not tokenized
            )

            if not text or not text.strip():
                if self.verbose:
                    print(f"[slime-router] Warning: Messages template resulted in empty text")
                return {
                    "tokens": [],
                    "response": text,
                    "loss_mask": [],
                    "token_length": 0,
                    "loss_mask_length": 0,
                    "rollout_logp": [],
                    "generation_versions": [],
                }

            # Use radix tree's get_or_create_tokenization_async method
            token_ids, logp, loss_mask, generation_versions = await radix_tree.get_or_create_tokenization_async(text)

            result = {
                "tokens": token_ids,
                "response": text,
                "loss_mask": loss_mask,
                "token_length": len(token_ids),
                "loss_mask_length": len(loss_mask),
                "rollout_logp": logp,
                "generation_versions": generation_versions,
            }

            return result

        except Exception as e:
            if self.verbose:
                print(f"[slime-router] Warning: Messages template processing error: {e}")
            return {
                "tokens": [],
                "response": "",
                "loss_mask": [],
                "token_length": 0,
                "loss_mask_length": 0,
                "rollout_logp": [],
                "generation_versions": [],
            }

    async def _use_url(self):
        """
        Select a worker URL using round-robin strategy.

        Thread-safe: Uses asyncio.Lock to prevent race conditions when multiple
        concurrent requests select workers simultaneously.
        """
        async with self._url_lock:
            assert len(self.worker_urls) > 0, "No workers available"

            # get the url with mininal count
            url = min(self.worker_urls, key=self.worker_urls.get)
            self.worker_urls[url] += 1
            return url

    async def _finish_url(self, url):
        """
        Mark the request to the given URL as finished.

        Thread-safe: Uses asyncio.Lock to prevent race conditions when decrementing
        worker counts.
        """
        async with self._url_lock:
            assert url in self.worker_urls, f"URL {url} not recognized"
            self.worker_urls[url] -= 1
            assert self.worker_urls[url] >= 0, f"URL {url} count went negative"

    async def chat_completions(self, request: Request):
        """
        OpenAI Chat Completion API endpoint.

        Provides 100% OpenAI-compatible interface while internally using
        Slime Router's token-based inference with Radix Cache optimization.
        """
        # Lazy import to avoid circular dependency
        from slime.router.openai_chat_completion import create_chat_completion_handler

        # Initialize handler on first use
        if not hasattr(self, '_chat_completion_handler'):
            self._chat_completion_handler = create_chat_completion_handler(self)

        # Handle request
        return await self._chat_completion_handler.handle_request(request)

    def get_component_registry(self) -> ComponentRegistry:
        """
        Get or create the thread-safe component registry for this router.

        Uses double-checked locking pattern for thread-safe lazy initialization.
        Each router instance has its own registry to ensure isolation.

        Returns:
            ComponentRegistry: The component registry for this router
        """
        if self._component_registry is None:
            with self._registry_lock:
                # Double-check lock pattern
                if self._component_registry is None:
                    self._component_registry = ComponentRegistry()
        return self._component_registry

    @property
    def component_registry(self) -> ComponentRegistry:
        """
        Convenient property access to the component registry.

        This provides backward compatibility with existing code that accesses
        self.component_registry directly.

        Returns:
            ComponentRegistry: The component registry for this router
        """
        return self.get_component_registry()

    def cleanup_components(self) -> None:
        """
        Cleanup all registered components.

        This method is primarily used for testing purposes to reset the state.
        """
        if self._component_registry is not None:
            self._component_registry.clear()

    def _get_tokenizer(self):
        """Get tokenizer from component registry."""
        return self.get_component_registry().get("tokenizer")

    def _get_radix_tree(self):
        """Get radix tree from component registry."""
        return self.get_component_registry().get("radix_tree")


    def _create_generate_handler(self):
        """Create handler for /generate API calls."""
        async def generate_handler(request_data: dict):
            """Forward request to /generate API."""
            # Get a worker URL
            worker_url = await self._use_url()

            try:
                # Forward to generate API
                response = await self.client.post(
                    f"{worker_url}/generate",
                    json=request_data
                )
                response.raise_for_status()
                result = response.json()

                return result

            finally:
                await self._finish_url(worker_url)

        # Add streaming support
        async def stream_handler(request_data: dict):
            """Forward streaming request to /generate API."""
            # For now, return a simple mock streaming response
            # In production, this would forward to the actual streaming endpoint
            import json

            class MockStreamResponse:
                def __init__(self, text="Hello there!"):
                    self.text = text
                    self.words = text.split()

                def __aiter__(self):
                    return self

                async def __anext__(self):
                    if self.words:
                        word = self.words.pop(0)
                        return {"text": word + " ", "finished": False}
                    else:
                        return {"text": "", "finished": True, "finish_reason": "stop"}

            # Get the text to generate
            text = request_data.get("text", "Hello there!")
            return MockStreamResponse(text)

        generate_handler.stream = stream_handler
        return generate_handler


if __name__ == "__main__":
    import argparse

    import uvicorn

    from slime.utils.arguments import get_slime_extra_args_provider

    parser = argparse.ArgumentParser(description="Slime Router - Token-based inference router with Radix Cache optimization")

    # SGLang backend configuration
    parser.add_argument("--sglang-host", type=str, required=True,
                       help="SGLang server host address")
    parser.add_argument("--sglang-port", type=int, required=True,
                       help="SGLang server port")
    # SGLang router configuration
    parser.add_argument("--sglang-router-ip", type=str, default="0.0.0.0",
                       help="Alias for --host for backward compatibility")
    parser.add_argument("--sglang-router-port", type=int, default=30000,
                       help="Alias for --port for backward compatibility")

    # Tokenizer configuration
    parser.add_argument("--hf-checkpoint", type=str,
                       help="HuggingFace model checkpoint for tokenizer")

    # Import all router-related arguments from global configuration
    parser.add_argument(
            "--slime-router-middleware-paths",
            type=str,
            nargs="+",
            default="",
        )
    parser.add_argument(
        "--enable-openai-chat-completion",
        action="store_true",
        default=False,
        help="Enable OpenAI-compatible Chat Completion API endpoint",
    )
    parser.add_argument(
        "--radix-tree-max-size",
        type=int,
        default=10000,
        help="Maximum cache size for RadixTree (default: 10000)",
    )

    # Logging configuration
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")

    args = parser.parse_args()

    # Run the router
    run_router(args)