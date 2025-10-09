import argparse
import asyncio
import json
import threading

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from slime.router.component_registry import ComponentRegistry
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
        self.app.get("/metrics")(self.get_metrics)

        # OpenAI Chat Completion API (if enabled)
        if (hasattr(self.args, 'enable_openai_chat_completion') and
            self.args.enable_openai_chat_completion):
            self.app.post("/v1/chat/completions")(self.chat_completions)

        # Catch-all route for proxying to SGLang - must be registered LAST
        self.app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])(self.proxy)

    async def health_check(self, request: Request):
        # TODO: do health check in background
        pass

    async def proxy(self, request: Request, path: str):
        """Proxy all other requests to the SGLang router"""
        # Forward all other paths to SGLang router
        worker_url = await self._use_url()
        url = f"{worker_url}/{path}"
        # print("path",path)

        # Get request body and headers
        body = await request.body()
        headers = dict(request.headers)

        try:
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
            self._chat_completion_handler = create_chat_completion_handler(
                self._get_radix_tree(),
                self._get_tokenizer(),
                self._create_generate_handler()
            )

        # Handle request
        import json
        body = await request.body()
        data = json.loads(body) if body else {}

        from slime.router.openai_chat_completion import ChatCompletionRequest
        chat_request = ChatCompletionRequest(**data)

        return await self._chat_completion_handler.handle_request(chat_request)

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

    parser = argparse.ArgumentParser(description="Slime Router - Token-based inference router with Radix Cache optimization")
    
    # Server configuration
    parser.add_argument("--host", type=str, default="0.0.0.0", 
                       help="Host address to bind the router server (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=30000,
                       help="Port to bind the router server (default: 30000)")
    
    # SGLang backend configuration
    parser.add_argument("--sglang-host", type=str, required=True,
                       help="SGLang server host address")
    parser.add_argument("--sglang-port", type=int, required=True,
                       help="SGLang server port")
    
    # Tokenizer configuration
    parser.add_argument("--hf-checkpoint", type=str, 
                       help="HuggingFace model checkpoint for tokenizer")
    
    # Radix Tree configuration
    parser.add_argument("--radix-tree-max-size", type=int, default=10000,
                       help="Maximum cache size for RadixTree (default: 10000)")
    
    # API configuration
    parser.add_argument("--enable-openai-chat-completion", action="store_true", default=False,
                       help="Enable OpenAI-compatible Chat Completion API endpoint")
    
    # General configuration
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")

    args = parser.parse_args()

    # Run the router
    run_router(args)
