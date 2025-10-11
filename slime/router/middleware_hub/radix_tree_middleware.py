import json
from typing import List, Dict, Optional

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from tenacity import AsyncRetrying, RetryError, stop_after_attempt, wait_fixed
from transformers import AutoTokenizer

from slime.router.component_registry import ComponentRegistry

from .radix_tree import StringRadixTrie


def _is_response_aborted(response_data: dict) -> bool:
    """Check if SGLang response indicates abort.

    Performs defensive checks to handle malformed responses gracefully.
    Returns False for any non-dict or missing/invalid nested fields.
    """
    if not isinstance(response_data, dict):
        return False

    meta_info = response_data.get("meta_info")
    if not isinstance(meta_info, dict):
        return False

    finish_reason = meta_info.get("finish_reason")
    if not isinstance(finish_reason, dict):
        return False

    return finish_reason.get("type") == "abort"


# Hop-by-hop headers that should not be forwarded
HOP_BY_HOP = {
    "content-length",
    "transfer-encoding",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "upgrade",
}


def _filter_headers(headers):
    """Filter out hop-by-hop headers that should not be forwarded."""
    return {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP}


async def _materialize_response(resp):
    """Convert streaming-like Response into a regular Response/JSONResponse safely."""
    # Collect all bytes from the streaming response
    body = b""
    async for chunk in resp.body_iterator:
        body += chunk

    # Try to parse as JSON based on content-type
    ct = resp.headers.get("content-type", "")
    headers = _filter_headers(resp.headers)

    if "application/json" in ct:
        # If it's JSON, try to parse and return as JSONResponse
        try:
            data = json.loads(body.decode("utf-8"))
            return JSONResponse(content=data, status_code=resp.status_code, headers=headers)
        except Exception:
            # JSON parsing failed, fall back to raw bytes
            pass

    # Other types: return as raw bytes (without content-length)
    return Response(content=body, status_code=resp.status_code, headers=headers, media_type=resp.media_type)


class RadixTreeMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, router, component_registry=None):
        super().__init__(app)
        self.router = router
        self.args = router.args

        # Determine which component registry to use
        if component_registry is not None:
            self.component_registry = component_registry
        else:
            # Use router's thread-safe component registry
            self.component_registry = router.get_component_registry()

        # Check required parameters
        if not hasattr(self.args, 'hf_checkpoint') or not self.args.hf_checkpoint:
            raise ValueError(
                "Missing required argument: --hf-checkpoint. "
                "Please specify the HuggingFace model checkpoint for tokenizer."
            )

        # Check if components are already registered to avoid duplicate initialization
        if self.component_registry.has("tokenizer") and self.component_registry.has("radix_tree"):
            # Use existing components
            self.tokenizer = self.component_registry.get("tokenizer")
            self.radix_tree = self.component_registry.get("radix_tree")
            if getattr(self.router, 'verbose', False) or getattr(self.args, 'verbose', False):
                print(f"[slime-router] RadixTreeMiddleware using existing components:")
                print(f"  - Component registry: existing")
        else:
            # Initialize components only if not already registered
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.args.hf_checkpoint,
                    trust_remote_code=True
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load tokenizer from checkpoint '{self.args.hf_checkpoint}': {e}"
                )

            # Create radix tree with unified verbose configuration
            router_verbose = getattr(self.router, 'verbose', False) or getattr(self.args, 'verbose', False)
            self.radix_tree = StringRadixTrie(
                max_cache_size=getattr(self.args, 'radix_tree_max_size', 10000),
                tokenizer=self.tokenizer,
                verbose=router_verbose
            )

            # Thread-safe component registration
            self.component_registry.register("tokenizer", self.tokenizer)
            self.component_registry.register("radix_tree", self.radix_tree)

            if getattr(self.router, 'verbose', False) or getattr(self.args, 'verbose', False):
                print(f"[slime-router] RadixTreeMiddleware initialized new components:")
                print(f"  - Tokenizer: {self.args.hf_checkpoint}")
                print(f"  - Max cache size: {getattr(self.args, 'radix_tree_max_size', 10000)}")
                print(f"  - Verbose: {router_verbose}")
                print(f"  - Component registry: created")

        # Maintain backward compatibility
        self.router.radix_tree = self.radix_tree

        if getattr(self.router, 'verbose', False) or getattr(self.args, 'verbose', False):
            print(f"[slime-router] RadixTreeMiddleware initialized:")
            print(f"  - Tokenizer: {self.args.hf_checkpoint}")
            print(f"  - Max cache size: {getattr(self.args, 'radix_tree_max_size', 10000)}")
            print(f"  - Verbose: {getattr(self.router, 'verbose', False) or getattr(self.args, 'verbose', False)}")
            print(f"  - Component registry: {'provided' if component_registry is not None else 'created'}")

    def _parse_response(self, response: Response) -> dict | None:
        """
        Extract response_data from FastAPI Response object.

        Handles both JSONResponse (content dict) and Response (body bytes).
        Returns None if parsing fails.

        Args:
            response: FastAPI Response object (JSONResponse or Response)

        Returns:
            dict | None: Parsed response data, or None if parsing fails
        """
        try:
            if hasattr(response, "body") and isinstance(response.body, (bytes, bytearray)):
                return json.loads(response.body.decode("utf-8"))
            elif hasattr(response, "content") and isinstance(response.content, dict):
                return response.content
        except Exception:
            pass
        return None

    async def query_cache_by_text(self, input_text: str) -> tuple:
        """
        Query cache by text input, equivalent to get_or_create_tokenization_async semantics.

        Args:
            input_text: Input text string to query from cache

        Returns:
            tuple: (token_ids, logp, loss_mask, generation_versions)
                   with format consistent to get_or_create_tokenization_async
                   Returns ([], [], [], []) on error
        """
        try:
            return await self.radix_tree.get_or_create_tokenization_async(input_text)
        except ValueError as e:
            # Phase 2 TODO: Implement structured error handling with security validation
            # Specific exception from radix_tree.py:618 - empty tokenizer or text
            # Should implement:
            # - Input validation and sanitization
            # - Error categorization (validation vs system errors)
            # - Security audit logging for suspicious inputs
            # - Rate limiting for repeated validation failures
            if getattr(self.router, "verbose", False):
                print(f"[slime-router] Warning: Text cache query validation error: {e}")
            return ([], [], [], [])
        except (AttributeError, KeyError) as e:
            # Phase 2 TODO: Implement secure exception handling for data structure errors
            # Data structure access errors could indicate:
            # - Corruption attacks
            # - Memory issues
            # - Race conditions
            # Should implement:
            # - Data integrity validation
            # - Automatic recovery mechanisms
            # - Security incident logging
            if getattr(self.router, "verbose", False):
                print(f"[slime-router] Warning: Text cache query data error: {e}")
            return ([], [], [], [])
        except Exception as e:
            # Phase 2 TODO: Implement comprehensive exception handling and monitoring
            # Catch-all for unexpected errors - should implement:
            # - Error classification and prioritization
            # - Circuit breaker pattern for cascade failures
            # - Detailed error logging with context
            # - Health check integration
            # - Graceful degradation strategies
            if getattr(self.router, "verbose", False):
                print(f"[slime-router] Warning: Unexpected text cache query error: {e}")
            return ([], [], [], [])

    async def query_cache_by_messages_template(
        self,
        messages: List[Dict],
        tools: Optional[List] = None,
        add_generation_prompt: bool = True
    ) -> tuple:
        """
        Query cache by messages template using tokenizer.apply_chat_template.

        This method converts OpenAI-format messages to text using tokenizer.apply_chat_template
        and then queries the radix tree cache, enabling semantic caching for chat completions.

        Args:
            messages: OpenAI format messages list, e.g., [{"role": "user", "content": "..."}]
            tools: Optional list of tool definitions for function calling
            add_generation_prompt: Whether to add generation prompt for model continuation
                - True: Prepare for model generation (default for Chat Completion)
                - False: For cache lookup only, no generation prompt

        Returns:
            tuple: (token_ids, logp, loss_mask, generation_versions)
                   with format consistent to get_or_create_tokenization_async
                   Returns ([], [], [], []) on error
        """
        try:
            # Convert messages to text using tokenizer.apply_chat_template
            # This enables semantic caching based on conversation content rather than raw text
            text = self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=add_generation_prompt,
                tokenize=False  # Return string, not tokenized
            )

            if not text or not text.strip():
                if getattr(self.router, "verbose", False):
                    print(f"[slime-router] Warning: Messages template resulted in empty text")
                return ([], [], [], [])

            # Reuse existing radix tree cache infrastructure
            return await self.radix_tree.get_or_create_tokenization_async(text)

        except Exception as e:
            if getattr(self.router, "verbose", False):
                print(f"[slime-router] Warning: Messages template cache query error: {e}")
            return ([], [], [], [])

    async def _generate_with_retry(
        self, request: Request, call_next
    ) -> tuple[Response, dict | None]:
        """
        Responsibility 2: Generation with retry on abort.

        Uses tenacity for robust retry logic:
        - Retries up to 5 times (stop_after_attempt)
        - Waits 30 seconds between retries (wait_fixed)
        - Raises exception on abort to trigger retry
        - Returns last response if all retries exhausted

        Returns:
            tuple[Response, dict | None]: FastAPI Response and parsed dict
        """
        # Variables to store last attempt result
        last_response = None
        last_response_data = None

        async def _single_attempt() -> tuple[Response, dict | None]:
            """
            Single generation attempt.

            Always updates last_response/last_response_data before checking abort.
            This ensures we have a response even when all retries are exhausted.
            """
            nonlocal last_response, last_response_data

            response = await call_next(request)

            # Materialize streaming response
            if response.__class__.__name__ == "_StreamingResponse":
                response = await _materialize_response(response)

            # Parse response
            response_data = self._parse_response(response)

            # Always save the response before potentially raising
            last_response = response
            last_response_data = response_data

            # Check if we should retry
            if response_data and _is_response_aborted(response_data):
                # Abort detected, signal retry needed
                raise Exception("SGLang abort - retry needed")

            return (response, response_data)

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(5),
                wait=wait_fixed(30),
                reraise=False,  # Don't raise RetryError
            ):
                with attempt:
                    await _single_attempt()
        except RetryError:
            # All retries exhausted, return last response
            pass

        return last_response, last_response_data

    async def _insert_cache(
        self,
        full_text: str,
        full_token_ids: list,
        full_logprobs: list,
        full_loss_mask: list,
        weight_version: int,
    ) -> bool:
        """Responsibility 3: Cache insertion.

        Uses async insert for better concurrency performance.

        Returns:
            bool: True if insertion was successful, False otherwise
        """
        try:
            # Use async insert for better concurrency
            result = await self.radix_tree.insert_async(
                full_text,
                full_token_ids,
                full_logprobs,
                full_loss_mask,
                weight_version=weight_version,
            )
            if getattr(self.router, "verbose", False):
                if result:
                    print(
                        f"[slime-router] Successfully cached trajectory with {len(full_token_ids)} tokens"
                    )
                else:
                    print(
                        f"[slime-router] Failed to cache trajectory with {len(full_token_ids)} tokens (empty text or validation failed)"
                    )
            return result
        except Exception as e:
            if getattr(self.router, "verbose", False):
                print(f"[slime-router] Warning: Failed to cache trajectory: {e}")
            return False

    async def dispatch(self, request: Request, call_next):
        """Main orchestration: Compose 3 responsibilities."""
        path = request.url.path

        if path != "/generate":
            return await call_next(request)

        # Parse request
        request_json = await request.json()
        if "text" in request_json:
            input_text = request_json.pop("text", "")
        elif "input_ids" in request_json:
            input_text = self.tokenizer.decode(request_json["input_ids"])
        else:
            input_text = None

        if not input_text:
            return await call_next(request)

        # Responsibility 1: Retrieve from cache
        input_tokens, input_logprobs, input_loss_mask, _ = await self.query_cache_by_text(
            input_text
        )

        # Modify request with cached tokens
        request_json["input_tokens"] = input_tokens
        request_json["stream"] = False
        request._json = request_json

        # Responsibility 2: Generate with retry
        response, response_data = await self._generate_with_retry(request, call_next)

        # Responsibility 3: Insert into cache
        if (
            isinstance(response_data, dict)
            and "text" in response_data
            and "output_ids" in response_data
        ):
            generated_text = response_data["text"]
            full_text = input_text + generated_text

            if full_text:
                if "output_token_logprobs" in response_data.get("meta_info", {}):
                    generated_token_logprobs = [
                        item[0]
                        for item in response_data["meta_info"]["output_token_logprobs"]
                    ]
                    generated_token_ids = [
                        item[1]
                        for item in response_data["meta_info"]["output_token_logprobs"]
                    ]
                    full_logprobs = input_logprobs + generated_token_logprobs
                    full_token_ids = input_tokens + generated_token_ids
                    full_loss_mask = input_loss_mask + [1] * len(generated_token_ids)
                else:
                    generated_token_ids = self.tokenizer(
                        generated_text, add_special_tokens=False
                    )["input_ids"]
                    full_token_ids = input_tokens + generated_token_ids
                    full_logprobs = [0.0] * len(full_token_ids)  # Default logprobs when not available
                    full_loss_mask = input_loss_mask + [1] * len(generated_token_ids)

                await self._insert_cache(
                    full_text,
                    full_token_ids,
                    full_logprobs,
                    full_loss_mask,
                    weight_version=response_data["meta_info"]["weight_version"],
                )

        return response
