"""
OpenAI Chat Completion API implementation for Slime Router.

This module provides 100% OpenAI-compatible Chat Completion API while leveraging
Slime Router's Radix Cache for optimal performance in multi-turn conversations.

Key Features:
- Full OpenAI API compatibility (text in/out)
- Internal token in/out via existing /generate API
- HuggingFace chat template integration
- Radix Cache integration for conversation prefix reuse
- Streaming and non-streaming support
- TDD-driven development approach

Architecture:
- Messages → apply_chat_template → formatted text → cache lookup → /generate API
- Cache combination: cached tokens + new tokens → OpenAI response format
- Single shared Radix Tree instance for all Chat Completion requests
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


@dataclass
class ChatCompletionRequest:
    """OpenAI Chat Completion API request format."""
    model: str
    messages: List[Dict[str, str]]
    stream: bool = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stop: Optional[Union[str, List[str]]] = None
    user: Optional[str] = None


@dataclass
class ChatCompletionChoice:
    """Single choice in Chat Completion response."""
    index: int
    message: Dict[str, str]
    finish_reason: Optional[str] = None


@dataclass
class ChatCompletionUsage:
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class ChatCompletionResponse:
    """OpenAI Chat Completion API response format."""
    id: str = field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = field(default_factory=lambda: int(time.time()))
    model: str = "slime-model"
    choices: List[ChatCompletionChoice] = field(default_factory=list)
    usage: Optional[ChatCompletionUsage] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": [
                {
                    "index": choice.index,
                    "message": choice.message,
                    "finish_reason": choice.finish_reason
                }
                for choice in self.choices
            ]
        }
        if self.usage:
            result["usage"] = {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens
            }
        return result


@dataclass
class ChatCompletionStreamDelta:
    """Delta content for streaming responses."""
    role: Optional[str] = None
    content: Optional[str] = None


@dataclass
class ChatCompletionStreamChoice:
    """Choice for streaming responses."""
    index: int
    delta: ChatCompletionStreamDelta
    finish_reason: Optional[str] = None


@dataclass
class ChatCompletionStreamResponse:
    """Streaming response chunk."""
    request_id: str
    chunk: Dict[str, Any]
    id: str = field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion.chunk"
    created: int = field(default_factory=lambda: int(time.time()))
    model: str = "slime-model"

    def to_sse_format(self) -> str:
        """Convert to Server-Sent Events format."""
        if self.chunk.get("finished", False):
            # Final chunk with finish reason
            data = {
                "id": self.id,
                "object": self.object,
                "created": self.created,
                "model": self.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": self.chunk.get("finish_reason", "stop")
                }]
            }
        else:
            # Content chunk
            data = {
                "id": self.id,
                "object": self.object,
                "created": self.created,
                "model": self.model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": self.chunk.get("text", "")
                    },
                    "finish_reason": None
                }]
            }

        return f"data: {json.dumps(data)}\n\n"


class ChatCompletionHandler:
    """
    Core Chat Completion handler implementing OpenAI API compatibility
    while leveraging Slime Router's Radix Cache for performance.
    """

    def __init__(self, radix_tree, tokenizer, generate_api_handler):
        """
        Initialize Chat Completion handler.

        Args:
            radix_tree: Shared Radix Tree instance for caching
            tokenizer: HuggingFace tokenizer with chat template support
            generate_api_handler: Handler for /generate API calls
        """
        self.radix_tree = radix_tree
        self.tokenizer = tokenizer
        self.generate_api_handler = generate_api_handler

    async def handle_request(self, request: ChatCompletionRequest) -> Union[Dict[str, Any], StreamingResponse]:
        """
        Handle Chat Completion request.

        Args:
            request: Chat Completion request

        Returns:
            Either JSON response (non-streaming) or StreamingResponse
        """
        # Validate request
        if not validate_chat_completion_request(request):
            raise HTTPException(status_code=400, detail="Invalid request parameters")

        # Format messages using HuggingFace chat template
        formatted_text = format_messages_with_hf_template(request.messages, self.tokenizer)

        if request.stream:
            return StreamingResponse(
                self._handle_streaming_request(request, formatted_text),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache"}
            )
        else:
            return await self._handle_non_streaming_request(request, formatted_text)

    async def _handle_non_streaming_request(self, request: ChatCompletionRequest, formatted_text: str) -> Dict[str, Any]:
        """Handle non-streaming Chat Completion request."""
        # Query cache for prefix
        cached_result = await self._query_cache(formatted_text)

        # Prepare generate API request
        generate_request = self._prepare_generate_request(request, formatted_text, cached_result)

        # Call generate API
        generate_response = await self.generate_api_handler(generate_request)

        # Extract generated content
        generated_text = generate_response.get("output_text", "")
        generated_tokens = generate_response.get("output_token_ids", [])

        # Update cache with complete conversation
        await self._update_cache(formatted_text, generated_text, generated_tokens)

        # Convert to OpenAI format
        openai_response = convert_generate_to_openai_response(
            generate_response, cached_result.token_ids, request.messages, self.tokenizer
        )

        return openai_response.to_dict()

    async def _handle_streaming_request(self, request: ChatCompletionRequest, formatted_text: str) -> AsyncGenerator[str, None]:
        """Handle streaming Chat Completion request."""
        # Query cache for prefix
        cached_result = await self._query_cache(formatted_text)

        # Prepare generate API request
        generate_request = self._prepare_generate_request(request, formatted_text, cached_result)

        # Create stream handler
        stream_response = ChatCompletionStreamResponse(request.id, {})

        # Accumulate generated content for cache update
        generated_content = []

        # Stream generation
        async for chunk in self.generate_api_handler.stream(generate_request):
            if chunk.get("text"):
                generated_content.append(chunk.get("text", ""))
                stream_response.chunk = chunk
                yield stream_response.to_sse_format()

        # Final chunk
        stream_response.chunk = {"finished": True, "finish_reason": "stop"}
        yield stream_response.to_sse_format()

        # Update cache with complete conversation
        generated_text = "".join(generated_content)
        if generated_text:
            await self._update_cache_safe(formatted_text, generated_text)

    async def _query_cache(self, formatted_text: str):
        """Query Radix Cache for conversation prefix."""
        try:
            return await self.radix_tree.find_longest_prefix_async(formatted_text)
        except Exception as e:
            logger.error(f"Cache query error: {e}")
            # Return empty result on cache error
            from slime.router.middleware_hub.radix_tree import MatchResult, StringTreeNode
            return MatchResult(
                matched_prefix="",
                token_ids=[],
                logp=[],
                loss_mask=[],
                remaining_string=formatted_text,
                last_node=StringTreeNode()
            )

    def _prepare_generate_request(self, request: ChatCompletionRequest, formatted_text: str, cached_result) -> Dict[str, Any]:
        """Prepare request for /generate API."""
        # Prepare sampling parameters
        sampling_params = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_new_tokens": request.max_tokens,
            "frequency_penalty": request.frequency_penalty,
            "presence_penalty": request.presence_penalty,
            "stop": request.stop,
        }

        # Remove None values
        sampling_params = {k: v for k, v in sampling_params.items() if v is not None}

        # Handle cached tokens and remaining text
        if cached_result.token_ids:
            # We have cached tokens, use them as starting point
            input_tokens = cached_result.token_ids
            remaining_text = cached_result.remaining_string

            # Tokenize the remaining text if needed
            if remaining_text.strip():
                try:
                    remaining_tokens = self.tokenizer.encode(remaining_text)
                    # Combine cached tokens with remaining tokens
                    input_tokens = input_tokens + remaining_tokens
                except Exception as e:
                    logger.error(f"Tokenization error for remaining text: {e}")
                    # Fallback: treat as text input
                    remaining_text = cached_result.remaining_string
            else:
                # No remaining text, just use cached tokens
                remaining_text = ""
        else:
            # No cached tokens, tokenize the full formatted text
            try:
                input_tokens = self.tokenizer.encode(formatted_text)
                remaining_text = ""
            except Exception as e:
                logger.error(f"Tokenization error for formatted text: {e}")
                # Fallback: use text input
                input_tokens = []
                remaining_text = formatted_text

        return {
            "input": input_tokens if input_tokens else None,  # None if empty
            "text": remaining_text if remaining_text else None,  # None if empty
            "sampling_params": sampling_params,
            "request_id": f"chatcomp-{uuid.uuid4().hex[:8]}",
            "stream": request.stream,
        }

    async def _update_cache(self, formatted_text: str, generated_text: str, generated_tokens: List[int]):
        """Update cache with complete conversation."""
        try:
            complete_text = formatted_text + generated_text
            await self.radix_tree.insert_async(
                complete_text,
                generated_tokens,
                logp=[0.0] * len(generated_tokens),  # Default logp
                loss_mask=[1] * len(generated_tokens)  # Response tokens
            )
        except Exception as e:
            logger.warning(f"Cache update failed: {e}. Continuing without cache.")
            # Continue without failing the request

    async def _update_cache_safe(self, formatted_text: str, generated_text: str):
        """
        Safely update cache with proper tokenization fallback.

        Args:
            formatted_text: The formatted conversation prompt
            generated_text: The generated response text
        """
        try:
            # Calculate actual tokens using tokenizer
            generated_tokens = self.tokenizer.encode(generated_text)
            await self._update_cache(formatted_text, generated_text, generated_tokens)
        except Exception as e:
            # Log tokenization failure but continue without cache
            logger.error(f"Tokenization failed during cache update: {e}")
            # Continue without failing the request

def format_messages_with_hf_template(messages: List[Dict[str, str]], tokenizer) -> str:
    """
    Format messages using HuggingFace tokenizer's apply_chat_template.

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        tokenizer: HuggingFace tokenizer with chat template support

    Returns:
        Formatted conversation string

    Raises:
        ValueError: If messages list is empty or contains invalid roles
    """
    if not messages:
        raise ValueError("Messages list cannot be empty")

    # Validate message roles
    valid_roles = {"system", "user", "assistant"}
    for message in messages:
        role = message.get("role")
        if role not in valid_roles:
            raise ValueError(f"Invalid message role: {role}")

    try:
        # Use HuggingFace tokenizer's apply_chat_template
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return formatted
    except Exception as e:
        raise ValueError(f"Failed to format messages: {e}")


def convert_generate_to_openai_response(generate_response: Dict[str, Any],
                                      cached_tokens: List[int],
                                      messages: List[Dict[str, str]],
                                      tokenizer) -> ChatCompletionResponse:
    """
    Convert /generate API response to OpenAI Chat Completion format.

    Args:
        generate_response: Response from /generate API
        cached_tokens: Tokens retrieved from cache
        messages: Original messages from the request
        tokenizer: Tokenizer for accurate token counting

    Returns:
        OpenAI Chat Completion response
    """
    output_text = generate_response.get("output_text", "")
    output_tokens = generate_response.get("output_token_ids", [])

    # Calculate accurate token counts
    try:
        # Calculate prompt tokens by formatting messages and tokenizing
        formatted_prompt = format_messages_with_hf_template(messages, tokenizer)
        prompt_tokens = len(tokenizer.encode(formatted_prompt))

        # Calculate completion tokens
        if output_tokens:
            completion_tokens = len(output_tokens)
        else:
            completion_tokens = len(tokenizer.encode(output_text))
    except Exception as e:
        logger.error(f"Token calculation error: {e}")
        # Fallback to simple counting
        prompt_tokens = len(cached_tokens)
        completion_tokens = len(output_tokens) if output_tokens else len(output_text.split())

    # Create response
    response = ChatCompletionResponse(
        choices=[
            ChatCompletionChoice(
                index=0,
                message={
                    "role": "assistant",
                    "content": output_text
                },
                finish_reason="stop"
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
    )

    return response


def validate_chat_completion_request(request: ChatCompletionRequest) -> bool:
    """
    Validate Chat Completion request parameters.

    Args:
        request: Chat Completion request

    Returns:
        True if valid, False otherwise
    """
    # Validate required fields
    if not request.model:
        return False

    if not request.messages:
        return False

    # Validate parameter ranges
    if request.temperature is not None and (request.temperature < 0.0 or request.temperature > 2.0):
        return False

    if request.top_p is not None and (request.top_p < 0.0 or request.top_p > 1.0):
        return False

    if request.max_tokens is not None and request.max_tokens <= 0:
        return False

    if request.frequency_penalty is not None and (request.frequency_penalty < -2.0 or request.frequency_penalty > 2.0):
        return False

    if request.presence_penalty is not None and (request.presence_penalty < -2.0 or request.presence_penalty > 2.0):
        return False

    return True


def create_chat_completion_handler(radix_tree, tokenizer, generate_api_handler) -> ChatCompletionHandler:
    """
    Factory function to create Chat Completion handler.

    Args:
        radix_tree: Shared Radix Tree instance
        tokenizer: HuggingFace tokenizer
        generate_api_handler: /generate API handler

    Returns:
        Configured Chat Completion handler
    """
    return ChatCompletionHandler(radix_tree, tokenizer, generate_api_handler)