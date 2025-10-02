from time import sleep

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from transformers import AutoTokenizer

from .radix_tree import StringRadixTrie


class RadixTreeMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, router):
        super().__init__(app)
        self.router = router
        self.args = router.args
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
        self.radix_tree = StringRadixTrie(max_cache_size=10000, tokenizer=self.tokenizer, verbose=False)
        self.router.radix_tree = self.radix_tree

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path != "/generate":
            return await call_next(request)

        request_json = await request.json()
        input_text = request_json.pop("text", "")
        if not input_text:
            return await call_next(request)
        input_tokens, input_logprobs, input_loss_mask = self.radix_tree.retrieve_from_text(
            input_text, return_logprob=True
        )
        request_json["input_tokens"] = input_tokens
        request._json = request_json
        while _ in range(5):
            response = await call_next(request)
            if (
                "meta_info" in response
                and "finish_reason" in response["meta_info"]
                and response["meta_info"]["finish_reason"]["type"] != "abort"
            ):
                break
            sleep(30)

        if "text" in response and "output_ids" in response:
            generated_text = response["text"]
            full_text = input_text + generated_text
            if full_text:
                try:
                    if "output_token_logprobs" in response.get("meta_info", {}):
                        generated_token_logprobs = [item[0] for item in response["meta_info"]["output_token_logprobs"]]
                        generated_token_ids = [item[1] for item in response["meta_info"]["output_token_logprobs"]]
                        full_logprobs = input_logprobs + generated_token_logprobs
                        full_token_ids = input_tokens + generated_token_ids
                        full_loss_mask = input_loss_mask + [1] * len(generated_token_ids)
                        self.radix_tree.insert(
                            full_text,
                            full_token_ids,
                            full_logprobs,
                            full_loss_mask,
                            weight_version=response["meta_info"]["weight_version"],
                        )
                    else:
                        print("Warning: output token logprobs not in response")
                        generated_token_ids = self.tokenizer(generated_text, add_special_tokens=False)["input_ids"]
                        full_token_ids = input_tokens + generated_token_ids
                        full_loss_mask = input_loss_mask + [1] * len(generated_token_ids)
                        self.radix_tree.insert(
                            full_text,
                            full_token_ids,
                            None,
                            full_loss_mask,
                            weight_version=response["meta_info"]["weight_version"],
                        )

                    if getattr(self.router, "verbose", False):
                        print(f"[slime-router] Successfully cached trajectory with {len(full_token_ids)} tokens")
                except Exception as e:
                    if getattr(self.router, "verbose", False):
                        print(f"[slime-router] Warning: Failed to cache trajectory: {e}")
        return response
