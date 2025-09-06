import asyncio
import json
from typing import Dict, Any, Optional, List
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Import radix tree
from .radix_tree import StringRadixTrie


class SlimeRouter:
    def __init__(self, sglang_host: str, sglang_port: int):
        """Initialize the SlimeRouter with SGLang router address"""
        self.sglang_host = sglang_host
        self.sglang_port = sglang_port
        self.sglang_router_url = f"http://{sglang_host}:{sglang_port}"
        self.app = FastAPI()
        
        # Initialize radix tree for caching
        self.radix_tree = StringRadixTrie(max_cache_size=10000)
        
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup all the HTTP routes"""
        # IMPORTANT: Register specific routes BEFORE the catch-all route
        self.app.post("/generate")(self.generate)
        self.app.post("/get_token_from_text")(self.get_token_from_text)
        # Catch-all route for proxying to SGLang - must be registered LAST
        self.app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])(self.proxy)
        
    async def generate(self, request: Request):
        """Wrapper for SGLang router's /generate endpoint"""
        # Get the request body
        body = await request.body()
        payload = json.loads(body) if body else {}
        
        # Extract text from payload for radix tree operations
        input_text = payload.get("text", "")
        
        # Forward request to SGLang router
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Modify the payload to use input_ids instead of text for token-in token-out
                sglang_payload = payload.copy()
                if input_text:
                    # Get tokens for the input text from radix tree
                    match_result = self.radix_tree.find_longest_prefix(input_text) if input_text else None
                    input_tokens = match_result.token_ids if match_result else []
                    # Replace "text" with "input_ids" 
                    sglang_payload.pop("text", None)
                    sglang_payload["input_ids"] = input_tokens
                
                response = await client.post(
                    f"{self.sglang_router_url}/generate",
                    json=sglang_payload
                )
                response_data = response.json()
                
                # Extract data for radix tree insertion
                if "text" in response_data and "output_ids" in response_data:
                    generated_text = response_data["text"]
                    generated_token_ids = response_data["output_ids"]
                    
                    # Combine input tokens and generated tokens
                    full_text = input_text + generated_text
                    full_token_ids = input_tokens + generated_token_ids
                    
                    # Insert the full trajectory into radix tree
                    if full_text and full_token_ids:
                        # Use default log probabilities (0.0) if not provided
                        self.radix_tree.insert(full_text, full_token_ids)
                
                return response_data
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={"error": str(e)}
                )
    
    async def proxy(self, request: Request, path: str):
        """Proxy all other requests to the SGLang router"""
        # Forward all other paths to SGLang router
        url = f"{self.sglang_router_url}/{path}"
        
        # Get request body and headers
        body = await request.body()
        headers = dict(request.headers)
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                if request.method == "GET":
                    response = await client.get(url, headers=headers)
                elif request.method == "POST":
                    response = await client.post(url, content=body, headers=headers)
                elif request.method == "PUT":
                    response = await client.put(url, content=body, headers=headers)
                elif request.method == "DELETE":
                    response = await client.delete(url, headers=headers)
                else:
                    return JSONResponse(
                        status_code=405,
                        content={"error": "Method not allowed"}
                    )
                
                # Try to return JSON response, fallback to text
                try:
                    content = response.json()
                except:
                    content = response.text
                    
                return JSONResponse(
                    status_code=response.status_code,
                    content=content
                )
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={"error": str(e)}
                )


if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--sglang-host", type=str, required=True)
    parser.add_argument("--sglang-port", type=int, required=True)
    
    args = parser.parse_args()
    
    # Initialize the router
    slime_router = SlimeRouter(args.sglang_host, args.sglang_port)
    
    # Start the server
    uvicorn.run(slime_router.app, host=args.host, port=args.port)