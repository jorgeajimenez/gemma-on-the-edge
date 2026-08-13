import os
import json
from flask import Flask, request, Response, jsonify
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

app = Flask(__name__)

# --- 1. Load & Cache Local LLM Weights ---
MODEL_PATH = "./gemma-2-2b-it-Q4_K_M.gguf"
if not os.path.exists(MODEL_PATH):
    print("Downloading Gemma-2B-IT GGUF model from Hugging Face...")
    MODEL_PATH = hf_hub_download(
        repo_id="bartowski/gemma-2-2b-it-GGUF",
        filename="gemma-2-2b-it-Q4_K_M.gguf",
        local_dir="."
    )
print(f"Model loaded and cached locally at: {MODEL_PATH}")

# --- 2. Initialize the Inference Engine ---
print("Loading model into memory...")
n_gpu_layers = int(os.environ.get("N_GPU_LAYERS", "-1"))
print(f"Configuring Llama with n_gpu_layers={n_gpu_layers}")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=512,      # Maximum context window size
    n_threads=4,    # CPU threads to allocate
    n_gpu_layers=n_gpu_layers, # GPU layer offloading (-1 for all layers)
    verbose=False   # Suppress internal llama.cpp logs
)

# --- 3. API Endpoints ---

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint to verify the server and model are ready."""
    return jsonify({
        "status": "healthy",
        "model": "gemma-2-2b-it-Q4_K_M.gguf",
        "context_window": 512
    })

@app.route("/api/generate", methods=["POST"])
def generate():
    """
    Standard generation endpoint.
    Accepts:
      {
        "prompt": "Q: Tell me a fact...\nA:",
        "temperature": 0.7,
        "max_tokens": 128,
        "stream": false
      }
    """
    data = request.json or {}
    prompt = data.get("prompt", "Q: Tell me a fact about Sunnyvale, CA. \nA:")
    temperature = data.get("temperature", 0.0)  # Default to deterministic greedy decoding
    top_p = data.get("top_p", 0.9)
    max_tokens = data.get("max_tokens", 128)
    stream = data.get("stream", False)

    if stream:
        def stream_response():
            response_stream = llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True
            )
            for chunk in response_stream:
                token_text = chunk["choices"][0]["text"]
                # Use Server-Sent Events (SSE) format
                yield f"data: {json.dumps({'text': token_text})}\n\n"
            yield "data: [DONE]\n\n"

        return Response(stream_response(), content_type="text/event-stream")
    else:
        response = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False
        )
        return jsonify({
            "text": response["choices"][0]["text"],
            "usage": response.get("usage", {})
        })


@app.route("/v1/chat/completions", methods=["POST"])
def openai_chat_completions():
    """
    OpenAI-compatible Chat Completions endpoint.
    Allows standard client libraries (like openai-python) or dev tools (like Cursor/Continue)
    to connect directly to this local edge server.
    """
    data = request.json or {}
    messages = data.get("messages", [])
    temperature = data.get("temperature", 0.0)
    top_p = data.get("top_p", 0.9)
    max_tokens = data.get("max_tokens", 128)
    stream = data.get("stream", False)

    # Convert chat history into Gemma instructions template
    # Format: <start_of_turn>user\n{message}<end_of_turn>\n<start_of_turn>model\n
    prompt = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        prompt += f"<start_of_turn>{role}\n{content}<end_of_turn>\n"
    prompt += "<start_of_turn>model\n"

    if stream:
        def stream_chat():
            response_stream = llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True
            )
            for chunk in response_stream:
                token_text = chunk["choices"][0]["text"]
                sse_data = {
                    "choices": [{
                        "delta": {"content": token_text},
                        "finish_reason": None,
                        "index": 0
                    }]
                }
                yield f"data: {json.dumps(sse_data)}\n\n"
            
            # Final completion chunk
            sse_final = {
                "choices": [{
                    "delta": {},
                    "finish_reason": "stop",
                    "index": 0
                }]
            }
            yield f"data: {json.dumps(sse_final)}\n\n"
            yield "data: [DONE]\n\n"

        return Response(stream_chat(), content_type="text/event-stream")
    else:
        response = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False
        )
        return jsonify({
            "id": "chatcmpl-local",
            "object": "chat.completion",
            "model": "gemma-2-2b-it-Q4_K_M.gguf",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response["choices"][0]["text"]
                },
                "finish_reason": "stop"
            }],
            "usage": response.get("usage", {})
        })


if __name__ == "__main__":
    # Host on 0.0.0.0 to allow other devices on the same local network to query this machine
    app.run(host="0.0.0.0", port=5002)
