import os
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# ANSI escape codes for beautiful terminal formatting
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"

# --- 1. Load & Cache Local LLM Weights ---
model_path = "./gemma-2-2b-it-Q4_K_M.gguf"
if not os.path.exists(model_path):
    print(f"{YELLOW}Downloading Gemma-2B-IT GGUF model from Hugging Face...{RESET}")
    model_path = hf_hub_download(
        repo_id="bartowski/gemma-2-2b-it-GGUF",
        filename="gemma-2-2b-it-Q4_K_M.gguf",
        local_dir="."
    )
print(f"{GREEN}{BOLD}✓{RESET} {GREEN}Model loaded and cached locally at: {model_path}{RESET}")

# --- 2. Initialize the Inference Engine ---
print(f"\n{YELLOW}⏳ Loading model into memory...{RESET}")
llm = Llama(
    model_path=model_path,
    n_ctx=512,      # Maximum context window size
    n_threads=4,    # CPU threads to allocate
    verbose=False   # Suppress internal llama.cpp logs
)

# --- 3. Prompt and Run Inference (Streaming) ---
prompt = "Q: Tell me a fact about Sunnyvale, CA. \nA:"
print(f"\n{BOLD}{CYAN}Prompt:{RESET} {prompt}\n")
print(f"{BOLD}{MAGENTA}Response:{RESET} ", end="", flush=True)

# Stream the output token-by-token
response_stream = llm(
    prompt,
    max_tokens=128,
    stream=True,
)

for chunk in response_stream:
    token_text = chunk["choices"][0]["text"]
    print(f"{GREEN}{token_text}{RESET}", end="", flush=True)
print("\n")
