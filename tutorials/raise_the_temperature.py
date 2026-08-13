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

# Prompt for temperature value with color highlight
print(f"{BOLD}{CYAN}🛠️ Setup Temperature Parameters{RESET}")
temp_input = input(f"Enter temperature value [0.0 - 2.0] or press Enter for '{YELLOW}none{RESET}' (uses engine default 0.8): ").strip()

if not temp_input or temp_input.lower() == "none":
    temp = None
else:
    try:
        temp = float(temp_input)
    except ValueError:
        print(f"{YELLOW}Invalid input, defaulting to 'none'{RESET}")
        temp = None

temp_label = f"{YELLOW}{temp}{RESET}" if temp is not None else f"{YELLOW}default (0.8){RESET}"
print(f"\n{BOLD}{CYAN}Prompt:{RESET} {prompt} (Temperature: {temp_label})\n")
print(f"{BOLD}{MAGENTA}Response:{RESET} ", end="", flush=True)

# Stream the output token-by-token
gen_params = {
    "prompt": prompt,
    "max_tokens": 128,
    "stream": True
}
if temp is not None:
    gen_params["temperature"] = temp

response_stream = llm(**gen_params)

for chunk in response_stream:
    token_text = chunk["choices"][0]["text"]
    print(f"{GREEN}{token_text}{RESET}", end="", flush=True)
print("\n")
