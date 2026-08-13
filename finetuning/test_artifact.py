import os
from llama_cpp import Llama 

# Terminal colors for aesthetics
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"

model_path = "./my_finetuned_gemma_gguf/gemma-2-2b-it.Q4_K_M.gguf"

if not os.path.exists(model_path):
    print(f"{YELLOW}Error: Artifact not found at {model_path}. Run finetune.py first!{RESET}")
    exit(1)

print(f"{GREEN}{BOLD}✓{RESET} {GREEN}Found GGUF Artifact at: {model_path}{RESET}")
print(f"\n{YELLOW}⏳ Loading GGUF model into C++ engine...{RESET}")

llm = Llama(
    model_path=model_path,
    n_ctx=512,
    n_threads=4,
    verbose=False
)

# --- CHOOSE YOUR QUESTION ---
# Here are a few from your lore dataset. Just uncomment the one you want to test!

question = "What kind of pets do people have in Vuldor?"
# question = "What is the primary export of the subterranean city of Vuldor?"
# question = "How does the city of Vuldor generate its power?"
# question = "Who is Kaelen the Deep-Seer in Vuldorian history?"

# The exact formatting Gemma 2 expects (WITHOUT the manual <bos>)
prompt = f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"

print(f"\n{BOLD}{CYAN}Prompt:{RESET} {question}\n")
print(f"{BOLD}{MAGENTA}Response:{RESET} ", end="", flush=True)

# Generate the response
response_stream = llm(
    prompt,
    max_tokens=128,
    stream=True,
)

for chunk in response_stream:
    token_text = chunk["choices"][0]["text"]
    print(f"{GREEN}{token_text}{RESET}", end="", flush=True)
print("\n")