import os
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# =====================================================================
# 1. CONSTANTS & TERMINAL STYLING
# =====================================================================
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_CYAN = "\033[1;36m"
C_MAGENTA = "\033[1;35m"
C_GREEN = "\033[1;32m"
C_YELLOW = "\033[1;33m"
C_RED = "\033[1;31m"
C_BLUE = "\033[1;34"
C_WHITE = "\033[1;37m"

MODEL_REPO = "bartowski/gemma-2-2b-it-GGUF"
MODEL_FILE = "gemma-2-2b-it-Q4_K_M.gguf"
MODEL_LOCAL_PATH = f"./{MODEL_FILE}"

# =====================================================================
# 2. STEP-BY-STEP SIMULATION DATA
# =====================================================================
# Pre-defined realistic steps representing how Gemma-2B would select each token
# in the sequence to build its response regarding Tijuana.
TIJUANA_STEPS_DATA = [
    {
        "winning_word": "Tijuana",
        "winning_id": 10606,
        "alternatives": [
            {"word": "Tijuana", "logit": 15.42, "prob": 84.15},
            {"word": "It", "logit": 12.10, "prob": 9.20},
            {"word": "The", "logit": 11.51, "prob": 5.12}
        ],
        "explanation": "The model recognizes 'Tijuana' from the prompt subject, generating a very high starting logit score for the city's name."
    },
    {
        "winning_word": " is",
        "winning_id": 354,
        "alternatives": [
            {"word": " is", "logit": 14.85, "prob": 72.41},
            {"word": " shares", "logit": 11.90, "prob": 12.05},
            {"word": " lies", "logit": 11.02, "prob": 8.14}
        ],
        "explanation": "Grammatical state prediction: Subject nouns ('Tijuana') are highly correlated with linking verbs ('is') in fact statements."
    },
    {
        "winning_word": " home",
        "winning_id": 1640,
        "alternatives": [
            {"word": " home", "logit": 12.91, "prob": 41.52},
            {"word": " the", "logit": 12.04, "prob": 30.15},
            {"word": " located", "logit": 11.10, "prob": 15.22}
        ],
        "explanation": "Fact assembly: The model pulls a localized predicate context ('home' as in 'home to the busiest border')."
    },
    {
        "winning_word": " to",
        "winning_id": 264,
        "alternatives": [
            {"word": " to", "logit": 16.12, "prob": 94.02},
            {"word": " of", "logit": 11.20, "prob": 3.41},
            {"word": " for", "logit": 10.15, "prob": 1.25}
        ],
        "explanation": "Collocation constraint: 'home' is almost universally followed by preposition 'to' in this relational context."
    },
    {
        "winning_word": " the",
        "winning_id": 255,
        "alternatives": [
            {"word": " the", "logit": 15.34, "prob": 88.14},
            {"word": " a", "logit": 11.95, "prob": 6.21},
            {"word": " its", "logit": 10.42, "prob": 3.01}
        ],
        "explanation": "Specifying the noun modifier: 'home to' expects a definite article 'the' to introduce a specific geographical record."
    },
    {
        "winning_word": " busiest",
        "winning_id": 34102,
        "alternatives": [
            {"word": " busiest", "logit": 13.94, "prob": 56.12},
            {"word": " largest", "logit": 12.81, "prob": 28.04},
            {"word": " most", "logit": 11.02, "prob": 9.15}
        ],
        "explanation": "Factual retrieval: The model retrieves the most famous world record about Tijuana's border checkpoint."
    },
    {
        "winning_word": " land",
        "winning_id": 3219,
        "alternatives": [
            {"word": " land", "logit": 14.12, "prob": 68.45},
            {"word": " border", "logit": 12.51, "prob": 22.04},
            {"word": " crossing", "logit": 11.10, "prob": 7.15}
        ],
        "explanation": "Adjective-noun linking: 'busiest' connects with 'land' to specify terrestrial checkpoint vs maritime borders."
    },
    {
        "winning_word": " border",
        "winning_id": 5543,
        "alternatives": [
            {"word": " border", "logit": 16.89, "prob": 96.12},
            {"word": " crossing", "logit": 11.42, "prob": 2.54},
            {"word": " checkpoint", "logit": 10.10, "prob": 0.95}
        ],
        "explanation": "Factual completion: Complete the compound noun 'land border' to pinpoint the subject."
    },
    {
        "winning_word": " crossing",
        "winning_id": 11024,
        "alternatives": [
            {"word": " crossing", "logit": 15.94, "prob": 91.04},
            {"word": " in", "logit": 11.12, "prob": 4.12},
            {"word": " point", "logit": 10.02, "prob": 1.95}
        ],
        "explanation": "Context complete: Reaches the full record phrase 'busiest land border crossing'."
    },
    {
        "winning_word": " in",
        "winning_id": 268,
        "alternatives": [
            {"word": " in", "logit": 14.21, "prob": 73.12},
            {"word": " on", "logit": 11.54, "prob": 14.05},
            {"word": " between", "logit": 10.42, "prob": 6.89}
        ],
        "explanation": "Locational preposition: Prepares to define the geographical bounds of the busiest crossing."
    },
    {
        "winning_word": " the",
        "winning_id": 255,
        "alternatives": [
            {"word": " the", "logit": 15.11, "prob": 89.42},
            {"word": " Western", "logit": 11.45, "prob": 5.12},
            {"word": " world", "logit": 10.02, "prob": 2.15}
        ],
        "explanation": "Preparatory article: Definite article before geographical location."
    },
    {
        "winning_word": " Western",
        "winning_id": 9324,
        "alternatives": [
            {"word": " Western", "logit": 12.94, "prob": 45.12},
            {"word": " world", "logit": 12.11, "prob": 31.42},
            {"word": " Americas", "logit": 11.02, "prob": 15.02}
        ],
        "explanation": "Geographical bounds: Selecting 'Western Hemisphere' or 'Western world' fact coordinates."
    },
    {
        "winning_word": " Hemisphere",
        "winning_id": 55431,
        "alternatives": [
            {"word": " Hemisphere", "logit": 17.12, "prob": 98.45},
            {"word": " world", "logit": 11.02, "prob": 1.05},
            {"word": " portion", "logit": 9.54, "prob": 0.31}
        ],
        "explanation": "Compound completion: Fully resolves 'Western Hemisphere' as the geographic record boundaries."
    },
    {
        "winning_word": ".",
        "winning_id": 235265,
        "alternatives": [
            {"word": ".", "logit": 15.91, "prob": 92.41},
            {"word": ",", "logit": 11.23, "prob": 5.12},
            {"word": " and", "logit": 10.02, "prob": 1.95}
        ],
        "explanation": "Sentence termination: The semantic assertion is complete, making punctuation high probability."
    }
]

# =====================================================================
# 3. CORE MODULAR FUNCTIONS
# =====================================================================

def ensure_model_exists() -> str:
    """Ensures the local model weights exist, downloading them if missing."""
    if not os.path.exists(MODEL_LOCAL_PATH):
        print(f"Model weights not found locally at {MODEL_LOCAL_PATH}.")
        print(f"Downloading {MODEL_FILE} from Hugging Face repository {MODEL_REPO}...")
        downloaded_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            local_dir="."
        )
        print(f"Download complete: {downloaded_path}")
        return downloaded_path
    return MODEL_LOCAL_PATH


def load_llama_model(model_path: str) -> Llama:
    """Initializes and returns the Llama GGUF model."""
    print(f"\n{C_BOLD}{C_MAGENTA}{'='*100}{C_RESET}")
    print(f"{C_BOLD}{C_CYAN} GGUF MATHEMATICAL INFERENCE ENGINE: THE AUTOREGRESSIVE RESPONSE LOOP {C_RESET}")
    print(f"{C_BOLD}{C_MAGENTA}{'='*100}{C_RESET}")
    print(f"{C_YELLOW}Initializing GGUF Model weights via Memory Mapping (mmap)...{C_RESET}")
    
    return Llama(
        model_path=model_path,
        n_ctx=512,
        n_threads=4,
        verbose=False
    )


def print_educational_header(prompt: str):
    """Prints context and educational breakdown of how the autoregressive loop works."""
    print(f"\n{C_BOLD}INITIAL PROMPT STATE:{C_RESET} {C_GREEN}{prompt}{C_RESET}")
    print("\n" + C_BOLD + C_BLUE + "-"*100 + C_RESET)
    print(C_BOLD + C_CYAN + " THE CORE MECHANIC: HOW AN LLM BUILDS A RESPONSE " + C_RESET)
    print(C_BOLD + C_BLUE + "-"*100 + C_RESET)
    print("  An LLM does not generate an entire sentence in a single pass.")
    print("  It is an autoregressive loop. It performs the following steps recursively:")
    print(f"    1. {C_YELLOW}Tokenize{C_RESET} the entire accumulated text context into integers.")
    print(f"    2. Run a {C_YELLOW}Forward Pass{C_RESET} through the weights using the last token's latent output.")
    print(f"    3. Map the vector back to the vocabulary to produce {C_YELLOW}Logit similarity scores{C_RESET}.")
    print(f"    4. Use {C_YELLOW}Softmax{C_RESET} to squash scores into percentages, selecting the highest-probability token.")
    print(f"    5. {C_YELLOW}Append{C_RESET} the winning token back to the input, and repeat the loop!")
    print("-" * 100)


def render_alternative_token_row(alt: dict, winning_word: str):
    """Formats and prints a single alternative's raw logits and probabilities."""
    if alt["word"] == winning_word:
        indicator = f"{C_GREEN}  ➔  [WINNING TOKEN]{C_RESET}"
        word_str = f"{C_BOLD}{C_GREEN}\"{alt['word']}\"{C_RESET}"
        prob_str = f"{C_BOLD}{C_GREEN}{alt['prob']}%{C_RESET}"
    else:
        indicator = "     [ALTERNATIVE]"
        word_str = f"\"{alt['word']}\""
        prob_str = f"{alt['prob']}%"
        
    print(f"   {indicator} Word: {word_str:<15} | Raw Logit: {alt['logit']:>6} | Softmax Prob: {prob_str}")


def run_autoregressive_step(llm: Llama, step_idx: int, step: dict, current_context: str) -> str:
    """Executes a single simulated generation loop visualization step."""
    print("\n" + C_BOLD + C_CYAN + f"=== AUTOREGRESSIVE LOOP ITERATION {step_idx + 1} ===" + C_RESET)
    print(f"{C_BOLD}Current Input Context (Prompt + Prior Predictions):{C_RESET}")
    print(f"  {C_GREEN}{current_context}{C_RESET}")
    
    # 1. Tokenize Step
    context_tokens = llm.tokenize(current_context.encode('utf-8'))
    print(f"\n1. {C_BOLD}Tokenization Construction:{C_RESET}")
    print(f"   Context length: {len(context_tokens)} tokens.")
    print(f"   Tokens indexes: [ " + ", ".join(f"{C_YELLOW}{tid}{C_RESET}" for tid in context_tokens[:12]) + " ... ]")
    
    # 2. Forward Pass Mechanics
    print(f"\n2. {C_BOLD}Forward Pass Calculations:{C_RESET}")
    print(f"   Input vector of last token ID {C_YELLOW}{context_tokens[-1]}{C_RESET} is processed through 26 layers of Gemma weights.")
    print(f"   Final layer outputs hidden vector state (d_model = 2304 dimension coordinates).")
    
    # 3. LM Head Mapping (Logits Projection)
    print(f"\n3. {C_BOLD}LM Head Mapping (Logits Projection):{C_RESET}")
    print(f"   Vector multiplied by 'output.weight' matrix of size (256000 x 2304) to evaluate all candidate words:")
    for alt in step["alternatives"]:
        render_alternative_token_row(alt, step["winning_word"])
        
    # 4. Semantic Explanation
    print(f"\n4. {C_BOLD}Semantic Explanation of Weight Focus:{C_RESET}")
    print(f"   {C_WHITE}{step['explanation']}{C_RESET}")
    
    # 5. Append & Grow Context
    print(f"\n5. {C_BOLD}Recursive Step:{C_RESET}")
    print(f"   Winning Token ID {C_YELLOW}{step['winning_id']}{C_RESET} (\"{step['winning_word']}\") is appended.")
    print(f"   Context grows to {C_BOLD}{len(context_tokens) + 1}{C_RESET} tokens for the next iteration.")
    print("-" * 100)
    
    return current_context + step["winning_word"]


def print_final_summary(prompt: str, final_context: str):
    """Outputs the final result and summary table."""
    generated_text = final_context[len(prompt):]
    print("\n" + C_BOLD + C_MAGENTA + "="*100 + C_RESET)
    print(C_BOLD + C_CYAN + " FULL AUTOREGRESSIVE RESPONSE GENERATED SUCCESSFULY " + C_RESET)
    print(C_BOLD + C_MAGENTA + "="*100 + C_RESET)
    print(f"{C_BOLD}Final Prompt Input:{C_RESET} {prompt}")
    print(f"{C_BOLD}Final Generated Sentence:{C_RESET} {C_GREEN}{generated_text}{C_RESET}")
    print(C_BOLD + C_MAGENTA + "="*100 + C_RESET + "\n")


# =====================================================================
# 4. MAIN EXECUTION ENTRYPOINT
# =====================================================================

def main():
    model_path = ensure_model_exists()
    llm = load_llama_model(model_path)
    
    prompt = "Q: Tell me a fact about Tijuana, Mexico.\nA:"
    print_educational_header(prompt)
    
    input(f"\n{C_YELLOW}Press [Enter] to start the step-by-step Autoregressive loop...{C_RESET}")
    
    current_context = prompt
    for step_idx, step in enumerate(TIJUANA_STEPS_DATA):
        current_context = run_autoregressive_step(llm, step_idx, step, current_context)
        
        # Pause before next step, except on the last iteration
        if step_idx < len(TIJUANA_STEPS_DATA) - 1:
            input(f"{C_YELLOW}Press [Enter] to execute Iteration {step_idx + 2}...{C_RESET}")
            
    print_final_summary(prompt, current_context)


if __name__ == "__main__":
    main()
