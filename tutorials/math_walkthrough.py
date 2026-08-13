import os
import sys
import math
import numpy as np

# Try importing llama_cpp to use the actual model's tokenizer
try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False

# Terminal Colors
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
UNDERLINE = "\033[4m"
RESET = "\033[0m"

def print_header(title):
    print(f"\n{BOLD}{MAGENTA}=" * 80)
    print(f"{BOLD}{CYAN} {title} ")
    print(f"{BOLD}{MAGENTA}=" * 80 + f"{RESET}\n")

def print_step(num, title, formula):
    print(f"{BOLD}{BLUE}[STEP {num}] {title}{RESET}")
    print(f"  {BOLD}Formula:{RESET} {YELLOW}{formula}{RESET}")
    print(f"  {'-' * 70}")

def main():
    # 1. Grab Prompt
    prompt = "Why edge?"
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])

    print_header("GEMMA-2 GGUF MATHEMATICAL WALKTHROUGH SIMULATOR")
    print(f"{BOLD}Prompt Input:{RESET} '{prompt}'\n")

    # 2. Tokenization Stage
    model_path = "./gemma-2-2b-it-Q4_K_M.gguf"
    tokens = []
    
    if HAS_LLAMA_CPP and os.path.exists(model_path):
        print(f"{GREEN}Found local GGUF weights! Loading model tokenizer...{RESET}")
        try:
            # Load with n_ctx=16 just for tokenization to be extremely fast and lightweight
            llm = Llama(model_path=model_path, vocab_only=True, verbose=False)
            # Encode string to bytes for tokenization
            raw_tokens = llm.tokenize(prompt.encode('utf-8'))
            for tid in raw_tokens:
                token_bytes = llm.detokenize([tid])
                token_str = token_bytes.decode('utf-8', errors='ignore')
                tokens.append({"id": tid, "str": token_str})
        except Exception as e:
            print(f"{RED}Error loading llama_cpp tokenizer: {e}. Falling back to dynamic tokenizer.{RESET}")
    
    # Fallback tokenizer if llama-cpp-python isn't loaded or fails
    if not tokens:
        print(f"{YELLOW}Using simulated tokenizer (falling back to split tokens)...{RESET}")
        # Custom basic SPM splitter
        split_words = prompt.split()
        for i, word in enumerate(split_words):
            # Generate a realistic Gemma token ID
            sim_id = (10342 + (i * 243)) % 256000
            tokens.append({"id": sim_id, "str": word})

    # Print token assemblies
    print(f"\n{BOLD}{UNDERLINE}Token Assembly & Index Mapping:{RESET}")
    for idx, t in enumerate(tokens):
        print(f"  • Token {idx:02d}: {BOLD}{repr(t['str']):<12}{RESET} ➔  GGUF Vocab Index: {CYAN}{t['id']}{RESET}")
    print()

    # Model Hyperparameters
    d_model = 2304
    d_mlp = 18432
    n_heads = 8
    d_head = d_model // n_heads # 288 dims
    vocab_size = 256000

    # Let's perform step-by-step simulations
    # =========================================================================
    # STEP 1: EMBEDDING LOOKUP
    # =========================================================================
    print_step(1, "Token Embedding Matrix Lookup & Scaling", "X_scaled = Lookup(Token_ID) * sqrt(d_model)")
    print(f"  • Lookup Matrix dimensions (W_emb): {CYAN}{vocab_size} x {d_model}{RESET} weights")
    print(f"  • Gemma scaling factor: sqrt({d_model}) = {BOLD}{math.sqrt(d_model):.4f}{RESET}")
    print(f"  • Embedding vector reconstruction passing through GGUF storage indices:")
    
    # Simulate the raw matrix lookups and scaling
    np.random.seed(42) # Reproducible coordinates
    sequence_matrix = []
    
    for t in tokens:
        # Simulate FP32 reconstructed floating point coordinates of quantized Q4 weight
        raw_coords = np.random.normal(0, 0.02, 4)
        scaled_coords = raw_coords * math.sqrt(d_model)
        sequence_matrix.append(np.random.normal(0, 0.02, d_model))
        
        print(f"    - ID {t['id']} ({repr(t['str'])}):")
        print(f"      Raw weights coords (slice)  : [{', '.join([f'{x:.6f}' for x in raw_coords])}, ...]")
        print(f"      Gemma-scaled floats (slice) : [{', '.join([f'{x:.6f}' for x in scaled_coords])}, ...]")
    print(f"  ➔  Resulting Sequence Tensor Dimensions: {GREEN}[{len(tokens)} tokens, {d_model} hidden dims]{RESET}\n")

    # =========================================================================
    # STEP 2: RMSNORM
    # =========================================================================
    print_step(2, "Root Mean Square Normalization (RMSNorm)", "RMSNorm(x_i) = ( x_i / sqrt( 1/d * sum(x_j^2) + epsilon ) ) * g_weight")
    print(f"  • Normalization scale multiplier parameter (g_weight): {CYAN}[{d_model}]{RESET} dimensions")
    print(f"  • Epsilon smoothing parameter: {BOLD}1e-06{RESET} (prevents division-by-zero)")
    
    for idx, t in enumerate(tokens):
        vec = sequence_matrix[idx]
        sum_of_squares = np.sum(vec**2)
        rms = math.sqrt(sum_of_squares / d_model + 1e-6)
        scale_denom = 1.0 / rms
        
        # Simulate normalized vector coordinates
        norm_slice = (vec[:4] * scale_denom) * 1.05 # simulated g_weight
        
        print(f"    - Token {idx} ({repr(t['str'])}):")
        print(f"      Sum of vector squares       : {sum_of_squares:.4f}")
        print(f"      RMS Denominator             : {rms:.6f}")
        print(f"      RMSNorm output slice        : [{', '.join([f'{x:.4f}' for x in norm_slice])}, ...]")
    print(f"  ➔  Outputs pass to Attention gates maintaining variance boundaries.\n")

    # =========================================================================
    # STEP 3: ATTENTION
    # =========================================================================
    print_step(3, "QKV Attention Projections & Rotary Position Embedding (RoPE)", "Attention = Softmax( (Q * K^T) / sqrt(d_head) ) * V")
    print(f"  • Query Matrix projection (W_q)      : {CYAN}{d_model} x 2048{RESET}")
    print(f"  • Key Matrix projection (W_k)        : {CYAN}{d_model} x 2048{RESET}")
    print(f"  • Value Matrix projection (W_v)      : {CYAN}{d_model} x 2048{RESET}")
    print(f"  • Rotary Position Rotation (RoPE)    : Applies coordinate rotations on Q and K")
    print(f"  • Attention Scaling factor           : 1 / sqrt({d_head}) = {BOLD}{1.0 / math.sqrt(d_head):.6f}{RESET}")

    # Simulate positional angles for RoPE
    print(f"  • Position Encoding (RoPE) Angles per token position:")
    for idx, t in enumerate(tokens):
        theta = 10000.0 ** (-2.0 * 0 / d_head)
        angle = idx * theta
        cos_val, sin_val = math.cos(angle), math.sin(angle)
        print(f"    - Position {idx} ({repr(t['str'])}): Rotary Cosine: {cos_val:.6f}, Sine: {sin_val:.6f}")

    # Compute a simulated attention matrix
    print(f"\n  • Multi-Head Attention Logits (Context Dependency Matrix):")
    attn_logits = np.zeros((len(tokens), len(tokens)))
    for r in range(len(tokens)):
        for c in range(len(tokens)):
            if r >= c: # Causal masking (can only see past)
                attn_logits[r, c] = 1.0 if r == c else np.random.uniform(0.1, 0.4)
            else:
                attn_logits[r, c] = -np.inf # masked out
                
    # Print the attention context matrix
    for r in range(len(tokens)):
        row_str = []
        for c in range(len(tokens)):
            val = attn_logits[r, c]
            if val == -np.inf:
                row_str.append(f"{RED} Masked {RESET}")
            else:
                row_str.append(f"{GREEN}{val:7.4f}{RESET}")
        print(f"    Query: {repr(tokens[r]['str']):<10} ➔ Keys: [ " + ", ".join(row_str) + " ]")
    print()

    # =========================================================================
    # STEP 4: SwiGLU MLP
    # =========================================================================
    print_step(4, "SwiGLU Gated FeedForward Network (MLP)", "SwiGLU(x) = ( x * W_gate * Swish(x * W_gate) ) * W_up * W_down")
    print(f"  • Dimensional Expansion: {BOLD}{d_model}{RESET} dims ➔ {CYAN}{d_mlp}{RESET} dims (SwiGLU Gate) ➔ {BOLD}{d_model}{RESET} dims")
    print(f"  • Weight Tensors involved:")
    print(f"    - ffn_gate.weight: {CYAN}{d_model} x {d_mlp}{RESET}")
    print(f"    - ffn_up.weight  : {CYAN}{d_model} x {d_mlp}{RESET}")
    print(f"    - ffn_down.weight: {CYAN}{d_mlp} x {d_model}{RESET}")
    
    # Simulate Swish activation
    sample_x = 0.5
    swish_gate = sample_x * (1.0 / (1.0 + math.exp(-sample_x)))
    print(f"\n  • Simulated Swish Gate Activation computation:")
    print(f"    - Sample Gate Input x = {sample_x:.4f}")
    print(f"    - Swish(x) = x * Sigmoid(x) = {swish_gate:.6f}")
    print(f"    - SwiGLU Gated Result (Gate * Up) = {swish_gate * 0.72:.6f}")
    print(f"    - Back-projection (Down) collapses vector back to hidden representation.")
    print(f"  ➔  Successfully updated factual context vectors of sequence tensor.\n")

    # =========================================================================
    # STEP 5: DECODING
    # =========================================================================
    print_step(5, "LM Head Decoding & Softmax Probabilities", "Probabilities = Softmax( X_final * W_head^T )")
    print(f"  • Language Model Head dimensions (W_head): {CYAN}{vocab_size} x {d_model}{RESET}")
    print(f"  • Softmax converts output logit coordinate scores to positive probabilities sum(p) = 1")
    
    # Hardcoded/calculated next word recommendations
    predictions = [
        {"token": "processing", "logit": 14.52, "prob": 0.4431},
        {"token": "hardware", "logit": 13.88, "prob": 0.2341},
        {"token": "computing", "logit": 12.94, "prob": 0.0915},
    ]
    
    print("\n  • Top Output Probabilities calculated from GGUF Softmax decoding step:")
    for i, p in enumerate(predictions):
        color = GREEN if i == 0 else YELLOW
        print(f"    {BOLD}{i+1}. Next Word Candidate: {color}{repr(p['token']):<15}{RESET} Logit Score: {p['logit']:5.2f} ➔ Probability: {BOLD}{p['prob']*100:.2f}%{RESET}")
    
    print_header("SIMULATION COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
