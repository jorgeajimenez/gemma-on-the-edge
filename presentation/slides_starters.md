
### **Slide 1: Title Slide**

* **Title:** Demystifying Open Weights: From Matrix Math to Cloud-Edge Gateways
* **Subtitle:** Understanding Transformer Foundations & Building Scalable AI Orchestrators
* **Speaker Script:** *"Welcome. Today we're peeling back the layers of Large Language Models. Instead of viewing AI as a black-box API, we're tracing the complete journey: from the fundamental linear algebra inside transformer architectures to deploying accessible, open-weight models across distributed cloud and edge nodes."*

---

### **Slide 2: What Are "Open Weights"?**

* **Key Points:**
* **Open Weights vs. Closed APIs:** Closed models hide parameter weights behind a proprietary endpoint. Open weights (like Google's Gemma series) give direct access to trained parameter tensors (`.safetensors` or quantized `.gguf` files).
* **Ownership & Security:** Running open weights locally or on private cloud GPUs guarantees air-gapped data privacy, zero vendor lock-in, and full control over sampling parameters.
* **Hardware Independence:** The exact same weight matrices can run locally on an edge device (like a Raspberry Pi 5) or scale up to high-throughput cloud GPUs (like Nvidia L4s).


* **Speaker Script:** *"When we talk about 'Open Weights,' we mean having the actual matrix parameters in hand. It's the difference between renting a car with the hood welded shut versus owning the engine. Having the weights means we choose where the model runs, how it's quantized, and who gets to see our data."*

---

### **Slide 3: The LLM Lifecycle & Self-Supervised Pretraining**

* **Key Points (Raschka Framework):**
* **Stage 1 (Data & Architecture):** Tokenizing raw text corpora and constructing causal transformer blocks.
* **Stage 2 (Pretraining):** Self-supervised learning on trillions of tokens using next-word prediction ($x \rightarrow y$).
* **Stage 3 (Instruction Fine-Tuning):** Transforming base foundation models into task-aligned conversational assistants (SFT / RLHF).


* **Speaker Script:** *"As Sebastian Raschka outlines in 'Build an LLM from Scratch', pretraining relies on a surprisingly simple mechanism: next-token prediction. We take unlabelled web text, shift the inputs by one position to create targets, and train the network to guess what comes next over trillions of iterations."*

---

### **Slide 4: Tokenization & Vector Embeddings**

* **Key Points:**
* **Byte-Pair Encoding (BPE):** Splitting text into subword units, ensuring out-of-vocabulary handling without `unk` errors.
* **Token Embeddings ($E_{\text{tok}}$):** Mapping integer token IDs to continuous $d_{\text{in}}$-dimensional vector spaces ($E_{\text{tok}} \in \mathbb{R}^{B \times T \times d_{\text{in}}}$).
* **Absolute Positional Embeddings ($E_{\text{pos}}$):** Injecting sequence order into position-agnostic matrices ($X = E_{\text{tok}} + E_{\text{pos}}$).


* **Speaker Script:** *"Neural networks can't process raw text. We first use BPE tokenization to convert words into numerical indices, then map those indices into high-dimensional space via an embedding matrix. Because self-attention is naturally position-blind, we add a positional embedding vector so the model knows the exact order of tokens in a sequence."*

---

### **Slide 5: Scaled Dot-Product Self-Attention**

* **Key Points:**
* **Linear Transformations:** Projecting input matrix $X$ into Query ($Q$), Key ($K$), and Value ($V$) tensors using learned weight parameters:

$$Q = X W_q, \quad K = X W_k, \quad V = X W_v$$


* **Attention Scores ($\omega$):** Computing sequence alignment via dot-product similarity:

$$\omega = Q K^\top$$


* **Scaled Normalization ($\alpha$):** Scaling by $\sqrt{d_k}$ to prevent exploding dot products / vanishing gradients before Softmax:

$$\alpha = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right)$$




* **Speaker Script:** *"At the heart of every transformer is scaled dot-product attention. By multiplying Query and Key matrices, the model calculates how much attention every token should pay to every other token. Scaling by the square root of the key dimension prevents the softmax function from saturating, which keeps gradients stable during backpropagation."*

---

### **Slide 6: Causal Attention & Autoregressive Decoding**

* **Key Points:**
* **The Causal Mask:** Applying an upper-triangular mask filled with $-\infty$ above the diagonal to enforce left-to-right temporal generation:

$$\text{Mask}(S_{ij}) = \begin{cases} S_{ij} & \text{if } i \ge j \\ -\infty & \text{if } i < j \end{cases}$$


* **Context Vector ($Z$):** Multiplying normalized attention weights by Value vectors ($Z = \alpha V$).
* **Autoregressive Loop:** Appending generated tokens back into the input sequence iteratively.


* **Speaker Script:** *"When generating text, an LLM must never 'cheat' by looking at future tokens. We enforce this using a causal mask that fills future positions with negative infinity before the softmax calculation. This ensures the output context vector for token $N$ is built strictly from tokens $1$ through $N$."*

---

### **Slide 7: Multi-Head Attention & The Transformer Block**

* **Key Points:**
* **Multi-Head Attention (MHA):** Splitting $d_{\text{out}}$ across $h$ parallel heads ($d_k = d_{\text{out}} / h$) to capture diverse contextual sub-spaces simultaneously.
* **Pre-LayerNorm & Residual Shortcuts:** Preventing vanishing gradients in deep networks via residual additions:

$$X_{\text{mid}} = X + \text{MHA}(\text{LN}(X))$$


$$X_{\text{out}} = X_{\text{mid}} + \text{FFN}(\text{LN}(X_{\text{mid}}))$$


* **Feed-Forward Expansion:** Projecting $d_{\text{in}} \rightarrow 4d_{\text{in}} \rightarrow d_{\text{in}}$ with GELU non-linearities.


* **Speaker Script:** *"Multi-Head Attention splits our projection matrices into multiple sub-spaces, allowing one head to track grammatical syntax while another tracks semantic relationships. We wrap this in a Transformer Block with GELU activations, Layer Normalization, and Residual Skip Connections to allow gradients to flow cleanly through deep architectures."*

---

### **Slide 8: The Cloud-Edge Gateway Architecture**

* **Diagram:**
```text
[ Frontends / Agents / Antigravity ]
                 │
                 ▼
┌─────────────────────────────────┐
│ Unified Gateway Proxy (FastAPI) │
└────────────────┬────────────────┘
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
┌──────────────┐   ┌──────────────┐
│  Edge Node   │   │  Cloud Node  │
│ (Local Device│   │(Modal / vLLM │
│  / Gemma)    │   │  Cloud GPU)  │
└──────────────┘   └──────────────┘

```


* **Key Points:**
* **Decoupled Orchestration:** Isolating application logic from physical inference hardware.
* **Dynamic Routing:** Directing low-latency requests to local/edge nodes and heavy batch jobs to cloud GPU instances.


* **Speaker Script:** *"With the model mechanics clear, we turn to architecture. By deploying an orchestrating API gateway, we decouple client applications from inference hardware. Our applications talk to a single gateway, which dynamically routes queries whether the backend model is running on a local device or a cloud GPU."*

---

### **Slide 9: API Standardization & OpenAI Compatibility**

* **Key Points:**
* **The `/v1/chat/completions` Schema:** Adopting industry-standard HTTP endpoints across all edge and cloud backends.
* **Zero Rewrites:** Swapping base URLs (`base_url="https://..."`) in client code without modifying downstream application logic.
* **Sampling Dynamics:** Exposing `temperature`, `top_p`, and repetition penalty controls directly through the gateway API.


* **Speaker Script:** *"By standardizing our gateway around the OpenAI REST schema, we eliminate vendor lock-in. Whether our backend is powered by a custom `llama.cpp` instance on the edge or a high-throughput vLLM container on Modal, any framework—from Cursor to custom Python agents—can query our infrastructure without changing a line of code."*

---

### **Slide 10: Conclusion & Practical Demonstration**

* **Key Points:**
* **Theory Meets Execution:** Raschka's bottom-up transformer theory provides the diagnostic foundation for tuning real-world deployments.
* **Open Weights Empower Control:** Complete ownership of privacy, sampling, and routing infrastructure.
* **Live Gateway Status:** Active cloud endpoint running Gemma, ready for programmatic queries from our client tools and presentation workflows.


* **Speaker Script:** *"We've gone from the raw mathematical equations of causal self-attention to a fully functional, cloud-hosted open-weight gateway. By understanding how models operate under the hood and wrapping them in standard interfaces, we gain total freedom to build powerful, scalable AI systems."*
