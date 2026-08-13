Role: Senior AI Educator & Explainer Specialist
Task: Research our uploaded GGUF weights, local execution codebase, and model metadata to write a zero-knowledge educational guide on what we are building.

Focus deeply on these two core concepts:
1. The Philosophy of "Open Weights":
   - What are weights (parameters/synaptic connections) in plain English?
   - What is the difference between closed APIs (cloud dependencies) and open weights (local ownership, complete privacy, offline deployment)?
   - Why do open weights democratize access to AI?

2. The Mechanics of Matrix Multiplication (MatMul):
   - Explain how LLM thinking is fundamentally massive linear algebra (Y = X * W).
   - Trace how a single word is transformed from characters to a 2304-dimensional vector coordinate, multiplied against attention weight columns (dot products), expanded via SwiGLU gates, and decoded back into words.
   - Demystify why local AI speed is "Memory-Bandwidth Bound" (reading weights from disk/RAM to processor caches) rather than compute-bound.

Educational Tone:
- Assume absolute zero prior machine learning knowledge.
- Use intuitive, clear analogies (e.g., comparing GGUF to a "portable suitcase brain", matrix multiplications to "coordinate maps", and quantization scales to "number clustering").
- Provide a summary and bullet points explaining why GGUF's on-the-fly dequantization enables giant models to run on ordinary laptops.
