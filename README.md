# GDG Sunnyvale Edge Meetup Code

This project runs a lightweight LLM (Gemma-2-2b-it GGUF) locally on the edge using `llama-cpp-python`.

## Repository Structure

The repository is organized as follows:
* 📁 **`tutorials/`**: Hands-on walkthrough scripts for learning local inference basics, sampling parameters, and mathematical concepts.
  * `start_here.py`: Simple model loading and streaming generation.
  * `main.py`: Interactive console chat loop.
  * `raise_the_temperature.py`: Exploration of temperature and top-p sampling.
  * `math_walkthrough.py`: Deep-dive into softmax, logits, and the probability math behind generation.
* 📁 **`visualizer/`**: A gorgeous web dashboard to introspect GGUF files and see how they partition parameters, quantize layers, and format prompts.
* 📁 **`server/`**: A local-network Flask server exposing custom and OpenAI-compatible API endpoints, Docker containers, and Cloud Run deploy setups.
* 📁 **`finetuning/`**: Python scripts for LoRA fine-tuning on custom lore data (Vuldor geode) and GGUF quantization.
* 📁 **`presentation/`**: Keynotes, slides, infographics, and references used during the GDG Sunnyvale Meetup presentation.
* 📁 **`notebooks/`**: Jupyter Notebooks showing end-to-end code for model workflows (e.g., Gemma 3 structural weights).
* 📁 **`archive/`**: Safely archived infrastructure and experiment code.

## Setup and Running with `uv`

Make sure you have [uv](https://github.com/astral-sh/uv) installed.

To run the script automatically installing all dependencies into an ephemeral or managed virtual environment:

```bash
uv run tutorials/main.py
```

Or to explicitly sync your environment and install dependencies:

```bash
uv sync
```

## Interactive GGUF Visualizer & Demystifier

We have built a gorgeous Flask web application that maps out, visualizes, and demystifies how a GGUF file arranges its parameters, quantization matrices, and layers!

To launch the interactive dashboard:

```bash
uv run visualizer/visualizer.py
```

Then open your browser and navigate to:
👉 **[http://localhost:5001](http://localhost:5001)**

### Features:
- **Real-Time Download Tracking**: If the model is not found locally, download it directly with dynamic, beautiful progress stats.
- **Structural Allocation Chart**: Instantly see how parameters split between Attention layers vs FeedForward layers vs Embeddings.
- **GGUF Quantization Breakdown**: Explore how weights are compressed using modern sub-byte techniques.
- **Deep Layer-by-Layer Matrix**: Interactive layer stacking diagram detailing parameter allocation sequentially.
- **Searchable Metadata Registry**: View key-value headers like dimensions, vocab sizes, architecture details, etc.
- **Interactive Sandbox**: Write some text and watch how the stored Jinja2 chat template processes it into tokenizable prompts.


## Determinism vs. Creativity in LLMs

When running `tutorials/main.py`, you may notice that you receive the exact same response every time. This is because local inference engines often default to **greedy decoding** (equivalent to `temperature=0.0`). 

In greedy decoding, the model always selects the token with the absolute highest probability. Since the model's weights and mathematical computations are entirely static, identical prompts yield identical outputs.

### Adding Creativity / Randomness

To make the output creative or non-deterministic on each run, adjust these generation parameters:

```python
response = llm(
    prompt,
    max_tokens=128,
    temperature=0.7,  # Higher = more creative/diverse; Lower = more focused/deterministic
    top_p=0.9         # Cumulative probability threshold (only consider top 90% candidates)
)
```
- **`temperature`**: Flat-lines or sharpens candidate token probabilities. Higher values give less-probable words a chance to be selected.
- **`top_p` (Nucleus Sampling)**: Dynamically truncates the active vocabulary to the minimal set of top tokens whose combined probability meets the threshold, filtering out nonsensical choices.


## Local Network Inference Server (Step-by-Step Guide)

To demonstrate true **network-based edge inference** (where one local device hosts the model and other local network devices like laptops, phones, or Raspberry Pis query it via HTTP APIs), follow these steps:

### Step 1: Launch the Local Inference Server

Run the Flask-based API server on your host machine:

```bash
uv run server/server.py
```

The server initializes on port `5002` and listens on `0.0.0.0` (all local network interfaces).

### Step 2: Retrieve your Local IP Address

To query this server from *another* device on the same local Wi-Fi or ethernet network, you need your host computer's local IP address:

* **macOS / Linux**: Open your terminal and run:
  ```bash
  ipconfig getifaddr en0  # (For Wi-Fi) or:
  ifconfig | grep "inet "
  ```
* **Windows**: Open Command Prompt and run:
  ```cmd
  ipconfig
  ```

Look for your local IPv4 Address (typically something like `192.168.1.XX` or `10.0.0.XX`).

### Step 3: Run the Zero-Dependency Client

You can run the client from your host machine (targeting `localhost`), or from **another computer on the same network** by replacing `localhost` with your host machine's IP inside `server/client.py`:

```python
# server/client.py (Line 5)
SERVER_URL = "http://YOUR_HOST_IP_HERE:5002/api/generate"
```

Then run the client:

```bash
uv run server/client.py
```

This client uses only Python's standard library (`urllib`), proving that edge-connected devices don't need heavy ML libraries to leverage a shared edge model.

### Step 4: Query the Edge Server with `curl`

You can also test the endpoints directly from your command-line interface:

**Test Custom Streaming Response (`/api/generate`):**
```bash
curl -N -X POST http://localhost:5002/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Q: Name three fruits that are yellow.\nA:",
    "temperature": 0.7,
    "max_tokens": 64,
    "stream": true
  }'
```

**Test OpenAI-Compatible Chat completions (`/v1/chat/completions`):**
```bash
curl -X POST http://localhost:5002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Explain gravity in one simple sentence."}
    ],
    "temperature": 0.0,
    "max_tokens": 64,
    "stream": false
  }'
```


## Google Cloud Run Deployment

You can deploy this lightweight inference server to **Google Cloud Run** to share your model on a publicly accessible API endpoint.

### 1. Build & Deploy
We have included an automated deployment shell script and optimized `.gcloudignore` boundaries (to prevent massive weight uploads during context packaging):

```bash
# 1. Make the script executable
chmod +x server/deploy_cloud_run.sh

# 2. Trigger the deployment
./server/deploy_cloud_run.sh
```

### 2. Live API Testing Suite

You can test your live Cloud Run endpoints by running the included executable test script:

```bash
./server/test_endpoints.sh
```

Alternatively, you can copy-paste and run these `curl` commands directly in your terminal to test your live service:

**Test Cloud Health check:**
```bash
curl -s https://gemma-edge-server-420844676062.us-central1.run.app/health
```

**Test Live Text Generation (Non-Streaming):**
```bash
curl -X POST https://gemma-edge-server-420844676062.us-central1.run.app/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Q: Name three fruits that are yellow.\nA:",
    "temperature": 0.7,
    "max_tokens": 32,
    "stream": false
  }'
```

**Test OpenAI-Compatible Streaming (SSE):**
```bash
curl -N -X POST https://gemma-edge-server-420844676062.us-central1.run.app/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Write a short haiku about running models in the cloud."}
    ],
    "temperature": 0.8,
    "max_tokens": 64,
    "stream": true
  }'
```






