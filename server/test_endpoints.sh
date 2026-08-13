#!/usr/bin/env bash
# =====================================================================
# Test Suite for Live Google Cloud Run Gemma Inference Server
# You can run this file directly: `./test_endpoints.sh`
# Or copy and paste individual curl commands below.
# =====================================================================

set -euo pipefail

# Live deployment URL
BASE_URL="https://gemma-edge-server-420844676062.us-central1.run.app"

echo "================================================================="
echo "[*] Testing Gemma Edge Server on Cloud Run"
echo "[*] Target URL: ${BASE_URL}"
echo "================================================================="

# --- 1. HEALTH CHECK ---
echo -e "\n[1] Testing GET /health endpoint..."
curl -s -w "\nHTTP Status: %{http_code}\n" "${BASE_URL}/health"

# --- 2. STANDARD GENERATION ---
echo -e "\n[2] Testing POST /api/generate (Standard JSON Response)..."
curl -X POST "${BASE_URL}/api/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Q: What is Tijuana, Mexico known for?\nA:",
    "temperature": 0.7,
    "max_tokens": 48,
    "stream": false
  }'
echo -e "\n"

# --- 3. STREAMING GENERATION (SSE) ---
echo -e "\n[3] Testing POST /api/generate (Streaming Event-Stream)..."
curl -N -X POST "${BASE_URL}/api/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Q: Name three yellow fruits.\nA:",
    "temperature": 0.7,
    "max_tokens": 48,
    "stream": true
  }'

# --- 4. OPENAI-COMPATIBLE CHAT COMPLETIONS (STREAMING) ---
echo -e "\n[4] Testing POST /v1/chat/completions (OpenAI Streaming Stream)..."
curl -N -X POST "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Write a short haiku about running models in the cloud."}
    ],
    "temperature": 0.8,
    "max_tokens": 64,
    "stream": true
  }'
echo ""
