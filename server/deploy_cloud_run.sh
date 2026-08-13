#!/usr/bin/env bash
# =====================================================================
# Google Cloud Run Deployment Script for Gemma Edge Server
# This script builds the container image via Google Cloud Builds 
# and deploys it to Cloud Run with optimized CPU and Memory allocations.
# =====================================================================

# Exit immediately if a command exits with a non-zero status
set -euo pipefail

# --- CONFIGURABLE VARIABLES ---
# Change these values to match your Google Cloud Platform setup
PROJECT_ID="YOUR_GCP_PROJECT_ID"  # <-- Set your Google Cloud Project ID here
REGION="us-central1"               # GCP Region for deployment
SERVICE_NAME="gemma-edge-server"   # Cloud Run Service Name
PORT="5002"                        # Container port

# Ensure gcloud CLI is installed
if ! command -v gcloud &> /dev/null; then
    echo "[-] Error: 'gcloud' command-line tool not found."
    echo "    Please install the Google Cloud SDK first: https://cloud.google.com/sdk"
    exit 1
fi

# Confirm project ID is set
if [ "$PROJECT_ID" == "YOUR_GCP_PROJECT_ID" ]; then
    echo "[!] Warning: PROJECT_ID is still set to 'YOUR_GCP_PROJECT_ID'."
    echo "    Checking active gcloud configuration..."
    ACTIVE_PROJECT=$(gcloud config get-value project 2>/dev/null || echo "")
    if [ -n "$ACTIVE_PROJECT" ]; then
        echo "[+] Found active gcloud project: $ACTIVE_PROJECT"
        PROJECT_ID="$ACTIVE_PROJECT"
    else
        echo "[-] Error: Please edit this script and set your PROJECT_ID, or run 'gcloud config set project <id>'."
        exit 1
    fi
fi

IMAGE_TAG="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest"

echo "================================================================="
echo "[*] TARGET GCP PROJECT: ${PROJECT_ID}"
echo "[*] CONTAINER IMAGE:   ${IMAGE_TAG}"
echo "[*] TARGET REGION:     ${REGION}"
echo "================================================================="

# 1. Ensure required Google Cloud services are enabled
echo -e "\n[*] Step 1: Enabling Artifact Registry and Cloud Run APIs..."
gcloud services enable artifactregistry.googleapis.com run.googleapis.com --project="${PROJECT_ID}"

# 2. Build the container image using Cloud Build
echo -e "\n[*] Step 2: Building and pushing container image to Artifact Registry..."
gcloud builds submit --tag "${IMAGE_TAG}" --project="${PROJECT_ID}"

# 3. Deploy the container to Cloud Run
echo -e "\n[*] Step 3: Deploying container to Google Cloud Run..."
# We allocate 4 vCPUs and 8GiB Memory to provide optimal execution performance for local CPU inference
gcloud run deploy "${SERVICE_NAME}" \
    --image "${IMAGE_TAG}" \
    --platform managed \
    --region "${REGION}" \
    --cpu=4 \
    --memory=8Gi \
    --port="${PORT}" \
    --project="${PROJECT_ID}" \
    --allow-unauthenticated

echo -e "\n[+] Success! Your local-style network inference server has been deployed to Google Cloud Run."
echo "    You can now update your client base URLs to target the Cloud Run service URL shown above."
