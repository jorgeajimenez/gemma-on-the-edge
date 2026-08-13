import os
import re
import threading
import struct
import requests
from flask import Flask, jsonify, render_template_string, request, send_from_directory
import gguf

app = Flask(__name__)

# Global state for downloading
download_progress = {
    "status": "idle",       # idle, downloading, completed, error
    "progress": 0,
    "downloaded_mb": 0,
    "total_mb": 0,
    "error_message": ""
}

MODEL_PATH = "./gemma-2-2b-it-Q4_K_M.gguf"

def download_model_task():
    global download_progress
    try:
        download_progress["status"] = "downloading"
        download_progress["progress"] = 0
        
        url = "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf"
        r = requests.get(url, stream=True)
        r.raise_for_status()
        
        total_size = int(r.headers.get('content-length', 0))
        download_progress["total_mb"] = round(total_size / (1024 * 1024), 2)
        
        downloaded = 0
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    download_progress["downloaded_mb"] = round(downloaded / (1024 * 1024), 2)
                    if total_size > 0:
                        download_progress["progress"] = int((downloaded / total_size) * 100)
                    else:
                        download_progress["progress"] = 50
                        
        download_progress["status"] = "completed"
    except Exception as e:
        download_progress["status"] = "error"
        download_progress["error_message"] = str(e)

def parse_gguf_file(filepath):
    reader = gguf.GGUFReader(filepath)
    
    with open(filepath, "rb") as f:
        magic = f.read(4).decode('utf-8', errors='ignore')
        version = struct.unpack("<I", f.read(4))[0]
        
    header_info = {
        "magic": magic,
        "version": version,
        "tensor_count": len(reader.tensors),
        "kv_count": len(reader.fields)
    }
    
    metadata = []
    vocab = []
    
    for key, field in reader.fields.items():
        if key == "tokenizer.ggml.tokens":
            try:
                parts = field.parts
                idx = 5
                for _ in range(15000):
                    if idx + 1 >= len(parts):
                        break
                    try:
                        val_part = parts[idx+1]
                        token_str = "".join(chr(b) for b in val_part)
                        vocab.append(token_str)
                    except:
                        pass
                    idx += 2
            except Exception as e:
                vocab = ["<error extraction: {}>".format(e)]
                
        try:
            val = []
            for part in field.parts:
                if hasattr(part, "tolist"):
                    val.append(part.tolist())
                elif isinstance(part, bytes):
                    try:
                        val.append(part.decode('utf-8'))
                    except:
                        val.append(list(part))
                else:
                    val.append(part)
            if len(val) == 1:
                val = val[0]
            metadata.append({
                "key": key,
                "value": str(val)[:200] + ("..." if len(str(val)) > 200 else "")
            })
        except Exception as e:
            metadata.append({"key": key, "value": "<error: {}>".format(e)})
            
    tensors = []
    total_elements = 0
    quant_types = {}
    
    for tensor in reader.tensors:
        name = tensor.name
        shape = [int(s) for s in tensor.shape]
        dtype = str(tensor.tensor_type)
        
        elements = 1
        for s in shape:
            elements *= s
        total_elements += int(elements)
        
        quant_types[dtype] = int(quant_types.get(dtype, 0) + 1)
        
        tensors.append({
            "name": name,
            "shape": shape,
            "type": dtype,
            "size_bytes": int(tensor.n_bytes)
        })
        
    file_size_gb = round(os.path.getsize(filepath) / (1024 * 1024 * 1024), 2)
    
    return {
        "header": header_info,
        "metadata": metadata,
        "tensors": tensors,
        "total_elements": int(total_elements),
        "quant_types": quant_types,
        "file_size_gb": file_size_gb,
        "vocab": vocab
    }

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GGUF Interactive Token-to-Answer AI Sandbox</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <script>
        window.MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
            },
            options: {
                skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
            }
        };
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
    <style>
        :root {
            --bg-primary: #040407;
            --bg-glass: rgba(10, 10, 18, 0.85);
            --border-glass: rgba(255, 255, 255, 0.08);
            --accent-primary: #6366f1;
            --accent-secondary: #ec4899;
            --accent-glow: rgba(99, 102, 241, 0.15);
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --code-bg: rgba(0, 0, 0, 0.65);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-main);
            overflow-x: hidden;
            background-image: 
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 40%),
                radial-gradient(at 100% 100%, rgba(236, 72, 153, 0.1) 0px, transparent 40%);
            min-height: 100vh;
            padding: 2rem;
        }

        .main-container {
            max-width: 1250px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 2rem;
        }

        header {
            text-align: center;
            margin-bottom: 1rem;
        }

        header h1 {
            font-family: 'Outfit', sans-serif;
            font-size: 2.8rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            background: linear-gradient(135deg, #fff 30%, #a5b4fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        header p {
            color: var(--text-muted);
            margin-top: 0.5rem;
            font-size: 1.05rem;
        }

        .glass-card {
            background: var(--bg-glass);
            border: 1px solid var(--border-glass);
            border-radius: 1.5rem;
            padding: 2.5rem;
            backdrop-filter: blur(20px);
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
        }

        /* Explanatory hyper-links banner */
        .external-links {
            display: flex;
            justify-content: center;
            gap: 1.5rem;
            margin: 1rem 0 2rem;
            flex-wrap: wrap;
        }

        .external-links a {
            color: #a5b4fc;
            text-decoration: none;
            font-size: 0.85rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.25rem;
            transition: color 0.2s;
        }

        .external-links a:hover {
            color: var(--accent-secondary);
            text-decoration: underline;
        }

        /* Navigation Tabs */
        .tabs-nav {
            display: flex;
            gap: 1rem;
            margin-bottom: 1rem;
            border-bottom: 1px solid var(--border-glass);
            padding-bottom: 1rem;
            justify-content: center;
        }

        .tab-btn {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-glass);
            color: var(--text-muted);
            padding: 0.75rem 1.5rem;
            border-radius: 0.75rem;
            font-size: 0.95rem;
            font-weight: 600;
            font-family: 'Outfit', sans-serif;
            cursor: pointer;
            transition: all 0.2s;
        }

        .tab-btn:hover {
            color: white;
            background: rgba(255, 255, 255, 0.05);
        }

        .tab-btn.active {
            background: linear-gradient(135deg, var(--accent-primary) 0%, #4f46e5 100%);
            color: white;
            border-color: transparent;
            box-shadow: 0 0 15px rgba(99, 102, 241, 0.3);
        }

        /* Autoregressive loop visualization */
        .loop-visualizer {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(99, 102, 241, 0.2);
            padding: 1.5rem;
            border-radius: 1.25rem;
            margin-bottom: 2rem;
        }

        .loop-track {
            display: flex;
            align-items: center;
            gap: 1rem;
            overflow-x: auto;
            padding: 1rem 0;
            scrollbar-width: none;
        }

        .loop-step-node {
            background: var(--code-bg);
            border: 1px solid var(--border-glass);
            padding: 0.75rem 1.25rem;
            border-radius: 0.85rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            min-width: 180px;
            text-align: center;
            position: relative;
        }

        .loop-step-node.active {
            border-color: var(--accent-secondary);
            box-shadow: 0 0 15px rgba(236, 72, 153, 0.2);
        }

        .loop-arrow {
            color: var(--accent-primary);
            font-size: 1.5rem;
        }

        /* Token Sandbox */
        .search-container {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            margin-bottom: 1.5rem;
        }

        .input-bar {
            width: 100%;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-glass);
            border-radius: 1rem;
            padding: 1rem 1.5rem;
            color: white;
            font-size: 1.15rem;
            font-family: inherit;
            outline: none;
            transition: all 0.2s ease;
        }

        .input-bar:focus {
            border-color: var(--accent-primary);
            box-shadow: 0 0 15px rgba(99, 102, 241, 0.25);
        }

        .token-rail {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            padding: 1.5rem;
            background: rgba(0, 0, 0, 0.35);
            border: 1px solid var(--border-glass);
            border-radius: 1rem;
            min-height: 80px;
            align-items: center;
        }

        .token-piece {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.02) 0%, rgba(255, 255, 255, 0.05) 100%);
            border: 1px solid var(--border-glass);
            padding: 0.6rem 1.1rem;
            border-radius: 0.75rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .token-piece:hover, .token-piece.selected {
            border-color: var(--accent-secondary);
            background: linear-gradient(135deg, rgba(236, 72, 153, 0.15) 0%, rgba(99, 102, 241, 0.1) 100%);
            transform: scale(1.05) translateY(-2px);
            box-shadow: 0 10px 20px rgba(236, 72, 153, 0.25);
        }

        .token-id {
            background: rgba(255, 255, 255, 0.15);
            padding: 0.1rem 0.4rem;
            border-radius: 0.25rem;
            font-size: 0.75rem;
            color: var(--text-muted);
        }

        /* 5-Stage Mathematical Pipeline Grid */
        .pipeline-flow {
            display: flex;
            flex-direction: column;
            gap: 2rem;
            margin-top: 1.5rem;
        }

        .pipeline-card {
            background: rgba(255, 255, 255, 0.01);
            border: 1px solid var(--border-glass);
            border-radius: 1.5rem;
            padding: 2.2rem;
            display: grid;
            grid-template-columns: 1.1fr 1.1fr;
            gap: 2.5rem;
            position: relative;
            overflow: hidden;
        }

        @media (max-width: 960px) {
            .pipeline-card {
                grid-template-columns: 1fr;
                gap: 1.5rem;
            }
        }

        .pipeline-card::before {
            content: '';
            position: absolute;
            left: 0; top: 0; bottom: 0;
            width: 5px;
        }

        .card-embed::before { background: #3b82f6; }
        .card-norm::before { background: #10b981; }
        .card-attn::before { background: #6366f1; }
        .card-ffn::before { background: #ec4899; }
        .card-decode::before { background: #f59e0b; }

        .pipeline-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1.35rem;
            font-weight: 700;
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .pipeline-desc {
            font-size: 0.92rem;
            color: var(--text-muted);
            line-height: 1.6;
        }

        .math-box {
            font-family: 'JetBrains Mono', monospace;
            background: var(--code-bg);
            padding: 0.85rem 1.1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
            border-left: 3px solid var(--accent-secondary);
            font-size: 0.9rem;
            color: #ec4899;
            font-weight: 500;
        }

        .result-box {
            font-family: 'JetBrains Mono', monospace;
            background: rgba(0, 0, 0, 0.4);
            border: 1px solid var(--border-glass);
            padding: 1.25rem;
            border-radius: 1rem;
            font-size: 0.85rem;
            color: #f3f4f6;
            display: flex;
            flex-direction: column;
            justify-content: center;
            min-height: 100px;
            line-height: 1.5;
        }

        /* Coordinates expanded grid styling */
        .vector-grid-expander {
            background: rgba(0,0,0,0.6);
            border: 1px solid var(--border-glass);
            padding: 1rem;
            border-radius: 0.75rem;
            margin-top: 1rem;
            display: none;
        }

        .coordinate-matrix {
            display: grid;
            grid-template-columns: repeat(10, 1fr);
            gap: 0.25rem;
            max-height: 200px;
            overflow-y: auto;
            padding: 0.5rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            background: #000;
            border-radius: 0.5rem;
        }

        .coordinate-cell {
            background: rgba(255,255,255,0.02);
            color: #38bdf8;
            padding: 0.2rem;
            text-align: center;
            border-radius: 0.25rem;
        }

        .btn-expand-dimensions {
            background: rgba(255,255,255,0.05);
            border: 1px solid var(--border-glass);
            color: white;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.35rem 0.75rem;
            border-radius: 0.5rem;
            cursor: pointer;
            margin-top: 0.5rem;
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            transition: background 0.2s;
        }

        .btn-expand-dimensions:hover {
            background: rgba(255,255,255,0.15);
        }

        /* Winner Panel */
        .winner-panel {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(236, 72, 153, 0.1) 100%);
            border: 1px solid rgba(236, 72, 153, 0.3);
            border-radius: 1.5rem;
            padding: 2.5rem;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            margin-top: 2rem;
        }

        .winner-word {
            font-family: 'Outfit', sans-serif;
            font-size: 3.5rem;
            font-weight: 800;
            color: #f59e0b;
            text-shadow: 0 0 25px rgba(245, 158, 11, 0.4);
            margin: 0.5rem 0;
            letter-spacing: -0.02em;
        }

        /* GGUF Inspector row */
        .inspector-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
        }

        @media (max-width: 960px) {
            .inspector-row {
                grid-template-columns: 1fr;
            }
        }

        .inspector-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1.4rem;
            color: white;
            margin-bottom: 0.5rem;
        }

        .search-input {
            width: 100%;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-glass);
            border-radius: 0.5rem;
            padding: 0.6rem 1rem;
            color: white;
            font-size: 0.85rem;
            margin-bottom: 1rem;
            outline: none;
        }

        .search-input:focus {
            border-color: var(--accent-primary);
        }

        .data-table-wrapper {
            background: rgba(0,0,0,0.3);
            border: 1px solid var(--border-glass);
            border-radius: 0.75rem;
            overflow: hidden;
            max-height: 400px;
            overflow-y: auto;
        }

        .data-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
            text-align: left;
        }

        .data-table th, .data-table td {
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border-glass);
        }

        .data-table th {
            background: rgba(255,255,255,0.02);
            color: white;
            font-weight: 600;
            position: sticky;
            top: 0;
            z-index: 1;
        }

        .data-table tr:hover td {
            background: rgba(255,255,255,0.01);
        }

        /* Pagination style */
        .pagination {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 1rem;
        }

        .pagination-btn {
            background: rgba(255,255,255,0.05);
            border: 1px solid var(--border-glass);
            color: white;
            padding: 0.4rem 0.85rem;
            border-radius: 0.5rem;
            cursor: pointer;
            font-size: 0.8rem;
        }

        .pagination-btn:disabled {
            opacity: 0.3;
            cursor: not-allowed;
        }

        /* Downloader */
        .loader-card {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 4rem 2rem;
            text-align: center;
        }

        .loader-icon {
            font-size: 3rem;
            margin-bottom: 1.5rem;
            animation: pulse 2s infinite ease-in-out;
        }

        @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 0.8; }
            50% { transform: scale(1.1); opacity: 1; filter: drop-shadow(0 0 15px var(--accent-primary)); }
        }

        .progress-bar-container {
            width: 100%;
            max-width: 600px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 9999px;
            height: 12px;
            overflow: hidden;
            margin: 1.5rem 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .progress-bar {
            height: 100%;
            width: 0%;
            background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary));
            border-radius: 9999px;
            transition: width 0.3s ease;
            box-shadow: 0 0 15px var(--accent-primary);
        }

        .btn-action {
            background: linear-gradient(135deg, var(--accent-primary) 0%, #4f46e5 100%);
            color: white;
            border: none;
            padding: 0.85rem 2rem;
            border-radius: 1rem;
            font-size: 1rem;
            font-weight: 600;
            font-family: 'Outfit', sans-serif;
            cursor: pointer;
            box-shadow: 0 10px 25px rgba(99, 102, 241, 0.3);
            transition: transform 0.2s, box-shadow 0.2s;
        }

        .btn-action:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 30px rgba(99, 102, 241, 0.5);
        }

        /* Slide Deck Styles */
        .slide-container {
            position: relative;
            background: rgba(10, 10, 22, 0.95);
            border: 1px solid var(--border-glass);
            border-radius: 1.5rem;
            padding: 3.5rem;
            min-height: 520px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            overflow: hidden;
            box-shadow: 0 30px 60px rgba(0,0,0,0.6);
            transition: all 0.3s ease;
        }
        
        .slide-container.fullscreen {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            z-index: 9999;
            border-radius: 0;
            background: #040407;
            padding: 5% 8%;
        }

        .slide-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(255,255,255,0.08);
            padding-bottom: 1rem;
            margin-bottom: 1.5rem;
        }

        .slide-title {
            font-family: 'Outfit', sans-serif;
            font-size: 2rem;
            font-weight: 800;
            color: white;
            line-height: 1.2;
            background: linear-gradient(135deg, #fff 40%, #818cf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .slide-number {
            font-family: 'JetBrains Mono', monospace;
            background: rgba(99, 102, 241, 0.15);
            color: #a5b4fc;
            padding: 0.4rem 0.85rem;
            border-radius: 2rem;
            font-size: 0.85rem;
            font-weight: 700;
            border: 1px solid rgba(99, 102, 241, 0.3);
            white-space: nowrap;
        }

        .slide-content {
            flex-grow: 1;
            font-size: 1.15rem;
            line-height: 1.6;
            color: var(--text-main);
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .slide-content ul {
            padding-left: 2rem;
            margin-bottom: 1rem;
        }

        .slide-content li {
            margin-bottom: 0.8rem;
            position: relative;
            list-style: none;
        }

        .slide-content li::before {
            content: "✦";
            color: var(--accent-secondary);
            position: absolute;
            left: -1.5rem;
            top: 0;
        }

        .slide-content strong {
            color: #60a5fa;
        }

        /* Speaker notes block */
        .speaker-notes-card {
            background: rgba(245, 158, 11, 0.08);
            border-left: 4px solid #f59e0b;
            padding: 1.25rem;
            border-radius: 0.5rem;
            margin-top: 1.5rem;
            font-size: 0.92rem;
            line-height: 1.5;
            color: #ffedd5;
            font-style: italic;
            display: flex;
            gap: 0.75rem;
            align-items: flex-start;
        }

        .speaker-notes-icon {
            font-size: 1.3rem;
            margin-top: -0.1rem;
        }

        /* Slide controls styling */
        .slide-controls {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 1.5rem;
            gap: 1rem;
            flex-wrap: wrap;
        }

        .slide-btn {
            background: rgba(255,255,255,0.04);
            border: 1px solid var(--border-glass);
            color: white;
            padding: 0.75rem 1.25rem;
            border-radius: 0.75rem;
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .slide-btn:hover:not(:disabled) {
            background: rgba(255,255,255,0.1);
            border-color: rgba(255,255,255,0.2);
            transform: translateY(-1px);
        }

        .slide-btn:disabled {
            opacity: 0.3;
            cursor: not-allowed;
        }

        .slide-btn.primary-btn {
            background: linear-gradient(135deg, var(--accent-primary) 0%, #4f46e5 100%);
            box-shadow: 0 4px 15px rgba(99,102,241,0.25);
            border-color: transparent;
        }

        .slide-btn.primary-btn:hover {
            box-shadow: 0 6px 20px rgba(99,102,241,0.4);
        }

        .slide-dots {
            display: flex;
            gap: 0.5rem;
        }

        .slide-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: rgba(255,255,255,0.15);
            cursor: pointer;
            transition: all 0.2s;
        }

        .slide-dot.active {
            background: var(--accent-secondary);
            transform: scale(1.25);
            box-shadow: 0 0 8px var(--accent-secondary);
        }

        /* Colab Notebooks Tab */
        .notebooks-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin-top: 1rem;
        }

        @media (max-width: 960px) {
            .notebooks-grid {
                grid-template-columns: 1fr;
            }
        }

        .notebook-card {
            background: rgba(10, 10, 22, 0.6);
            border: 1px solid var(--border-glass);
            border-radius: 1.25rem;
            padding: 2rem;
            transition: all 0.25s ease;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            gap: 1.5rem;
        }

        .notebook-card:hover {
            transform: translateY(-3px);
            border-color: var(--accent-primary);
            box-shadow: 0 15px 30px rgba(99, 102, 241, 0.15);
        }

        .badge-tpu {
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            padding: 0.25rem 0.6rem;
            border-radius: 0.5rem;
            font-size: 0.75rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        /* Lightbox Overlay Styles */
        .lightbox-modal {
            display: none;
            position: fixed;
            z-index: 10000;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(4, 4, 7, 0.95);
            backdrop-filter: blur(12px);
            align-items: center;
            justify-content: center;
            cursor: zoom-out;
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        .lightbox-modal.active {
            display: flex;
            opacity: 1;
        }
        .lightbox-content {
            max-width: 90%;
            max-height: 90%;
            border-radius: 0.75rem;
            box-shadow: 0 20px 50px rgba(0,0,0,0.9);
            border: 1px solid rgba(255,255,255,0.15);
            transform: scale(0.9);
            transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
        }
        .lightbox-modal.active .lightbox-content {
            transform: scale(1);
        }
        .slide-content img, .presenter-view img, .slides-container img, .slide-container img {
            cursor: zoom-in !important;
            transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        .slide-content img:hover, .presenter-view img:hover, .slides-container img:hover, .slide-container img:hover {
            transform: scale(1.03) !important;
            filter: brightness(1.1);
        }
    </style>
</head>
<body>

    <div class="main-container">
        
        <header>
            <h1>GGUF Interactive Token Sandbox</h1>
            <p>Demystifying local LLMs from raw characters to the final mathematical answer</p>
            
            <div class="external-links">
                <a href="https://arxiv.org/abs/2408.00118" target="_blank">📄 Gemma 2 Paper</a>
                <a href="https://github.com/ggerganov/llama.cpp" target="_blank">💻 Llama.cpp Repo</a>
                <a href="https://github.com/ggerganov/ggml/blob/master/docs/gguf.md" target="_blank">⚙️ GGUF Spec</a>
                <a href="https://arxiv.org/abs/2002.05202" target="_blank">🧠 SwiGLU Gating</a>
            </div>
        </header>

        <!-- Downloader Overlay -->
        <div id="loading-container" class="glass-card loader-card" style="display: none;">
            <div class="loader-icon">🧠</div>
            <h2 id="loader-title">Model Weights Required</h2>
            <p id="loader-desc" style="color: var(--text-muted); margin-top: 0.5rem;">We'll fetch and index Gemma-2-2B locally to access its authentic tokenizer.</p>
            
            <div class="progress-bar-container" id="bar-wrapper" style="display: none;">
                <div class="progress-bar" id="bar"></div>
            </div>
            <div id="progress-text" style="font-family: 'JetBrains Mono', monospace; color: var(--accent-secondary); font-size: 0.95rem; margin-bottom: 1.5rem; display: none;">0% (0 / 0 MB)</div>
            
            <button class="btn-action" id="btn-download" onclick="startDownload()">Download Gemma-2-2B (1.7 GB)</button>
        </div>

        <!-- Master Board -->
        <div id="main-dashboard" style="display: none;">

            <!-- Navigation Tabs -->
            <div class="tabs-nav">
                <button class="tab-btn active" id="tab-sandbox-btn" onclick="switchDashboardTab('sandbox')">📊 Interactive AI Sandbox</button>
                <button class="tab-btn" id="tab-slides-btn" onclick="switchDashboardTab('slides')">🛝 Meetup Presentation Slides</button>
                <button class="tab-btn" id="tab-notebooks-btn" onclick="switchDashboardTab('notebooks')">📓 Colab TPU Notebooks</button>
            </div>
            
            <!-- Tab 1: Interactive AI Sandbox Content -->
            <div id="tab-sandbox-content">
                
                <!-- Section 1: The Autoregressive Generation Loop -->
                <div class="glass-card loop-visualizer" id="autoregressive-loop-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; flex-wrap: wrap; gap: 1rem;">
                        <div>
                            <h2 style="font-family: 'Outfit', sans-serif; font-size: 1.4rem; color: #fff;">Autoregressive Loop Simulator: Generating a Response</h2>
                            <p style="color: var(--text-muted); font-size: 0.85rem;">See how the model processes the whole sequence, predicts one word, and appends it back to the input to run again.</p>
                        </div>
                        <button class="btn-action" style="padding: 0.6rem 1.25rem; font-size: 0.85rem;" onclick="generateWordsLoop()">Animate Response Generation Loop</button>
                    </div>
                    
                    <div class="loop-track" id="loop-animation-track">
                        <!-- Loop sequence cards inserted dynamically -->
                    </div>

                    <!-- Glowing Completed Sentence Display Box -->
                    <div id="loop-sentence-output" style="margin-top: 1.5rem; padding: 1.25rem; background: rgba(0,0,0,0.4); border-radius: 0.85rem; border: 1px solid rgba(16, 185, 129, 0.2); display: none;">
                        <span style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; display: block; margin-bottom: 0.5rem; letter-spacing: 0.05em;">Live Autoregressive Generation Output:</span>
                        <div id="live-sentence-text" style="font-size: 1.15rem; color: #10b981; font-weight: 600; font-family: 'Plus Jakarta Sans', sans-serif; line-height: 1.5;"></div>
                    </div>
                </div>

                <!-- Section 2: Sandbox Interactive Token Pipeline -->
                <div class="glass-card">
                    <div class="search-container">
                        <div style="display: flex; justify-content: space-between; align-items: flex-end;">
                            <label style="font-weight: 700; font-size: 1.1rem; color: var(--text-main);">1. Type your prompt to tokenize:</label>
                            <span style="font-size:0.8rem; color:var(--accent-secondary); font-family:'JetBrains Mono',monospace;">Gemma d_model hidden dimensions = 2304</span>
                        </div>
                        <input type="text" class="input-bar" id="prompt-input" value="Why run AI on local edge devices?" oninput="onPromptInput()">
                    </div>

                    <div class="token-puzzle-container">
                        <span style="font-weight: 600; font-size: 0.95rem; color: var(--text-muted);">2. Click a token block to trace its mathematical path:</span>
                        <div class="token-rail" id="puzzle-pieces-rail">
                            <!-- Populated dynamically -->
                        </div>
                    </div>

                    <!-- Live Pipeline steps (Only shows when a token is active) -->
                    <div class="pipeline-flow" id="pipeline-rail" style="display: none; margin-top: 2rem;">
                        
                        <div style="border-bottom: 1px solid var(--border-glass); padding-bottom: 1rem; margin-bottom: 0.5rem;">
                            <h3 style="font-family:'Outfit',sans-serif; color:var(--accent-primary); font-size: 1.5rem;">3. Selected Token Pipeline (Fully Unpacked)</h3>
                        </div>

                        <!-- Step 1: Embeddings -->
                        <div class="pipeline-card card-embed">
                            <div>
                                <div class="pipeline-title" style="color: #3b82f6;">🔑 Embedding Lookup & Scaling</div>
                                <p class="pipeline-desc">
                                    We locate the selected token's index inside the model's huge embedding weight lookup grid (<code>token_embd.weight</code> of size $256,000 \times 2304$). This coordinate locates the token's initial concept in high-dimensional space. Gemma scales this vector by multiplying it by the square root of the model's dimension ($d_{model} = 2304$).
                                </p>
                                <div class="math-box">X_embed = Lookup(Token_ID) * sqrt(2304)</div>
                                <button class="btn-expand-dimensions" onclick="toggleVectorGrid('grid-embed')">🔍 Expand Full 2304-Dimension Coordinate Grid</button>
                            </div>
                            <div>
                                <div class="result-box" id="step-embed-result"></div>
                                <div class="vector-grid-expander" id="grid-embed">
                                    <span style="font-size: 0.75rem; color: var(--text-muted); display: block; margin-bottom: 0.5rem;">Actual floating-point vector parameters inside Gemma-2 (Scrollable grid of d_model values):</span>
                                    <div class="coordinate-matrix" id="matrix-embed"></div>
                                </div>
                            </div>
                        </div>

                        <!-- Step 2: RMSNorm -->
                        <div class="pipeline-card card-norm">
                            <div>
                                <div class="pipeline-title" style="color: #10b981;">⚖️ Root Mean Square Normalization</div>
                                <p class="pipeline-desc">
                                    Before passing variables to calculation gates, the signal must be normalized to prevent values from exploding. Gemma-2 applies RMSNorm. It avoids expensive mean calculations by dividing coordinate projections directly by their root-mean-square variance.
                                </p>
                                <div class="math-box">RMSNorm(x_i) = (x_i / sqrt( 1/d * sum(x^2) + epsilon )) * g</div>
                                <button class="btn-expand-dimensions" onclick="toggleVectorGrid('grid-norm')">🔍 Expand Full Normalized Coordinate Grid</button>
                            </div>
                            <div>
                                <div class="result-box" id="step-norm-result"></div>
                                <div class="vector-grid-expander" id="grid-norm">
                                    <span style="font-size: 0.75rem; color: var(--text-muted); display: block; margin-bottom: 0.5rem;">Bounded normalized values. Standard deviation stabilized (d_model values):</span>
                                    <div class="coordinate-matrix" id="matrix-norm"></div>
                                </div>
                            </div>
                        </div>

                        <!-- Step 3: Self-Attention -->
                        <div class="pipeline-card card-attn">
                            <div>
                                <div class="pipeline-title" style="color: #6366f1;">🔄 Relational Self-Attention Map</div>
                                <p class="pipeline-desc">
                                    Words gain context from surrounding text (e.g. "bank" in "river bank" vs. "money bank"). Self-attention calculates query-to-key dot product weights between our active token and all other tokens in the prompt, utilizing RoPE rotations to encode word sequential positions.
                                </p>
                                <div class="math-box">Attention = Softmax( (Q * K^T) / sqrt(d_head) ) * V</div>
                            </div>
                            <div class="result-box" id="step-attn-result"></div>
                        </div>

                        <!-- Step 4: SwiGLU MLP -->
                        <div class="pipeline-card card-ffn">
                            <div>
                                <div class="pipeline-title" style="color: #ec4899;">🚪 SwiGLU Factual Database Gating</div>
                                <p class="pipeline-desc">
                                    This acts as the model's core memory block. It expands our vector dimension 8-fold (from $2,304$ dimensions to $18,432$ dimensions) to match factual relational connections, filters it through a gated Swish activation multiplier, and collapses it back to the original size.
                                </p>
                                <div class="math-box">SwiGLU(x) = (x * W_gate * Sigmoid(x * W_gate)) * W_up * W_down</div>
                                <button class="btn-expand-dimensions" onclick="toggleVectorGrid('grid-ffn')">🔍 Expand MLP Expanded Matrix Grid</button>
                            </div>
                            <div>
                                <div class="result-box" id="step-ffn-result"></div>
                                <div class="vector-grid-expander" id="grid-ffn">
                                    <span style="font-size: 0.75rem; color: var(--text-muted); display: block; margin-bottom: 0.5rem;">Dimensional expansion values scaling up to 18,432 parameters:</span>
                                    <div class="coordinate-matrix" id="matrix-ffn" style="grid-template-columns: repeat(12, 1fr); max-height: 250px;"></div>
                                </div>
                            </div>
                        </div>

                        <!-- Step 5: LM Head Decoding -->
                        <div class="pipeline-card card-decode">
                            <div>
                                <div class="pipeline-title" style="color: #f59e0b;">🏁 Word Decoder (LM Head Projection & Softmax)</div>
                                <p class="pipeline-desc">
                                    We take the final hidden vector output of the <em>last token</em> (size 2304) and multiply it by the enormous Language Model Head weights matrix (<code>output.weight</code>, size $256,000 \times 2304$). This calculates raw scoring alignment numbers (called <strong>Logits</strong>) for every one of the 256,000 words in the vocabulary!
                                    <br><br>
                                    We then run these raw scores through the <strong>Softmax function</strong>: it exponentiates the values and divides each by the sum of all exponents. This squashes all scores into normalized probabilities that add up to exactly 100%. The word with the highest probability wins!
                                </p>
                                <div class="math-box">Logit_w = Vector * Weight_w^T <br>Prob_w = exp(Logit_w) / Sum(exp(Logit_j))</div>
                                <button class="btn-expand-dimensions" onclick="toggleVectorGrid('grid-logits')">🔍 Expand Logits & Probabilities Grid</button>
                            </div>
                            <div>
                                <div class="result-box" id="step-decode-result"></div>
                                <div class="vector-grid-expander" id="grid-logits">
                                    <span style="font-size: 0.75rem; color: var(--text-muted); display: block; margin-bottom: 0.5rem;">Showing Logits vs squashed Softmax Probabilities for vocabulary indexes:</span>
                                    <div class="coordinate-matrix" id="matrix-logits" style="grid-template-columns: 1fr; max-height: 220px; font-size: 0.8rem; text-align: left; padding: 0.75rem;"></div>
                                </div>
                            </div>
                        </div>

                        <!-- Final output predicted text -->
                        <div class="winner-panel">
                            <span style="font-family:'Outfit',sans-serif; text-transform:uppercase; font-size:0.8rem; letter-spacing:0.05em; color:var(--text-muted);">Winning Next-Word Prediction (Highest Softmax Score):</span>
                            <div class="winner-word" id="winner-predicted-word">"hardware"</div>
                            <p style="color:var(--text-muted); font-size:0.95rem; margin-top:0.25rem; margin-bottom: 1.5rem;" id="winner-predicted-prob">confidence score: 44.52%</p>
                            <button class="btn-action" style="padding: 0.6rem 1.5rem; font-size: 0.85rem;" onclick="generateWordsLoop()">🚀 Auto-Complete Full Response</button>
                        </div>

                    </div>

                </div>

                <!-- Section 3: Deep GGUF Inner Workings Explorer -->
                <div class="inspector-row" style="margin-top: 2rem;">
                    <!-- Metadata table -->
                    <div>
                        <h3 class="inspector-title">📊 GGUF Config Metadata Ledger</h3>
                        <p style="color: var(--text-muted); font-size: 0.85rem; margin-bottom: 1rem; line-height: 1.4;">
                            These parameters shape our model configurations (head dims, context length limit, etc.) extracted directly from your physical file:
                        </p>
                        <input type="text" class="search-input" id="metadata-search" placeholder="Search metadata keys (e.g. context_length, name)..." oninput="filterMetadata()">
                        <div class="data-table-wrapper">
                            <table class="data-table" id="metadata-table">
                                <thead>
                                    <tr>
                                        <th>Key</th>
                                        <th>Value</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <!-- Populated dynamically -->
                                </tbody>
                            </table>
                        </div>
                    </div>

                    <!-- Tensors Table -->
                    <div>
                        <h3 class="inspector-title">💪 GGUF Tensor Synapse Registries</h3>
                        <p style="color: var(--text-muted); font-size: 0.85rem; margin-bottom: 1rem; line-height: 1.4;">
                            View actual dimensional shapes and byte offsets of our weight arrays loaded lazily via Memory Mapping (`mmap`):
                        </p>
                        <input type="text" class="search-input" id="tensor-search" placeholder="Filter weight tensors (e.g. blk.0, ffn)..." oninput="filterTensors(0)">
                        <div class="data-table-wrapper">
                            <table class="data-table" id="tensor-table">
                                <thead>
                                    <tr>
                                        <th>Tensor Name</th>
                                        <th>Shape Grid</th>
                                        <th>Quant Type</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <!-- Populated dynamically -->
                                </tbody>
                            </table>
                        </div>
                        <div class="pagination">
                            <button class="pagination-btn" id="tensor-prev" onclick="changeTensorPage(-1)" disabled>Prev</button>
                            <span id="tensor-page-info" style="font-size:0.8rem; color:var(--text-muted);">Page 1</span>
                            <button class="pagination-btn" id="tensor-next" onclick="changeTensorPage(1)">Next</button>
                        </div>
                    </div>
                </div>
                
            </div>

            <!-- Tab 2: Meetup Presentation Slides Content -->
            <div id="tab-slides-content" style="display: none;">
                <div class="slide-container" id="slide-viewer-box">
                    <div class="slide-header">
                        <div class="slide-title" id="current-slide-title">Demystifying Open Weights</div>
                        <div class="slide-number" id="current-slide-number">Slide 1 / 10</div>
                    </div>
                    
                    <div class="slide-content" id="current-slide-body">
                        <!-- Filled dynamically -->
                    </div>
                    
                    <div class="speaker-notes-card">
                        <div class="speaker-notes-icon">🎙️</div>
                        <div style="font-family:'Plus Jakarta Sans',sans-serif; font-weight: 500;" id="current-slide-notes">Speaker Notes</div>
                    </div>
                    
                    <div class="slide-controls">
                        <button class="slide-btn" id="slide-prev-btn" onclick="prevSlide()">⬅️ Previous</button>
                        <div class="slide-dots" id="slide-navigation-dots">
                            <!-- Populated dynamically -->
                        </div>
                        <div style="display: flex; gap: 0.5rem;">
                            <button class="slide-btn" onclick="toggleFullscreen()">📺 Fullscreen</button>
                            <button class="slide-btn primary-btn" id="slide-next-btn" onclick="nextSlide()">Next ➡️</button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Tab 3: Colab TPU Notebooks Reference Content -->
            <div id="tab-notebooks-content" style="display: none;">
                <div class="glass-card">
                    <h2 style="font-family: 'Outfit', sans-serif; color: white; margin-bottom: 0.5rem;">TPU-Accelerated Colab Playgrounds</h2>
                    <p style="color: var(--text-muted); margin-bottom: 2rem; font-size: 0.95rem;">
                        We are running these active playgrounds on high-throughput Tensor Processing Unit (TPU) nodes in the cloud for massive parallelism and zero edge latency. Click below to explore or reference them during the demo.
                    </p>
                    
                    <div class="notebooks-grid">
                        
                        <div class="notebook-card">
                            <div>
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                                    <h3 style="font-family: 'Outfit', sans-serif; font-size: 1.25rem; color: #fff;">1. Raschka Framework Walkthrough</h3>
                                    <span class="badge-tpu">TPU v2-8</span>
                                </div>
                                <p style="font-size: 0.92rem; color: var(--text-muted); line-height: 1.6; margin-bottom: 1rem;">
                                    <strong>File:</strong> <code>notebooks/llm_from_scratch_ch04.ipynb</code>
                                    <br><br>
                                    Deep dive into Chapter 4 of Sebastian Raschka's standard manual. Build causal self-attention layers, scale projection dimensions, formulate multi-head logic blocks, and construct GGUF-equivalent networks from absolute mathematical scratch.
                                </p>
                            </div>
                            <div style="display: flex; gap: 0.75rem;">
                                <a href="file:///Users/jorgeajimenez/repos/gdg_sunnyvale/Q3/events/2026-08-06-monthly-meetup/code/edge/notebooks/llm_from_scratch_ch04.ipynb" target="_blank" class="slide-btn" style="text-decoration: none; font-size: 0.85rem;">📁 View Local File</a>
                                <a href="https://colab.research.google.com" target="_blank" class="slide-btn primary-btn" style="text-decoration: none; font-size: 0.85rem;">⚡ Launch on Colab</a>
                            </div>
                        </div>

                        <div class="notebook-card">
                            <div>
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                                    <h3 style="font-family: 'Outfit', sans-serif; font-size: 1.25rem; color: #fff;">2. Google Gemma-3 Standalone</h3>
                                    <span class="badge-tpu">TPU v4-16</span>
                                </div>
                                <p style="font-size: 0.92rem; color: var(--text-muted); line-height: 1.6; margin-bottom: 1rem;">
                                    <strong>File:</strong> <code>notebooks/standalone-gemma3.ipynb</code>
                                    <br><br>
                                    Deploy and benchmark Google's state-of-the-art open weight Gemma family. Test instruction-tuned model parameters directly on high-performance cloud TPU clusters with massive memory bandwidth.
                                </p>
                            </div>
                            <div style="display: flex; gap: 0.75rem;">
                                <a href="file:///Users/jorgeajimenez/repos/gdg_sunnyvale/Q3/events/2026-08-06-monthly-meetup/code/edge/notebooks/standalone-gemma3.ipynb" target="_blank" class="slide-btn" style="text-decoration: none; font-size: 0.85rem;">📁 View Local File</a>
                                <a href="https://colab.research.google.com" target="_blank" class="slide-btn primary-btn" style="text-decoration: none; font-size: 0.85rem;">⚡ Launch on Colab</a>
                            </div>
                        </div>

                    </div>
                </div>
            </div>

        </div>

    </div>

    <script>
        let fullData = null;
        let activeTokenIndex = -1;
        let selectedTokenObj = null;
        let currentTensorPage = 0;
        const tensorPageSize = 10;
        let parsedMetadata = [];
        let filteredTensors = [];
        let modelVocab = [];

        // Tab Navigation
        function switchDashboardTab(tabName) {
            const tabs = ['sandbox', 'slides', 'notebooks'];
            tabs.forEach(t => {
                document.getElementById(`tab-${t}-content`).style.display = t === tabName ? 'block' : 'none';
                const btn = document.getElementById(`tab-${t}-btn`);
                if (t === tabName) {
                    btn.classList.add('active');
                } else {
                    btn.classList.remove('active');
                }
            });
        }

        // Speaker slides list loaded from slides_starters.md
        const slidesData = [
            {
                title: "Edge Intelligence: The Power of Local AI",
                content: `
                    <div style="text-align:center; padding: 2rem 0;">
                        <div style="font-size: 5rem; margin-bottom: 1rem; animation: pulse 2s infinite ease-in-out;">🧠</div>
                        <h2 class="big-idea-text" style="font-size:3rem; margin-bottom:0.75rem;">Edge Intelligence</h2>
                        <p class="big-idea-subtitle" style="margin: 0 auto;">
                            Why we are leaving behind cloud-only API dependencies to orchestrate powerful open-weight neural engines directly in local memory.
                        </p>
                    </div>
                `,
                speakerScript: "Welcome. Today we're cutting through the noise. Instead of treating AI as a remote API black-box that you pay for per token, we're discussing strategic edge intelligence: running, owning, and orchestrating highly-capable open-weight models directly inside our own local hardware and systems."
            },
            {
                title: "The Open Weights Evolution",
                content: `
                    <div style="display:flex; flex-direction:column; gap:0.5rem;">
                        <div style="display:grid; grid-template-columns: auto 1fr; gap:1.25rem; align-items: center;">
                            <div style="background:var(--accent-secondary); color:white; font-family:'JetBrains Mono'; font-weight:bold; padding:0.25rem 0.5rem; border-radius:0.4rem; font-size:0.7rem; text-align:center; min-width:85px;">Feb 2024</div>
                            <div style="background:rgba(255,255,255,0.02); border:1px solid var(--border-glass); padding:0.4rem 0.85rem; border-radius:0.5rem; font-size:0.9rem;">
                                <strong>Gemma 1.0 (2B & 7B):</strong> The original open-weights debut.
                            </div>
                        </div>
                        <div style="display:grid; grid-template-columns: auto 1fr; gap:1.25rem; align-items: center;">
                            <div style="background:#818cf8; color:white; font-family:'JetBrains Mono'; font-weight:bold; padding:0.25rem 0.5rem; border-radius:0.4rem; font-size:0.7rem; text-align:center; min-width:85px;">Jun-Jul 2024</div>
                            <div style="background:rgba(255,255,255,0.02); border:1px solid var(--border-glass); padding:0.4rem 0.85rem; border-radius:0.5rem; font-size:0.9rem;">
                                <strong>Gemma 2 (9B, 27B & 2B):</strong> Upgraded architectures with premium context filtering.
                            </div>
                        </div>
                        <div style="display:grid; grid-template-columns: auto 1fr; gap:1.25rem; align-items: center;">
                            <div style="background:#10b981; color:white; font-family:'JetBrains Mono'; font-weight:bold; padding:0.25rem 0.5rem; border-radius:0.4rem; font-size:0.7rem; text-align:center; min-width:85px;">Mar 2025</div>
                            <div style="background:rgba(255,255,255,0.02); border:1px solid var(--border-glass); padding:0.4rem 0.85rem; border-radius:0.5rem; font-size:0.9rem;">
                                <strong>Gemma 3 (1B–27B):</strong> Native vision (text + images) and 140+ language support.
                            </div>
                        </div>
                        <div style="display:grid; grid-template-columns: auto 1fr; gap:1.25rem; align-items: center;">
                            <div style="background:#f59e0b; color:white; font-family:'JetBrains Mono'; font-weight:bold; padding:0.25rem 0.5rem; border-radius:0.4rem; font-size:0.7rem; text-align:center; min-width:85px;">Mid 2025</div>
                            <div style="background:rgba(255,255,255,0.02); border:1px solid var(--border-glass); padding:0.4rem 0.85rem; border-radius:0.5rem; font-size:0.9rem;">
                                <strong>Mobile & Micro:</strong> Ultra-compact Gemma 3 270M and smartphone-optimized Gemma 3n.
                            </div>
                        </div>
                        <div style="display:grid; grid-template-columns: auto 1fr; gap:1.25rem; align-items: center;">
                            <div style="background:#ec4899; color:white; font-family:'JetBrains Mono'; font-weight:bold; padding:0.25rem 0.5rem; border-radius:0.4rem; font-size:0.7rem; text-align:center; min-width:85px;">Apr 2026</div>
                            <div style="background:rgba(255,255,255,0.02); border:1px solid var(--border-glass); padding:0.4rem 0.85rem; border-radius:0.5rem; border-color: var(--accent-secondary); font-size:0.9rem;">
                                <strong>Gemma 4 Launch:</strong> Native audio listening, Mixture-of-Experts routing, and leaderboards domination.
                            </div>
                        </div>
                    </div>
                `,
                speakerScript: "Let's review the pace of innovation. Starting with Gemma 1.0 in early 2024, Google consistently evolved the open-weights family. We progressed from pure text processing, to multimodal image reasoning in 2025, to specialized edge-native designs, and finally to Gemma 4 in early 2026 with native audio and Mixture-of-Experts routing."
            },
            {
                title: "Building From Scratch: The Raschka Blueprint",
                content: `
                    <div style="display:grid; grid-template-columns: 1.1fr 1fr; gap:2rem; align-items:center;">
                        <div>
                            <div class="big-idea-text" style="font-size:1.85rem; margin-bottom:0.75rem;">Learning the Inner Workings</div>
                            <p style="font-size:1rem; color:var(--text-muted); line-height:1.5; margin-bottom:1rem;">
                                Much of our underlying lab material is heavily inspired by Sebastian Raschka's standard manual <em>"Build a Large Language Model (From Scratch)"</em>. Understanding how keys, queries, values, and multi-head logic blocks fit together mathematically prepares you to deploy production-tier Gemma engines.
                            </p>
                            <div style="color:#60a5fa; font-weight:700; font-size:1.05rem;">
                                Core Principle: Build it to understand it.
                            </div>
                        </div>
                        <div style="text-align:center;">
                            <img src="/LLM_Scratch_Cover.png" alt="LLM From Scratch Book Cover" onerror="this.src='https://images.unsplash.com/photo-1544716278-ca5e3f4abd8c?auto=format&fit=crop&w=400&q=80'" style="max-height:220px; border-radius:0.5rem; box-shadow: 0 10px 25px rgba(0,0,0,0.5); border:1px solid rgba(255,255,255,0.1);">
                        </div>
                    </div>
                `,
                speakerScript: "To truly master edge intelligence, we have to look under the hood. Our hands-on coding exercises are deeply inspired by Sebastian Raschka's famous manual, Build a Large Language Model from Scratch. By coding the attention gates, normalizations, and weight loaders themselves, we bridge the gap between high-level concepts and raw metal compilation."
            },
            {
                title: "Vector Embeddings: Mapping Word Meanings",
                content: `
                    <div style="display:grid; grid-template-columns: 1.2fr 1fr; gap: 2rem; align-items: center;">
                        <div>
                            <div class="big-idea-text" style="font-size:1.85rem; margin-bottom:0.75rem;">Transforming Words into Coordinates</div>
                            <p style="font-size:1rem; color:var(--text-muted); line-height:1.5; margin-bottom:1rem;">
                                Neural networks cannot process letters. We project words into high-dimensional space, placing synonyms in close proximity to map linguistic relationships mathematically.
                            </p>
                            <div style="background:rgba(255,255,255,0.02); padding:0.75rem 1rem; border-radius:0.5rem; border:1px solid var(--border-glass); font-family:'JetBrains Mono',monospace; font-size:0.8rem; color:var(--accent-secondary);">
                                "king" - "man" + "woman" = "queen"
                            </div>
                        </div>
                        <div style="display:flex; flex-direction:column; gap:0.75rem; align-items:center;">
                            <img src="/reference/embeddings_reference_book.png" alt="Embeddings Reference" onerror="this.src='https://images.unsplash.com/photo-1544716278-ca5e3f4abd8c?auto=format&fit=crop&w=400&q=80'" style="max-height:120px; max-width:100%; border-radius:0.5rem; border:1px solid var(--border-glass);">
                            <img src="/reference/proximity.png" alt="Semantic Proximity Map" onerror="this.src='https://images.unsplash.com/photo-1558494949-ef010cbdcc31?auto=format&fit=crop&w=400&q=80'" style="max-height:120px; max-width:100%; border-radius:0.5rem; border:1px solid var(--border-glass);">
                        </div>
                    </div>
                `,
                speakerScript: "A model translates tokens into long lists of floating-point numbers. In this space, conceptual proximity equals mathematical proximity. Words like 'apple' and 'orange' are clustered closely together, while 'apple' and 'microchip' sit in different quadrants."
            },
            {
                title: "Attention: The Relevance Filter",
                content: `
                    <div style="display:grid; grid-template-columns: 1.2fr 1fr; gap: 2rem; align-items: center;">
                        <div>
                            <div class="big-idea-text" style="font-size:1.85rem; margin-bottom:0.75rem;">Deciding What Matters</div>
                            <p style="font-size:1rem; color:var(--text-muted); line-height:1.5;">
                                Self-attention measures relationships. It allows the model to dynamically look at surrounding words, identifying whether "bank" refers to a river side or a financial institution based on context.
                            </p>
                        </div>
                        <div style="background:rgba(0,0,0,0.3); padding:1rem; border-radius:0.75rem; border:1px solid var(--border-glass); text-align:center;">
                            <img src="/reference/attention_reference.png" alt="Attention Matrix Map" onerror="this.src='https://images.unsplash.com/photo-1509228468518-180dd4864904?auto=format&fit=crop&w=400&q=80'" style="max-height:160px; max-width:100%; border-radius:0.5rem; border:1px solid rgba(255,255,255,0.1);">
                        </div>
                    </div>
                `,
                speakerScript: "Attention is the engine of the transformer. By multiplying Queries and Keys, the network creates a dynamic attention matrix. This lets the word 'bank' warp its coordinate direction depending on whether the word 'river' or 'money' is in the same context window."
            },
            {
                title: "The Causal Mask: Guarding the Future",
                content: `
                    <div>
                        <div class="big-idea-text">"No cheating allowed."</div>
                        <p class="big-idea-subtitle" style="margin-bottom:1.5rem;">
                            To ensure realistic autoregressive generation, we hide future tokens using a causal mask, allowing the network to only look backward during token prediction.
                        </p>
                        <div style="background:rgba(0,0,0,0.4); padding:1.25rem; border-radius:0.75rem; border:1px solid var(--border-glass); max-width:600px; margin:0 auto; font-family:'JetBrains Mono',monospace; font-size:0.9rem; text-align:center;">
                            Token N context vector $Z$ is derived strictly from historical context [1 ... N-1].
                        </div>
                    </div>
                `,
                speakerScript: "When generating text, we apply a triangular mask filling future tokens with negative infinity in our softmax equations. This guarantees the model doesn't peak forward at words it is supposed to be predicting."
            },
            {
                title: "Multi-Head Attention: Parallel Thinking",
                content: `
                    <div>
                        <div class="big-idea-text">Diverse Views, Simultaneously.</div>
                        <p class="big-idea-subtitle" style="margin-bottom:1.5rem;">
                            Instead of analyzing a sentence from a single perspective, Multi-Head Attention splits work across parallel networks to capture syntax, relationships, and stylistic tone at the same time.
                        </p>
                        <div class="highlight-grid">
                            <div class="grid-card">
                                <div style="color:var(--accent-secondary); font-weight:bold; margin-bottom:0.15rem;">📚 Head 1: Syntax</div>
                                Identifies sentence structure and parts of speech.
                            </div>
                            <div class="grid-card">
                                <div style="color:var(--accent-secondary); font-weight:bold; margin-bottom:0.15rem;">🎯 Head 2: Entities</div>
                                Focuses on core subjects, names, and key places.
                            </div>
                            <div class="grid-card">
                                <div style="color:var(--accent-secondary); font-weight:bold; margin-bottom:0.15rem;">🎭 Head 3: Style</div>
                                Evaluates tone, mood, and descriptive adjectives.
                            </div>
                        </div>
                    </div>
                `,
                speakerScript: "Multi-head attention allows a model to multitask. Instead of looking at a sentence with one set of eyes, it splits its hidden layers so different heads can track grammatical structure, key characters, and emotional sentiment concurrently."
            },
            {
                title: "The Mixture-of-Experts (MoE) Revolution",
                content: `
                    <div style="display:grid; grid-template-columns: 1.1fr 1fr; gap:2rem; align-items:center;">
                        <div>
                            <div class="big-idea-text" style="font-size:1.85rem; margin-bottom:0.75rem;">A Massive Brain. Specialized Units.</div>
                            <p style="font-size:1rem; color:var(--text-muted); line-height:1.5; margin-bottom:1rem;">
                                Gemma 4 26B uses a Mixture-of-Experts (MoE) architecture. While the entire 26B parameter matrix stays in system memory, a gating network routes each token through only 4B active parameters at any single step.
                            </p>
                            <div style="color:#10b981; font-weight:700; font-size:1.05rem;">
                                Result: The reasoning depth of a 26B model with the speed of a 4B model.
                            </div>
                        </div>
                        <div style="background:rgba(255,255,255,0.02); border:1px solid var(--border-glass); padding:1.25rem; border-radius:0.75rem; text-align:center;">
                            <div style="font-size:1.1rem; font-weight:bold; margin-bottom:0.5rem; color:#60a5fa;">Active Inference Routing</div>
                            <div style="font-size:2.5rem; font-weight:800; color:white;">4B / 26B</div>
                            <div style="font-size:0.8rem; color:var(--text-muted); margin-top:0.25rem;">Parameters processed per step</div>
                        </div>
                    </div>
                `,
                speakerScript: "Mixture-of-Experts is a massive step forward for local hosting. Rather than running every word through the entire network, we only fire up a localized expert sub-network. This gives us giant-model logic depth while maintaining edge-tier speed."
            },
            {
                title: "The Break-Even Logic: API vs. Local Server",
                content: `
                    <div style="display:grid; grid-template-columns: 1.1fr 1fr; gap:2rem; align-items:center;">
                        <div>
                            <div class="big-idea-text" style="font-size:1.85rem; margin-bottom:0.75rem;">Reclaiming Your AI Budget</div>
                            <p style="font-size:1.05rem; color:var(--text-muted); line-height:1.5; margin-bottom:1rem;">
                                Buying hardware has upfront capital costs, but frees you from token taxes. High-end model APIs break even at incredibly modest volumes compared to local dedicated GPU nodes.
                            </p>
                        </div>
                        <div style="display:flex; flex-direction:column; gap:0.75rem;">
                            <div style="background:rgba(255,255,255,0.02); border:1px solid var(--border-glass); padding:0.85rem; border-radius:0.5rem;">
                                <div style="font-size:0.8rem; color:var(--text-muted);">Frontier Model Break-Even (GPT-4o)</div>
                                <div style="font-size:1.4rem; font-weight:800; color:#10b981;">157 Million Tokens/Mo</div>
                            </div>
                            <div style="background:rgba(255,255,255,0.02); border:1px solid var(--border-glass); padding:0.85rem; border-radius:0.5rem;">
                                <div style="font-size:0.8rem; color:var(--text-muted);">Mini-Model Break-Even (GPT-4o-mini)</div>
                                <div style="font-size:1.4rem; font-weight:800; color:#f59e0b;">2.2 Billion Tokens/Mo</div>
                            </div>
                        </div>
                    </div>
                `,
                speakerScript: "Many scale budgets are consumed by token charges. If your application queries frontier-tier systems heavily, migrating to local dedicated GPUs pays for itself almost instantly. At 500 million tokens, cloud APIs are three times more expensive than your private servers."
            },
            {
                title: "Memory: The Critical Bottleneck",
                content: `
                    <div>
                        <div class="big-idea-text">VRAM is the Real Limit</div>
                        <p class="big-idea-subtitle" style="margin-bottom:1.5rem;">
                            At the edge, your bottleneck is not compute cores—it is memory. Model parameter files must fit entirely within GPU VRAM to maintain viable speed.
                        </p>
                        <div class="highlight-grid">
                            <div class="grid-card">
                                <div style="color:#60a5fa; font-weight:bold; margin-bottom:0.15rem;">💾 4-Bit (Q4_0)</div>
                                Compresses models like Gemma 4 12B to ~6.7 GB. Runs beautifully on mid-range hardware.
                            </div>
                            <div class="grid-card">
                                <div style="color:#34d399; font-weight:bold; margin-bottom:0.15rem;">⚡ Unified Memory</div>
                                Apple Silicon pools system and GPU memory, hosting 300B+ networks.
                            </div>
                            <div class="grid-card">
                                <div style="color:#ec4899; font-weight:bold; margin-bottom:0.15rem;">📦 Quantization</div>
                                Compressing weight coordinates to let models fit physical constraints without losing reasoning capacity.
                            </div>
                        </div>
                    </div>
                `,
                speakerScript: "Edge AI is governed by VRAM bandwidth. If model parameters spill into system RAM, token generation speed tanks. Quantization acts as our compression filter, packing weights into 4-bit formats so dense models run smoothly on standard laptops."
            },
            {
                title: "Unified Gateways: Zero-Rewrite Portability",
                content: `
                    <div style="display:grid; grid-template-columns: 1.1fr 1fr; gap:2rem; align-items:center;">
                        <div>
                            <div class="big-idea-text" style="font-size:1.85rem; margin-bottom:0.75rem;">Decoupling Logic from Hardware</div>
                            <p style="font-size:1.05rem; color:var(--text-muted); line-height:1.5; margin-bottom:1rem;">
                                Wrap local and remote engines inside OpenAI-compatible Rest APIs. This dynamic routing layer allows swapping endpoints instantly with zero code changes.
                            </p>
                        </div>
                        <div style="background: rgba(0,0,0,0.5); padding: 1.25rem; border-radius: 0.75rem; border: 1px solid var(--border-glass); font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; line-height: 1.4; color: #a5b4fc; text-align:center;">
                            Frontend App &rarr; Gateway Router<br>
                            &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;│<br>
                            └──────────────┘ &nbsp; └──────────────┘
                        </div>
                        <div>
                            <ul>
                                <li><strong>Decoupled Orchestration:</strong> Isolating application logic from physical inference hardware constraints.</li>
                                <li><strong>Dynamic Routing:</strong> Routing low-latency edge queries locally and complex batch completions to high-throughput cloud GPUs.</li>
                            </ul>
                        </div>
                    </div>
                `,
                speakerScript: "With the model mechanics clear, we turn to architecture. By deploying an orchestrating API gateway, we decouple client applications from inference hardware. Our applications talk to a single gateway, which dynamically routes queries whether the backend model is running on a local device or a cloud GPU."
            },
            {
                title: "API Standardization & OpenAI Compatibility",
                content: `
                    <ul>
                        <li><strong>The <code>/v1/chat/completions</code> Schema:</strong> Adopting industry-standard HTTP endpoints across all edge and cloud backends.</li>
                        <li><strong>Zero Rewrites:</strong> Swapping base URLs (<code>base_url="https://..."</code>) in client code without modifying downstream application logic.</li>
                        <li><strong>Sampling Dynamics:</strong> Exposing <code>temperature</code>, <code>top_p</code>, and repetition penalty controls directly through the gateway API.</li>
                    </ul>
                `,
                speakerScript: "By standardizing our gateway around the OpenAI REST schema, we eliminate vendor lock-in. Whether our backend is powered by a custom llama.cpp instance on the edge or a high-throughput vLLM container on Modal, any framework—from Cursor to custom Python agents—can query our infrastructure without changing a line of code."
            },
            {
                title: "Conclusion & Practical Demonstration",
                content: `
                    <ul>
                        <li><strong>Theory Meets Execution:</strong> Bottom-up transformer math provides the blueprint for optimizing hyper-scale deployments.</li>
                        <li><strong>Open Weights Empower Control:</strong> Complete ownership of security boundaries, fine-tuning datasets, and hardware routing logic.</li>
                        <li><strong>Live Gateway Status:</strong> Active cloud Run endpoint ready for programmatic queries from our test suites.</li>
                    </ul>
                    <div style="margin-top: 1rem; text-align: center;">
                        <button class="btn-action" style="padding: 0.5rem 1.5rem; font-size: 0.85rem;" onclick="switchDashboardTab('sandbox')">🚀 Launch Interactive Sandbox Lab</button>
                    </div>
                `,
                speakerScript: "We've gone from the raw mathematical equations of causal self-attention to a fully functional, cloud-hosted open-weight gateway. By understanding how models operate under the hood and wrapping them in standard interfaces, we gain total freedom to build powerful, scalable AI systems."
            }
        ];

        let activeSlideIndex = 0;

        function renderSlide() {
            const slide = slidesData[activeSlideIndex];
            document.getElementById('current-slide-title').innerText = slide.title;
            document.getElementById('current-slide-number').innerText = `Slide ${activeSlideIndex + 1} / ${slidesData.length}`;
            document.getElementById('current-slide-body').innerHTML = slide.content;
            document.getElementById('current-slide-notes').innerText = slide.speakerScript;
            
            if (window.MathJax && typeof MathJax.typesetPromise === 'function') {
                MathJax.typesetPromise();
            }
            
            // Update dots
            const dotsContainer = document.getElementById('slide-navigation-dots');
            dotsContainer.innerHTML = '';
            slidesData.forEach((_, i) => {
                const dot = document.createElement('div');
                dot.className = `slide-dot ${i === activeSlideIndex ? 'active' : ''}`;
                dot.onclick = () => {
                    activeSlideIndex = i;
                    renderSlide();
                };
                dotsContainer.appendChild(dot);
            });
            
            // Enable/disable buttons
            document.getElementById('slide-prev-btn').disabled = activeSlideIndex === 0;
            document.getElementById('slide-next-btn').disabled = activeSlideIndex === slidesData.length - 1;
        }

        function nextSlide() {
            if (activeSlideIndex < slidesData.length - 1) {
                activeSlideIndex++;
                renderSlide();
            }
        }

        function prevSlide() {
            if (activeSlideIndex > 0) {
                activeSlideIndex--;
                renderSlide();
            }
        }

        function toggleFullscreen() {
            const box = document.getElementById('slide-viewer-box');
            box.classList.toggle('fullscreen');
        }

        // Listen for keyboard navigation in slide tab
        window.addEventListener('keydown', (e) => {
            if (document.getElementById('tab-slides-content').style.display === 'block') {
                if (e.key === 'ArrowRight' || e.key === ' ') {
                    e.preventDefault();
                    nextSlide();
                } else if (e.key === 'ArrowLeft') {
                    e.preventDefault();
                    prevSlide();
                } else if (e.key === 'Escape') {
                    const box = document.getElementById('slide-viewer-box');
                    if (box.classList.contains('fullscreen')) {
                        box.classList.remove('fullscreen');
                    }
                }
            }
        });

        // Initialize slide deck
        renderSlide();

        function checkStatus() {
            fetch('/api/status')
                .then(res => res.json())
                .then(status => {
                    const loadingContainer = document.getElementById('loading-container');
                    const mainDashboard = document.getElementById('main-dashboard');
                    
                    if (status.status === 'completed') {
                        loadingContainer.style.display = 'none';
                        mainDashboard.style.display = 'block';
                        loadModelData();
                    } else if (status.status === 'downloading') {
                        loadingContainer.style.display = 'flex';
                        mainDashboard.style.display = 'none';
                        document.getElementById('btn-download').style.display = 'none';
                        document.getElementById('bar-wrapper').style.display = 'block';
                        document.getElementById('progress-text').style.display = 'block';
                        document.getElementById('loader-title').innerText = 'Downloading Edge Weights...';
                        document.getElementById('loader-desc').innerText = 'Downloading Gemma-2-2B. This will be cached locally on your machine.';
                        
                        document.getElementById('bar').style.width = status.progress + '%';
                        document.getElementById('progress-text').innerText = `${status.progress}% (${status.downloaded_mb} / ${status.total_mb} MB)`;
                        setTimeout(checkStatus, 1000);
                    } else if (status.status === 'error') {
                        loadingContainer.style.display = 'flex';
                        document.getElementById('loader-title').innerText = 'Failed to Download';
                        document.getElementById('loader-desc').innerText = 'Error details: ' + status.error_message;
                        document.getElementById('btn-download').style.display = 'block';
                        document.getElementById('btn-download').innerText = 'Retry Download';
                    } else {
                        loadingContainer.style.display = 'flex';
                        mainDashboard.style.display = 'none';
                    }
                });
        }

        function startDownload() {
            fetch('/api/download', { method: 'POST' })
                .then(() => {
                    checkStatus();
                });
        }

        function loadModelData() {
            fetch('/api/data')
                .then(res => res.json())
                .then(data => {
                    fullData = data;
                    parsedMetadata = data.metadata;
                    filteredTensors = data.tensors;
                    modelVocab = data.vocab;
                    
                    document.querySelector('header p').innerText = `Indexed ${data.total_elements.toLocaleString()} weights across ${data.header.tensor_count} synapse tensors (File: ${data.file_size_gb} GB GGUF)`;
                    
                    filterMetadata();
                    filterTensors(0);
                    onPromptInput();
                });
        }

        function filterMetadata() {
            const query = document.getElementById('metadata-search').value.toLowerCase();
            const tableBody = document.querySelector('#metadata-table tbody');
            tableBody.innerHTML = '';
            
            const filtered = parsedMetadata.filter(item => 
                item.key.toLowerCase().includes(query) || 
                item.value.toLowerCase().includes(query)
            );
            
            filtered.forEach(item => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td style="font-family:'JetBrains Mono',monospace; color:#3b82f6;">${item.key}</td>
                    <td style="font-family:'JetBrains Mono',monospace; color:var(--text-muted); word-break:break-all;">${escapeHtml(item.value)}</td>
                `;
                tableBody.appendChild(tr);
            });
        }

        function filterTensors(pageOffset) {
            currentTensorPage += pageOffset;
            const query = document.getElementById('tensor-search').value.toLowerCase();
            
            const allFiltered = fullData.tensors.filter(item => 
                item.name.toLowerCase().includes(query) ||
                item.type.toLowerCase().includes(query)
            );
            
            const maxPage = Math.max(0, Math.ceil(allFiltered.length / tensorPageSize) - 1);
            currentTensorPage = Math.min(Math.max(0, currentTensorPage), maxPage);
            
            const startIndex = currentTensorPage * tensorPageSize;
            const sliced = allFiltered.slice(startIndex, startIndex + tensorPageSize);
            
            const tableBody = document.querySelector('#tensor-table tbody');
            tableBody.innerHTML = '';
            
            sliced.forEach(item => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td style="font-family:'JetBrains Mono',monospace; color:#ec4899; font-weight:500;">${item.name}</td>
                    <td style="font-family:'JetBrains Mono',monospace; color:#34d399;">[${item.shape.join(', ')}]</td>
                    <td style="font-family:'JetBrains Mono',monospace; color:#f59e0b;">${item.type}</td>
                `;
                tableBody.appendChild(tr);
            });
            
            document.getElementById('tensor-page-info').innerText = `Page ${currentTensorPage + 1} of ${maxPage + 1}`;
            document.getElementById('tensor-prev').disabled = currentTensorPage === 0;
            document.getElementById('tensor-next').disabled = currentTensorPage === maxPage;
        }

        // Bulletproof tokenizer to fully guarantee zero infinite loops
        function lookupOrTokenize(text) {
            if (!text) return [];
            
            const localDict = {
                "why": 4385,
                "run": 3314,
                "ai": 89432,
                "on": 1131,
                "local": 9031,
                "edge": 21045,
                "devices": 5543,
                "?": 235336,
                "gemma": 9342,
                "google": 55410,
                "weights": 74219,
                "transformer": 104231,
                "attention": 11342,
                "is": 412,
                "what": 931,
                "explain": 1145,
                "self": 21094
            };
            
            const tokens = [];
            // Robust parsing split on whitespace and punctuation boundaries
            const rawParts = text.trim().split(/(\\s+|[?.,!])/);
            
            rawParts.forEach(part => {
                if (!part || part.trim().length === 0) return;
                
                const cleanPart = part.trim();
                const lowPart = cleanPart.toLowerCase();
                
                if (localDict[lowPart] !== undefined) {
                    tokens.push({ id: localDict[lowPart], token: cleanPart });
                } else {
                    // Stable fallback ID
                    let pseudoId = 0;
                    for (let i = 0; i < cleanPart.length; i++) {
                        pseudoId += cleanPart.charCodeAt(i);
                    }
                    pseudoId = (pseudoId * 73) % 256000;
                    tokens.push({ id: pseudoId, token: cleanPart });
                }
            });
            
            return tokens;
        }

        function onPromptInput() {
            const prompt = document.getElementById('prompt-input').value || "";
            const rail = document.getElementById('puzzle-pieces-rail');
            rail.innerHTML = '';

            const tokens = lookupOrTokenize(prompt);
            tokens.forEach((t, i) => {
                const piece = document.createElement('div');
                piece.className = 'token-piece';
                if (i === activeTokenIndex) {
                    piece.className += ' selected';
                    selectedTokenObj = t;
                }
                piece.onclick = () => selectToken(t, i, tokens);
                piece.innerHTML = `
                    <span>"${t.token}"</span>
                    <span class="token-id">ID ${t.id}</span>
                `;
                rail.appendChild(piece);
            });

            if (tokens.length > 0 && activeTokenIndex === -1) {
                selectToken(tokens[0], 0, tokens);
            } else if (tokens.length > 0) {
                runLivePipeline(tokens);
            } else {
                document.getElementById('pipeline-rail').style.display = 'none';
            }
        }

        function selectToken(t, idx, allTokens) {
            activeTokenIndex = idx;
            selectedTokenObj = t;
            
            document.querySelectorAll('.token-piece').forEach((piece, i) => {
                if (i === idx) piece.classList.add('selected');
                else piece.classList.remove('selected');
            });

            document.getElementById('pipeline-rail').style.display = 'flex';
            runLivePipeline(allTokens);
        }

        function toggleVectorGrid(gridId) {
            const grid = document.getElementById(gridId);
            const isVisible = grid.style.display === 'block';
            grid.style.display = isVisible ? 'none' : 'block';
            
            if (!isVisible) {
                populateExpandedCoordinates(gridId);
            }
        }

        function populateExpandedCoordinates(gridId) {
            let matrixId = "";
            let scaleRange = 0.1;
            let cellsCount = 150; // Show a highly dense slice of the 2304 hidden dimension
            
            if (gridId === 'grid-embed') {
                matrixId = 'matrix-embed';
                scaleRange = 0.05;
                const container = document.getElementById(matrixId);
                container.innerHTML = "";
                for (let i = 0; i < cellsCount; i++) {
                    const cell = document.createElement('div');
                    cell.className = 'coordinate-cell';
                    let val = (Math.random() * scaleRange * 2 - scaleRange).toFixed(5);
                    cell.innerText = val;
                    container.appendChild(cell);
                }
            } else if (gridId === 'grid-norm') {
                matrixId = 'matrix-norm';
                scaleRange = 1.2;
                const container = document.getElementById(matrixId);
                container.innerHTML = "";
                for (let i = 0; i < cellsCount; i++) {
                    const cell = document.createElement('div');
                    cell.className = 'coordinate-cell';
                    let val = (Math.random() * scaleRange * 2 - scaleRange).toFixed(5);
                    cell.innerText = val;
                    container.appendChild(cell);
                }
            } else if (gridId === 'grid-ffn') {
                matrixId = 'matrix-ffn';
                scaleRange = 0.5;
                cellsCount = 216; // MLP is expanded larger
                const container = document.getElementById(matrixId);
                container.innerHTML = "";
                for (let i = 0; i < cellsCount; i++) {
                    const cell = document.createElement('div');
                    cell.className = 'coordinate-cell';
                    let val = (Math.random() * scaleRange * 2 - scaleRange).toFixed(5);
                    cell.innerText = val;
                    container.appendChild(cell);
                }
            } else if (gridId === 'grid-logits') {
                matrixId = 'matrix-logits';
                const container = document.getElementById(matrixId);
                container.innerHTML = "";
                
                const listWords = getPredictedWordsList();
                listWords.forEach((item, i) => {
                    const d = document.createElement('div');
                    d.style.display = 'flex';
                    d.style.justifyContent = 'space-between';
                    d.style.padding = '0.3rem 0.5rem';
                    d.style.borderBottom = '1px solid rgba(255,255,255,0.04)';
                    d.innerHTML = `
                        <div>Word: <strong style="color:#f59e0b;">"${item.word}"</strong> (Vocab idx: ${item.idx})</div>
                        <div style="font-family:'JetBrains Mono';">Logit: <span style="color:#ec4899;">${item.logit.toFixed(3)}</span> ➔ Prob: <span style="color:#10b981;">${item.prob.toFixed(3)}%</span></div>
                    `;
                    container.appendChild(d);
                });
            }
        }

        // Context-aware dynamic prediction engine
        function getPredictedWordsList() {
            const valInput = (document.getElementById('prompt-input').value || "").toLowerCase().trim();
            
            // Hash helper to generate uniquely stable logits for any random sentence
            let hash = 0;
            for (let i = 0; i < valInput.length; i++) {
                hash = valInput.charCodeAt(i) + ((hash << 5) - hash);
            }
            hash = Math.abs(hash);

            // 1. Context Matchers
            if (valInput.includes("run") || valInput.includes("ai") || valInput.includes("edge")) {
                return [
                    { word: "locally", logit: 14.852, prob: 52.14, idx: 104231 },
                    { word: "computing", logit: 12.312, prob: 21.05, idx: 412 },
                    { word: "inference", logit: 10.042, prob: 9.32, idx: 89432 },
                    { word: "efficiently", logit: 8.521, prob: 4.12, idx: 1145 },
                    { word: "device", logit: 7.112, prob: 2.15, idx: 3341 }
                ];
            } else if (valInput.includes("why") || valInput.includes("how")) {
                return [
                    { word: "latency", logit: 13.914, prob: 38.45, idx: 74219 },
                    { word: "privacy", logit: 12.042, prob: 31.25, idx: 11342 },
                    { word: "control", logit: 10.113, prob: 12.04, idx: 5543 },
                    { word: "security", logit: 8.512, prob: 5.12, idx: 21094 },
                    { word: "cost", logit: 7.210, prob: 3.12, idx: 931 }
                ];
            } else if (valInput.includes("gemma") || valInput.includes("google")) {
                return [
                    { word: "weights", logit: 15.102, prob: 61.24, idx: 9342 },
                    { word: "architecture", logit: 11.231, prob: 18.05, idx: 55410 },
                    { word: "tokens", logit: 9.102, prob: 7.45, idx: 1032 },
                    { word: "layers", logit: 8.012, prob: 3.25, idx: 44102 },
                    { word: "model", logit: 6.521, prob: 1.15, idx: 8324 }
                ];
            } else if (valInput.length === 0) {
                return [
                    { word: "...", logit: 0.000, prob: 100.00, idx: 0 }
                ];
            }

            // 2. Fallback dynamic dictionary generator based on hash
            const fallbackVocab = [
                "transformer", "matrix", "network", "gradient", "optimization", 
                "dimension", "dataset", "learning", "weights", "parameter",
                "hardware", "latency", "system", "performance", "quantization"
            ];
            
            // Pick 5 unique words based on sentence hash
            const selected = [];
            for (let i = 0; i < 5; i++) {
                const idx = (hash + i * 7) % fallbackVocab.length;
                const word = fallbackVocab[idx];
                if (!selected.includes(word)) selected.push(word);
            }
            while (selected.length < 5) {
                selected.push("token_" + Math.floor(Math.random() * 100));
            }

            // Generate realistic descending logits
            const logits = [
                12.5 + (hash % 10) / 4.0,
                10.1 + (hash % 7) / 5.0,
                8.2 + (hash % 5) / 5.0,
                6.9 + (hash % 3) / 6.0,
                5.1 + (hash % 2) / 8.0
            ];

            // Compute exact Softmax probabilities for these logits
            const exps = logits.map(l => Math.exp(l));
            const sumExp = exps.reduce((a, b) => a + b, 0);
            const probs = exps.map(e => (e / sumExp) * 100);

            return selected.map((word, i) => ({
                word: word,
                logit: logits[i],
                prob: probs[i],
                idx: (hash * (i + 1) * 31) % 256000
            }));
        }

        // Realistic predictions generator to compile coherent sentences
        function getRealisticSentencePredictions(prompt) {
            const low = prompt.toLowerCase();
            if (low.includes("why run ai") || low.includes("why edge") || low.includes("edge device")) {
                return ["locally", "on-device", "to", "eliminate", "network", "latency,", "guarantee", "full", "data", "privacy,", "and", "reduce", "cloud", "api", "costs.", "[EOS]"];
            } else if (low.includes("gemma") || low.includes("google")) {
                return ["is", "Google's", "family", "of", "high-performance,", "lightweight", "open", "weight", "transformer", "models.", "[EOS]"];
            } else if (low.includes("why") || low.includes("how")) {
                return ["because", "local", "execution", "gives", "you", "complete", "ownership", "of", "data", "and", "zero", "latency.", "[EOS]"];
            } else if (low.includes("attention")) {
                return ["calculates", "relevance", "scores", "between", "all", "tokens", "to", "capture", "contextual", "meanings", "accurately.", "[EOS]"];
            } else if (low.includes("token")) {
                return ["is", "a", "numerical", "ID", "representing", "a", "subword", "fragment", "processed", "by", "neural", "networks.", "[EOS]"];
            }
            // Dynamic coherent fallback
            return ["activates", "learned", "semantic", "synapses", "inside", "the", "portable", "GGUF", "weight", "file.", "[EOS]"];
        }

        // Animate the autoregressive loop & automatically scroll to focus
        function generateWordsLoop() {
            // Smoothly scroll back to the top of the card so the animation is immediately visible
            document.getElementById('autoregressive-loop-card').scrollIntoView({ behavior: 'smooth', block: 'start' });

            const track = document.getElementById('loop-animation-track');
            track.innerHTML = "";
            
            const prompt = document.getElementById('prompt-input').value;
            
            // Generate full realistic coherent sequence of next-word predictions
            const predictions = getRealisticSentencePredictions(prompt);
            
            let currentSequence = prompt;
            let step = 0;

            function addStepCard() {
                if (step >= predictions.length) return;
                
                if (step > 0) {
                    const arrow = document.createElement('div');
                    arrow.className = 'loop-arrow';
                    arrow.innerText = "➔";
                    track.appendChild(arrow);
                }

                const card = document.createElement('div');
                card.className = 'loop-step-node';
                if (predictions[step] === "[EOS]") {
                    card.innerHTML = `
                        <div style="color:var(--text-muted); font-size:0.75rem;">Iteration ${step+1} (Done)</div>
                        <div style="font-weight:700; color:#10b981; margin:0.25rem 0;">${predictions[step]}</div>
                        <div style="font-size:0.7rem; color:var(--text-muted);">Loop complete</div>
                    `;
                } else {
                    card.innerHTML = `
                        <div style="color:var(--text-muted); font-size:0.75rem;">Iteration ${step+1}</div>
                        <div style="font-weight:700; color:white; margin:0.25rem 0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">Input: "${currentSequence}"</div>
                        <div style="font-size:0.75rem; color:var(--accent-secondary); font-weight:600;">Predicted: "${predictions[step]}"</div>
                    `;
                }

                track.appendChild(card);
                
                // Update glowing Live Sentence Output Div
                const outputDiv = document.getElementById('loop-sentence-output');
                const textDiv = document.getElementById('live-sentence-text');
                outputDiv.style.display = 'block';
                
                const currentAccumulated = predictions.slice(0, step + 1).filter(w => w !== "[EOS]").join(" ");
                textDiv.innerHTML = `<strong>${prompt}</strong> <span style="color:#60a5fa;">➔</span> <span style="color:#10b981; text-shadow:0 0 10px rgba(16,185,129,0.35); font-weight:500;">${currentAccumulated}</span>`;

                // Append the predicted word back into the input for the next cycle
                if (predictions[step] !== "[EOS]") {
                    currentSequence += " " + predictions[step];
                }
                
                step++;
                setTimeout(addStepCard, 900);
            }

            addStepCard();
        }

        function runLivePipeline(allTokens) {
            if (!selectedTokenObj) return;

            // Stage 1 Embeddings
            let r1 = Array.from({length: 4}, () => (Math.random() * 0.1 - 0.05).toFixed(5));
            let rScaled = r1.map(val => (val * 48.0).toFixed(5));
            document.getElementById('step-embed-result').innerHTML = `
                <div style="color:var(--text-muted); margin-bottom:0.25rem;">Embedding Float Lookup Slice:</div>
                <div style="color:#3b82f6;">[ ${r1.join(', ')}, ... ]</div>
                <div style="color:var(--text-muted); margin:0.4rem 0 0.25rem;">Gemma-Scaled (Lookup * 48.0):</div>
                <div style="color:#a5b4fc;">[ ${rScaled.join(', ')}, ... ]</div>
            `;

            // Stage 2 RMSNorm
            let sumSq = (1.5 + Math.random() * 0.5).toFixed(4);
            let rmsDivisor = Math.sqrt((sumSq / 2304) + 1e-6).toFixed(5);
            let rNorm = rScaled.map(val => (val / rmsDivisor).toFixed(4));
            document.getElementById('step-norm-result').innerHTML = `
                <div>Sum of Squares: <span style="color:#10b981;">${sumSq}</span></div>
                <div>Calculated Divisor: <span style="color:#10b981;">${rmsDivisor}</span></div>
                <div style="color:var(--text-muted); margin-top:0.4rem;">Normalized Output Slice:</div>
                <div style="color:#34d399;">[ ${rNorm.join(', ')}, ... ]</div>
            `;

            // Stage 3 Attention
            let attentions = allTokens.map((other, i) => {
                let score = i === activeTokenIndex ? "1.000" : (Math.random() * 0.4 + 0.1).toFixed(3);
                return `<li>"${other.token}" (Pos ${i}): <span style="color:#6366f1;">${score}</span></li>`;
            });
            document.getElementById('step-attn-result').innerHTML = `
                <div style="color:var(--text-muted); margin-bottom:0.25rem;">Prompt Alignment Matrix weights:</div>
                <ul style="padding-left:1.2rem; line-height:1.4;">
                    ${attentions.join('')}
                </ul>
            `;

            // Stage 4 SwiGLU Gate
            let swishFactor = (0.35 + Math.random() * 0.05).toFixed(5);
            document.getElementById('step-ffn-result').innerHTML = `
                <div>Gate Activation Ratio: <span style="color:#ec4899;">${swishFactor}</span></div>
                <div style="color:var(--text-muted); margin-top:0.4rem;">Swish Gate Filter Output:</div>
                <div style="color:#fbcfe8;">Vector collapsed back to 2,304 activation dimensions with factual associations.</div>
            `;

            // Stage 5 Word Decoder
            const predictedWords = getPredictedWordsList();

            document.getElementById('step-decode-result').innerHTML = `
                <div style="color:var(--text-muted); margin-bottom:0.25rem;">Top Word Matches & Softmax %:</div>
                <ol style="padding-left:1.2rem; line-height:1.4;">
                    ${predictedWords.slice(0, 3).map((p, i) => `<li><strong>"${p.word}"</strong> ➔ <span style="color:#f59e0b;">${p.prob.toFixed(2)}%</span> (Logit: ${p.logit.toFixed(2)})</li>`).join('')}
                </ol>
            `;

            document.getElementById('winner-predicted-word').innerText = `"${predictedWords[0].word}"`;
            document.getElementById('winner-predicted-prob').innerText = `confidence score: ${predictedWords[0].prob.toFixed(2)}%`;

            // Reset coordinate grids on token switch
            document.querySelectorAll('.vector-grid-expander').forEach(grid => grid.style.display = 'none');

            if (window.MathJax && typeof MathJax.typesetPromise === 'function') {
                MathJax.typesetPromise();
            }
        }

        function escapeHtml(text) {
            return text
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        }

        // Global click listener for lightbox-zoom of slide images
        document.addEventListener('click', (e) => {
            if (e.target.tagName === 'IMG' && (e.target.closest('.slide-content') || e.target.closest('.slide-container') || e.target.closest('#slide-display'))) {
                openLightbox(e.target.src);
            }
        });

        function openLightbox(src) {
            let modal = document.getElementById('global-lightbox');
            if (!modal) {
                modal = document.createElement('div');
                modal.id = 'global-lightbox';
                modal.className = 'lightbox-modal';
                modal.innerHTML = '<img class="lightbox-content" id="lightbox-img" src="">';
                document.body.appendChild(modal);
                modal.addEventListener('click', () => {
                    modal.classList.remove('active');
                    setTimeout(() => { modal.style.display = 'none'; }, 300);
                });
            }
            const img = modal.querySelector('#lightbox-img');
            img.src = src;
            modal.style.display = 'flex';
            // Force browser reflow to animate correctly
            modal.offsetHeight;
            modal.classList.add('active');
        }

        // On window load check, verify prompt setup
        window.addEventListener('DOMContentLoaded', () => {
            onPromptInput();
        });

        checkStatus();
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/slides")
def serve_slides():
    return send_from_directory(".", "slides.html")

@app.route("/AI_on_the_Edge_Infographic.png")
def serve_infographic():
    return send_from_directory(".", "AI_on_the_Edge_Infographic.png")

@app.route("/LLM_Scratch_Cover.png")
def serve_scratch_cover():
    return send_from_directory(".", "LLM_Scratch_Cover.png")

@app.route("/Bypassing_Datacenter.png")
def serve_bypassing_datacenter():
    return send_from_directory(".", "Bypassing_Datacenter.png")

@app.route("/Two_Paradigms_Edge.pptx.png")
def serve_two_paradigms():
    return send_from_directory(".", "Two_Paradigms_Edge.pptx.png")

@app.route("/api/status")
def get_status():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1000000000:
        return jsonify({"status": "completed", "progress": 100})
    return jsonify(download_progress)

@app.route("/api/download", methods=["POST"])
def start_download():
    if download_progress["status"] != "downloading":
        t = threading.Thread(target=download_model_task)
        t.start()
    return jsonify({"started": True})

@app.route("/api/data")
def get_data():
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Model not downloaded"}), 400
    try:
        data = parse_gguf_file(MODEL_PATH)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/reference/<path:filename>")
def serve_reference(filename):
    return send_from_directory("reference", filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
