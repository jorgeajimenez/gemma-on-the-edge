import json
import urllib.request

# Address of our local inference server (replace 'localhost' with the host machine's IP for local network inference)
SERVER_URL = "http://localhost:5002/api/generate"

def query_local_server(prompt: str, stream: bool = True):
    payload = {
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": 128,
        "stream": stream
    }
    
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(
        SERVER_URL, 
        data=json.dumps(payload).encode("utf-8"), 
        headers=headers, 
        method="POST"
    )

    print(f"Prompt: {prompt}\n")
    print("Response: ", end="", flush=True)

    try:
        with urllib.request.urlopen(req) as response:
            if stream:
                # Handle SSE (Server-Sent Events) stream
                while True:
                    line = response.readline()
                    if not line:
                        break
                    
                    decoded_line = line.decode("utf-8").strip()
                    if decoded_line.startswith("data:"):
                        data_content = decoded_line[5:].strip()
                        if data_content == "[DONE]":
                            break
                        
                        try:
                            chunk_json = json.loads(data_content)
                            token_text = chunk_json.get("text", "")
                            print(token_text, end="", flush=True)
                        except json.JSONDecodeError:
                            pass
            else:
                # Handle standard non-streaming response
                result = json.loads(response.read().decode("utf-8"))
                print(result.get("text", ""))
                
    except Exception as e:
        print(f"\nError connecting to inference server: {e}")
        print("Make sure your server is running via: 'uv run server.py'")
    print("\n")


if __name__ == "__main__":
    prompt_input = "Q: What makes Sunnyvale, California unique?\nA:"
    # Run the streaming client (zero third-party dependencies required!)
    query_local_server(prompt_input, stream=True)
