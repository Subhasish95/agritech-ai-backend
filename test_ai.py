import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5vl:3b"

payload = {
    "model": OLLAMA_MODEL,
    "prompt": "Say exactly: Ollama AI is working",
    "stream": False
}

try:
    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    print("\nSUCCESS!")
    print(data.get("response", "").strip())

except requests.exceptions.ConnectionError:
    print("\nFAILED!")
    print("Could not connect to Ollama. Make sure Ollama is running.")

except Exception as e:
    print("\nFAILED!")
    print(type(e).__name__)
    print(e)