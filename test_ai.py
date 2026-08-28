import sys
import requests

# Reconfigure stdout to force UTF-8 printing in Windows terminals
sys.stdout.reconfigure(encoding='utf-8')

URL = "http://localhost:5000/api/ai-advisor"
payload = {
    "query": "আমার ২ বিঘা জমি আছে, আমি কি কৃষক বন্ধু প্রকল্পের সুবিধা পাব?"
}

print("Sending question to your local AI server...")
try:
    response = requests.post(URL, json=payload, timeout=30)
    if response.status_code == 200:
        result = response.json()
        print("\n--- AI RESPONSE RECEIVED ---")
        print(result.get("ai_response"))
    else:
        print(f"Error {response.status_code}: {response.text}")
except Exception as e:
    print(f"Connection failed: {e}")