import sys
import requests

# Ensure UTF-8 output for terminal display
sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = "http://localhost:5000/api"

def test_community_pipeline():
    print("--- 1. Testing Farmer Registration ---")
    reg_payload = {
        "full_name": "Ramesh Das",
        "phone": "9876543210",
        "password": "password123",
        "location": "Bardhaman, West Bengal",
        "primary_crop": "Paddy"
    }
    res = requests.post(f"{BASE_URL}/register", json=reg_payload)
    print("Register Response:", res.json())

    print("\n--- 2. Testing Farmer Login ---")
    login_payload = {
        "phone": "9876543210",
        "password": "password123"
    }
    res = requests.post(f"{BASE_URL}/login", json=login_payload)
    print("Login Response:", res.json())

    print("\n--- 3. Testing Creating a Community Post ---")
    post_payload = {
        "author_name": "Ramesh Das",
        "location": "Bardhaman",
        "content": "বৃষ্টির পর ধানের ফলন বেশ ভালোই মনে হচ্ছে! (Paddy crops are looking good after the rain!)",
        "media_url": "https://example.com/sample_paddy_image.jpg"
    }
    res = requests.post(f"{BASE_URL}/posts", json=post_payload)
    print("Create Post Response:", res.json())

    print("\n--- 4. Testing Fetching All Community Posts ---")
    res = requests.get(f"{BASE_URL}/posts")
    print("Get Posts Response:", res.json())

if __name__ == "__main__":
    try:
        test_community_pipeline()
    except Exception as e:
        print("Failed to connect to backend server. Make sure ai_assistant.py is running on http://localhost:5000")
        print("Error details:", e)