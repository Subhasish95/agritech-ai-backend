import json
import requests

# Your official personal API key from data.gov.in
API_KEY = "579b464db66ec23bdd000001adbecb0aa918434e68318d643dd0dcba"

# Endpoint fetching up to 2,000 records across ALL Indian states (no state filter)
URL = f"https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070?api-key={API_KEY}&format=json&limit=2000"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def fetch_all_india_mandi_prices():
    print("Connecting to data.gov.in for All-India Mandi Rates...")
    try:
        response = requests.get(URL, headers=HEADERS, timeout=20)
        
        if response.status_code == 200:
            data = response.json()
            records = data.get('records', [])
            
            if records:
                # 1. Export raw records to JSON file
                output_filename = "all_india_mandi_prices.json"
                with open(output_filename, "w", encoding="utf-8") as f:
                    json.dump(records, f, indent=4, ensure_ascii=False)
                
                # 2. Extract and display summary stats
                states_found = sorted(list(set(item.get('state', 'Unknown') for item in records)))
                
                print(f"\n--- SUCCESS ---")
                print(f"Fetched {len(records)} live mandi records across India.")
                print(f"Covering {len(states_found)} States/UTs.")
                print(f"Data saved to: '{output_filename}'")
                
                print("\n--- STATES COVERED IN THIS BATCH ---")
                print(", ".join(states_found))
                
                print("\n--- SAMPLE DATA PREVIEW ---")
                for item in records[:5]:
                    state = item.get('state', 'N/A')
                    market = item.get('market', 'N/A')
                    commodity = item.get('commodity', 'N/A')
                    min_price = item.get('min_price', 'N/A')
                    max_price = item.get('max_price', 'N/A')
                    print(f"[{state}] {market} Mandi -> {commodity}: ₹{min_price} - ₹{max_price} / Quintal")
            else:
                print("Connected successfully, but no records were returned.")
                
        else:
            print(f"Failed to fetch data. Server HTTP status code: {response.status_code}")
            
    except requests.exceptions.Timeout:
        print("Error: Request timed out. Check your internet connection or try again shortly.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    fetch_all_india_mandi_prices()