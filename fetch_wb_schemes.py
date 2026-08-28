import json
import requests
from bs4 import BeautifulSoup

TARGET_URL = "https://matirkatha.net/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5"
}

# Verified local fallback database of West Bengal schemes
FALLBACK_SCHEMES = [
    {
        "title": "Krishak Bandhu (Natun) Scheme",
        "description": "Financial assistance up to ₹10,000 per year per acre for WB farmers, plus ₹2 Lakh death benefit cover.",
        "eligibility": "All farmers possessing arable land in West Bengal.",
        "link": "https://krishakbandhu.wb.gov.in/",
        "state": "West Bengal"
    },
    {
        "title": "Bangla Shasya Bima (Crop Insurance)",
        "description": "Fully state-sponsored crop insurance covering yield loss due to natural calamities.",
        "eligibility": "Farmers growing Paddy, Jute, Potato, Sugarcane, Maize, and pulses.",
        "link": "https://banglashasyabima.net/",
        "state": "West Bengal"
    },
    {
        "title": "Matir Katha Farmer Portal",
        "description": "Official agricultural advisory, soil testing results, and seasonal crop package of practices.",
        "eligibility": "Open access to all WB farmers.",
        "link": "https://matirkatha.net/",
        "state": "West Bengal"
    },
    {
        "title": "West Bengal Farm Mechanization Scheme",
        "description": "Subsidies on farm machinery, tractors, and modern equipment purchase (FMI/CHS/FMMC).",
        "eligibility": "Registered farmers and Farmer Producer Organizations (FPOs).",
        "link": "https://matirkatha.net/farm-mechanization/",
        "state": "West Bengal"
    }
]

def fetch_wb_schemes():
    print("Connecting to West Bengal Matir Katha Portal...")
    schemes_list = []
    
    try:
        response = requests.get(TARGET_URL, headers=HEADERS, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            for link in soup.find_all('a', href=True):
                title = link.text.strip()
                href = link['href']
                if len(title) > 15:
                    if href.startswith('/'):
                        href = f"https://matirkatha.net{href}"
                    schemes_list.append({
                        "title": title,
                        "description": "Scraped from official portal news/bulletins.",
                        "eligibility": "Check portal details.",
                        "link": href,
                        "state": "West Bengal"
                    })
            print(f"Successfully scraped {len(schemes_list)} links live from website.")
        else:
            print(f"Server returned HTTP Code: {response.status_code}. Activating verified local fallback...")
            schemes_list = FALLBACK_SCHEMES
            
    except Exception as e:
        print(f"Network request failed ({e}). Activating verified local fallback...")
        schemes_list = FALLBACK_SCHEMES

    # Save output to JSON file
    output_filename = "wb_schemes.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(schemes_list, f, indent=4, ensure_ascii=False)
        
    print(f"\n--- SUCCESS ---")
    print(f"Loaded {len(schemes_list)} West Bengal government schemes into '{output_filename}'.")
    
    print("\n--- SCHEMES PREVIEW ---")
    for item in schemes_list[:3]:
        print(f"• {item['title']} | Link: {item['link']}")

if __name__ == "__main__":
    fetch_wb_schemes()