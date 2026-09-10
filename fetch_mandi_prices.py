import json
import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DATA_GOV_API_KEY")

if not API_KEY:
    raise ValueError(
        "DATA_GOV_API_KEY is missing. Add it to your .env file."
    )

URL = (
    "https://api.data.gov.in/resource/"
    "9ef84268-d588-465a-a308-a864a43d0070"
    f"?api-key={API_KEY}&format=json&limit=2000"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def fetch_all_india_mandi_prices():
    print("Connecting to data.gov.in for All-India Mandi Rates...")

    try:
        response = requests.get(
            URL,
            headers=HEADERS,
            timeout=20
        )

        if response.status_code == 200:
            data = response.json()
            records = data.get("records", [])

            if records:
                output_filename = "all_india_mandi_prices.json"

                with open(
                    output_filename,
                    "w",
                    encoding="utf-8"
                ) as f:
                    json.dump(
                        records,
                        f,
                        indent=4,
                        ensure_ascii=False
                    )

                states_found = sorted(
                    list(
                        set(
                            item.get("state", "Unknown")
                            for item in records
                        )
                    )
                )

                print("\n--- SUCCESS ---")
                print(
                    f"Fetched {len(records)} live mandi "
                    "records across India."
                )
                print(
                    f"Covering {len(states_found)} States/UTs."
                )
                print(
                    f"Data saved to: '{output_filename}'"
                )

                print("\n--- STATES COVERED IN THIS BATCH ---")
                print(", ".join(states_found))

                print("\n--- SAMPLE DATA PREVIEW ---")

                for item in records[:5]:
                    state = item.get("state", "N/A")
                    market = item.get("market", "N/A")
                    commodity = item.get("commodity", "N/A")
                    min_price = item.get("min_price", "N/A")
                    max_price = item.get("max_price", "N/A")

                    print(
                        f"[{state}] {market} Mandi -> "
                        f"{commodity}: ₹{min_price} - "
                        f"₹{max_price} / Quintal"
                    )

            else:
                print(
                    "Connected successfully, but no records "
                    "were returned."
                )

        else:
            print(
                "Failed to fetch data. Server HTTP status code: "
                f"{response.status_code}"
            )

    except requests.exceptions.Timeout:
        print(
            "Error: Request timed out. Check your internet "
            "connection or try again shortly."
        )

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    fetch_all_india_mandi_prices()