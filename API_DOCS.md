# AgriTech AI Backend API Documentation

Base URL: `http://localhost:5000`

## 1. Fetch All-India Mandi Rates
* **Endpoint:** `/api/mandi`
* **Method:** `GET`
* **Description:** Returns real-time market rates collected from official government sources across India.
* **Sample Response:**
```json
{
  "status": "success",
  "count": 500,
  "data": [
    {
      "state": "West Bengal",
      "district": "Purba Bardhaman",
      "market": "Katwa APMC",
      "commodity": "Potato",
      "min_price": "750",
      "max_price": "850"
    }
  ]
}