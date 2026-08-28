import base64
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from google import genai
import requests

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

# Database Configuration (SQLite)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///agritech.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Gemini API Key configuration (loaded securely from environment)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)


# =========================================================
# DATABASE MODELS
# =========================================================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(15), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(100), nullable=False)
    primary_crop = db.Column(db.String(100), nullable=True)


class CommunityPost(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    author_name = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(100), nullable=False)
    content = db.Column(db.Text, nullable=False)
    media_url = db.Column(db.String(255), nullable=True)  # Image/Video link
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# Create tables on startup
with app.app_context():
    db.create_all()


def load_data():
    try:
        with open("wb_schemes.json", "r", encoding="utf-8") as f:
            schemes = json.load(f)
    except Exception:
        schemes = []

    try:
        with open("all_india_mandi_prices.json", "r", encoding="utf-8") as f:
            mandi_prices = json.load(f)
    except Exception:
        mandi_prices = []

    return schemes, mandi_prices


# =========================================================
# AUTHENTICATION & USER MANAGEMENT ENDPOINTS
# =========================================================
@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    if not data.get("phone") or not data.get("password"):
        return (
            jsonify({"error": "Phone number and password are required"}),
            400,
        )

    if User.query.filter_by(phone=data.get("phone")).first():
        return jsonify({"error": "Farmer already registered"}), 400

    new_user = User(
        full_name=data.get("full_name", "Farmer"),
        phone=data.get("phone"),
        password=data.get("password"),
        location=data.get("location", "West Bengal"),
        primary_crop=data.get("primary_crop", "Paddy"),
    )
    db.session.add(new_user)
    db.session.commit()

    return jsonify(
        {"status": "success", "message": "Farmer registered successfully"}
    )


@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    user = User.query.filter_by(
        phone=data.get("phone"), password=data.get("password")
    ).first()

    if not user:
        return jsonify({"error": "Invalid phone number or password"}), 401

    return jsonify({
        "status": "success",
        "user": {
            "id": user.id,
            "full_name": user.full_name,
            "phone": user.phone,
            "location": user.location,
            "primary_crop": user.primary_crop,
        },
    })


@app.route('/api/farmers', methods=['GET'])
def get_farmers():
    farmers = User.query.all()
    farmers_data = [
        {
            "id": f.id,
            "full_name": f.full_name,
            "location": f.location,
            "primary_crop": f.primary_crop,
        }
        for f in farmers
    ]
    return jsonify({"status": "success", "data": farmers_data})


# =========================================================
# COMMUNITY FEED ENDPOINTS
# =========================================================
@app.route('/api/posts', methods=['GET'])
def get_posts():
    posts = CommunityPost.query.order_by(CommunityPost.created_at.desc()).all()
    posts_data = [
        {
            "id": p.id,
            "author_name": p.author_name,
            "location": p.location,
            "content": p.content,
            "media_url": p.media_url,
            "created_at": p.created_at.strftime("%Y-%m-%d %H:%M"),
        }
        for p in posts
    ]
    return jsonify({"status": "success", "data": posts_data})


@app.route('/api/posts', methods=['POST'])
def create_post():
    data = request.json
    if not data.get("content") or not data.get("author_name"):
        return jsonify({"error": "Author and content required"}), 400

    new_post = CommunityPost(
        author_name=data.get("author_name"),
        location=data.get("location", "West Bengal"),
        content=data.get("content"),
        media_url=data.get("media_url", ""),
    )
    db.session.add(new_post)
    db.session.commit()

    return jsonify({"status": "success", "message": "Post published"})


# =========================================================
# AGRICULTURAL CONTENT & TOOLS ENDPOINTS
# =========================================================
@app.route('/api/mandi', methods=['GET'])
def get_mandi_prices():
    _, mandi_prices = load_data()
    return jsonify({
        "status": "success",
        "count": len(mandi_prices),
        "data": mandi_prices,
    })


@app.route('/api/schemes', methods=['GET'])
def get_schemes():
    schemes, _ = load_data()
    return jsonify({
        "status": "success",
        "count": len(schemes),
        "data": schemes,
    })


@app.route('/api/crops', methods=['GET'])
def get_crops():
    try:
        with open("crop_calendar.json", "r", encoding="utf-8") as f:
            crops = json.load(f)
        return jsonify({"status": "success", "data": crops})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/tools', methods=['GET'])
def get_tools():
    try:
        with open("agri_tools.json", "r", encoding="utf-8") as f:
            tools = json.load(f)
        return jsonify({"status": "success", "data": tools})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/location-recommendation', methods=['GET'])
def get_location_recommendation():
    lat = request.args.get('lat', '22.5726')
    lon = request.args.get('lon', '88.3639')
    
    recommendations = {
        "location": "Detected Region",
        "current_climate_risk": "High summer heat / seasonal water scarcity",
        "traditional_crop_warning": "Avoid excessive paddy farming due to high water consumption.",
        "recommended_hybrid_crops": [
            {"crop": "Hybrid Maize", "water_requirement": "Low", "suitable_months": "March - June"},
            {"crop": "Drought-Resistant Millets (Ragi/Jowar)", "water_requirement": "Very Low", "suitable_months": "April - July"},
            {"crop": "Short-Duration Hybrid Pulses (Moong)", "water_requirement": "Low", "suitable_months": "March - May"}
        ]
    }
    return jsonify({"status": "success", "data": recommendations})


@app.route('/api/about', methods=['GET'])
def get_about():
    about_info = {
        "platform_name": "AgriTech Modern Farming Platform",
        "objective": "Empowering traditional farmers with modern techniques, crop disease AI, and local crop advice to prevent crop failure.",
        "team": "Subhasish Sarkar & Team"
    }
    return jsonify({"status": "success", "data": about_info})


# =========================================================
# WEATHER & AI ADVISOR ENDPOINTS
# =========================================================
@app.route('/api/weather', methods=['GET'])
def get_weather():
    lat = request.args.get('lat', '22.5726')
    lon = request.args.get('lon', '88.3639')
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=auto"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            return jsonify({"status": "success", "weather_data": res.json()})
        return jsonify({"error": "Failed to fetch weather data"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/ai-advisor', methods=['POST'])
def ai_advisor():
    user_query = request.json.get("query", "")
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    schemes, mandi = load_data()
    prompt = f"""
    You are an expert AI Agriculture Advisor for West Bengal farmers.
    Answer the farmer's question accurately using ONLY the government context below.
    Provide the response in simple, clear Bengali (or English if requested).
    
    --- WEST BENGAL SCHEMES DATA ---
    {json.dumps(schemes, ensure_ascii=False)}
    
    --- LIVE MANDI PRICES DATA (SAMPLE) ---
    {json.dumps(mandi[:20], ensure_ascii=False)}
    
    Farmer Question: {user_query}
    Answer:
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash', contents=prompt
        )
        return jsonify({"status": "success", "ai_response": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/predict-disease', methods=['POST'])
def predict_disease():
    data = request.json
    image_base64 = data.get("image")
    if not image_base64:
        return jsonify({"error": "No image provided"}), 400
    prompt = """
    Analyze this crop leaf image for pests or diseases. 
    Provide the response in structured JSON with these keys:
    1. disease_name (in Bengali & English)
    2. symptoms
    3. organic_treatment
    4. chemical_treatment
    5. recommended_youtube_search_query
    """
    try:
        image_bytes = base64.b64decode(image_base64)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                {"mime_type": "image/jpeg", "data": image_bytes},
                prompt,
            ],
        )
        return jsonify({"status": "success", "analysis": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("Starting AgriTech AI Backend on http://localhost:5000...")
    app.run(port=5000, debug=True)