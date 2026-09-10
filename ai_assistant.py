import base64
import json
import os
import re
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash


# =========================================================
# PATHS / ENVIRONMENT
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate"
)
OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5vl:3b"
)

MAX_IMAGE_SIZE_MB = 8
ALLOWED_IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


# =========================================================
# FLASK APP CONFIGURATION
# =========================================================

app = Flask(__name__)
CORS(app)

app.config["SQLALCHEMY_DATABASE_URI"] = (
    f"sqlite:///{BASE_DIR / 'agritech.db'}"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_SIZE_MB * 1024 * 1024

db = SQLAlchemy(app)


# =========================================================
# DATABASE MODELS
# =========================================================

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(15), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    location = db.Column(db.String(100), nullable=False)
    primary_crop = db.Column(db.String(100), nullable=True)


class CommunityPost(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    author_name = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(100), nullable=False)
    content = db.Column(db.Text, nullable=False)
    media_url = db.Column(db.String(255), nullable=True)
    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )


with app.app_context():
    db.create_all()


# =========================================================
# HELPERS
# =========================================================

def read_json_file(filename, fallback=None):
    """Safely read a JSON file from the project directory."""
    if fallback is None:
        fallback = []

    file_path = BASE_DIR / filename

    try:
        with file_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Warning: {filename} was not found.")
    except json.JSONDecodeError as exc:
        print(f"Warning: Invalid JSON in {filename}: {exc}")
    except Exception as exc:
        print(f"Warning: Could not read {filename}: {exc}")

    return fallback


def load_data():
    schemes = read_json_file("wb_schemes.json", [])
    mandi_prices = read_json_file(
        "all_india_mandi_prices.json",
        []
    )
    return schemes, mandi_prices


def get_current_season(month=None):
    """
    Basic Indian cropping-season mapping.
    Kharif: June-October
    Rabi: November-March
    Zaid: April-May
    """
    month = month or datetime.now().month

    if 6 <= month <= 10:
        return "Kharif"
    if month in (11, 12, 1, 2, 3):
        return "Rabi"
    return "Zaid"


def normalize_water_level(value):
    value = (value or "medium").strip().lower()

    aliases = {
        "very low": "very low",
        "very_low": "very low",
        "low": "low",
        "medium": "medium",
        "moderate": "medium",
        "high": "high",
    }
    return aliases.get(value, "medium")


def get_water_score(available, required):
    order = {
        "very low": 0,
        "low": 1,
        "medium": 2,
        "high": 3,
    }

    available_value = order.get(
        normalize_water_level(available),
        2
    )

    required_value = order.get(
        normalize_water_level(required),
        2
    )

    difference = available_value - required_value

    if difference >= 0:
        return 3

    if difference == -1:
        return -3

    if difference == -2:
        return -6

    return -8


def get_weather_data(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": "true",
        "daily": (
            "temperature_2m_max,"
            "temperature_2m_min,"
            "precipitation_sum"
        ),
        "forecast_days": 7,
        "timezone": "auto",
    }

    response = requests.get(
        url,
        params=params,
        timeout=15
    )
    response.raise_for_status()
    return response.json()


def build_crop_recommendations(
    crops,
    season,
    temperature,
    water_available,
    soil_type=None,
):
    """
    Scores crops using structured agricultural data.
    The LLM is not used to choose the crops.
    """
    recommendations = []

    for crop in crops:
        score = 0
        reasons = []

        crop_name = crop.get("crop", "Unknown")
        crop_seasons = [
            str(item).strip().lower()
            for item in crop.get("season", [])
        ]

        # Season is a hard requirement.
        if season.lower() in crop_seasons:
            score += 4
            reasons.append(
                f"Suitable for the {season} season"
            )
        else:
            continue

        # Temperature suitability.
        min_temp = crop.get("min_temperature")
        max_temp = crop.get("max_temperature")

        if (
            temperature is not None
            and isinstance(min_temp, (int, float))
            and isinstance(max_temp, (int, float))
        ):
            if min_temp <= temperature <= max_temp:
                score += 3
                reasons.append(
                    f"Current temperature ({temperature}°C) "
                    "is within its preferred range"
                )
            elif (
                min_temp - 3
                <= temperature
                <= max_temp + 3
            ):
                score += 1
                reasons.append(
                    "Current temperature is close to its "
                    "preferred range"
                )
            else:
                score -= 2
                reasons.append(
                    "Current temperature is outside its "
                    "preferred range"
                )

        # Water suitability. A larger shortage gets a larger penalty.
        required_water = crop.get(
            "water_requirement",
            "Medium"
        )

        water_score = get_water_score(
            water_available,
            required_water
        )

        score += water_score

        if water_score == 3:
            reasons.append(
                f"Water requirement ({required_water}) "
                "fits the selected availability"
            )
        elif water_score == -3:
            reasons.append(
                f"Requires {required_water} water, which is "
                "slightly above the selected availability"
            )
        elif water_score == -6:
            reasons.append(
                f"Requires {required_water} water, which is "
                "much higher than the selected availability"
            )
        else:
            reasons.append(
                f"Requires {required_water} water, which is "
                "far above the selected availability"
            )

        # Soil suitability.
        soil_types = [
            str(item).strip().lower()
            for item in crop.get("soil_types", [])
        ]

        if soil_type and soil_types:
            if soil_type.strip().lower() in soil_types:
                score += 2
                reasons.append(
                    f"Suitable for {soil_type} soil"
                )
            else:
                score -= 1
                reasons.append(
                    f"{soil_type} soil is not listed as a "
                    "preferred soil type"
                )

        recommendations.append({
            "crop": crop_name,
            "score": score,
            "season": crop.get("season", []),
            "temperature_range_c": {
                "min": min_temp,
                "max": max_temp,
            },
            "water_requirement": required_water,
            "soil_types": crop.get(
                "soil_types",
                []
            ),
            "reasons": reasons,
        })

    recommendations.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return recommendations

def looks_like_password_hash(value):
    if not value:
        return False

    return (
        value.startswith("pbkdf2:")
        or value.startswith("scrypt:")
    )


def ollama_generate(
    prompt,
    images=None,
    json_mode=False,
    timeout=180,
):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    if images:
        payload["images"] = images

    if json_mode:
        payload["format"] = "json"

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=timeout
    )

    if response.status_code != 200:
        raise RuntimeError(
            "Ollama request failed: "
            + response.text[:1000]
        )

    result = response.json()
    text = result.get("response", "").strip()

    if not text:
        raise RuntimeError(
            "Ollama returned an empty response"
        )

    return text


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "success",
        "message": "AgriTech AI Backend is running",
        "model": OLLAMA_MODEL,
    })


@app.route("/api/health", methods=["GET"])
def health():
    ollama_status = "unavailable"

    try:
        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=3
        )
        if response.status_code == 200:
            ollama_status = "available"
    except requests.RequestException:
        pass

    return jsonify({
        "status": "success",
        "backend": "available",
        "ollama": ollama_status,
        "model": OLLAMA_MODEL,
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),
    })


# =========================================================
# AUTHENTICATION & USER MANAGEMENT
# =========================================================

@app.route("/api/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}

    phone = str(data.get("phone", "")).strip()
    password = str(data.get("password", ""))
    full_name = str(
        data.get("full_name", "Farmer")
    ).strip()
    location = str(
        data.get("location", "West Bengal")
    ).strip()
    primary_crop = str(
        data.get("primary_crop", "Paddy")
    ).strip()

    if not phone or not password:
        return jsonify({
            "error":
                "Phone number and password are required"
        }), 400

    if not re.fullmatch(r"\d{10,15}", phone):
        return jsonify({
            "error":
                "Phone number must contain 10 to 15 digits"
        }), 400

    if len(password) < 6:
        return jsonify({
            "error":
                "Password must be at least 6 characters"
        }), 400

    if User.query.filter_by(phone=phone).first():
        return jsonify({
            "error": "Farmer already registered"
        }), 409

    password_hash = generate_password_hash(
        password,
        method="pbkdf2:sha256"
    )

    new_user = User(
        full_name=full_name or "Farmer",
        phone=phone,
        password=password_hash,
        location=location or "West Bengal",
        primary_crop=primary_crop or "Paddy",
    )

    db.session.add(new_user)
    db.session.commit()

    return jsonify({
        "status": "success",
        "message": "Farmer registered successfully",
        "user_id": new_user.id,
    }), 201


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}

    phone = str(data.get("phone", "")).strip()
    password = str(data.get("password", ""))

    if not phone or not password:
        return jsonify({
            "error":
                "Phone number and password are required"
        }), 400

    user = User.query.filter_by(phone=phone).first()

    if not user:
        return jsonify({
            "error": "Invalid phone number or password"
        }), 401

    valid_password = False

    if looks_like_password_hash(user.password):
        try:
            valid_password = check_password_hash(
                user.password,
                password
            )
        except ValueError:
            valid_password = False
    else:
        # Backward compatibility for users created by the
        # older version that stored passwords as plain text.
        valid_password = user.password == password

        # Transparently upgrade the old password after
        # a successful login.
        if valid_password:
            user.password = generate_password_hash(
                password,
                method="pbkdf2:sha256"
            )
            db.session.commit()

    if not valid_password:
        return jsonify({
            "error": "Invalid phone number or password"
        }), 401

    return jsonify({
        "status": "success",
        "message": "Login successful",
        "user": {
            "id": user.id,
            "full_name": user.full_name,
            "phone": user.phone,
            "location": user.location,
            "primary_crop": user.primary_crop,
        }
    })


@app.route("/api/farmers", methods=["GET"])
def get_farmers():
    farmers = User.query.order_by(User.id.desc()).all()

    farmers_data = [
        {
            "id": farmer.id,
            "full_name": farmer.full_name,
            "location": farmer.location,
            "primary_crop": farmer.primary_crop,
        }
        for farmer in farmers
    ]

    return jsonify({
        "status": "success",
        "count": len(farmers_data),
        "data": farmers_data,
    })


# =========================================================
# COMMUNITY FEED
# =========================================================

@app.route("/api/posts", methods=["GET"])
def get_posts():
    posts = CommunityPost.query.order_by(
        CommunityPost.created_at.desc()
    ).all()

    posts_data = [
        {
            "id": post.id,
            "author_name": post.author_name,
            "location": post.location,
            "content": post.content,
            "media_url": post.media_url,
            "created_at": post.created_at.strftime(
                "%Y-%m-%d %H:%M"
            ),
        }
        for post in posts
    ]

    return jsonify({
        "status": "success",
        "count": len(posts_data),
        "data": posts_data,
    })


@app.route("/api/posts", methods=["POST"])
def create_post():
    data = request.get_json(silent=True) or {}

    author_name = str(
        data.get("author_name", "")
    ).strip()
    content = str(
        data.get("content", "")
    ).strip()

    if not author_name or not content:
        return jsonify({
            "error": "Author and content are required"
        }), 400

    new_post = CommunityPost(
        author_name=author_name,
        location=str(
            data.get("location", "West Bengal")
        ).strip(),
        content=content,
        media_url=str(
            data.get("media_url", "")
        ).strip(),
    )

    db.session.add(new_post)
    db.session.commit()

    return jsonify({
        "status": "success",
        "message": "Post published",
        "post_id": new_post.id,
    }), 201


# =========================================================
# AGRICULTURAL CONTENT & TOOLS
# =========================================================

@app.route("/api/mandi", methods=["GET"])
def get_mandi_prices():
    _, mandi_prices = load_data()

    state = request.args.get("state", "").strip()
    district = request.args.get(
        "district",
        ""
    ).strip()
    market = request.args.get("market", "").strip()
    commodity = request.args.get(
        "commodity",
        ""
    ).strip()

    try:
        limit = min(
            max(
                int(request.args.get("limit", 100)),
                1
            ),
            2000
        )
    except ValueError:
        limit = 100

    filtered = mandi_prices

    filters = {
        "state": state,
        "district": district,
        "market": market,
        "commodity": commodity,
    }

    for key, value in filters.items():
        if value:
            filtered = [
                item for item in filtered
                if value.lower() in str(
                    item.get(key, "")
                ).lower()
            ]

    return jsonify({
        "status": "success",
        "count": len(filtered),
        "returned": min(len(filtered), limit),
        "data": filtered[:limit],
    })


@app.route("/api/schemes", methods=["GET"])
def get_schemes():
    schemes, _ = load_data()

    query = request.args.get("q", "").strip().lower()

    if query:
        schemes = [
            scheme for scheme in schemes
            if query in json.dumps(
                scheme,
                ensure_ascii=False
            ).lower()
        ]

    return jsonify({
        "status": "success",
        "count": len(schemes),
        "data": schemes,
    })


@app.route("/api/crops", methods=["GET"])
def get_crops():
    crops = read_json_file("crop_calendar.json", [])

    return jsonify({
        "status": "success",
        "count": len(crops),
        "data": crops,
    })


@app.route("/api/tools", methods=["GET"])
def get_tools():
    tools = read_json_file("agri_tools.json", [])

    return jsonify({
        "status": "success",
        "count": len(tools),
        "data": tools,
    })


# =========================================================
# WEATHER API
# =========================================================

@app.route("/api/weather", methods=["GET"])
def get_weather():
    lat = request.args.get("lat", "22.5726")
    lon = request.args.get("lon", "88.3639")

    try:
        lat_float = float(lat)
        lon_float = float(lon)

        if not (-90 <= lat_float <= 90):
            raise ValueError("Invalid latitude")

        if not (-180 <= lon_float <= 180):
            raise ValueError("Invalid longitude")

        weather_data = get_weather_data(
            lat_float,
            lon_float
        )

        return jsonify({
            "status": "success",
            "weather_data": weather_data,
        })

    except ValueError as exc:
        return jsonify({
            "error": str(exc)
        }), 400

    except requests.RequestException as exc:
        return jsonify({
            "error":
                "Unable to fetch weather data",
            "details": str(exc),
        }), 502


# =========================================================
# LOCATION / CROP RECOMMENDATION
# =========================================================

@app.route(
    "/api/location-recommendation",
    methods=["GET"]
)
def get_location_recommendation():
    try:
        lat = float(
            request.args.get("lat", "22.5726")
        )
        lon = float(
            request.args.get("lon", "88.3639")
        )

        if not (-90 <= lat <= 90):
            raise ValueError("Invalid latitude")

        if not (-180 <= lon <= 180):
            raise ValueError("Invalid longitude")

        water_available = normalize_water_level(
            request.args.get("water", "medium")
        )
        soil_type = request.args.get(
            "soil",
            ""
        ).strip() or None

        weather = get_weather_data(lat, lon)

        current_weather = weather.get(
            "current_weather",
            {}
        )
        temperature = current_weather.get(
            "temperature"
        )

        season = get_current_season()

        crop_requirements = read_json_file(
            "crop_requirements.json",
            []
        )

        if not crop_requirements:
            return jsonify({
                "error":
                    "crop_requirements.json contains no crop data"
            }), 500

        recommendations = build_crop_recommendations(
            crops=crop_requirements,
            season=season,
            temperature=temperature,
            water_available=water_available,
            soil_type=soil_type,
        )

        suitable = [
            item for item in recommendations
            if item["score"] > 0
        ][:5]

        return jsonify({
            "status": "success",
            "data": {
                "coordinates": {
                    "latitude": lat,
                    "longitude": lon,
                },
                "season": season,
                "current_temperature_c":
                    temperature,
                "water_availability":
                    water_available,
                "soil_type": soil_type,
                "recommendations": suitable,
                "note": (
                    "Recommendations are generated from "
                    "structured crop requirements and "
                    "current weather. Local soil testing "
                    "and agriculture-office guidance "
                    "should be considered before planting."
                ),
            }
        })

    except ValueError as exc:
        return jsonify({
            "error": str(exc)
        }), 400

    except requests.RequestException as exc:
        return jsonify({
            "error":
                "Could not fetch weather data for "
                "crop recommendation",
            "details": str(exc),
        }), 502


# =========================================================
# ABOUT
# =========================================================

@app.route("/api/about", methods=["GET"])
def get_about():
    return jsonify({
        "status": "success",
        "data": {
            "platform_name":
                "AgriTech Modern Farming Platform",
            "objective": (
                "Empowering farmers with modern "
                "techniques, local crop guidance, "
                "weather information, mandi prices, "
                "government schemes and AI-assisted "
                "crop disease analysis."
            ),
            "team": "Subhasish Sarkar & Team",
            "ai": (
                f"Local Ollama model: {OLLAMA_MODEL}"
            ),
        }
    })


# =========================================================
# AI AGRICULTURE ADVISOR - LOCAL OLLAMA
# =========================================================

@app.route("/api/ai-advisor", methods=["POST"])
def ai_advisor():
    data = request.get_json(silent=True) or {}
    user_query = str(
        data.get("query", "")
    ).strip()

    if not user_query:
        return jsonify({
            "error": "No query provided"
        }), 400

    schemes, mandi = load_data()

    # Keep the local model context reasonably small.
    schemes_context = schemes[:10]
    mandi_context = mandi[:25]

    prompt = f"""
You are a practical AI farming assistant for Indian farmers,
with special attention to West Bengal.

Answer the farmer's question clearly and simply.

Use the supplied local data when it is relevant. Do not claim
that stored data is live unless the supplied data explicitly
contains a current date or the application has fetched it live.

Rules:
1. Prefer practical, low-cost and water-conscious advice.
2. Clearly distinguish facts from suggestions.
3. Do not invent government-scheme eligibility, mandi prices,
   pesticide dosages or guaranteed crop yields.
4. For pesticide, fertilizer, disease or safety-sensitive advice,
   recommend confirming locally with an agriculture officer or
   qualified agricultural expert.
5. If the farmer asks for current weather but no weather data is
   supplied here, tell them to use the platform weather feature.
6. Be concise but useful.
7. Answer in the same language as the farmer when practical.

WEST BENGAL GOVERNMENT SCHEMES DATA:
{json.dumps(schemes_context, ensure_ascii=False)}

MANDI PRICE DATA SAMPLE:
{json.dumps(mandi_context, ensure_ascii=False)}

FARMER QUESTION:
{user_query}

Answer:
"""

    try:
        ai_text = ollama_generate(
            prompt,
            timeout=180
        )

        return jsonify({
            "status": "success",
            "ai_response": ai_text,
            "model": OLLAMA_MODEL,
        })

    except requests.exceptions.ConnectionError:
        return jsonify({
            "error": (
                "Could not connect to Ollama. Make sure "
                "Ollama is installed and running on "
                "http://localhost:11434."
            )
        }), 503

    except requests.exceptions.Timeout:
        return jsonify({
            "error":
                "The local AI model took too long "
                "to respond. Please try again."
        }), 504

    except Exception as exc:
        print(
            "Ollama AI Advisor Error:",
            str(exc)
        )
        return jsonify({
            "error": str(exc)
        }), 500


# =========================================================
# CROP DISEASE PREDICTION - LOCAL OLLAMA VISION
# =========================================================

@app.route("/api/predict-disease", methods=["POST"])
def predict_disease():
    try:
        image_base64 = None
        image_mime_type = None
        crop_hint = ""

        # ---------------------------------------------
        # Option 1: normal multipart/form-data upload
        # ---------------------------------------------
        if "image" in request.files:
            image_file = request.files["image"]
            crop_hint = request.form.get("crop", "").strip()   
            if not image_file.filename:
                return jsonify({
                    "error": "No image selected"
                }), 400

            image_mime_type = (
                image_file.mimetype or ""
            ).lower()

            if (
                image_mime_type
                and image_mime_type
                not in ALLOWED_IMAGE_MIME_TYPES
            ):
                return jsonify({
                    "error": (
                        "Unsupported image type. "
                        "Use JPG, PNG or WEBP."
                    )
                }), 415

            image_bytes = image_file.read()

            if not image_bytes:
                return jsonify({
                    "error": "Uploaded image is empty"
                }), 400

            if (
                len(image_bytes)
                > MAX_IMAGE_SIZE_MB * 1024 * 1024
            ):
                return jsonify({
                    "error": (
                        f"Image must be smaller than "
                        f"{MAX_IMAGE_SIZE_MB} MB"
                    )
                }), 413

            image_base64 = base64.b64encode(
                image_bytes
            ).decode("utf-8")

        # ---------------------------------------------
        # Option 2: JSON containing base64 image
        # ---------------------------------------------
        else:
            data = request.get_json(
                silent=True
            ) or {}

            image_base64 = data.get("image")
            crop_hint = str(data.get("crop", "")).strip()

            if not image_base64:
                return jsonify({
                    "error": (
                        "No image provided. Send an "
                        "image file using multipart/form-data "
                        "or a base64 image in JSON."
                    )
                }), 400

            if "," in image_base64:
                header, image_base64 = (
                    image_base64.split(",", 1)
                )

                if header.startswith("data:image/"):
                    image_mime_type = (
                        header.split(";", 1)[0]
                    ).replace("data:", "")

            try:
                decoded = base64.b64decode(
                    image_base64,
                    validate=True
                )
            except Exception:
                return jsonify({
                    "error":
                        "Invalid base64 image data"
                }), 400

            if (
                len(decoded)
                > MAX_IMAGE_SIZE_MB * 1024 * 1024
            ):
                return jsonify({
                    "error": (
                        f"Image must be smaller than "
                        f"{MAX_IMAGE_SIZE_MB} MB"
                    )
                }), 413

        prompt = f"""
You are an AI-assisted crop health and disease identification assistant.

The farmer says the crop is:
{crop_hint if crop_hint else "Not provided"}

Carefully inspect the supplied plant image.

IMPORTANT:
Before diagnosing any disease, FIRST decide whether the image
actually shows visible evidence of disease or pest damage.

Return ONLY valid JSON with exactly these keys:

{{
  "crop": "",
  "is_diseased": false,
  "disease_name": "",
  "disease_name_bengali": "",
  "confidence": "",
  "symptoms": "",
  "organic_treatment": "",
  "chemical_treatment": "",
  "prevention": "",
  "recommended_youtube_search_query": "",
  "warning": ""
}}

DECISION PROCESS:

STEP 1 - HEALTH CHECK

Look for actual visible abnormal symptoms such as:
- abnormal brown, black, yellow or orange lesions
- fungal growth or mildew
- abnormal leaf spots
- severe yellowing
- pest holes or feeding damage
- abnormal curling or deformation
- rotting or necrotic tissue

Natural leaf texture, veins, shadows, lighting differences,
minor imperfections or normal colour variation are NOT enough
to classify a plant as diseased.

If no convincing abnormal symptoms are visible:

"is_diseased": false
"disease_name": "Healthy / No obvious disease detected"
"disease_name_bengali": "কোনো স্পষ্ট রোগ শনাক্ত হয়নি"
"symptoms": "No obvious disease symptoms are visible."
"organic_treatment": "No treatment required."
"chemical_treatment": "No chemical treatment recommended."
"prevention": "Continue normal crop care and regular monitoring."
"recommended_youtube_search_query": ""
"warning": "Image-based screening cannot rule out problems that are not visibly apparent."

STOP disease diagnosis when the plant appears healthy.

STEP 2 - DISEASE IDENTIFICATION

Only when convincing visible disease or pest symptoms exist,
set "is_diseased" to true.

Then:
1. Identify the most likely problem using the crop context and
   visible symptoms.
2. Give a cautious confidence percentage.
3. Describe only symptoms that can actually be seen.
4. If several conditions are possible, give the most likely one
   and mention alternatives in "warning".
5. Never invent symptoms that are not visible in the image.
6. Never claim laboratory certainty from a photograph.

TREATMENT SAFETY:

1. Prefer non-chemical cultural or organic management first.
2. Do NOT invent pesticide, fungicide or bactericide names.
3. If you are not highly confident about an appropriate chemical,
   write:
   "Consult a local agriculture officer or qualified agricultural expert for an appropriate registered treatment."
4. Never invent pesticide dosage or application rate.
5. Do not recommend an insecticide for a bacterial or fungal
   disease unless an insect pest is actually identified.

OUTPUT RULES:

1. "is_diseased" must be a JSON boolean: true or false.
2. Do not use Markdown.
3. Return one JSON object only.
4. Do not add commentary before or after the JSON.
"""

        analysis_text = ollama_generate(
            prompt,
            images=[image_base64],
            json_mode=True,
            timeout=300,
        )

        try:
            analysis_json = json.loads(
                analysis_text
            )
        except json.JSONDecodeError:
            return jsonify({
                "status": "partial_success",
                "analysis": analysis_text,
                "warning": (
                    "The image was analysed, but "
                    "the model response was not valid JSON."
                ),
                "model": OLLAMA_MODEL,
            }), 200

        expected_keys = [
            "crop",
            "is_diseased",
            "disease_name",
            "disease_name_bengali",
            "confidence",
            "symptoms",
            "organic_treatment",
            "chemical_treatment",
            "prevention",
            "recommended_youtube_search_query",
            "warning",
        ]

        cleaned_analysis = {
            key: analysis_json.get(key, "")
            for key in expected_keys
        }

        disease_value = analysis_json.get("is_diseased", False)

        if isinstance(disease_value, bool):
            cleaned_analysis["is_diseased"] = disease_value
        elif isinstance(disease_value, str):
            cleaned_analysis["is_diseased"] = (
                disease_value.strip().lower() == "true"
            )
        else:
            cleaned_analysis["is_diseased"] = False

        return jsonify({
            "status": "success",
            "analysis": cleaned_analysis,
            "model": OLLAMA_MODEL,
            "image_type": image_mime_type,
            "disclaimer": (
                "AI image analysis is advisory and is "
                "not a laboratory diagnosis."
            ),
        })

    except requests.exceptions.ConnectionError:
        return jsonify({
            "error": (
                "Could not connect to Ollama. Make sure "
                "Ollama is running locally."
            )
        }), 503

    except requests.exceptions.Timeout:
        return jsonify({
            "error": (
                "Disease analysis took too long. "
                "Try a smaller or clearer image."
            )
        }), 504

    except Exception as exc:
        print(
            "Ollama Disease Prediction Error:",
            str(exc)
        )
        return jsonify({
            "error": str(exc)
        }), 500


# =========================================================
# ERROR HANDLERS
# =========================================================

@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({
        "error": (
            f"Uploaded file is too large. "
            f"Maximum allowed size is "
            f"{MAX_IMAGE_SIZE_MB} MB."
        )
    }), 413


@app.errorhandler(404)
def not_found(_error):
    return jsonify({
        "error": "API endpoint not found"
    }), 404


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":
    debug_mode = (
        os.getenv("FLASK_DEBUG", "true")
        .strip()
        .lower()
        in {"1", "true", "yes"}
    )

    print(
        "Starting AgriTech AI Backend with local Ollama AI "
        "on http://localhost:5000..."
    )
    print(f"Ollama model: {OLLAMA_MODEL}")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=debug_mode,
    )
