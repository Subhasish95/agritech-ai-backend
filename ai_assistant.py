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
OLLAMA_CHAT_URL = os.getenv(
    "OLLAMA_CHAT_URL",
    "http://localhost:11434/api/chat"
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


def ollama_chat(messages, timeout=180):
    """
    Send a real multi-message conversation to Ollama's /api/chat
    endpoint. This is used by the farming chatbot so conversation
    history stays structurally separate from system instructions.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 700,
        },
    }

    response = requests.post(
        OLLAMA_CHAT_URL,
        json=payload,
        timeout=timeout
    )

    if response.status_code != 200:
        raise RuntimeError(
            "Ollama chat request failed: "
            + response.text[:1000]
        )

    result = response.json()
    message = result.get("message", {})
    text = str(message.get("content", "")).strip()

    if not text:
        raise RuntimeError(
            "Ollama chat returned an empty response"
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


def get_user_messages_from_history(history):
    """
    Return only valid user messages from a normalized conversation history.
    """
    return [
        item.get("content", "").strip()
        for item in history
        if (
            isinstance(item, dict)
            and item.get("role") == "user"
            and str(item.get("content", "")).strip()
        )
    ]


def get_deterministic_history_answer(user_query, conversation_history):
    """
    Answer exact conversation-recall questions in Python instead of
    asking the language model to reconstruct them.

    Returns:
        str: Direct answer when a supported recall pattern is found.
        None: Let the AI model handle the question.
    """
    query = re.sub(
        r"\s+",
        " ",
        user_query.strip().lower()
    )

    user_messages = get_user_messages_from_history(
        conversation_history
    )

    if not user_messages:
        return None

    # "Before that" must be checked before broader "before" patterns.
    before_previous_patterns = [
        r"\bwhat did i ask before that\b",
        r"\bwhat was the question before that\b",
        r"\bwhat was my question before the last one\b",
        r"\bwhat did i ask one question ago\b",
    ]

    if any(
        re.search(pattern, query)
        for pattern in before_previous_patterns
    ):
        if len(user_messages) >= 2:
            return (
                'The question before that was: '
                f'"{user_messages[-2]}"'
            )

        return (
            "There is only one earlier question in this "
            "conversation."
        )

    # Explicit first / second / third question recall.
    ordinal_map = {
        "first": 0,
        "1st": 0,
        "second": 1,
        "2nd": 1,
        "third": 2,
        "3rd": 2,
        "fourth": 3,
        "4th": 3,
        "fifth": 4,
        "5th": 4,
    }

    ordinal_match = re.search(
        r"\b(?:what was|what is|repeat|tell me)\s+"
        r"(?:my\s+)?"
        r"(first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th)"
        r"\s+question\b",
        query
    )

    if ordinal_match:
        ordinal_word = ordinal_match.group(1)
        index = ordinal_map[ordinal_word]

        if len(user_messages) > index:
            labels = [
                "first",
                "second",
                "third",
                "fourth",
                "fifth",
            ]
            return (
                f'Your {labels[index]} question was: '
                f'"{user_messages[index]}"'
            )

        return (
            f"You have not asked that many questions in "
            f"this conversation."
        )

    first_patterns = [
        r"\bwhat was the first question i asked\b",
        r"\bwhat did i ask first\b",
    ]

    if any(
        re.search(pattern, query)
        for pattern in first_patterns
    ):
        return (
            'Your first question was: '
            f'"{user_messages[0]}"'
        )

    previous_patterns = [
        r"\bwhat was my previous question\b",
        r"\bwhat was my last question\b",
        r"\bwhat did i ask previously\b",
        r"\bwhat did i ask before\b",
        r"\brepeat my previous question\b",
        r"\brepeat my last question\b",
        r"\bwhat did i just ask\b",
    ]

    if any(
        re.search(pattern, query)
        for pattern in previous_patterns
    ):
        return (
            'Your previous question was: '
            f'"{user_messages[-1]}"'
        )

    return None


def is_scheme_query(query):
    q = query.lower()
    keywords = [
        "scheme",
        "schemes",
        "subsidy",
        "subsidies",
        "government help",
        "yojana",
        "loan",
        "credit",
        "insurance",
        "pm-kisan",
        "kisan credit",
    ]
    return any(keyword in q for keyword in keywords)


def is_mandi_query(query):
    q = query.lower()
    keywords = [
        "mandi",
        "market price",
        "crop price",
        "commodity price",
        "selling price",
        "wholesale price",
        "market rate",
        "price today",
    ]
    return any(keyword in q for keyword in keywords)


def build_crop_reference_context(user_query):
    """
    Pull only relevant crop/calendar/tool records from local project JSON.
    This keeps the small local model focused and prevents unrelated data
    from being treated as conversational context.
    """
    query = user_query.lower()

    crop_calendar = read_json_file(
        "crop_calendar.json",
        []
    )
    crop_requirements = read_json_file(
        "crop_requirements.json",
        []
    )
    agri_tools = read_json_file(
        "agri_tools.json",
        []
    )

    matched_calendar = []
    matched_requirements = []

    def crop_name(record):
        return str(
            record.get("crop")
            or record.get("name")
            or record.get("crop_name")
            or ""
        ).strip()

    for record in crop_calendar:
        if not isinstance(record, dict):
            continue
        name = crop_name(record)
        if name and name.lower() in query:
            matched_calendar.append(record)

    for record in crop_requirements:
        if not isinstance(record, dict):
            continue
        name = crop_name(record)
        if name and name.lower() in query:
            matched_requirements.append(record)

    matched_tools = []
    if any(
        word in query
        for word in [
            "tool",
            "tools",
            "equipment",
            "machine",
            "machinery",
            "implement",
        ]
    ):
        matched_tools = [
            item
            for item in agri_tools
            if isinstance(item, dict)
        ][:15]

    location_notes = []

    if (
        "kolkata" in query
        or "calcutta" in query
    ):
        location_notes.append(
            "For Kolkata-specific crop advice, do not use a "
            "'last expected frost' or frost-date rule as the basis "
            "for sowing guidance. Use the crop season and verified "
            "local agricultural guidance instead."
        )

    if "west bengal" in query:
        location_notes.append(
            "For West Bengal-specific timing or input rates, use "
            "verified local crop-calendar data or advise confirmation "
            "with the local agriculture office when that data is not "
            "available."
        )

    # A small, conservative deterministic fact to prevent the model from
    # inventing frost-based wheat timing in eastern India.
    if "wheat" in query:
        location_notes.append(
            "Wheat is generally treated as a Rabi/cool-season crop "
            "in India. Do not invent a frost-based sowing date. "
            "If an exact local sowing window is not present in the "
            "application data, say that the farmer should confirm the "
            "district/local recommendation."
        )

    return {
        "matched_crop_calendar": matched_calendar[:5],
        "matched_crop_requirements": matched_requirements[:5],
        "matched_tools": matched_tools,
        "location_notes": location_notes,
    }


def contains_kolkata_frost_mismatch(user_query, text):
    """
    Detect generic frost-based advice when the farmer explicitly asked
    about Kolkata/Calcutta.
    """
    q = user_query.lower()
    t = (text or "").lower()

    asks_kolkata = (
        "kolkata" in q
        or "calcutta" in q
    )

    return asks_kolkata and "frost" in t


def remove_kolkata_frost_mismatch(text):
    """
    Deterministic fallback for Kolkata answers if the model still inserts
    frost-based planting advice after regeneration.
    """
    if not text:
        return text

    parts = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    kept = [
        part
        for part in parts
        if "frost" not in part.lower()
    ]

    replacement = (
        "For Kolkata, base sowing time on the crop's local season "
        "and verified district guidance rather than a frost-date rule."
    )

    if replacement not in kept:
        kept.append(replacement)

    return " ".join(kept).strip()


def normalize_ai_formatting(text):
    """
    Preserve readable numbered/bulleted answers from small local models.
    """
    if not text:
        return text

    cleaned = text.strip()

    # Put numbered Markdown list items on separate lines when the model
    # compresses them into one paragraph.
    cleaned = re.sub(
        r"(?<!^)(?<!\n)\s+(?=\d{1,2}\.\s+\*\*)",
        "\n\n",
        cleaned
    )

    cleaned = re.sub(
        r"(?<!^)(?<!\n)\s+(?=\d{1,2}\.\s+[A-Z])",
        "\n\n",
        cleaned
    )

    # Prevent excessive blank lines.
    cleaned = re.sub(
        r"\n{3,}",
        "\n\n",
        cleaned
    )

    return cleaned.strip()

def contains_unsupported_agrochemical_dosage(text):
    """
    Detect likely unsupported numeric fertilizer/pesticide/
    agrochemical application-rate advice.
    """
    if not text:
        return False

    lowered = text.lower()

    agrochemical_terms = [
        "fertilizer",
        "fertiliser",
        "npk",
        "pesticide",
        "fungicide",
        "herbicide",
        "insecticide",
        "agrochemical",
    ]

    if not any(
        term in lowered
        for term in agrochemical_terms
    ):
        return False

    dosage_patterns = [
        (
            r"\b\d+(?:\.\d+)?\s*-\s*"
            r"\d+(?:\.\d+)?\s*kg\s*"
            r"(?:per|/)\s*"
            r"(?:acre|hectare|ha)\b"
        ),
        (
            r"\b\d+(?:\.\d+)?\s*kg\s*"
            r"(?:per|/)\s*"
            r"(?:acre|hectare|ha)\b"
        ),
        (
            r"\b\d+(?:\.\d+)?\s*g\s*"
            r"(?:per|/)\s*"
            r"(?:litre|liter|l)\b"
        ),
        (
            r"\b\d+(?:\.\d+)?\s*ml\s*"
            r"(?:per|/)\s*"
            r"(?:litre|liter|l)\b"
        ),
        (
            r"\b\d+(?:\.\d+)?\s*"
            r"(?:kg|g|ml)\s*/\s*"
            r"(?:acre|ha|hectare|l)\b"
        ),
    ]

    return any(
        re.search(pattern, lowered)
        for pattern in dosage_patterns
    )


def remove_unsupported_agrochemical_dosages(text):
    """
    Final fallback sanitization. Replace sentences that appear to
    contain unsupported numeric agrochemical rates with a safe note.
    """
    if not text:
        return text

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    safe_sentences = []
    replaced = False

    for sentence in sentences:
        if contains_unsupported_agrochemical_dosage(sentence):
            if not replaced:
                safe_sentences.append(
                    "For fertilizer or pesticide application rates, "
                    "the correct dosage depends on the crop, soil "
                    "condition and registered product. Please confirm "
                    "the rate using a soil test, the registered product "
                    "label, or a local agriculture officer."
                )
                replaced = True
            continue

        safe_sentences.append(sentence)

    return " ".join(safe_sentences).strip()


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

    # Conversation history is supplied by the client.
    # Only valid user/assistant messages are accepted.
    raw_history = data.get("history", [])

    if not isinstance(raw_history, list):
        raw_history = []

    conversation_history = []

    # Keep the prompt small enough for the local 3B model.
    for item in raw_history[-30:]:
        if not isinstance(item, dict):
            continue

        role = str(
            item.get("role", "")
        ).strip().lower()

        content = str(
            item.get("content", "")
        ).strip()

        if role not in {"user", "assistant"}:
            continue

        if not content:
            continue

        # Avoid unexpectedly huge individual messages.
        conversation_history.append({
            "role": role,
            "content": content[:4000],
        })

    # Handle exact conversation-recall questions directly.
    # This prevents the local language model from guessing or
    # hallucinating previous user messages.
    deterministic_answer = get_deterministic_history_answer(
        user_query,
        conversation_history
    )

    if deterministic_answer is not None:
        return jsonify({
            "status": "success",
            "ai_response": deterministic_answer,
            "model": "deterministic-history",
            "history_messages_used": len(
                conversation_history
            ),
        })

    schemes, mandi = load_data()

    # Supply only reference data that is relevant to the current query.
    # This reduces distraction and hallucination for the small 3B model.
    schemes_context = (
        schemes[:10]
        if is_scheme_query(user_query)
        else []
    )

    mandi_context = (
        mandi[:25]
        if is_mandi_query(user_query)
        else []
    )

    crop_reference_context = build_crop_reference_context(
        user_query
    )

    system_prompt = f"""
You are a practical AI farming assistant for Indian farmers,
with special attention to West Bengal.

Your job is to answer the farmer's current question while using
the recent conversation and only the relevant application data.

IMPORTANT CONVERSATION RULES:
1. Treat user and assistant chat messages as the authoritative
   conversation history.
2. Resolve follow-up phrases such as "it", "that", "this",
   "the first one", "the second one", "my first question",
   "your previous answer" and similar references from chat history.
3. Never replace conversation history with scheme, mandi, weather,
   crop-calendar or other reference data.
4. If the conversation does not contain enough information, say so
   instead of inventing an earlier message.
5. Stay on the topic the farmer is asking about.

LOCALITY AND AGRONOMY QUALITY RULES:
1. Do not invent local sowing dates, frost dates, rainfall,
   temperature, irrigation schedules, crop varieties, yield figures,
   soil-test values or district-specific recommendations.
2. Use exact local facts only when they appear in the supplied
   application reference data.
3. If exact local information is unavailable, give general guidance
   and clearly say that exact local timing/rates should be confirmed
   from the platform's verified data or local agriculture office.
4. For Kolkata-specific advice, never use a generic "last frost"
   rule as a planting cue.
5. Do not claim that a crop is suitable for a specific location
   unless the supplied data supports that claim. You may explain
   general crop requirements and limitations.
6. If the user asks for current weather and no live weather data is
   supplied in this conversation, direct them to the platform's
   weather feature instead of inventing current conditions.

AGRICULTURAL SAFETY RULES:
1. Prefer practical, low-cost and water-conscious guidance.
2. Prefer cultural, mechanical and non-chemical management first
   when practical.
3. Never invent government-scheme eligibility, mandi prices,
   pesticide dosages, fertilizer dosages, agrochemical application
   rates or guaranteed yields.
4. NEVER provide a numeric fertilizer, pesticide, fungicide,
   herbicide, insecticide or other agrochemical dose/rate unless that
   exact value is explicitly present in reliable application data.
5. When a farmer asks for such a rate and no verified value is
   supplied, explain that the correct rate depends on the crop, soil
   condition and registered product, and recommend a soil test,
   registered product label or local agriculture officer.
6. Do not invent pesticide/fungicide names or mix products casually.
7. For disease, pesticide, fertilizer and other safety-sensitive
   advice, recommend qualified local confirmation when appropriate.

RESPONSE QUALITY RULES:
1. Answer the current question directly.
2. Use short paragraphs or numbered steps.
3. Put every numbered step on its own line.
4. Avoid repeating the same point.
5. Keep the answer concise but useful.
6. Answer in the same language as the farmer when practical.
7. If a supplied application record conflicts with your general
   knowledge, use the supplied record and do not silently override it.

RELEVANT CROP / TOOL / LOCATION CONTEXT:
{json.dumps(crop_reference_context, ensure_ascii=False)}

RELEVANT WEST BENGAL SCHEME DATA:
{json.dumps(schemes_context, ensure_ascii=False)}

RELEVANT MANDI DATA:
{json.dumps(mandi_context, ensure_ascii=False)}

The reference sections above are application data, NOT conversation
history. If a section is empty, do not invent data for it.
""".strip()

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    messages.extend(conversation_history)

    messages.append({
        "role": "user",
        "content": user_query,
    })

    try:
        ai_text = ollama_chat(
            messages,
            timeout=180
        )

        safety_regenerated = False
        safety_sanitized = False

        # Do not trust prompt instructions alone for numeric
        # fertilizer/pesticide rates. If such a rate appears,
        # ask the model to rewrite the answer without it.
        if contains_unsupported_agrochemical_dosage(ai_text):
            safety_regenerated = True

            safety_messages = messages + [
                {
                    "role": "assistant",
                    "content": ai_text,
                },
                {
                    "role": "system",
                    "content": (
                        "The previous answer contained an unsupported "
                        "numeric fertilizer, pesticide, fungicide, "
                        "herbicide or other agrochemical dosage/application "
                        "rate. Rewrite the answer without ANY numeric "
                        "agrochemical dosage or application rate. Do not "
                        "repeat or paraphrase the unsupported number. "
                        "For fertilizer or pesticide rates, explain that "
                        "the correct dosage depends on crop, soil condition "
                        "and product, and should be confirmed using a soil "
                        "test, the registered product label, or a local "
                        "agriculture officer."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Rewrite your previous answer now, keeping the "
                        "useful farming guidance but removing unsupported "
                        "numeric agrochemical dosages."
                    ),
                },
            ]

            ai_text = ollama_chat(
                safety_messages,
                timeout=180
            )

        # Final deterministic fallback: if the model still gives a
        # prohibited numeric rate, remove that sentence before
        # returning the answer to the farmer.
        if contains_unsupported_agrochemical_dosage(ai_text):
            safety_sanitized = True
            ai_text = remove_unsupported_agrochemical_dosages(
                ai_text
            )

        locality_regenerated = False
        locality_sanitized = False

        # Catch the specific generic-climate error seen in testing:
        # frost-based sowing advice for Kolkata.
        if contains_kolkata_frost_mismatch(
            user_query,
            ai_text
        ):
            locality_regenerated = True

            locality_messages = messages + [
                {
                    "role": "assistant",
                    "content": ai_text,
                },
                {
                    "role": "system",
                    "content": (
                        "The previous answer used frost-based planting "
                        "advice for a Kolkata-specific question. Rewrite "
                        "the answer without any frost/last-frost rule. "
                        "Use only verified supplied local data for exact "
                        "timing. If no exact local sowing window is "
                        "supplied, give general seasonal guidance and say "
                        "that the exact local timing should be confirmed "
                        "from verified district guidance."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Rewrite the previous answer with the incorrect "
                        "Kolkata frost-based advice removed."
                    ),
                },
            ]

            ai_text = ollama_chat(
                locality_messages,
                timeout=180
            )

        if contains_kolkata_frost_mismatch(
            user_query,
            ai_text
        ):
            locality_sanitized = True
            ai_text = remove_kolkata_frost_mismatch(
                ai_text
            )

        # Run the agrochemical check again after any locality rewrite.
        if contains_unsupported_agrochemical_dosage(ai_text):
            safety_sanitized = True
            ai_text = remove_unsupported_agrochemical_dosages(
                ai_text
            )

        ai_text = normalize_ai_formatting(ai_text)

        return jsonify({
            "status": "success",
            "ai_response": ai_text,
            "model": OLLAMA_MODEL,
            "history_messages_used": len(
                conversation_history
            ),
            "safety_regenerated": safety_regenerated,
            "safety_sanitized": safety_sanitized,
            "locality_regenerated": locality_regenerated,
            "locality_sanitized": locality_sanitized,
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