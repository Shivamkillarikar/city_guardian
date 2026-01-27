from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from dotenv import load_dotenv
import requests, base64, os, json, re, math, time
import pandas as pd
from datetime import datetime
import uuid
from typing import Optional, Dict, List
import asyncio

# 1. INITIALIZATION
load_dotenv(override=True)

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash-exp')

MAILEROO_API_KEY = os.getenv("MAILEROO_API_KEY")

# API Keys for Social Listening (Agent 2)
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")  # Optional
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")  # Get from newsapi.org
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")  # Optional
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")  # Optional

app = FastAPI(title="CityGuardian Pro – Multi-Agent Backend")

# --- CORS ---
origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5173",
    "https://city-guardian-yybm.vercel.app",
    "https://city-guardian-n8n-integration.vercel.app",
    "https://cityguardian-react.vercel.app",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIG & DATA ---
OFFICERS = [
    {"name": "Water Dept", "email": "shivamkillarikar007@gmail.com", "keywords": ["water", "leak", "pipe", "burst"]},
    {"name": "Sewage Dept", "email": "shivamkillarikar22@gmail.com", "keywords": ["sewage", "drain", "gutter", "overflow"]},
    {"name": "Roads Dept", "email": "aishanidolan@gmail.com", "keywords": ["road", "pothole", "traffic", "pavement"]},
    {"name": "Electric Dept", "email": "adityakillarikar@gmail.com", "keywords": ["light", "wire", "pole", "shock", "power"]},
]

# Emergency Services (Agent 3)
EMERGENCY_SERVICES = {
    "hospitals": [
        {"name": "KEM Hospital", "lat": 19.0075, "lon": 72.8445, "type": "government", "emergency": True},
        {"name": "Sion Hospital", "lat": 19.0433, "lon": 72.8626, "type": "government", "emergency": True},
        {"name": "Lilavati Hospital", "lat": 19.0520, "lon": 72.8326, "type": "private", "emergency": True},
    ],
    "fire_stations": [
        {"name": "Mumbai Fire Brigade HQ", "lat": 18.9388, "lon": 72.8354, "phone": "101"},
    ],
    "police_stations": [
        {"name": "Mumbai Police HQ", "lat": 18.9397, "lon": 72.8353, "phone": "100"},
    ]
}

DEFAULT_EMAIL = "shivamkillarikar22@gmail.com"
SHEET_ID = '1yHcKcLdv0TEEpEZ3cAWd9A_t8MBE-yk4JuWqJKn0IeI'

# --- UTILS ---
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371000 
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def clean_gemini_json(text):
    """Removes markdown code blocks from Gemini response."""
    clean = re.sub(r"```json\s?|\s?```", "", text).strip()
    return clean

def find_nearest_facility(lat: float, lon: float, facility_type: str = "hospitals") -> Dict:
    """Find nearest emergency facility using distance calculation."""
    facilities = EMERGENCY_SERVICES.get(facility_type, [])
    if not facilities:
        return None
    
    nearest = min(facilities, key=lambda f: calculate_distance(lat, lon, f['lat'], f['lon']))
    nearest['distance'] = calculate_distance(lat, lon, nearest['lat'], nearest['lon'])
    return nearest

# =============================================================================
# AGENT 2: SOCIAL LISTENING & SENTIMENT ANALYSIS
# =============================================================================

async def sentiment_analyzer(text: str) -> Dict:
    """Analyze sentiment of social media posts using Gemini."""
    try:
        prompt = f"""Analyze the sentiment and urgency of this civic-related social media post.
        
Post: {text}

Respond ONLY in JSON format:
{{
    "sentiment": "positive|neutral|negative",
    "urgency": "low|medium|high",
    "category": "Water|Sewage|Roads|Electric|Health|Emergency|Other",
    "is_civic_issue": true/false,
    "summary": "brief one-line summary"
}}"""
        
        response = model.generate_content(prompt)
        return json.loads(clean_gemini_json(response.text))
    except:
        return {"sentiment": "neutral", "urgency": "low", "category": "Other", "is_civic_issue": False}

async def monitor_twitter(keywords: List[str] = ["Mumbai pothole", "Mumbai roads", "civic issue Mumbai"]):
    """Monitor Twitter/X for civic issues (requires Twitter API access)."""
    if not TWITTER_BEARER_TOKEN:
        return []
    
    # Twitter API v2 endpoint
    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"}
    
    results = []
    for keyword in keywords:
        params = {
            "query": f"{keyword} -is:retweet lang:en",
            "max_results": 10,
            "tweet.fields": "created_at,public_metrics"
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                for tweet in data.get('data', []):
                    sentiment = await sentiment_analyzer(tweet['text'])
                    if sentiment['is_civic_issue']:
                        results.append({
                            "source": "twitter",
                            "text": tweet['text'],
                            "timestamp": tweet['created_at'],
                            "sentiment": sentiment
                        })
        except Exception as e:
            print(f"Twitter API error: {e}")
    
    return results

async def monitor_reddit(subreddits: List[str] = ["mumbai", "india"]):
    """Monitor Reddit for civic issues."""
    results = []
    
    for subreddit in subreddits:
        url = f"https://www.reddit.com/r/{subreddit}/new.json"
        headers = {"User-Agent": "CityGuardian/1.0"}
        
        try:
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                for post in data['data']['children'][:10]:
                    post_data = post['data']
                    text = f"{post_data['title']} {post_data.get('selftext', '')}"
                    
                    # Check if civic related
                    if any(kw in text.lower() for kw in ['pothole', 'civic', 'municipal', 'garbage', 'water']):
                        sentiment = await sentiment_analyzer(text)
                        if sentiment['is_civic_issue']:
                            results.append({
                                "source": "reddit",
                                "text": text,
                                "url": f"https://reddit.com{post_data['permalink']}",
                                "timestamp": datetime.fromtimestamp(post_data['created_utc']).isoformat(),
                                "sentiment": sentiment
                            })
        except Exception as e:
            print(f"Reddit API error: {e}")
    
    return results

async def monitor_news_api(query: str = "Mumbai civic issues"):
    """Monitor news articles for civic issues."""
    if not NEWS_API_KEY:
        return []
    
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 10,
        "apiKey": NEWS_API_KEY
    }
    
    results = []
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            for article in data.get('articles', []):
                text = f"{article['title']} {article.get('description', '')}"
                sentiment = await sentiment_analyzer(text)
                if sentiment['is_civic_issue']:
                    results.append({
                        "source": "news",
                        "text": text,
                        "url": article['url'],
                        "timestamp": article['publishedAt'],
                        "sentiment": sentiment
                    })
    except Exception as e:
        print(f"NewsAPI error: {e}")
    
    return results

@app.get("/api/social-listening")
async def social_listening_endpoint(background_tasks: BackgroundTasks):
    """
    Agent 2: Social Listening Endpoint
    Returns aggregated civic issues from social media
    """
    try:
        # Run all monitoring tasks concurrently
        twitter_task = asyncio.create_task(monitor_twitter())
        reddit_task = asyncio.create_task(monitor_reddit())
        news_task = asyncio.create_task(monitor_news_api())
        
        # Wait for all tasks
        twitter_results, reddit_results, news_results = await asyncio.gather(
            twitter_task, reddit_task, news_task, return_exceptions=True
        )
        
        # Handle exceptions
        twitter_results = twitter_results if not isinstance(twitter_results, Exception) else []
        reddit_results = reddit_results if not isinstance(reddit_results, Exception) else []
        news_results = news_results if not isinstance(news_results, Exception) else []
        
        # Combine all results
        all_issues = twitter_results + reddit_results + news_results
        
        # Sort by urgency
        high_urgency = [i for i in all_issues if i['sentiment']['urgency'] == 'high']
        medium_urgency = [i for i in all_issues if i['sentiment']['urgency'] == 'medium']
        low_urgency = [i for i in all_issues if i['sentiment']['urgency'] == 'low']
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "total_issues": len(all_issues),
            "by_urgency": {
                "high": len(high_urgency),
                "medium": len(medium_urgency),
                "low": len(low_urgency)
            },
            "issues": {
                "high": high_urgency[:5],
                "medium": medium_urgency[:5],
                "low": low_urgency[:5]
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# =============================================================================
# AGENT 3: EMERGENCY & HEALTHCARE RESPONSE
# =============================================================================

def classify_emergency(complaint: str) -> Dict:
    """Classify emergency type and severity."""
    try:
        prompt = f"""Classify this report as an emergency type.

Report: {complaint}

Emergency types:
- medical: Health crisis, injury, medical emergency
- fire: Fire, explosion, smoke
- accident: Vehicle accident, structural collapse
- crime: Assault, theft, violence
- natural: Flood, earthquake, landslide
- civic: Non-emergency civic issue

Respond ONLY in JSON:
{{
    "type": "medical|fire|accident|crime|natural|civic",
    "severity": "critical|high|medium|low",
    "requires_ambulance": true/false,
    "requires_fire_brigade": true/false,
    "requires_police": true/false,
    "estimated_response_time": "immediate|15min|30min|1hour"
}}"""
        
        response = model.generate_content(prompt)
        return json.loads(clean_gemini_json(response.text))
    except:
        return {
            "type": "civic",
            "severity": "medium",
            "requires_ambulance": False,
            "requires_fire_brigade": False,
            "requires_police": False,
            "estimated_response_time": "1hour"
        }

@app.post("/api/emergency-report")
async def emergency_report(
    name: str = Form(...),
    phone: str = Form(...),
    complaint: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    address: Optional[str] = Form(None)
):
    """
    Agent 3: Emergency Response Endpoint
    Handles health/emergency situations with immediate routing
    """
    # Classify emergency
    classification = classify_emergency(complaint)
    
    # Route to appropriate service
    response_data = {
        "status": "success",
        "id": str(uuid.uuid4())[:8],
        "classification": classification,
        "location": {"lat": latitude, "lon": longitude, "address": address},
        "recommended_actions": []
    }
    
    # Find nearest facilities
    if classification['requires_ambulance'] or classification['type'] == 'medical':
        nearest_hospital = find_nearest_facility(latitude, longitude, "hospitals")
        if nearest_hospital:
            response_data['nearest_hospital'] = nearest_hospital
            response_data['recommended_actions'].append(
                f"Ambulance dispatched to {nearest_hospital['name']} (Distance: {nearest_hospital['distance']:.0f}m)"
            )
    
    if classification['requires_fire_brigade'] or classification['type'] == 'fire':
        nearest_fire = find_nearest_facility(latitude, longitude, "fire_stations")
        if nearest_fire:
            response_data['nearest_fire_station'] = nearest_fire
            response_data['recommended_actions'].append(
                f"Fire brigade from {nearest_fire['name']} alerted. Call: {nearest_fire['phone']}"
            )
    
    if classification['requires_police'] or classification['type'] == 'crime':
        nearest_police = find_nearest_facility(latitude, longitude, "police_stations")
        if nearest_police:
            response_data['nearest_police_station'] = nearest_police
            response_data['recommended_actions'].append(
                f"Police from {nearest_police['name']} notified. Call: {nearest_police['phone']}"
            )
    
    # Log to sheet for tracking
    try:
        requests.post("https://shivam2212.app.n8n.cloud/webhook/emergency-intake", json={
            "ID": response_data['id'],
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "name": name,
            "phone": phone,
            "issue": complaint,
            "type": classification['type'],
            "severity": classification['severity'],
            "location": f"{latitude},{longitude}",
            "Status": "Active"
        }, timeout=5)
    except Exception as e:
        print(f"n8n sync failed: {e}")
    
    return response_data

@app.get("/api/emergency-facilities")
async def get_emergency_facilities(
    lat: float,
    lon: float,
    radius_km: float = 5.0
):
    """Get all emergency facilities within radius."""
    facilities_in_range = {
        "hospitals": [],
        "fire_stations": [],
        "police_stations": []
    }
    
    radius_m = radius_km * 1000
    
    for facility_type, facilities in EMERGENCY_SERVICES.items():
        for facility in facilities:
            distance = calculate_distance(lat, lon, facility['lat'], facility['lon'])
            if distance <= radius_m:
                facility_copy = facility.copy()
                facility_copy['distance'] = distance
                facilities_in_range[facility_type].append(facility_copy)
    
    return {
        "status": "success",
        "location": {"lat": lat, "lon": lon},
        "radius_km": radius_km,
        "facilities": facilities_in_range
    }

# =============================================================================
# ORIGINAL AGENT 1: CIVIC REPORT (Enhanced)
# =============================================================================

def vision_verifier(image_data: bytes):
    """Verifies if the image shows a legitimate civic issue."""
    try:
        prompt = """Analyze this image carefully. Does it show a civic/municipal issue that needs government attention?

Valid civic issues include:
- Potholes, road damage, cracks
- Broken/dim streetlights
- Water leaks, burst pipes
- Sewage overflow, drain blockages
- Garbage accumulation, littering
- Damaged sidewalks/pavements
- Fallen trees, overgrown vegetation blocking paths
- Broken traffic signs/signals
- Graffiti on public property

Invalid (not civic issues):
- Personal portraits/selfies
- Landscapes, nature photos
- Food, products, objects
- Private property issues
- Animals, pets
- Indoor scenes

Respond ONLY in JSON format: {"valid": true} or {"valid": false}"""
        
        contents = [prompt, {"mime_type": "image/jpeg", "data": image_data}]
        res = model.generate_content(contents)
        result = json.loads(clean_gemini_json(res.text))
        print(f"Vision Verifier Result: {result}")
        return result
    except Exception as e:
        print(f"Vision Verifier Error: {e}")
        return {"valid": True}

def vision_description_agent(image_data: bytes):
    """Generates a text description from an image for zero-click reporting."""
    try:
        prompt = """Describe the civic/municipal issue visible in this photo in one clear, professional sentence.

Format: "[Type of issue] at [visible location details]. [Brief description of severity/impact]"

Example: "Large pothole on asphalt road. Approximately 2 feet wide, poses risk to vehicles."

If no civic issue is visible, respond with exactly: "None"
"""
        contents = [prompt, {"mime_type": "image/jpeg", "data": image_data}]
        res = model.generate_content(contents)
        desc = res.text.strip()
        print(f"Vision Description: {desc}")
        return None if "none" in desc.lower() else desc
    except Exception as e:
        print(f"Vision Description Error: {e}")
        return None

def classification_agent(complaint: str):
    """Categorizes the issue and sets urgency."""
    try:
        prompt = f"""Classify this civic complaint into one category and assign urgency level.

Categories:
- Water: leaks, burst pipes, water quality, supply issues
- Sewage: drains, gutters, overflow, blockages
- Roads: potholes, cracks, traffic, pavement damage
- Electric: streetlights, power lines, poles, electrical hazards

Urgency levels:
- high: immediate safety risk (live wires, large potholes, sewage overflow)
- medium: significant inconvenience or minor safety concern
- low: cosmetic issues, minor maintenance

Respond ONLY in JSON: {{"category": "Water|Sewage|Roads|Electric", "urgency": "low|medium|high"}}

Complaint: {complaint}"""
        
        res = model.generate_content(prompt)
        result = json.loads(clean_gemini_json(res.text))
        print(f"Classification Result: {result}")
        return result
    except Exception as e:
        print(f"Classification Error: {e}")
        return {"category": "Roads", "urgency": "medium"}

def drafting_agent(name, email, complaint, location, category, urgency):
    """Drafts a formal municipal email body."""
    try:
        prompt = f"""Write a formal municipal complaint email based on the following details:

Citizen: {name} ({email})
Location: {location}
Category: {category}
Urgency: {urgency}
Issue: {complaint}

Rules:
1. Use a professional, respectful, yet firm tone.
2. Explain the public hazard or inconvenience caused by this issue.
3. Keep the email concise but formal (3 paragraphs max).
4. Do NOT use placeholders like [Date] - use actual information provided.

End the email exactly with:
Thank you,
{name}
{email}
Reported Location: {location}
"""
        res = model.generate_content(prompt)
        return res.text
    except Exception as e:
        print(f"Drafting Agent Error: {e}")
        return f"Formal report for {category} issue at {location}. Details: {complaint}."

@app.post("/send-report")
async def send_report(
    name: str = Form(...),
    email: str = Form(...),
    complaint: str = Form(None),
    latitude: float = Form(...),
    longitude: float = Form(...),
    address: str = Form(None),
    image: UploadFile = File(None)
):
    """Agent 1: Standard civic report submission (original endpoint)."""
    print(f"\n=== New Report Received ===")
    print(f"Name: {name}, Email: {email}")
    print(f"Location: {latitude}, {longitude}")
    print(f"Complaint: {complaint}")
    print(f"Image: {image.filename if image else 'None'}")
    
    # 1. IMAGE & VISION PROCESSING
    img_b64 = None
    image_bytes = None
    
    if image:
        image_bytes = await image.read()
        img_b64 = base64.b64encode(image_bytes).decode()
        
        print("Running vision verification...")
        v_check = vision_verifier(image_bytes)
        if not v_check.get("valid"):
            print("Image rejected: Not a civic issue")
            raise HTTPException(
                status_code=400, 
                detail="Image rejected: Not a civic issue. Please upload a photo showing potholes, broken lights, water leaks, garbage, or similar municipal problems."
            )

        if not complaint or complaint.strip() == "" or complaint.lower() == "undefined":
            print("Generating description from image...")
            complaint = vision_description_agent(image_bytes)
            if not complaint:
                raise HTTPException(
                    status_code=400, 
                    detail="Could not identify a civic issue from the uploaded image. Please add a text description or upload a clearer photo."
                )
            print(f"Generated description: {complaint}")

    if not complaint:
        raise HTTPException(
            status_code=400, 
            detail="Please provide a description OR upload a photo of the civic issue."
        )

    # 2. DUPLICATE CHECK
    try:
        print("Checking for duplicates...")
        SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&t={int(time.time())}"
        df = pd.read_csv(SHEET_URL)
        df.columns = [c.strip() for c in df.columns]

        if {"Status", "Location", "issue"}.issubset(df.columns):
            pending = df[df["Status"].astype(str).str.lower().str.strip() == "pending"]
            keywords = {"pothole", "drainage", "leak", "garbage", "light", "sewage", "wire"}
            current_keywords = {k for k in keywords if k in complaint.lower()}

            for _, row in pending.iterrows():
                loc_str = str(row["Location"]).replace(" ", "")
                if ',' in loc_str:
                    ex_lat, ex_lon = map(float, loc_str.split(','))
                    distance = calculate_distance(latitude, longitude, ex_lat, ex_lon)
                    if distance < 50:
                        existing_issue = str(row.get("issue", "")).lower()
                        if any(k in existing_issue for k in current_keywords):
                            print(f"Duplicate found: {distance}m away")
                            raise HTTPException(
                                status_code=409, 
                                detail="A similar report is already active in this area."
                            )
        print("No duplicates found")
    except HTTPException: 
        raise
    except Exception as e: 
        print(f"Duplicate check error (non-fatal): {e}")

    # 3. CLASSIFICATION & ROUTING
    print("Classifying complaint...")
    cl = classification_agent(complaint)
    cat = cl.get('category', 'Roads')
    urg = cl.get('urgency', 'medium')
    print(f"Category: {cat}, Urgency: {urg}")

    tokens = set(re.findall(r"\b[a-z]+\b", complaint.lower()))
    dept = next((d for d in OFFICERS if any(k in tokens for k in d['keywords'])), None)
    
    if not dept: 
        dept = next((d for d in OFFICERS if d['name'].split()[0].lower() in cat.lower()), OFFICERS[0])
    
    print(f"Routed to: {dept['name']} ({dept['email']})")

    # 4. DATA SYNC
    report_id = str(uuid.uuid4())[:8]
    loc_display = address if address else f"{latitude}, {longitude}"
    full_loc_info = f"{loc_display}\nMaps: https://www.google.com/maps?q={latitude},{longitude}"

    try:
        print("Syncing to n8n...")
        requests.post("https://shivam2212.app.n8n.cloud/webhook/city-report-intake", json={
            "ID": report_id, "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "name": name, "email": email, "issue": complaint,
            "category": cat, "urgency": urg, "location": f"{latitude},{longitude}", "Status": "Pending"
        }, timeout=5)
        print("n8n sync successful")
    except Exception as e:
        print(f"n8n sync failed (non-fatal): {e}")

    print("Drafting email...")
    email_body = drafting_agent(name, email, complaint, full_loc_info, cat, urg)
    
    try:
        print("Sending email...")
        payload = {
            "from": {"address": "no-reply@ead86fd4bcfd6c15.maileroo.org", "display_name": "CityGuardian"},
            "to": [{"address": dept['email']}],
            "subject": f"[{urg.upper()}] New {cat} Report at {loc_display[:20]}...",
            "html": email_body.replace("\n", "<br>")
        }
        if img_b64:
            payload["attachments"] = [{"file_name": "issue.jpg", "content": img_b64, "type": "image/jpeg"}]

        email_response = requests.post(
            "https://smtp.maileroo.com/api/v2/emails", 
            headers={"Authorization": f"Bearer {MAILEROO_API_KEY}", "Content-Type": "application/json"},
            json=payload, 
            timeout=10
        )
        print(f"Email sent: {email_response.status_code}")
    except Exception as e: 
        print(f"Email dispatch failed (non-fatal): {e}")

    print(f"=== Report {report_id} Processed Successfully ===\n")
    
    return {
        "status": "success", 
        "id": report_id,
        "department": dept['name'], 
        "urgency": urg,
        "ai_description": complaint if image else None 
    }

@app.get("/")
def health(): 
    return {
        "status": "active", 
        "message": "CityGuardian Multi-Agent System",
        "agents": {
            "agent_1": "Civic Reporting (Active)",
            "agent_2": "Social Listening (Active)",
            "agent_3": "Emergency Response (Active)"
        }
    }
