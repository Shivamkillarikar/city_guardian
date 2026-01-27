from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from dotenv import load_dotenv
import requests, base64, os, json, re, math, time
import pandas as pd
from datetime import datetime
import uuid

# 1. INITIALIZATION
load_dotenv(override=True)

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
# Using gemini-1.5-flash for speed and project efficiency
model = genai.GenerativeModel('gemini-1.5-flash')

MAILEROO_API_KEY = os.getenv("MAILEROO_API_KEY")

app = FastAPI(title="CityGuardian Pro – Agentic Backend")

# --- CORS ---
origins = [
    "http://127.0.0.1:5500",
    "http://localhost:3000",
    "https://city-guardian-yybm.vercel.app",
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

# --- UTILS ---
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371000 
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def clean_gemini_json(text):
    """Removes markdown code blocks to isolate JSON string."""
    return re.sub(r"```json\s?|\s?```", "", text).strip()

# --- AI AGENTS ---
def vision_verifier(image_data: bytes):
    try:
        prompt = "Is this a civic issue (garbage, pothole, leak, broken light)? Respond ONLY in JSON: {'valid': true/false}"
        contents = [prompt, {"mime_type": "image/jpeg", "data": image_data}]
        res = model.generate_content(contents)
        return json.loads(clean_gemini_json(res.text))
    except: return {"valid": True}

def vision_description_agent(image_data: bytes):
    try:
        prompt = "Describe the civic issue in this photo in one clear, formal sentence."
        contents = [prompt, {"mime_type": "image/jpeg", "data": image_data}]
        res = model.generate_content(contents)
        return res.text.strip()
    except: return None

def classification_agent(complaint: str):
    try:
        prompt = f"Classify this civic complaint. Categories: Water, Sewage, Roads, Electric. Respond ONLY in JSON: {{'category': '...', 'urgency': 'low|medium|high'}}\n\nComplaint: {complaint}"
        res = model.generate_content(prompt)
        return json.loads(clean_gemini_json(res.text))
    except: return {"category": "Roads", "urgency": "medium"}

def drafting_agent(name, email, complaint, location, category, urgency):
    try:
        prompt = f"Write a formal 3-paragraph municipal complaint email. Citizen: {name}, Location: {location}, Issue: {complaint}. End with: Thank you, {name}, {email}."
        res = model.generate_content(prompt)
        return res.text
    except: return f"Formal report for {category} issue at {location}."

# --- MAIN ROUTE ---
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
    # 1. PROCESS IMAGE & TEXT
    img_b64 = None
    if image:
        image_bytes = await image.read()
        img_b64 = base64.b64encode(image_bytes).decode()
        
        if not vision_verifier(image_bytes).get("valid"):
            raise HTTPException(status_code=400, detail="Image rejected: Not a civic issue.")

        if not complaint or complaint.strip() in ["", "undefined"]:
            complaint = vision_description_agent(image_bytes)

    if not complaint:
        raise HTTPException(status_code=400, detail="No description or valid image provided.")

    # 2. GENERATE IDENTITY & CLASSIFICATION
    report_id = str(uuid.uuid4())[:8]
    cl = classification_agent(complaint)
    cat = cl.get('category', 'Roads')
    urg = cl.get('urgency', 'medium')

    # 3. AGENT 3: EMERGENCY DISPATCH (Telegram/Hospital Finder)
    if urg == 'high' or cat == 'Electric':
        try:
            n8n_agent_3_url = "https://shi22.app.n8n.cloud/webhook-test/emergency-dispatch"
            requests.post(n8n_agent_3_url, json={
                "report_id": report_id,
                "category": cat,
                "latitude": latitude,
                "longitude": longitude,
                "issue": complaint,
                "name": name
            }, timeout=3)
        except Exception as e: print(f"Agent 3 Trigger Failed: {e}")

    # 4. DUPLICATE CHECK
    SHEET_ID = '1yHcKcLdv0TEEpEZ3cAWd9A_t8MBE-yk4JuWqJKn0IeI'
    try:
        SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&t={int(time.time())}"
        df = pd.read_csv(SHEET_URL)
        df.columns = [c.strip() for c in df.columns]
        if {"Status", "Location", "issue"}.issubset(df.columns):
            pending = df[df["Status"].astype(str).str.lower().str.strip() == "pending"]
            for _, row in pending.iterrows():
                loc_str = str(row["Location"]).replace(" ", "")
                if ',' in loc_str:
                    ex_lat, ex_lon = map(float, loc_str.split(','))
                    if calculate_distance(latitude, longitude, ex_lat, ex_lon) < 50:
                        raise HTTPException(status_code=409, detail="A similar report is active.")
    except HTTPException: raise
    except: pass

    # 5. ROUTING & DATA SYNC
    tokens = set(re.findall(r"\b[a-z]+\b", complaint.lower()))
    dept = next((d for d in OFFICERS if any(k in tokens for k in d['keywords'])), None)
    if not dept:
        dept = next((d for d in OFFICERS if d['name'].split()[0].lower() in cat.lower()), OFFICERS[0])

    loc_display = address if address else f"{latitude}, {longitude}"
    full_loc_info = f"{loc_display}\nMaps: https://www.google.com/maps?q={latitude},{longitude}"

    # Sync to n8n (Agent 1 Flow)
    try:
        requests.post("https://shivam2212.app.n8n.cloud/webhook/city-report-intake", json={
            "ID": report_id, "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "name": name, "email": email, "issue": complaint,
            "category": cat, "urgency": urg, "location": f"{latitude},{longitude}", "Status": "Pending"
        }, timeout=5)
    except: pass

    # Email Dispatch
    email_body = drafting_agent(name, email, complaint, full_loc_info, cat, urg)
    try:
        payload = {
            "from": {"address": "no-reply@ead86fd4bcfd6c15.maileroo.org", "display_name": "CityGuardian"},
            "to": [{"address": dept['email']}],
            "subject": f"[{urg.upper()}] New {cat} Report at {loc_display[:15]}...",
            "html": email_body.replace("\n", "<br>")
        }
        if img_b64:
            payload["attachments"] = [{"file_name": "issue.jpg", "content": img_b64, "type": "image/jpeg"}]
        requests.post("https://smtp.maileroo.com/api/v2/emails", 
                      headers={"Authorization": f"Bearer {MAILEROO_API_KEY}", "Content-Type": "application/json"},
                      json=payload, timeout=10)
    except Exception as e: print(f"Email Dispatch failed: {e}")

    return {"status": "success", "id": report_id, "department": dept['name'], "urgency": urg}

@app.get("/")
def health(): return {"status": "active"}
        
