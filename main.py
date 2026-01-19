# from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from openai import OpenAI
# from dotenv import load_dotenv
# import requests
# import base64
# import os

# # ---------------- LOAD ENV ----------------
# load_dotenv(override=True)

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# MAILEROO_API_KEY = os.getenv("MAILEROO_API_KEY")

# if not OPENAI_API_KEY:
#     raise RuntimeError("OPENAI_API_KEY missing")

# if not MAILEROO_API_KEY:
#     raise RuntimeError("MAILEROO_API_KEY missing")

# # ---------------- CLIENTS ----------------
# client = OpenAI(api_key=OPENAI_API_KEY)

# # ---------------- APP ----------------
# app = FastAPI(title="CityGuardian API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://127.0.0.1:5500"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ---------------- MAILEROO FUNCTION ----------------
# def send_email_maileroo(subject: str, body: str, to_email: str, attachment=None):
#     url = "https://smtp.maileroo.com/api/v2/emails"
#     headers = {
#         "Authorization": f"Bearer {MAILEROO_API_KEY}",
#         "Content-Type": "application/json",
#     }

#     payload = {
#         "from": {
#             "address": "no-reply@ead86fd4bcfd6c15.maileroo.org",
#             "display_name": "CityGuardian"
#         },
#         "to": [
#             {
#                 "address": to_email
#             }
#         ],
#         "subject": subject,
#         "text": body,
#         "html": body.replace("\n", "<br>")
#     }

#     # ✅ Attach image if provided
#     if attachment:
#         # attachment["filename"] = attachment.get("filename") or "complaint-image.jpg"
#         payload["attachments"] = [
#             {
#                 "file_name": attachment["file_name"],
#                 "content": attachment["content"],
#                 "type": attachment["type"]
#             }
#         ]

#     response = requests.post(url, json=payload, headers=headers, timeout=15)

#     print("Maileroo status:", response.status_code)
#     print("Maileroo response:", response.text)

#     if response.status_code not in [200, 201, 202]:
#         raise RuntimeError(
#             f"Maileroo failed: {response.status_code} {response.text}"
#         )


# # ---------------- ROUTES ----------------
# @app.post("/send-report")
# async def send_report(
#     name: str = Form(...),
#     email: str = Form(...),
#     complaint: str = Form(...),
#     latitude: float = Form(...),
#     longitude: float = Form(...),
#     image: UploadFile = File(None),
# ):
#     # Image is optional — just read if present
#     attachment = None

#     if image:
#         image_bytes = await image.read()

#         # ✅ FORCE a safe filename
#         safe_filename = "complaint-image.jpg"

#         if image.filename and image.filename.strip():
#             safe_filename = image.filename.strip()

#         attachment = {
#             "file_name": "complaint-image.jpg",
#             "content": base64.b64encode(image_bytes).decode("utf-8"),
#             "type": image.content_type or "image/jpeg"
#         }



#     # -------- PROMPT --------
#     prompt = f"""
# You are an AI assistant for a municipal corporation.

# Draft a formal, polite civic complaint email.

# Citizen Name: {name}
# Citizen Email: {email}

# Issue:
# {complaint}

# Location:
# Latitude {latitude}
# Longitude {longitude}

# End the email professionally.
# """

#     # -------- OPENAI --------
#     ai_response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[{"role": "user", "content": prompt}],
#     )

#     email_body = ai_response.choices[0].message.content

#     # -------- SEND EMAIL (MAILEROO) --------
#     try:
#         send_email_maileroo(
#             subject="Civic Complaint Report",
#             body=email_body,
#             to_email="shivamkillarikar22@gmail.com",
#             attachment=attachment  # change if needed
#         )
#     except Exception as e:
#         # Do NOT crash demo
#         print("Email sending failed:", e)

#     return {
#         "status": "success",
#         "email_body": email_body,
#         "note": "Email generated and sent using Maileroo"
#     }

# @app.get("/")
# def health():
#     return {"status": "CityGuardian backend running"}


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
# Using gemini-1.5-flash for speed and free tier availability
model = genai.GenerativeModel('gemini-1.5-flash-latest')

MAILEROO_API_KEY = os.getenv("MAILEROO_API_KEY")

app = FastAPI(title="CityGuardian Pro – Agentic Backend (Gemini Edition)")

# --- CORS ---
origins = [
    "http://127.0.0.1:5500",
    "https://city-guardian-yybm.vercel.app",
    "https://city-guardian-n8n-integration.vercel.app",
    "https://cityguardian-react.vercel.app/",
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

DEFAULT_EMAIL = "shivamkillarikar22@gmail.com"

# --- UTILS ---
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371000 
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def clean_gemini_json(text):
    """Removes markdown code blocks from Gemini response to get clean JSON."""
    clean = re.sub(r"```json\s?|\s?```", "", text).strip()
    return clean

# --- AI AGENTS (GEMINI POWERED) ---

def vision_verifier(image_data: bytes):
    """Verifies if the image shows a legitimate civic issue."""
    try:
        prompt = "Is this a civic issue (garbage, pothole, leak, broken light)? Respond ONLY in JSON: {'valid': true/false}"
        contents = [prompt, {"mime_type": "image/jpeg", "data": image_data}]
        res = model.generate_content(contents)
        return json.loads(clean_gemini_json(res.text))
    except Exception as e:
        print(f"Vision Verifier Error: {e}")
        return {"valid": True}

def vision_description_agent(image_data: bytes):
    """Generates a text description from an image for zero-click reporting."""
    try:
        prompt = "Describe the civic issue in this photo in one clear, formal sentence. If none found, say 'None'."
        contents = [prompt, {"mime_type": "image/jpeg", "data": image_data}]
        res = model.generate_content(contents)
        desc = res.text.strip()
        return None if "none" in desc.lower() else desc
    except: return None

def classification_agent(complaint: str):
    """Categorizes the issue and sets urgency."""
    try:
        prompt = f"Classify this civic complaint. Categories: Water, Sewage, Roads, Electric. Respond ONLY in JSON: {{'category': '...', 'urgency': 'low|medium|high'}}\n\nComplaint: {complaint}"
        res = model.generate_content(prompt)
        return json.loads(clean_gemini_json(res.text))
    except: return {"category": "Roads", "urgency": "medium"}

def drafting_agent(name, email, complaint, location, category, urgency):
    """Drafts a formal municipal email body."""
    try:
       #prompt = f"Write a formal based on Citizen: {name}({email}} Location: {location} Category : {category}  Urgency : {urgency} Issue: {complaint} End exactly with : Thank You /n {name} /n {email} /n Reported Location : {location} ."
        prompt = f"""
Write a formal municipal complaint email based on the following details:

Citizen: {name} ({email})
Location: {location}
Category: {category}
Urgency: {urgency}
Issue: {complaint}

Rules:
1. Use a professional, respectful, yet firm tone.
2. Explain the public hazard caused by this issue.
3. Keep the email concise but formal (3 paragraphs).

End the email exactly with:
Thank you,
{name}
{email}
Reported Location: {location}
"""
        res = model.generate_content(prompt)
        return res.text
    except: return f"Formal report for {category} issue at {location}. Details: {complaint}."

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
    # 1. IMAGE & VISION PROCESSING
    img_b64 = None
    image_bytes = None
    
    if image:
        image_bytes = await image.read()
        # Keep base64 for Maileroo/email attachment
        img_b64 = base64.b64encode(image_bytes).decode()
        
        # Vision validation with Gemini
        v_check = vision_verifier(image_bytes)
        if not v_check.get("valid"):
            raise HTTPException(status_code=400, detail="Image rejected: Not a civic issue.")

        # Zero-Click Logic: Generate text from image if complaint is empty
        if not complaint or complaint.strip() == "" or complaint.lower() == "undefined":
            complaint = vision_description_agent(image_bytes)
            if not complaint:
                raise HTTPException(status_code=400, detail="Could not identify issue from image.")

    if not complaint:
        raise HTTPException(status_code=400, detail="No complaint text or image provided.")

    # 2. DUPLICATE CHECK (Geospatial + Keyword)
    SHEET_ID = '1yHcKcLdv0TEEpEZ3cAWd9A_t8MBE-yk4JuWqJKn0IeI'
    try:
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
                    if calculate_distance(latitude, longitude, ex_lat, ex_lon) < 50:
                        existing_issue = str(row.get("issue", "")).lower()
                        if any(k in existing_issue for k in current_keywords):
                            raise HTTPException(status_code=409, detail="A similar report is already active in this area.")
    except HTTPException: raise
    except Exception as e: print(f"Duplicate check log: {e}")

    # 3. CLASSIFICATION & ROUTING
    cl = classification_agent(complaint)
    cat = cl.get('category', 'Roads')
    urg = cl.get('urgency', 'medium')

    tokens = set(re.findall(r"\b[a-z]+\b", complaint.lower()))
    dept = next((d for d in OFFICERS if any(k in tokens for k in d['keywords'])), None)
    
    if not dept: 
        dept = next((d for d in OFFICERS if d['name'].split()[0].lower() in cat.lower()), OFFICERS[0])

    # 4. DATA SYNC (n8n & Email)
    report_id = str(uuid.uuid4())[:8]
    loc_display = address if address else f"{latitude}, {longitude}"
    full_loc_info = f"{loc_display}\nMaps: https://www.google.com/maps?q={latitude},{longitude}"

    try:
        requests.post("https://shivam2212.app.n8n.cloud/webhook/city-report-intake", json={
            "ID": report_id, "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "name": name, "email": email, "issue": complaint,
            "category": cat, "urgency": urg, "location": f"{latitude},{longitude}", "Status": "Pending"
        }, timeout=5)
    except: pass

    email_body = drafting_agent(name, email, complaint, full_loc_info, cat, urg)
    
    try:
        payload = {
            "from": {"address": "no-reply@ead86fd4bcfd6c15.maileroo.org", "display_name": "CityGuardian"},
            "to": [{"address": dept['email']}],
            "subject": f"[{urg.upper()}] New {cat} Report at {loc_display[:20]}...",
            "html": email_body.replace("\n", "<br>")
        }
        if img_b64:
            payload["attachments"] = [{"file_name": "issue.jpg", "content": img_b64, "type": "image/jpeg"}]

        requests.post("https://smtp.maileroo.com/api/v2/emails", 
                      headers={"Authorization": f"Bearer {MAILEROO_API_KEY}", "Content-Type": "application/json"},
                      json=payload, timeout=10)
    except Exception as e: print(f"Email Dispatch failed: {e}")

    return {
        "status": "success", 
        "id": report_id,
        "department": dept['name'], 
        "urgency": urg,
        "ai_description": complaint if image else None 
    }

@app.get("/")
def health(): return {"status": "active"}
    








