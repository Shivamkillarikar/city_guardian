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
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv
import requests, base64, os, json, re, math
import pandas as pd
from datetime import datetime
import uuid

# ================= ENV =================
load_dotenv(override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MAILEROO_API_KEY = os.getenv("MAILEROO_API_KEY")

if not GEMINI_API_KEY or not MAILEROO_API_KEY:
    raise RuntimeError("Missing API keys")

genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-1.5-flash"

# ================= SAFETY SETTINGS (CRITICAL FIX) =================
# This stops Gemini from blocking emails about "sewage" or "accidents"
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# ================= APP =================
app = FastAPI(title="CityGuardian Backend (Gemini Powered)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for debugging
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= UTILS =================
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))

def clean_json_response(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()
    return text

# ================= DEPARTMENTS =================
OFFICERS = [
    {"name": "Water Dept", "email": "shivamkillarikar007@gmail.com", "keywords": ["water", "leak", "pipe", "burst"]},
    {"name": "Sewage Dept", "email": "shivamkillarikar22@gmail.com", "keywords": ["sewage", "drain", "gutter", "overflow"]},
    {"name": "Roads Dept", "email": "aishanidolan@gmail.com", "keywords": ["road", "pothole", "traffic", "pavement"]},
    {"name": "Electric Dept", "email": "adityakillarikar@gmail.com", "keywords": ["light", "wire", "pole", "shock", "power"]},
]

# ================= AI AGENTS =================

def vision_verifier(img_bytes: bytes):
    """Checks if image is relevant. Returns True/False."""
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        image_part = {"mime_type": "image/jpeg", "data": img_bytes}

        # Simpler prompt without complex JSON strictness
        prompt = (
            "Analyze this image. Is it a civic issue like garbage, pothole, water leak, broken street light, traffic, or construction? "
            "Respond with exactly one word: YES or NO."
        )

        response = model.generate_content(
            [prompt, image_part], 
            safety_settings=SAFETY_SETTINGS
        )
        
        # Simple string check is more robust than JSON parsing
        text = response.text.strip().upper()
        print(f"Vision Agent says: {text}")
        
        if "YES" in text:
            return {"valid": True}
        else:
            return {"valid": False}

    except Exception as e:
        print(f"Vision Error: {e}")
        return {"valid": True} # Fallback to True to not block user

def classification_agent(complaint: str):
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        prompt = f"""
        Classify this civic complaint into: Water, Sewage, Roads, Electric.
        Complaint: "{complaint}" """
        Respond ONLY in JSON: {{"category": "...", "urgency": "low|medium|high"}}
        """
        response = model.generate_content(prompt, safety_settings=SAFETY_SETTINGS)
        return json.loads(clean_json_response(response.text))
    except Exception as e:
        print(f"Classification Error: {e}")
        return {"category": "Roads", "urgency": "medium"}
        

def drafting_agent(name, email, complaint, location, category, urgency):
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        prompt = f"""
        You are a Municipal Correspondence AI. Write a formal 3-paragraph email to the {category} Department.

        Details:
        - Citizen Name: {name}
        - Email: {email}
        - Location: {location}
        - Complaint: {complaint}
        - Urgency: {urgency}

        Structure:
        1. Introduction stating the issue clearly.
        2. Impact on the neighborhood and public safety.
        3. Request for immediate resolution.

        Sign off exactly like this:
        Thank you,
        {name}
        {email}
        """
        
        response = model.generate_content(prompt, safety_settings=SAFETY_SETTINGS)
        
        if not response.text:
            raise ValueError("Empty response from AI")
            
        return response.text

    except Exception as e:
        print(f"Drafting Error (CRITICAL): {e}")
        # Only uses this if AI completely crashes
        return f"To the {category} Department,\n\nI am writing to report a serious issue regarding {complaint} at {location}.\n\nPlease resolve this.\n\nThank you,\n{name}\n{email}"

# ================= MAIN ROUTE =================
@app.post("/send-report")
async def send_report(
    name: str = Form(...),
    email: str = Form(...),
    complaint: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    address: str = Form(""),
    image: UploadFile = File(None),
):
    # 1. IMAGE CHECK
    img_b64 = None
    if image:
        img_bytes = await image.read()
        img_b64 = base64.b64encode(img_bytes).decode()
        
        # Verify
        check = vision_verifier(img_bytes)
        if not check.get("valid"):
             # If you want to force block invalid images, uncomment the next line:
             # raise HTTPException(status_code=400, detail="Image does not look like a civic issue.")
             print("Warning: Vision agent flagged this image, but processing anyway.")

    # 2. CLASSIFICATION
    cl = classification_agent(complaint)
    category = cl.get("category", "Roads")
    urgency = cl.get("urgency", "medium")

    # 3. DUPLICATE CHECK
    try:
        SHEET_ID = "1yHcKcLdv0TEEpEZ3cAWd9A_t8MBE-yk4JuWqJKn0IeI"
        SHEET_URL = f"[https://docs.google.com/spreadsheets/d/](https://docs.google.com/spreadsheets/d/){SHEET_ID}/export?format=csv"
        df = pd.read_csv(SHEET_URL)
        if {"Status", "Location"}.issubset(df.columns):
            pending = df[df["Status"].astype(str).str.lower() == "pending"]
            for _, row in pending.iterrows():
                try:
                    lat, lon = map(float, str(row["Location"]).split(","))
                    if calculate_distance(latitude, longitude, lat, lon) < 50:
                        # Check category match
                        row_cat = str(row.get("Category", "")).lower()
                        if category.lower() in row_cat:
                             raise HTTPException(status_code=409, detail=f"Duplicate report exists.")
                except: continue
    except HTTPException: raise
    except: pass

    # 4. PREPARE DATA
    loc_display = address if address else f"{latitude}, {longitude}"
    full_location = f"{loc_display} (Map: [http://maps.google.com/?q=](http://maps.google.com/?q=){latitude},{longitude})"

    # 5. N8N
    report_id = str(uuid.uuid4())[:8]
    try:
        requests.post(
            "[https://shivam2212.app.n8n.cloud/webhook/city-report-intake](https://shivam2212.app.n8n.cloud/webhook/city-report-intake)",
            json={
                "ID": report_id,
                "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "name": name,
                "email": email,
                "issue": complaint,
                "category": category,
                "urgency": urgency,
                "location": f"{latitude},{longitude}",
                "address": loc_display,
                "Status": "Pending",
            },
            timeout=5
        )
    except: pass

    # 6. EMAIL
    # Fix: Search for category INSIDE department name
    dept = next((d for d in OFFICERS if category.lower() in d["name"].lower()), OFFICERS[0])

    # Generate Email
    email_body = drafting_agent(name, email, complaint, full_location, category, urgency)

    # Send Email
    payload = {
        "from": {"address": "no-reply@ead86fd4bcfd6c15.maileroo.org", "display_name": "CityGuardian"},
        "to": [{"address": dept["email"]}],
        "subject": f"[{urgency.upper()}] New {category} Report",
        "html": email_body.replace("\n", "<br>")
    }
    if img_b64:
        payload["attachments"] = [{"file_name": "issue.jpg", "content": img_b64, "type": "image/jpeg"}]

    try:
        requests.post(
            "[https://smtp.maileroo.com/api/v2/emails](https://smtp.maileroo.com/api/v2/emails)",
            headers={"Authorization": f"Bearer {MAILEROO_API_KEY}"},
            json=payload,
            timeout=10
        )
    except Exception as e:
        print(f"Maileroo Error: {e}")

    return {
        "status": "success",
        "ticket": report_id,
        "department": dept["name"],
        "message": "Report submitted."
    }

@app.get("/")
def health(): return {"status": "Active"}
