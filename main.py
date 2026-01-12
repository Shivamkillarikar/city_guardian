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
from google import genai
from google.genai import types
from dotenv import load_dotenv
import requests, base64, os, json, math, time, random
import pandas as pd
from datetime import datetime
import uuid

# ================= ENV =================
load_dotenv(override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MAILEROO_API_KEY = os.getenv("MAILEROO_API_KEY")

if not GEMINI_API_KEY or not MAILEROO_API_KEY:
    raise RuntimeError("Missing API keys")

# NEW SDK INITIALIZATION
client = genai.Client(api_key=GEMINI_API_KEY)

# Use stable flash model for best balance
MODEL_NAME = "gemini-1.5-flash"
origins=[
    "http://127.0.0.1:5500",
    "https://city-guardian-yybm.vercel.app",
    "https://city-guardian-yybm.vercel.app/"
]
app = FastAPI(title="CityGuardian Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

OFFICERS = [
    {"name": "Water Dept", "email": "shivamkillarikar007@gmail.com", "keywords": ["water", "leak", "pipe", "burst"]},
    {"name": "Sewage Dept", "email": "shivamkillarikar22@gmail.com", "keywords": ["sewage", "drain", "gutter", "overflow"]},
    {"name": "Roads Dept", "email": "aishanidolan@gmail.com", "keywords": ["road", "pothole", "traffic", "pavement"]},
    {"name": "Electric Dept", "email": "adityakillarikar@gmail.com", "keywords": ["light", "wire", "pole", "shock", "power"]},
]

# ================= ROBUST AI WRAPPER (Handles 429 Errors) =================
def generate_with_retry(model_id, contents, config=None, retries=3):
    """
    Wraps the API call to handle Rate Limits (429) automatically.
    """
    delay = 2
    for attempt in range(retries):
        try:
            return client.models.generate_content(
                model=model_id,
                contents=contents,
                config=config
            )
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if attempt < retries - 1:
                    wait_time = delay + random.uniform(0, 1)
                    print(f"⚠️ Quota hit. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    delay *= 2
                    continue
            # Handle Model alias not found
            if "404" in error_str and "not found" in error_str:
                print("⚠️ Model alias issue, trying specific version...")
                return client.models.generate_content(
                    model="gemini-1.5-flash-001",
                    contents=contents,
                    config=config
                )
            raise e
    raise RuntimeError("Max retries exceeded for AI generation")

# ================= AI AGENTS =================
def vision_verifier(img_bytes: bytes):
    try:
        image_part = types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
        
        # STRICT PROMPT: Reject blurry or irrelevant images
        prompt = """
        Analyze this image for civic complaints (potholes, garbage dumps, broken streetlights, sewage overflow).
        
        CRITICAL RULES:
        1. If the image is just a generic photo of a street, park, or people walking with no obvious damage/garbage, respond NO.
        2. If the image is too blurry, dark, or shaky to clearly identify a specific issue, respond NO.
        3. Only respond YES if you clearly see a maintenance issue or hazard.
        
        Respond only YES or NO.
        """
        
        # Use retry wrapper
        response = generate_with_retry(MODEL_NAME, [prompt, image_part])
        print(f"🤖 Vision AI says: {response.text.strip()}")
        
        return {"valid": "YES" in response.text.strip().upper()}
    except Exception as e:
        print(f"⚠️ Vision Check Error: {e}")
        # FAIL-SAFE: If AI crashes, we usually allow it, BUT you can set this to False if you want strict blocking.
        return {"valid": True} 

def classification_agent(complaint: str):
    try:
        prompt ="""Classify this complaint: "{complaint}". Categories: Water, Sewage, Roads, Electric. Respond ONLY in JSON: {{"category": "...", "urgency": "low|medium|high"}}"""
        
        response = generate_with_retry(
            MODEL_NAME, 
            prompt, 
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        return json.loads(clean_json_response(response.text))
    except Exception as e:
        print(f"Classification Error: {e}")
        return {"category": "Roads", "urgency": "medium"}

def drafting_agent(name, email, complaint, location, category, urgency):
    try:
        prompt = f"""Write a formal email to {category} Dept. Citizen: {name} ({email}). Issue: {complaint} at {location}. Urgency: {urgency}. Sign off: Thank you, {name}, {email}."""
        # Add small delay to help quota
        time.sleep(1)
        response = generate_with_retry(MODEL_NAME, prompt)
        return response.text
    except Exception as e:
        print(f"Drafting Error: {e}")
        return f"To {category} Dept,\n\nReporting issue: {complaint} at {location}.\n\nThank you,\n{name}"

# ================= SYNCHRONOUS PROCESSING =================
def process_external_integrations(report_id, name, email, complaint, category, urgency, latitude, longitude, loc_display, full_location, img_b64):
    """
    Runs immediately (BLOCKING) to ensure execution before server shutdown.
    """
    print("⏳ Starting Integrations...")
    
    # 1. N8N TRIGGER
    try:
        requests.post(
            "https://shivam2212.app.n8n.cloud/webhook/city-report-intake", 
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
    except Exception as e:
        print(f"N8N Error: {e}")

    # 2. EMAIL DISPATCH
    try:
        dept = next((d for d in OFFICERS if category.lower() in d["name"].lower()), OFFICERS[0])
        email_body = drafting_agent(name, email, complaint, full_location, category, urgency)

        payload = {
            "from": {"address": "no-reply@ead86fd4bcfd6c15.maileroo.org", "display_name": "CityGuardian"},
            "to": [{"address": dept["email"]}],
            "subject": f"[{urgency.upper()}] New {category} Report",
            "html": email_body.replace("\n", "<br>")
        }
        if img_b64:
            payload["attachments"] = [{"file_name": "issue.jpg", "content": img_b64, "type": "image/jpeg"}]

        # UPDATED URL: Using the standard v1 transactional endpoint
        response = requests.post(
            "https://smtp.maileroo.com/api/v2/emails", 
            headers={"Authorization": f"Bearer {MAILEROO_API_KEY}"},
            json=payload,
            timeout=15
        )
        
        if response.status_code >= 200 and response.status_code < 300:
            print(f"✅ Email sent successfully to {dept['name']}")
        else:
            print(f"❌ Maileroo Failed: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"❌ Email Dispatch failed: {e}")

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
    print(f"📥 New Report: {complaint[:30]}...")

    # --- 1. IMAGE CHECK (BLOCKING) ---
    img_b64 = None
    if image:
        img_bytes = await image.read()
        img_b64 = base64.b64encode(img_bytes).decode()
        
        # Check image validity
        check = vision_verifier(img_bytes)
        
        # STOP HERE if the image is invalid
        if not check.get("valid"):
             print(f"⛔ Report Rejected: Image invalid.")
             raise HTTPException(
                 status_code=400, 
                 detail="The uploaded image does not appear to show a valid civic issue. Please upload a clear photo of the problem."
             )

    # --- 2. CLASSIFICATION ---
    cl = classification_agent(complaint)
    category = cl.get("category", "Roads")
    urgency = cl.get("urgency", "medium")

    # --- 3. DUPLICATE CHECK ---
    try:
        SHEET_ID = "1yHcKcLdv0TEEpEZ3cAWd9A_t8MBE-yk4JuWqJKn0IeI"
        SHEET_URL = f"[https://docs.google.com/spreadsheets/d/](https://docs.google.com/spreadsheets/d/){SHEET_ID}/export?format=csv"
        
        df = pd.read_csv(SHEET_URL)
        if {"Status", "Location"}.issubset(df.columns):
            pending = df[df["Status"].astype(str).str.lower().str.strip() == "pending"]
            for _, row in pending.iterrows():
                try:
                    lat, lon = map(float, str(row["Location"]).split(","))
                    if calculate_distance(latitude, longitude, lat, lon) < 50:
                        row_cat = str(row.get("Category", "")).lower()
                        if category.lower() in row_cat:
                             raise HTTPException(status_code=409, detail="Duplicate report exists.")
                except: continue
    except HTTPException: raise
    except Exception: pass

    # --- 4. PREPARE DATA ---
    loc_display = address if address else f"{latitude}, {longitude}"
    google_maps_link=f"https://www.google.com/maps?q={latitude},{longitude}"
    full_location = f"[{loc_display}\nGoogle Maps:{google_maps_link}"
    report_id = str(uuid.uuid4())[:8]

    # --- 5. EXECUTE SYNCHRONOUSLY ---
    # The user waits here until emails are actually sent
    process_external_integrations(
        report_id, name, email, complaint, category, urgency, latitude, longitude, loc_display, full_location, img_b64
    )

    dept_name = next((d["name"] for d in OFFICERS if category.lower() in d["name"].lower()), OFFICERS[0]["name"])

    return {
        "status": "success", 
        "ticket": report_id, 
        "department": dept_name, 
        "message": "Report submitted."
    }

@app.get("/")
def health(): return {"status": "Active"}
