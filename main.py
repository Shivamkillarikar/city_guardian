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

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from dotenv import load_dotenv
import requests, base64, os, json, math, time
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

# Use Flash for speed and cost efficiency
MODEL_NAME = "gemini-1.5-flash"

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

# ================= AI AGENTS (UPDATED FOR NEW SDK) =================
def vision_verifier(img_bytes: bytes):
    try:
        # Prepare the image for the new SDK
        # Convert bytes to base64 if needed, or pass bytes directly if supported by PIL. 
        # The new SDK handles Pillow images or bytes nicely in 'contents'.
        
        # Simpler approach: Create a part object manually if passing raw bytes
        # Or encode to base64 string
        b64_img = base64.b64encode(img_bytes).decode('utf-8')
        
        image_part = types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")

        prompt = "Is this a civic issue (like a pothole, garbage, broken light)? Respond YES or NO."
        
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[prompt, image_part]
        )
        return {"valid": "YES" in response.text.strip().upper()}
    except Exception as e:
        print(f"Vision Error: {e}")
        # Fail safe: Accept image if AI fails
        return {"valid": True}

def classification_agent(complaint: str):
    try:
        prompt = f
        """Classify this complaint: "{complaint}". Categories: Water, Sewage, Roads, Electric. Respond ONLY in JSON: {{"category": "...", "urgency": "low|medium|high"}}"""
        
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Classification Error: {e}")
        return {"category": "Roads", "urgency": "medium"}

def drafting_agent(name, email, complaint, location, category, urgency):
    try:
        prompt = f"""Write a formal email to {category} Dept. Citizen: {name} ({email}). Issue: {complaint} at {location}. Urgency: {urgency}. Sign off: Thank you, {name}, {email}."""
        
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        return response.text
    except Exception as e:
        print(f"Drafting Error: {e}")
        return f"To {category} Dept,\n\nReporting issue: {complaint} at {location}.\n\nThank you,\n{name}"

# ================= BACKGROUND TASKS =================
def process_external_integrations(report_id, name, email, complaint, category, urgency, latitude, longitude, loc_display, full_location, img_b64):
    """
    Handles N8N and Email sending in the background so the user gets a fast response.
    """
    
    # 1. N8N TRIGGER
    try:
        requests.post(
            "[https://shivam2212.app.n8n.cloud/webhook/city-report-intake](https://shivam2212.app.n8n.cloud/webhook/city-report-intake)", # FIXED URL
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
            timeout=10
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

        requests.post(
            "https://smtp.maileroo.com/api/v2/emails](https://smtp.maileroo.com/api/v2/emails", # FIXED URL
            headers={"Authorization": f"Bearer {MAILEROO_API_KEY}"},
            json=payload,
            timeout=10
        )
        print(f"Email sent to {dept['name']}")
    except Exception as e:
        print(f"Email Dispatch failed: {e}")

# ================= MAIN ROUTE =================
@app.post("/send-report")
async def send_report(
    background_tasks: BackgroundTasks,
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
        check = vision_verifier(img_bytes)
        if not check.get("valid"):
             print("Warning: Vision agent flagged this image.")

    # 2. CLASSIFICATION
    cl = classification_agent(complaint)
    category = cl.get("category", "Roads")
    urgency = cl.get("urgency", "medium")


    # 4. PREPARE DATA
    loc_display = address if address else f"{latitude}, {longitude}"
    # Fixed URL in string
    full_location = f"{loc_display} (Map: [http://maps.google.com/?q=](http://maps.google.com/?q=){latitude},{longitude})"
    report_id = str(uuid.uuid4())[:8]

    # 5. DELEGATE TO BACKGROUND TASK
    # This prevents the request from timing out while waiting for N8N and Maileroo
    background_tasks.add_task(
        process_external_integrations,
        report_id, name, email, complaint, category, urgency, latitude, longitude, loc_display, full_location, img_b64
    )

    dept_name = next((d["name"] for d in OFFICERS if category.lower() in d["name"].lower()), OFFICERS[0]["name"])

    return {
        "status": "success", 
        "ticket": report_id, 
        "department": dept_name, 
        "message": "Report submitted. Processing emails in background."
    }

@app.get("/")
def health(): return {"status": "Active"}
        

