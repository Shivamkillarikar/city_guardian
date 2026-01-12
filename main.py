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
from openai import OpenAI
from dotenv import load_dotenv
import requests, base64, os, json, re, math
import pandas as pd
from datetime import datetime

# 1. INITIALIZATION & CONFIG
load_dotenv(override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MAILEROO_API_KEY = os.getenv("MAILEROO_API_KEY")

app = FastAPI(title="CityGuardian Backend")

# --- CORS SETTINGS ---
# Ensure these match your Vercel deployment exactly
origins = [
    "http://127.0.0.1:5500",
    "https://city-guardian-yybm.vercel.app",
    "https://city-guardian-yybm.vercel.app/",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- UTILS ---
def calculate_distance(lat1, lon1, lat2, lon2):
    """Haversine formula to calculate distance in meters."""
    R = 6371000 
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

# --- OFFICERS / DEPARTMENT DATA ---
OFFICERS = [
    {"name": "Water Dept", "email": "shivamkillarikar007@gmail.com", "keywords": ["water", "leak", "pipe", "burst"]},
    {"name": "Sewage Dept", "email": "shivamkillarikar22@gmail.com", "keywords": ["sewage", "drain", "gutter", "overflow"]},
    {"name": "Roads Dept", "email": "aishanidolan@gmail.com", "keywords": ["road", "pothole", "traffic", "pavement"]},
    {"name": "Electric Dept", "email": "adityakillarikar@gmail.com", "keywords": ["light", "wire", "pole", "shock", "power"]},
]

# --- AI AGENTS ---
def vision_verifier(img_b64: str):
    """Agent 1: Checks if the image is actually a civic issue."""
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Is this a civic issue (garbage, pothole, leak, fallen tree, etc)? Respond ONLY in JSON: {'valid': true/false}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]}],
            response_format={"type": "json_object"}
        )
        return json.loads(res.choices[0].message.content)
    except: return {"valid": True} # Fallback to true to avoid blocking valid reports

def classification_agent(complaint: str):
    """Agent 2: Categorizes the text and assesses urgency."""
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Classify this civic complaint. Use only these categories: Water, Sewage, Roads, Electric. Respond ONLY in JSON: {{'category': '...', 'urgency': 'low|medium|high'}}\n\nComplaint: {complaint}"}],
            response_format={"type": "json_object"}
        )
        return json.loads(res.choices[0].message.content)
    except: return {"category": "General", "urgency": "medium"}

def drafting_agent(name, email, complaint, location, category, urgency):
    """Agent 3: Drafts a professional municipal email."""
    system_msg = "You are a professional Municipal Correspondence AI. Write a formal 3-paragraph email."
    user_msg = f"""
    Write a formal email based on:
    Citizen: {name} ({email})
    Location: {location}
    Category: {category}
    Urgency: {urgency}
    Issue: {complaint}

    End exactly with:
    Thank you,
    {name}
    {email}
    Reported Location: {location}
    """
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
    )
    return res.choices[0].message.content

# --- MAIN ROUTE ---
@app.post("/send-report")
async def send_report(
    name: str = Form(...),
    email: str = Form(...),
    complaint: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    address: str = Form(None),
    image: UploadFile = File(None)
):
    # 1. IMAGE HANDLING & VISION CHECK
    img_b64 = None
    if image:
        content = await image.read()
        img_b64 = base64.b64encode(content).decode()
        v_check = vision_verifier(img_b64)
        if not v_check.get("valid"):
            # Return a 400 error if AI determines image is not a civic issue
            return {"status": "error", "message": "AI rejected image: This does not appear to be a civic issue."}

    # 2. AI CLASSIFICATION
    cl = classification_agent(complaint)
    category = cl.get('category', 'Roads') # Default to Roads if unknown

    # 3. GEOSPATIAL DUPLICATE DETECTION
    SHEET_ID = '1yHcKcLdv0TEEpEZ3cAWd9A_t8MBE-yk4JuWqJKn0IeI'
    SHEET_URL = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv'

    # 3. SMART DUPLICATE DETECTION (Location + Keywords)
    try:
        SHEET_ID = "1yHcKcLdv0TEEpEZ3cAWd9A_t8MBE-yk4JuWqJKn0IeI"
        SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"
        
        df = pd.read_csv(SHEET_URL)
        if {"Status", "Location", "issue"}.issubset(df.columns):
            # Only check 'Pending' reports
            pending = df[df["Status"].astype(str).str.lower().str.strip() == "pending"]
            
            # Define keywords to look for
            issue_keywords = ["pothole", "drainage", "leak", "garbage", "light", "sewage", "wire"]
            
            # Find keywords in the CURRENT complaint
            current_keywords = {k for k in issue_keywords if k in complaint.lower()}
            
            for _, row in pending.iterrows():
                try:
                    # 1. Check Distance
                    lat, lon = map(float, str(row["Location"]).split(","))
                    if calculate_distance(latitude, longitude, lat, lon) < 50:
                        
                        # 2. Check for matching keywords in the existing 'issue' column
                        existing_issue = str(row.get("issue", "")).lower()
                        matching_keywords = [k for k in current_keywords if k in existing_issue]
                        
                        if matching_keywords:
                            # Flag as duplicate only if location AND at least one keyword match
                            raise HTTPException(
                                status_code=409, 
                                detail=f"Duplicate report: A ticket for a '{matching_keywords[0]}' issue already exists at this location."
                            )
                except (ValueError, TypeError): continue
    except HTTPException: raise
    except Exception as e: print(f"Duplicate check log: {e}")
        

    # 4. PREPARE LOCATION & ROUTING
    # Prefer fetched address, fallback to coordinates
    loc_display = address if address else f"{latitude}, {longitude}"
    google_maps_link = f"https://www.google.com/maps?q={latitude},{longitude}"
    full_location_info = f"{loc_display}\nGoogle Maps: {google_maps_link}"
    
    # Standardize department routing
    dept = next((d for d in OFFICERS if d['name'].lower() in category.lower() or any(k in complaint.lower() for k in d['keywords'])), OFFICERS[0])
    
    # 5. DRAFT EMAIL & SEND VIA MAILEROO
    email_body = drafting_agent(name, email, complaint, full_location_info, category, cl.get('urgency', 'medium'))
    
    try:
        payload = {
            "from": {"address": "no-reply@ead86fd4bcfd6c15.maileroo.org", "display_name": "CityGuardian"},
            "to": [{"address": dept['email']}],
            "subject": f"[{cl.get('urgency', 'MED').upper()}] New {category} Report at {loc_display[:30]}...",
            "html": email_body.replace("\n", "<br>")
        }
        if img_b64:
            payload["attachments"] = [{"file_name": "issue.jpg", "content": img_b64, "type": "image/jpeg"}]

        requests.post(
            "https://smtp.maileroo.com/api/v2/emails", 
            headers={"Authorization": f"Bearer {MAILEROO_API_KEY}", "Content-Type": "application/json"},
            json=payload,
            timeout=10
        )
    except Exception as e:
        print(f"Email Dispatch failed: {e}")

    # 6. RETURN SUCCESS
    return {
        "status": "success", 
        "department": dept['name'], 
        "urgency": cl.get('urgency', 'medium'),
        "message": "Report submitted successfully."
    }

@app.get("/")
def health(): return {"status": "active"}
    
    


