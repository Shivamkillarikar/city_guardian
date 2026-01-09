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

load_dotenv(override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MAILEROO_API_KEY = os.getenv("MAILEROO_API_KEY")

app = FastAPI()

# --- CORS SETTINGS ---
origins = [
    "http://127.0.0.1:5500",
    "https://city-guardian-yybm.vercel.app",
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
    R = 6371000  # Radius of Earth in meters
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

# --- OFFICERS DATA ---
OFFICERS = [
    {"name": "Water Dept", "email": "shivamkillarikar007@gmail.com", "keywords": ["water", "leak", "pipe"]},
    {"name": "Sewage Dept", "email": "shivamkillarikar22@gmail.com", "keywords": ["sewage", "drain", "smell"]},
    {"name": "Roads Dept", "email": "aishanidolan@gmail.com", "keywords": ["road", "pothole", "traffic"]},
    {"name": "Electric Dept", "email": "adityakillarikar@gmail.com", "keywords": ["light", "wire", "pole"]},
]

# --- AGENTS ---
def vision_verifier(img_b64: str):
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Is this a civic issue (garbage, pothole, leak, etc)? Respond ONLY in JSON: {'valid': true/false}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]}],
            response_format={"type": "json_object"}
        )
        return json.loads(res.choices[0].message.content)
    except: return {"valid": True}

def classification_agent(complaint: str):
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Classify this complaint. Use only these categories: Water, Sewage, Roads, Electric. JSON ONLY: {{'category': '...', 'urgency': 'low|medium|high'}}\n\n{complaint}"}],
        response_format={"type": "json_object"}
    )
    return json.loads(res.choices[0].message.content)

def drafting_agent(name, email, complaint, location, category, urgency):
    system_msg = "You are an official Municipal Correspondence AI. Write a formal 3-paragraph email."
    user_msg = f"Citizen: {name}\nEmail: {email}\nLocation: {location}\nCategory: {category}\nUrgency: {urgency}\nIssue: {complaint}\n\nEnd with name, email, and location."
    
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
    )
    return res.choices[0].message.content

# --- ROUTES ---
@app.post("/send-report")
async def send_report(
    name: str = Form(...),
    email: str = Form(...),
    complaint: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    image: UploadFile = File(None)
):
    # 1. Image Verification
    img_b64 = None
    if image:
        content = await image.read()
        img_b64 = base64.b64encode(content).decode()
        v_check = vision_verifier(img_b64)
        if not v_check.get("valid"):
            return {"status": "error", "message": "AI rejected image: Not a civic issue."}

    # 2. AI Classification
    cl = classification_agent(complaint)
    category = cl['category']

    # 3. DUPLICATE DETECTION (Google Sheet Check)
    SHEET_ID = '1yHcKcLdv0TEEpEZ3cAWd9A_t8MBE-yk4JuWqJKn0IeI'
    SHEET_URL = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv'
    
    try:
        df = pd.read_csv(SHEET_URL)
        df.columns = df.columns.str.strip()
        # Filter Pending issues of the same category
        pending = df[df['Status'].str.strip().str.capitalize() == 'Pending']
        
        for _, row in pending.iterrows():
            ex_lat, ex_lon = map(float, str(row['Location']).split(','))
            if calculate_distance(latitude, longitude, ex_lat, ex_lon) < 50:
                # If duplicate found, throw 409 error
                raise HTTPException(status_code=409, detail=f"Duplicate Request: This {category} issue is already being tracked (Ticket #{row['ID']}).")
    except HTTPException as e: raise e
    except Exception as e: print(f"Sheet check skipped: {e}")

    # 4. Routing and Email Prep
    google_maps_link = f"https://www.google.com/maps?q={latitude},{longitude}"
    location_text = f"{latitude}, {longitude}\nMap: {google_maps_link}"
    
    dept = next((d for d in OFFICERS if d['name'].lower() in category.lower() or any(k in complaint.lower() for k in d['keywords'])), OFFICERS[0])
    
    email_body = drafting_agent(name, email, complaint, location_text, category, cl['urgency'])
    
    # 5. Send via Maileroo
    payload = {
        "from": {"address": "no-reply@ead86fd4bcfd6c15.maileroo.org", "display_name": "CityGuardian"},
        "to": [{"address": dept['email']}],
        "subject": f"[{cl['urgency'].upper()}] New {category} Report",
        "html": email_body.replace("\n", "<br>") # Fixed 'body' to 'email_body'
    }
    if img_b64:
        payload["attachments"] = [{"file_name": "issue.jpg", "content": img_b64, "type": "image/jpeg"}]

    requests.post("https://smtp.maileroo.com/api/v2/emails", 
                  headers={"Authorization": f"Bearer {MAILEROO_API_KEY}", "Content-Type": "application/json"},
                  json=payload)

    return {"status": "success", "department": dept['name'], "urgency": cl['urgency']}

@app.get("/")
def health(): return {"status": "active"}
    
        
