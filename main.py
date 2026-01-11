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

# ================= APP =================
app = FastAPI(title="CityGuardian Backend (Gemini Powered)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "https://city-guardian-yybm.vercel.app",
        "https://city-guardian-yybm.vercel.app/",
    ],
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
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        image_part = {"mime_type": "image/jpeg", "data": img_bytes}

        prompt = (
            "Is this a real civic issue like pothole, garbage, leak, fallen tree, broken light?\n"
            "Respond ONLY in JSON:\n"
            '{"valid": true|false}'
        )

        response = model.generate_content([prompt, image_part])
        return json.loads(clean_json_response(response.text))
    except Exception as e:
        print("Vision agent fallback:", e)
        return {"valid": True}


def classification_agent(complaint: str):
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        prompt = f"""
Classify the civic complaint.

Allowed categories: Water, Sewage, Roads, Electric.

Complaint:
{complaint}

Respond ONLY in JSON:
{{"category":"...", "urgency":"low|medium|high"}}
"""
        response = model.generate_content(prompt)
        return json.loads(clean_json_response(response.text))
    except Exception as e:
        print("Classification fallback:", e)
        return {"category": "Roads", "urgency": "medium"}


def drafting_agent(name, email, complaint, location, category, urgency):
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        prompt = f"""
Write a formal 3-paragraph municipal complaint email.

Citizen: {name} ({email})
Category: {category}
Urgency: {urgency}
Location:
{location}

Complaint:
{complaint}

End exactly with:
Thank you,
{name}
{email}
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print("Drafting fallback:", e)
        return f"Complaint regarding {category} at {location}.\n\nThank you,\n{name}\n{email}"


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
    # ---------- IMAGE CHECK ----------
    img_b64, img_bytes = None, None
    if image:
        img_bytes = await image.read()
        img_b64 = base64.b64encode(img_bytes).decode()
        if not vision_verifier(img_bytes).get("valid"):
            raise HTTPException(status_code=400, detail="Image is not a civic issue")

    # ---------- CLASSIFICATION ----------
    cl = classification_agent(complaint)
    category = cl.get("category", "Roads")
    urgency = cl.get("urgency", "medium")

    # ---------- DUPLICATE CHECK ----------
    SHEET_ID = "1yHcKcLdv0TEEpEZ3cAWd9A_t8MBE-yk4JuWqJKn0IeI"
    SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

    try:
        df = pd.read_csv(SHEET_URL)
        if {"Status", "Location"}.issubset(df.columns):
            pending = df[df["Status"].astype(str).str.lower() == "pending"]
            for _, row in pending.iterrows():
                try:
                    lat, lon = map(float, str(row["Location"]).split(","))
                    if calculate_distance(latitude, longitude, lat, lon) < 50:
                        raise HTTPException(
                            status_code=409,
                            detail=f"Duplicate report already exists (ID: {row.get('ID','N/A')})",
                        )
                except Exception:
                    continue
    except HTTPException:
        raise
    except Exception as e:
        print("Duplicate check skipped:", e)

    # ---------- LOCATION ----------
    loc_display = address if address else f"{latitude}, {longitude}"
    google_maps_link = f"https://www.google.com/maps?q={latitude},{longitude}"
    full_location = f"{loc_display}\nGoogle Maps: {google_maps_link}"

    # ---------- N8N ----------
    report_id = str(uuid.uuid4())[:8]
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
            timeout=5,
        )
    except Exception as e:
        print("n8n failed:", e)

    # ---------- EMAIL ----------
    dept = next(
        (d for d in OFFICERS if d["name"].lower() in category.lower()),
        OFFICERS[0],
    )

    email_body = drafting_agent(name, email, complaint, full_location, category, urgency)

    payload = {
        "from": {
            "address": "no-reply@ead86fd4bcfd6c15.maileroo.org",
            "display_name": "CityGuardian",
        },
        "to": [{"address": dept["email"]}],
        "subject": f"[{urgency.upper()}] New {category} Report",
        "html": email_body.replace("\n", "<br>"),
    }

    if img_b64:
        payload["attachments"] = [
            {"file_name": "issue.jpg", "content": img_b64, "type": "image/jpeg"}
        ]

    try:
        requests.post(
            "https://smtp.maileroo.com/api/v2/emails",
            headers={
                "Authorization": f"Bearer {MAILEROO_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=10,
        )
    except Exception as e:
        print("Email failed:", e)

    return {
        "status": "success",
        "ticket": report_id,
        "department": dept["name"],
        "urgency": urgency,
        "message": "Report submitted successfully",
    }


@app.get("/")
def health():
    return {"status": "CityGuardian (Gemini) Active"}
