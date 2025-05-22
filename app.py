from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import torch

# Load model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# FastAPI setup
app = FastAPI()

# Mount static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Pydantic request/response
class SimilarityRequest(BaseModel):
    text1: str
    text2: str

class SimilarityResponse(BaseModel):
    similarity_score: float

# Serve homepage (index.html)
@app.get("/", response_class=HTMLResponse)
async def serve_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# API Endpoint for external use (e.g., Postman, curl)
@app.post("/predict", response_model=SimilarityResponse)
async def predict_similarity(request: SimilarityRequest):
    with torch.no_grad():
        emb1 = model.encode(request.text1, convert_to_tensor=True)
        emb2 = model.encode(request.text2, convert_to_tensor=True)

    sim_score = util.cos_sim(emb1, emb2).item()
    normalized_score = (sim_score + 1) / 2
    return {"similarity_score": round(normalized_score, 4)}

# HTML form handler
@app.post("/predict_form", response_class=HTMLResponse)
async def handle_form(request: Request, text1: str = Form(...), text2: str = Form(...)):
    with torch.no_grad():
        emb1 = model.encode(text1, convert_to_tensor=True)
        emb2 = model.encode(text2, convert_to_tensor=True)

    sim_score = util.cos_sim(emb1, emb2).item()
    normalized_score = (sim_score + 1) / 2

    return templates.TemplateResponse("index.html", {
        "request": request,
        "score": round(normalized_score, 4),
        "text1": text1,
        "text2": text2
    })
