from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import torch


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


app = FastAPI()


class SimilarityRequest(BaseModel):
    text1: str
    text2: str


class SimilarityResponse(BaseModel):
    similarity_score: float


@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Semantic Similarity API!",
        "usage": "Send a POST request to /predict with 'text1' and 'text2'."
    }


@app.post("/predict", response_model=SimilarityResponse)
async def predict_similarity(request: SimilarityRequest):
 
    with torch.no_grad():
        emb1 = model.encode(request.text1, convert_to_tensor=True)
        emb2 = model.encode(request.text2, convert_to_tensor=True)


    sim_score = util.cos_sim(emb1, emb2).item()


    normalized_score = (sim_score + 1) / 2


    return {"similarity_score": round(normalized_score, 4)}
