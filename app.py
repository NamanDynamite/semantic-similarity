from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


app = FastAPI()


class SimilarityRequest(BaseModel):
    text1: str
    text2: str


class SimilarityResponse(BaseModel):
    similarity_score: float

@app.post("/predict", response_model=SimilarityResponse)
async def predict_similarity(request: SimilarityRequest):
 
    emb1 = model.encode(request.text1, convert_to_tensor=True)
    emb2 = model.encode(request.text2, convert_to_tensor=True)
    

    sim_score = util.cos_sim(emb1, emb2).item()


    normalized_score = (sim_score + 1) / 2

    return {"similarity_score": round(normalized_score, 4)}
