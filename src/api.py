import os 
from fastapi.middleware.cors import CORSMiddleware

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.pitch_predictor import predict_pitch_probabilities

app = FastAPI(title="Pitch Predictor API")
allowed_origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class PitchSituation(BaseModel):
    balls: int = Field(ge=0, le=3)
    strikes: int = Field(ge=0, le=2)
    stand: Literal["L", "R"]
    inning: int = Field(ge=1)
    outs_when_up: int = Field(ge=0, le=2)
    previous_pitch: Literal["START", "UNKNOWN", "CH", "FC", "FF", "KC", "SI", "SL", "ST"]


@app.get("/")
def root():
    return {"message": "Pitch Predictor API is running"}


@app.post("/predict")
def predict(situation: PitchSituation):
    try:
        results = predict_pitch_probabilities(situation.model_dump())
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Saved model not found. Run python -m scripts.train first.",
        )

    return {
        "pitcher": "Dylan Cease",
        "probabilities": [
            {"pitch_type": pitch_type, "probability": float(probability)}
            for pitch_type, probability in results
        ],
    }