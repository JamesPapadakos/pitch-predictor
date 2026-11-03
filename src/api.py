import os 
from fastapi.middleware.cors import CORSMiddleware

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.pitch_predictor import (
    list_available_pitchers,
    predict_pitch_probabilities,
)
from src.model import DEFAULT_PITCHER_ID, PITCH_TYPES

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
    pitcher_id: int = Field(default=DEFAULT_PITCHER_ID, gt=0)
    balls: int = Field(ge=0, le=3)
    strikes: int = Field(ge=0, le=2)
    stand: Literal["L", "R"]
    inning: int = Field(ge=1)
    outs_when_up: int = Field(ge=0, le=2)
    previous_pitch: str

    @field_validator("previous_pitch")
    @classmethod
    def validate_previous_pitch(cls, value):
        if value not in {*PITCH_TYPES, "START", "UNKNOWN"}:
            raise ValueError("Unrecognized previous pitch type.")
        return value


@app.get("/")
def root():
    return {"message": "Pitch Predictor API is running"}


@app.post("/predict")
def predict(situation: PitchSituation):
    available_pitchers = {
        pitcher["pitcher_id"]: pitcher
        for pitcher in list_available_pitchers()
    }
    pitcher = available_pitchers.get(situation.pitcher_id)

    if pitcher is None:
        raise HTTPException(
            status_code=404,
            detail="No saved model is available for this pitcher.",
        )

    # The pitcher ID selects the model; it is not a model input feature.
    inputs = situation.model_dump(exclude={"pitcher_id"})

    try:
        results = predict_pitch_probabilities(
            inputs,
            pitcher_id=situation.pitcher_id,
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="The selected pitcher's saved model is unavailable.",
        )

    return {
        "pitcher_id": situation.pitcher_id,
        "pitcher": pitcher["name"],
        "probabilities": [
            {"pitch_type": pitch_type, "probability": float(probability)}
            for pitch_type, probability in results
        ],
    }

@app.get("/pitchers")
def get_pitchers():
    return list_available_pitchers()