from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from pathlib import Path
from app.schemas import ChurnInput, PredictionOutput

app = FastAPI(
    title="Churn Predictor API",
    description="End-to-end ML deployment - Capstone project",
    version="1.0.0",
)

# Allow Streamlit (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load artifacts ONCE at startup ----
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
model = joblib.load(MODEL_DIR / "churn_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
encoder = joblib.load(MODEL_DIR / "encoders.pkl")
print("✅ Model + scaler + encoder loaded")


@app.get("/")
def root():
    return {"message": "Churn Predictor API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict(data: ChurnInput):
    try:
        gender_encoded = encoder.transform([data.gender])[0]

        features = np.array([[
            data.tenure,
            data.monthly_charges,
            data.total_charges,
            gender_encoded,
        ]])

        features_scaled = scaler.transform(features)

        pred = int(model.predict(features_scaled)[0])
        prob = float(model.predict_proba(features_scaled)[0][pred])

        return PredictionOutput(
            prediction=pred,
            label="Churn" if pred == 1 else "No Churn",
            probability=round(prob, 4),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
