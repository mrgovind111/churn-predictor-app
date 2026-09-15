# Customer Churn Predictor — End-to-End ML Deployment

Capstone project: a scikit-learn model served via FastAPI, containerized with Docker, and deployed to the cloud with a Streamlit frontend.

## Live URLs

- Frontend UI: https://churn-predictor-app-kqzwypxsnz3y4ttwsznzc2.streamlit.app/
- API docs: https://churn-predictor-app-i5pq.onrender.com/docs
- API health: https://churn-predictor-app-i5pq.onrender.com/health

> The Render free tier sleeps after 15 minutes of inactivity. The first request may take 30-60 seconds to wake the service.

## Project Structure

churn-predictor-app/
  app/                  FastAPI backend
    main.py             API endpoints
    schemas.py          Pydantic validation
    __init__.py
  frontend/             Streamlit UI
    streamlit_app.py
  model/                Serialized ML artifacts
    churn_model.pkl
    scaler.pkl
    encoders.pkl
  Dockerfile
  requirements.txt
  README.md

## Run Locally

1. Create and activate a virtual environment:
   python -m venv venv
   venv\Scripts\activate

2. Install dependencies:
   pip install -r requirements.txt

3. Start the API:
   uvicorn app.main:app --reload --port 8000

4. Start the frontend (in a new terminal):
   streamlit run frontend/streamlit_app.py

Open http://localhost:8501

## Run with Docker

   docker build -t churn-api .
   docker run -p 8000:8000 churn-api

## API Endpoints

GET /health
  Response: {"status": "ok"}

POST /predict
  Request:
    {"tenure": 12, "monthly_charges": 70.5, "total_charges": 850.0, "gender": "Male"}
  Response:
    {"prediction": 0, "label": "No Churn", "probability": 0.7234}

## Tech Stack

- ML: scikit-learn, pandas, numpy, joblib
- Backend: FastAPI, Uvicorn, Pydantic
- Frontend: Streamlit, requests
- DevOps: Docker, Render, Streamlit Cloud, GitHub