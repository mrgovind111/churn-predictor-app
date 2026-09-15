# Customer Churn Predictor API

End-to-end ML deployment capstone: scikit-learn model + FastAPI + Docker, deployed on Render.

## Project Structure

- app/          - FastAPI backend
- frontend/     - Streamlit UI
- model/        - Serialized ML artifacts (.pkl files)
- Dockerfile    - Container build recipe
- requirements.txt

## Run Locally

1. Install dependencies:
   pip install -r requirements.txt

2. Start the API:
   uvicorn app.main:app --reload --port 8000

3. Start the frontend (new terminal):
   streamlit run frontend/streamlit_app.py

Open http://localhost:8501

## Run with Docker

   docker build -t churn-api .
   docker run -p 8000:8000 churn-api

## API Endpoints

GET /health
Returns: {"status": "ok"}

POST /predict
Request:
{"tenure": 12, "monthly_charges": 70.5, "total_charges": 850.0, "gender": "Male"}

Response:
{"prediction": 0, "label": "No Churn", "probability": 0.7234}

## Tech Stack

scikit-learn, FastAPI, Uvicorn, Streamlit, Docker, Render