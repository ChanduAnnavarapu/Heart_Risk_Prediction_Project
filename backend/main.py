from fastapi import FastAPI
from pydantic import BaseModel

from backend.training import train_model
from backend.predict import predict

app = FastAPI(
    title="Heart Disease Prediction API",
    description="API for training a heart disease prediction model and making predictions based on input",
    version="1.0.0"
)

class HeartDiseasePredictionRequest(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/train")
def train():
        train_model()
        return {"message": "Model training completed successfully."}
    
@app.post("/predict")
def make_prediction(input_data: HeartDiseasePredictionRequest):
        input_data = input_data.model_dump()
        result = predict(input_data)
        return {
            "prediction": result["prediction"],
            "probability": result["probability"]
        }
