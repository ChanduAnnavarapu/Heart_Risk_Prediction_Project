from logger_config import setup_logger
from backend.training import train_model
import os
from pathlib import Path
from dotenv import load_dotenv
import logging
from joblib import load
import pandas as pd

def predict(input_data):
    try:
        load_dotenv()
        PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))
        MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_DIR") / os.getenv("MODEL_NAME")
        LOG_PATH =  PROJECT_ROOT / os.getenv("LOG_DIR") / os.getenv("LOG_NAME")

        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        logger = setup_logger(LOG_PATH)
        
        model=load(MODEL_PATH)
        logger.info("Model loaded successfully.")
        
        input_df=pd.DataFrame([input_data])
        logger.info(f"Input data received for prediction: {input_df}")
        prediction_value=int(model.predict(input_df)[0])
        if prediction_value==1:
            prediction="Presence of heart disease"
        else:
            prediction="Absence of heart disease"
            
        probality=round(float(model.predict_proba(input_df)[0][1]*100), 2)
        logger.info(f"Prediction made: {prediction}, probability: {probality}")
        return {
            "prediction": prediction_value, 
            "probability": probality
        }

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise e
    
# sample_input={
#     "age": 52,
#     "sex": 1,
#     "cp": 0,
#     "trestbps": 125,
#     "chol": 212,
#     "fbs": 0,
#     "restecg": 1,
#     "thalach": 168,
#     "exang": 0,
#     "oldpeak": 1.0,
#     "slope": 2,
#     "ca": 0,
#     "thal": 2
# }

# result=predict(sample_input)
# print(result)
