from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from typing import List, Dict
from backend.car_model import load_car_model

app = FastAPI()

model, feature_names = load_car_model()

class CarPredictionRequest(BaseModel):
    """input  for car mpg prediction"""
    cylinders: int
    displacement: float
    horsepower: float
    weight: float
    acceleration: float
    model_year: int
    origin: int  # 1: American, 2: European, 3: Asian

@app.get("/")
def read_root():
    return {"message": "Car MPG Prediction API", "status": "active"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def predict_mpg(request: CarPredictionRequest):
    
    try:
        print("trying to run")
        # Convert input to appropriate format for the model
        features = {
            'cylinders': request.cylinders,
            'displacement': request.displacement,
            'horsepower': request.horsepower,
            'weight': request.weight,
            'acceleration': request.acceleration,
            'model_year': request.model_year,  # Renamed for clarity
            'origin': request.origin
        }
        
        input_data = [[
            features['cylinders'],
            features['displacement'],
            features['horsepower'],
            features['weight'],
            features['acceleration'],
            features['model_year'],
            features['origin']
        ]]
        
        print("processed features")
        # df = pd.DataFrame([d.model_dump() for d in data])
        # print("did dump")
        
        print("Feature names:", feature_names)
        print("Type of feature_names:", type(feature_names))

        
        input_df = pd.DataFrame(input_data, columns=feature_names)
        print("converted to df")
        
        print(input_df)
        
        prediction = model.predict(input_df)
        
        print(prediction)
        
        return {"predicted_mpg": float(prediction[0])}
    
    except Exception as e: 
        print("Error encountered")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # import uvicorn
    # uvicorn.run("car_api:app", host="0.0.0.0", port=8000, reload=True)
    pass