from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import joblib
import pandas as pd
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'nutrition_model.pkl')
model = joblib.load(MODEL_PATH)

# Updated Data Structure with strict validation
class PredictionRequest(BaseModel):
    # Age: Must be between 0 and 120 years based on NHANES ranges
    age: float = Field(..., ge=0, le=120, description="Age must be between 0 and 120")
    
    # Gender: Strictly limited to 1 (Male) or 2 (Female)
    gender: Literal[1, 2] = Field(..., description="1 for Male, 2 for Female")
    
    # Intake: Cannot be negative
    iron_intake: float = Field(..., ge=0, description="Iron intake cannot be negative")
    vit_d_intake: float = Field(..., ge=0, description="Vitamin D intake cannot be negative")

@app.post("/predict")
def predict(data: PredictionRequest):
    try:
        input_data = pd.DataFrame([{
            'RIDAGEYR': data.age,
            'RIAGENDR': data.gender,
            'DR1TIRON': data.iron_intake,
            'DR1TVD': data.vit_d_intake
        }])

        prediction = int(model.predict(input_data)[0])
        probability = model.predict_proba(input_data).tolist()[0]

        recommendations = []
        if prediction == 1:
            if data.iron_intake < 8.0:
                recommendations.append("Increase consumption of iron-rich foods like lean meats, beans, and spinach.")
            if data.vit_d_intake < 15.0:
                recommendations.append("Incorporate Vitamin D sources like fatty fish, egg yolks, or fortified cereals.")
            recommendations.append("Clinical Action: Consult a dietitian for a Serum Ferritin or Vitamin D blood test.")
        else:
            recommendations.append("Continue maintaining a diverse and balanced diet.")
            recommendations.append("Monitor your nutrient intake regularly.")

        return {
            "deficiency_risk": "High" if prediction == 1 else "Low",
            "confidence": round(max(probability) * 100, 2),
            "raw_prediction": prediction,
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))