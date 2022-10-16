from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import predict_pipeline
from app.model.model import __version__ as model_version

app = FastAPI()

class TextIn(BaseModel):
    age: int
    sex: str
    marital_status: str
    Education: str
    Medication_preparation_by: str
    medication: int
    SAMS_item1: int
    SAMS_item3: int
    SAMS_item6: int
    SAMS_item10: int
    SAMS_item11: int
    SAMS_item15: int
    SAMS_item16: int
    SAMS_item17: int
    SAMS_item19: int

class PredictionOut(BaseModel):
    output: str

@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}

@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    output = predict_pipeline(payload)
    return {"output": output}
