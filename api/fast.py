"""
FastAPI application for serving the zoologist model.
"""
import os

import pandas as pd
from fastapi import FastAPI
import joblib

# FastAPI instance
app = FastAPI()

# Root endpoint
@app.get("/")
def root():
    """Root endpoint returning a greeting."""
    return {"greeting": "hello"}

# Dummy endpoint
@app.get("/dummy")
def dummy(number: int):
    """Dummy endpoint that returns the square of the input number."""
    return {"result": number ** 2}

# Predict endpoint
@app.get("/predict")
def predict_endpoint(island: str, bill_length_mm : float, bill_depth_mm : float, flipper_length_mm : float, body_mass_g : float, sex: str) :
    """ Predict endpoint that returns a prediction of the pinguin"""
    X_pred = pd.DataFrame([dict(
            island=island,
            bill_length_mm=bill_length_mm,
            bill_depth_mm=bill_depth_mm,
            flipper_length_mm=flipper_length_mm,
            body_mass_g=body_mass_g,
            sex=sex
    )])
    model_path= "models/pipeline_logreg.joblib"
    model = joblib.load(model_path)
    prediction = model.predict(X_pred)
    return {"prediction": prediction[0]}

    # return (model.predict(X_pred))
