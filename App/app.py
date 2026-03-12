from fastapi import FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse
from typing import Union, List
from pydantic import BaseModel, Field
from loguru import logger
from App.preprocess_data import preprocess
from App.explain import compute_shap_values

import pandas as pd
import numpy as np

import skops.io as sio
import uuid
import uvicorn
import sys
import yaml
import os
import joblib



logger.remove()
logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
    level=10
)

logger.add(
    "log.log", rotation="1 MB", level='DEBUG', compression="zip"
)

CONFIG_PATH = './config_prod.yml'

with open(CONFIG_PATH, 'r', encoding='utf-8') as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.SafeLoader)

MODEL_DIR = config['MODEL_DIR']
CATEGORY_MAP_DIR = config['CATEGORY_MAP_DIR']

# 모델 로드
model = joblib.load(MODEL_DIR)

# 카테고리 맵 로드
category_dict = joblib.load(CATEGORY_MAP_DIR)

app = FastAPI(
    title='Sample API for ML Model Serving',
    version = config['VERSION'],
    description = 'Based on ML with FastAPI Serving'
)

class PredictionInput(BaseModel):
    HomeService: object
    EncarDiagnosis: object
    Manufacturer: object
    Model: object
    Badge: object
    Transmission: object
    FuelType: object
    Year: float
    Mileage: float
    SellType: object
    OfficeCityState: object

class ResponseModel(BaseModel):
    prediction_Id: object
    predict: object
    explain_feature_labels: List[str]
    explain_feature_values: List[float]

@app.get('/', include_in_schema=False)
async def redirect():
    return RedirectResponse('/docs')

@app.post('/predict', response_model = ResponseModel, status_code=status.HTTP_200_OK)
async def predict(input: PredictionInput):
    result = {
        'prediction_Id': str(uuid.uuid4()),
        'explain_feature_labels': [],
        'explain_feature_values': [],
        'predict': ""
    }
    input_df = pd.DataFrame([input.dict()])
    input_df = preprocess(input_df, category_dict)

    logger.info(input_df)

    prediction = str(model.predict(input_df))
    logger.info(prediction)

    result['predict'] = prediction
    result['explain_feature_labels'], result['explain_feature_values'] = compute_shap_values(model, input_df)

    logger.info(result)

    return result