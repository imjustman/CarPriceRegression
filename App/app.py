from fastapi import FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse
from typing import Union, List
from pydantic import BaseModel, Field
from loguru import logger

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

@app.get('/', include_in_schema=False)
async def redirect():
    return RedirectResponse('/docs')
