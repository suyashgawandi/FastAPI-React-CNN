import uvicorn
from fastapi import FastAPI, File, UploadFile
import cv2
import PIL
from PIL import Image
from pydantic import BaseModel
import backend.model as model
from io import BytesIO
from backend.model import CNN
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import os

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def read_image(image_encoded):
    """
        Returns a grayscale PIL image.
        Resizing handled by the predict_image method
    """
    pil_image = Image.open(BytesIO(image_encoded)).convert('L')
    return pil_image


@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    image = read_image(await file.read())
    prediction = model.predict_image(image)
    return {"prediction": prediction}


if __name__ == "__main__":
    config = uvicorn.Config("main:app", host="0.0.0.0",
                            port=8000, log_level="debug")
    server = uvicorn.Server(config)
    server.run()
