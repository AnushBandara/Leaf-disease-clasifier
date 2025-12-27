from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH_POTATO = os.path.join(
    BASE_DIR, "Models", "Final", "Potato", "potato_model.keras"
)
MODEL_PATH_TOMATO = os.path.join(
    BASE_DIR, "Models", "Final", "Tomato", "tomato_model.keras"
)

POTATO_MODEL = tf.keras.models.load_model(MODEL_PATH_POTATO)
TOMATO_MODEL = tf.keras.models.load_model(MODEL_PATH_TOMATO)

POTATO_CLASS_NAMES = ['Bacteria', 'Fungi', 'Healthy', 'Nematode', 'Pest', 'Phytopthora', 'Virus']
TOMATO_CLASS_NAMES = ['Tomato_Bacterial_spot','Early_blight','Late_blight','Leaf_Mold','Septoria_leaf_spot','Spider_mites_Two_spotted_spider_mite','Target_Spot','YellowLeaf__Curl_Virus', 'mosaic_virus', 'healthy']

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    return np.array(image)


app = FastAPI()


@app.get("/")
async def read_root():
    return {"message": "Welcome to the Leaf Disease Classifier API!"}


@app.post("/potato/")
async def predict_potato_blight(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = read_file_as_image(contents)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img_batch = np.expand_dims(image, 0)
    predictions = POTATO_MODEL.predict(img_batch)

    predicted_class = POTATO_CLASS_NAMES[int(np.argmax(predictions[0]))]
    confidence = float(np.max(predictions[0]))

    return {"disease": predicted_class, "confidence": confidence}

@app.post("/tomato/")
async def predict_tomato_disease(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = read_file_as_image(contents)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img_batch = np.expand_dims(image, 0)
    predictions = TOMATO_MODEL.predict(img_batch)

    predicted_class = TOMATO_CLASS_NAMES[int(np.argmax(predictions[0]))]
    confidence = float(np.max(predictions[0]))

    return {"disease": predicted_class, "confidence": confidence}