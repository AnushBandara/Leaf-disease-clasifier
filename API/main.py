from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "Models",
    "potato_earlyblight_lateblight_healthy_custom_model",
    "3.keras",
)

BLIGHT_MODEL = tf.keras.models.load_model(MODEL_PATH)
print(BLIGHT_MODEL.summary())
BLIGHT_CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


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
    predictions = BLIGHT_MODEL.predict(img_batch)

    predicted_class = BLIGHT_CLASS_NAMES[int(np.argmax(predictions[0]))]
    confidence = float(np.max(predictions[0]))

    return {"disease": predicted_class, "confidence": confidence}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)