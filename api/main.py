from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from api.inference import predict_class
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

MODEL = keras.models.load_model('models/ResNet152V2-600.h5', custom_objects={'Addons': tfa.metrics.F1Score(num_classes=6)})
CLASS_DICT = {
    0: 'scab',
    1: 'healthy',
    2: 'frog_eye_leaf_spot',
    3: 'rust',
    4: 'complex',
    5: 'powdery_mildew',
}

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/inference/")
async def inference_image(file: UploadFile):
    contents = await file.read()
    pred_classes = predict_class(MODEL, contents)
    res = {'results': [{'name': CLASS_DICT[i], 'percentage': int(p*100)} for i, p in enumerate(pred_classes)]}
    return JSONResponse(content=res)
