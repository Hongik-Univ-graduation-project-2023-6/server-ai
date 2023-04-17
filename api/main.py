from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from api.inference import predict_class
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

MODEL = keras.models.load_model('models/inception_v3-demo1.h5')
# CLASS_DICT = {
#     0: '더뎅이병, 검은별무늬병', # scab
#     1: '건강함', # healthy
#     2: '점무늬병', # frog_eye_leaf_spot
#     3: '녹병', # rust
#     4: '복합성 질병', # complex
#     5: '흰가루병', # powdery_mildew
# }
CLASS_DICT = {
    0: '복합성 질병', # complex
    1: '점무늬병', # frog_eye_leaf_spot
    2: '복합성 점무늬병', # frog_eye_leaf_spot complex
    3: '건강함', # healthy
    4: '흰가루병', # powdery_mildew
    5: '복합성 흰가루병', # powdery_mildew complex
    6: '녹병', # rust
    7: '복합성 녹병', # rust complex
    8: '녹병 및 점무늬병', # rust frog_eye_leaf_spot
    9: '검은별무늬병', # scab
    10: '검은별무늬병 및 점무늬병', # scab frog_eye_leaf_spot
    11: '복합성 검은별무늬병 및 점무늬병', # scab frog_eye_leaf_spot complex
}

app = FastAPI()

origins = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/inference/")
async def inference_image(file: UploadFile):
    contents = await file.read()
    pred_classes = predict_class(MODEL, contents)
    pred_list = sorted(
        [{'name': CLASS_DICT[i], 'percentage': int(p*100)} for i, p in enumerate(pred_classes)],
        key=lambda d: d['percentage'],
        reverse=True
    )
    res = {'results': pred_list[:5]}
    return JSONResponse(content=res)
