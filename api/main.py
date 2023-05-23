import joblib
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from api.inference import predict_class, read_image, verify_image, softmax
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

MODEL = keras.models.load_model('models/inception_v3-230502-91.h5')
BINARY_MODEL = joblib.load('models/leafs_binary_svm.pkl') 
CLASS_DICT = {
    0: '복합성 질병', # complex
    1: '점무늬병', # frog_eye_leaf_spot
    2: '건강함', # healthy
    3: '흰가루병', # powdery_mildew
    4: '녹병', # rust
    5: '더뎅이병, 검은별무늬병', # scab
}
# CLASS_DICT = {
#     0: '복합성 질병', # complex
#     1: '점무늬병', # frog_eye_leaf_spot
#     2: '복합성 점무늬병', # frog_eye_leaf_spot complex
#     3: '건강함', # healthy
#     4: '흰가루병', # powdery_mildew
#     5: '복합성 흰가루병', # powdery_mildew complex
#     6: '녹병', # rust
#     7: '복합성 녹병', # rust complex
#     8: '녹병 및 점무늬병', # rust frog_eye_leaf_spot
#     9: '검은별무늬병', # scab
#     10: '검은별무늬병 및 점무늬병', # scab frog_eye_leaf_spot
#     11: '복합성 검은별무늬병 및 점무늬병', # scab frog_eye_leaf_spot complex
# }

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
    img = read_image(contents)
    image_class = verify_image(BINARY_MODEL, img)

    if image_class == 0:
        res = {'results': [{'name': '식물 잎 이미지가 아닙니다.', 'percentage': 0}]}
        return JSONResponse(content=res)
    else:
        pred_classes = predict_class(MODEL, img)

        pred_list = sorted([(CLASS_DICT[i], p) for i, p in enumerate(pred_classes)], key=lambda tup: tup[1], reverse=True)[:6]
        prob_list = softmax([p for c, p in pred_list])
        pred_list = [{'name': c, 'percentage': round(p*100, 1)} for c, p in zip([c for c, p in pred_list], prob_list)]
        
        res = {'results': pred_list}
        return JSONResponse(content=res)
