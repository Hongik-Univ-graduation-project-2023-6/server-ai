import numpy as np
import cv2

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def read_image(contents):
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = np.float32(img)
    return img

def verify_image(model, img):
    img = cv2.resize(img, (32, 32)) / 255.0
    nx, ny, nrgb = img.shape
    img = img.reshape(1, nx*ny*nrgb)
    pred_class = model.predict(img)[0]
    return pred_class

def predict_class(model, img):
    img = cv2.resize(img, (299, 299))
    img = img[np.newaxis, :]
    pred_classes = model.predict(img)[0]
    return pred_classes
