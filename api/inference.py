import numpy as np
import cv2

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def predict_class(model, contents):
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = np.float32(img)
    img = cv2.resize(img, (299, 299))
    img = img[np.newaxis, :]

    pred_classes = softmax(model.predict(img)[0])
    print(pred_classes.sum())
    return pred_classes
