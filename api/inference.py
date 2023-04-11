import numpy as np
import cv2

def predict_class(model, contents):
  nparr = np.fromstring(contents, np.uint8)
  img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  img = np.float32(img)
  img = cv2.resize(img, (600, 600))
  img = img[np.newaxis, :]

  pred_classes = model.predict(img)[0]
  return pred_classes
