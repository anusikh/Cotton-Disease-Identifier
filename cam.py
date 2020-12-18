import cv2
import os
import numpy as np
from PIL import Image
import tensorflow
from tensorflow.keras.models import load_model

model = load_model('model.h5')
video = cv2.VideoCapture(0)

categories = ["diseased cotton leaf","diseased cotton plant","fresh cotton leaf","fresh cotton plant"]
font = cv2.FONT_HERSHEY_SIMPLEX 

while True:
    _, frame = video.read()
    im = Image.fromarray(frame,"RGB")
    im = im.resize((100,100))
    img_array = np.array(im)
    img_array = np.expand_dims(img_array,axis=0)
    preds = model.predict(img_array)
    pred_index = np.argmax(preds)
    print(categories[pred_index])
    cv2.putText(frame,categories[pred_index],(50, 50),  
                font, 1,  
                (0, 255, 255),  
                2,  
                cv2.LINE_4)
    cv2.imshow("Capturing",frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
