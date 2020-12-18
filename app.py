from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from flask import Flask, render_template, request, url_for, redirect
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='template')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

MODEL_PATH = './model.h5'

model = load_model(MODEL_PATH)

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(100,100))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, mode='caffe')
    preds = model.predict(x)
    return preds


@app.route('/', methods = ['GET'])
def home():
    return render_template('home.html')

@app.route('/team', methods = ['GET'])
def team():
    return render_template('team.html')

@app.route('/predict', methods=['GET','POST'])
def upload():
    categories = ["diseased cotton leaf","diseased cotton plant","fresh cotton leaf","fresh cotton plant"]
    if request.method=='POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'static', secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path, model)
        #pred_class = decode_predictions(preds, top=1)
        #res = str(pred_class[0][0][1])
        pred_index = np.argmax(preds)
         
        return render_template('predict.html', result = categories[pred_index] , user_image = file_path)
    return None

if __name__ == '__main__':
    app.run()
