from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
# import torch
from PIL import Image
# import albumentations as aug
# from efficientnet_pytorch import EfficientNet

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# model = torch.load("best_model.pth")
# model.eval()

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model=load_model('CropDoc.h5')

def model_predict(file, model):
    my_image=Image.open(file)
    my_img_arr=np.array(my_image)
    my_img_arr=np.expand_dims(my_img_arr,axis=0)
    if my_img_arr.max()>1:
        preds=model.predict(my_img_arr/256)
    else:
        preds=model.predict(my_img_arr)

    preds=np.argmax(preds)
    return preds


# def model_predict(file, model):
#     image = Image.open(file)
#     image = np.array(image)
#     transforms = aug.Compose([
#             aug.Resize(256,256),
#             aug.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225),max_pixel_value=255.0,always_apply=True),
#             ])
#     image = transforms(image=image)["image"]
#     image = np.transpose(image, (2, 0, 1)).astype(np.float32)
#     image = torch.tensor([image], dtype=torch.float)
#     preds = model(image)
#     preds = np.argmax(preds.detach())
#     return preds


@app.route('/')
def index():
    # Main page
    return render_template('index.html')


@app.route('/diagnose')
def index1():
    # Main page
    return render_template('diagnose.html')


@app.route('/predict', methods=['POST'])
def upload():
    # Get the file from post request
    f = request.files['file']
    cat = np.load("cat.npy", allow_pickle=True)
    # Make prediction
        # with open(filename, 'rb') as file:
        #     binaryData = file.read()
    preds = model_predict(f, model)
    result = cat[preds]
    return result


if __name__ == '__main__':
    app.run(debug=True)
