from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import json

# Tensorflow
import cv2 as cv
import matplotlib.pyplot as plt

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__, static_url_path='', static_folder='static')
app.config['RESULTS_FOLDER'] = os.path.join('static', 'results')

# Model saved
net = cv.dnn.readNetFromTensorflow('models/graph_opt.pb')

print('Model loaded. Check http://127.0.0.1:5000/')

inWidth = 368
inHeight = 368
thr = 0.2

# COCO Dataset
BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
              "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
              ["Neck", "RHip"], ["RHip", "RKnee"], [
                  "RKnee", "RAnkle"], ["Neck", "LHip"],
              ["LHip", "LKnee"], ["LKnee", "LAnkle"], [
                  "Neck", "Nose"], ["Nose", "REye"],
              ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]


def pose_estimation(frame):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight),
                 (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    # MobileNet output [1, 57, -1, -1], we only need the first 19 elements
    out = out[:, :19, :, :]

    assert(len(BODY_PARTS) == out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > thr else None)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3),
                       0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0,
                       0, 360, (0, 0, 255), cv.FILLED)

    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    return frame


def model_predict(img_path, model):
    img = cv.imread(img_path)
    plt.figure(num=None, figsize=(10, 10), dpi=80,
               facecolor='w', edgecolor='k')
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

    estimated_img = pose_estimation(img)
    plt.figure(num=None, figsize=(10, 10), dpi=80,
               facecolor='w', edgecolor='k')
    plt.imshow(cv.cvtColor(estimated_img, cv.COLOR_BGR2RGB))

    return estimated_img


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, net)
        # Save the image to results

        # result_img_fname = 'result_' + secure_filename(f.filename)
        result_img_fname = "final.jpeg"

        result_img_path = os.path.join(basepath, 'static', result_img_fname)
        # result_img_path = os.path.join(app.config['RESULTS_FOLDER'], result_img_fname)
        if cv.imwrite(result_img_path, preds):
            print('Image saved successfully')
            print(result_img_path)
            return render_template('index.html', result_img=result_img_path)
        else:
            print('Image not saved')
            print(result_img_path)
            return render_template('index.html', result_img=None)

    return None


if __name__ == '__main__':
    app.run(debug=True)
