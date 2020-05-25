"""
flask server
"""
import numpy as np
import cv2
from flask import Flask, request, jsonify
from autocrop import Cropper

app = Flask(__name__)
host = '0.0.0.0'
port = 8080
img_WIDTH = 224
img_HEIGH = 224
img_chanels = 3

model_real_age = None
model_gender = None

@app.route('/')
def home_endpoint():
    return 'Welcome to Flask endpoint! Here you can predict your real age - wich age you look, and your gender!'

def load_model_age_real():
    global model_real_age
    prototxtFilePath = 'age.prototxt'
    modelFilePath = 'dex_imdb_wiki.caffemodel'
    model_real_age = cv2.dnn.readNetFromCaffe(prototxtFilePath, modelFilePath)

def load_model_gender():
    global model_gender
    prototxtFilePath = 'gender.prototxt'
    modelFilePath = 'gender.caffemodel'
    model_gender = cv2.dnn.readNetFromCaffe(prototxtFilePath, modelFilePath)

load_model_age_real()
load_model_gender()

def normalize(img):
    return (img /255)


def preprocessingIMG(img):
    """
    preprocessing the img. crop face, normalize, scale picture, reshaping...
    """
    # crop face
    cropper = Cropper(width=img_WIDTH, height=img_HEIGH, face_percent=60)
    cropped_array = cropper.crop(img)
    # normalize
    img = normalize(cropped_array)
    # scale
    img = cv2.resize(img, (img_WIDTH, img_HEIGH))
    # reshaping
    img = img.reshape((img_chanels, img_WIDTH, img_HEIGH))
    data = np.array(img)[np.newaxis, :] # converts shape from (X,) to (1, X)
    return data


def responseBuilder(preds_real_age, preds_gender):
    """
    build the response to the front end giving the results
    :param preds_real_age:
    :param preds_gender:
    :return:
    """
    real_age = str(np.argmax(preds_real_age))
    gender = 'male' if np.argmax(preds_gender) else 'female'

    res = jsonify(real_age=real_age,
                  gender=gender)
    return res


@app.route('/predict', methods=['POST'])
def get_prediction():
    if request.method == 'POST':
        try:
            filestr = request.files['image'].read()
            npimg = np.fromstring(filestr, np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            data = preprocessingIMG(img)

            # get real age prediction
            model_real_age.setInput(data)
            preds_real_age = model_real_age.forward()

            # get gender prediction
            model_gender.setInput(data)
            preds_gender = model_gender.forward()
        except:
            return jsonify('-1')

    return responseBuilder(preds_real_age,preds_gender)
