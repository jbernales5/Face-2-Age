"""
flask server
"""
import numpy as np
import cv2
from flask import Flask, request, jsonify
from autocrop import Cropper
from flask_cors import CORS, cross_origin
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)

host = '0.0.0.0'
port = 80
img_WIDTH = 224
img_HEIGH = 224
img_chanels = 3
NORMALIZATION_PARAM = 1
DEBUG_MODE = False


model_age_apparent = None
model_gender = None
mean_file = 'trained_models/ilsvrc_2012_mean.npy'

@app.route('/')
def home_endpoint():
    return 'Welcome to Flask endpoint! Here you can predict wich age you look, and your gender!'

def load_model_age_apparent():
    global model_age_apparent
    prototxtFilePath = 'trained_models/apparent_age/apparent_age.prototxt'
    modelFilePath = 'trained_models/apparent_age/dex_chalearn_iccv2015.caffemodel'
    model_age_apparent = cv2.dnn.readNetFromCaffe(prototxtFilePath, modelFilePath)


def load_model_gender():
    global model_gender
    prototxtFilePath = 'trained_models/gender/gender.prototxt'
    modelFilePath = 'trained_models/gender/gender.caffemodel'
    model_gender = cv2.dnn.readNetFromCaffe(prototxtFilePath, modelFilePath)


def preprocessingIMG(img):
    """
    preprocessing the img. crop face, normalize, scale picture, reshaping...
    """
    # crop face
    cropper = Cropper(width=img_WIDTH, height=img_HEIGH, face_percent=60)
    cropped_array = cropper.crop(img)

    # create blob
    blob = cv2.dnn.blobFromImage(cropped_array, scalefactor=1.0, mean=np.load(mean_file).mean(1).mean(1), swapRB=True)

    return blob


def responseBuilder(preds_age_apparent, preds_gender):
    """
    build the response to the front end giving the results
    :param preds_real_age:
    :param preds_age_apparent:
    :param preds_gender:
    :return:
    """
    #factorVector = np.arange(0,101,1)
    #multiplos = factorVector * preds_age_apparent
    #apparent_age = np.sum(multiplos)
    apparent_age = str(np.argmax(preds_age_apparent))
    gender = 'male' if np.argmax(preds_gender) else 'female'

    res = jsonify(apparent_age=apparent_age,
                  gender=gender)
    print("age:{0} - prob age:{1} - gender: {2} - prob gender: {3}".format(apparent_age,
                    preds_age_apparent[0][np.argmax(preds_age_apparent)],
                    gender,
                    preds_gender[0][np.argmax(preds_gender)]))
    return res


@app.route('/predict', methods=['POST'])
def get_prediction():
    if request.method == 'POST':
        try:
            filestr = request.files['image'].read()
            npimg = np.frombuffer(filestr, np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            data = preprocessingIMG(img)

            # get apparent age prediction
            model_age_apparent.setInput(data)
            preds_age_apparent = model_age_apparent.forward()

            # get gender prediction
            model_gender.setInput(data)
            preds_gender = model_gender.forward()

        except Exception as exc:
            print(exc)
            return jsonify(-1)

    return responseBuilder(preds_age_apparent,preds_gender)


if __name__ == '__main__':
    print("Launching server on {0}:{1}".format(host, port))
    load_model_age_apparent()
    load_model_gender()
    app.run(host=host, port=port, debug=DEBUG_MODE)
