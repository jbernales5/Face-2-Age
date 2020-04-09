import os
from Project import Project
from logger import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from preprocessing.Preprocessing import load_and_preprocessing
from models.resNet50.ResNet50 import ResNet50

def info_data(X_train, Y_train, X_test, Y_test):
    print("number of training examples = " + str(X_train.shape[0]))
    print("number of test examples = " + str(X_test.shape[0]))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(Y_test.shape))

if __name__ == '__main__':
    project = Project()
    logging.info('RUNNING APP')
    # load data
    X_train, Y_train, X_test, Y_test = load_and_preprocessing()

    # load model
    model = ResNet50(input_shape=(project.IMG_HEIGHT, project.IMG_WIDTH, project.CHANELS), classes=project.NUM_CLASSES)
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    #info_data(X_train, Y_train, X_test, Y_test)


    model.summary()

    model.fit(X_train, Y_train, epochs = project.EPOCHS, batch_size = project.BATCH_SIZE)
    preds = model.evaluate(X_test, Y_test)