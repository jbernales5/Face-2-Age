import os
from datetime import time

from IPython import display

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


def save_checkpoint(epoch, checkpoint):
    # saving (checkpoint) the model every 20 epochs
    if (epoch + 1) % 20 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)

    checkpoint.save(file_prefix=checkpoint_prefix)

def train(X_train, X_test):
    # load model
    model = ResNet50(input_shape=(project.IMG_HEIGHT, project.IMG_WIDTH, project.CHANELS), classes=project.NUM_CLASSES)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    checkpoint_prefix = os.path.join(project.checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint()

    # TODO CALLBACKS, SAVE MODEL, SAVE LOG, EARLYSTOPPING

    model.fit(X_train, steps_per_epoch=4000, epochs=project.EPOCHS, batch_size=project.BATCH_SIZE, validation_data=X_test)
    # preds = model.evaluate(X_test, Y_test)


if __name__ == '__main__':
    project = Project()
    logging.info('RUNNING TRAINING')
    # load data
    X_train, X_test = load_and_preprocessing()

    X_train = X_train.repeat()
    X_test = X_test.repeat()

    X_train = X_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    X_test = X_test.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # TODO  info_data(X_train, Y_train, X_test, Y_test)

    train(X_train, X_test)


