import tensorflow as tf
import tensorflow.keras
from keras.callbacks import tensorboard_v1
from tensorflow_core.python.keras.callbacks import ModelCheckpoint, History, CSVLogger, EarlyStopping
import os
from config import Config
from logger import logging
from models.resNet50.ResNet50 import ResNet50
from preprocessing.Preprocessing import load_and_preprocessing

def callbacks():

    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',
                                            baseline=None, restore_best_weights=False)

    checkpoint = ModelCheckpoint('best_model.hdf5', monitor='loss',
                                 verbose=1,
                                 save_best_only=True, mode='auto')
    history = History()

    csvlogger = CSVLogger('csvlog.csv', separator=',', append=False)


    return [checkpoint, history, csvlogger]

def train(train, test):
    # load model
    model = ResNet50(input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.CHANELS), classes=config.NUM_CLASSES)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(train, epochs=config.EPOCHS, validation_data=test, verbose=1, callbacks=callbacks())
    preds = model.evaluate(test)


if __name__ == '__main__':
    tf.executing_eagerly()
    config = Config()
    logging.info('RUNNING TRAINING')
    # load data
    trainSet, testSet = load_and_preprocessing()
    train(trainSet, testSet)
