import tensorflow as tf
from tensorflow_core.python.keras.callbacks import ModelCheckpoint, History, CSVLogger, EarlyStopping, TensorBoard
from tensorflow_core.python.keras.optimizer_v2.adam import Adam
from tensorflow_core.python.keras.models import load_model

from ResNet import ResNet, ResNet18
from config import Config
from logger import logging
from models.resNet50.ResNet50 import ResNet50
from preprocessing.Preprocessing import load_and_preprocessing

def loadModel(name):
    model = load_model(name)
    return model
def callbacks():

    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',
                                            baseline=None, restore_best_weights=False)

    checkpoint = ModelCheckpoint('best_model.hdf5', monitor='loss',
                                 verbose=1,
                                 save_best_only=True, mode='auto')
    history = History()

    csvlogger = CSVLogger('csvlog.csv', separator=',', append=False)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=1)

    return [checkpoint, history, csvlogger, tensorboard_callback]

def train(train, test):
    # load model
    lr = config.LEARNING_RATE
    optimizer = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,name='Adam')

    model = ResNet50(input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.CHANELS), classes=config.NUM_CLASSES)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(train, epochs=config.EPOCHS, validation_data=test, verbose=1, callbacks=callbacks())
    preds = model.evaluate(test)

def train2(train, test):
    lr = config.LEARNING_RATE
    optimizer = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')

    model = ResNet18(input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 1), classes=config.NUM_CLASSES)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(train, epochs=config.EPOCHS, validation_data=test, verbose=1, callbacks=callbacks())
    preds = model.evaluate(test)

def pred(model, data):
    data = data.take(1)
    for img in data:
        img = img[0]
        pred = model.predict(data)

    return pred
if __name__ == '__main__':
    tf.executing_eagerly()
    config = Config()
    logging.info('RUNNING TRAINING')
    # load data
    trainSet, testSet = load_and_preprocessing()
    train2(trainSet, testSet)
    #model = loadModel('best_model.hdf5')
    #pred(model, testSet)
    print("fin")
