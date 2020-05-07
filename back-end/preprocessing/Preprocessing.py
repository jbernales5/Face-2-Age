"""
Preprocessing functions
References:
https://towardsdatascience.com/building-efficient-data-pipelines-using-tensorflow-8f647f03b4ce
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import sys
from Balancer import balancer_Data
from config import Config as config
from logger import logging


def getNumberOfPeopleRange(data, min, max):
    """
    from the ageGroupBar list
    return the number of people in a range of ages
    :return:
    """
    ageCount = 0
    for i in range(min, max):
        ageCount += data[i]

    return ageCount


def resize(img, height, width):
    """
    resize a img givign new height and width
    :param img:
    :param height:
    :param width:
    :return:
    """
    return tf.image.resize(img, [height, width])


def normalize(img):
    """
    return a img with values from -1 to 1 - Important, matrix must be in float32
    :param img:
    :return:
    """
    #return (img / 127.5) - 1
    return (img / 255)


def disnormalize(img):
    #return (img + 1) / 2
    return img * 255


def random_jitter(img):
    """
    include a random jitter to make data augmentation
    flip randomly the img
    Required: input img size 200x200
    :return:
    """
    img = resize(img, 220, 220)

    img = tf.image.random_crop(img, [config.IMG_WIDTH, config.IMG_HEIGHT, config.CHANELS])

    if tf.random.uniform(()) > 0.5:
        img = tf.image.random_flip_left_right(img)

    if tf.random.uniform(()) > 0.5:
        img = tf.image.random_flip_up_down(img)

    return img


def get_label_from_path(filename):
    """
    return a int  with the correspond number of the folder
    :param filename:
    :return:
    """
    p_level_1 = os.path.split(filename)
    img_name = p_level_1[1]
    p_level_2 = os.path.split(p_level_1[0])
    folder_name = p_level_2[1]
    label = int(folder_name)

    return folder_name, img_name, label


def get_label_from_Balanced_data(files, folder_name, img_name):
    return files[folder_name][img_name]


def load_image(filename, augment=True):
    """
    load image from a file, and preprocess the img [resize - ranfom_jitter - normalize]
    :param filename:
    :param augment:
    :return:
    """
    img = tf.image.decode_png(tf.io.read_file(filename))
    if config.CHANELS == 1:
        img = tf.image.rgb_to_grayscale(img)

    img = tf.cast(img, tf.float32)[..., : config.CHANELS]
    img = resize(img, config.IMG_HEIGHT, config.IMG_WIDTH)

    if augment:
        img = random_jitter(img)

    img = normalize(img)

    return img


def load_train_image(filename, age):
    return load_image(filename, True), one_hot_tag(age)


def load_test_image(filename, age):
    return load_image(filename, False), one_hot_tag(age)


def split_data(img_paths, percentage):
    """
    split  data in train and test given a percentage
    :param img_paths:
    :param percentage:
    :return:
    """
    num = len(img_paths)
    train_n = round(num * percentage)
    random_img = np.copy(img_paths)
    np.random.shuffle(random_img)

    train_imgs = random_img[:train_n]
    test_imgs = random_img[train_n:]

    return train_imgs, test_imgs


def load_label_img(data_paths, files):
    """
    Recive the image paths of the train/test and return the right tag for the img (np.array)
    :param data_paths:
    :return: np.array
    """
    infoFile = [get_label_from_path(filename) for filename in data_paths]
    label_img = [get_label_from_Balanced_data(files, folder_name, img_name) for folder_name, img_name, label in
                 infoFile]
    # replace age by the index in rules list, to later be processed to onehot
    new_labels = [replace_age_by_index(int(label)) for label in label_img]

    return np.asarray(new_labels)


def replace_age_by_index(age):
    """
    return the index where the age is on the list
    :param age:
    :return:
    """
    rules = config.RULES_BALANCER
    count = 0
    for range, tag in rules:
        if tag == age:
            return count
        count = count + 1

    return 0  # if ins't  found 0...


def one_hot_tag(tag):
    """
    process in a map function one_hot of the index given
    :param tag:
    :return:
    """
    return tf.one_hot(tag, config.NUM_CLASSES)


def load_and_preprocessing():
    """
    Call all functions to load data, x and y, preprocess and make a tensorflow dataset
    :return: x and y, train and test tensorflow dataset
    """
    logging.debug('Loading path from files')
    imgs_paths_DATASET = glob.glob(config.DATASET_PATH + '/*/*')
    imgs_paths_DATASET.sort()

    if not imgs_paths_DATASET:  # check if is empty
        logging.error('--- IMAGES NOT FOUND ---')

    logging.debug('Spligin data ' + str(config.SPLIT_PERCENTAGE * 100) + ' %')
    train_img_paths, test_img_paths = split_data(imgs_paths_DATASET, config.SPLIT_PERCENTAGE)

    logging.debug('Balancing tags')
    files = balancer_Data()  # load a balanced set of tags with the correspond img path

    # load tags for each image
    train_img_Y = load_label_img(train_img_paths, files)
    test_img_Y = load_label_img(test_img_paths, files)

    #pd.DataFrame(train_img_Y).to_csv("train_img_Y.csv", header=None, index=None)
    #pd.DataFrame(test_img_Y).to_csv("test_img_Y.csv", header=None, index=None)

    logging.debug('Making tensorflow dataset')
    # train
    train = tf.data.Dataset.from_tensor_slices((train_img_paths, train_img_Y))

    # test
    test = tf.data.Dataset.from_tensor_slices((test_img_paths, test_img_Y))

    logging.debug('Preprocessing and Augmentation data..')
    # preprocessing IMG
    train = train.map(load_train_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train = train.batch(config.BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test = test.map(load_test_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test = test.batch(config.BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train, test


if __name__ == "__main__":

    config.DATASET_PATH = '../DataSet/face_age'

    train, test = load_and_preprocessing()

    for img in train.take(1):
        img = img[0][0, ...]
        plt.imshow(disnormalize(tf.squeeze(img)))
        plt.show()

    print('done')
