"""
Preprocessing functions
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from Balancer import balancer_Data


DATASET_PATH = '../DataSet/face_age'
IMG_HEIGHT = 200
IMG_WIDTH = 200
BATCH_SIZE = 1


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
    return (img / 127.5) - 1


def disnormalize(img):
    return (img + 1) / 2


def random_jitter(img):
    """
    include a random jitter to make data augmentation
    flip randomly the img
    Required: input img size 200x200
    :return:
    """
    img = resize(img, 220, 220)
    img = tf.image.random_crop(img, [IMG_WIDTH, IMG_HEIGHT, 3])

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


def get_label_from_Balanced_data(folder_name, img_name):
    print(len(files))
    return files[folder_name][img_name]

def load_image(filename, augment=True):
    """
    load image from a file, and preprocess the img [resize - ranfom_jitter - normalize]
    :param filename:
    :param augment:
    :return:
    """
    filename = bytes(filename.numpy()).decode("utf-8")
    folder_name, img_name, label = get_label_from_path(filename)

    label_img = get_label_from_Balanced_data(folder_name, img_name)

    img = tf.cast(tf.image.decode_png(tf.io.read_file(filename)), tf.float32)[..., :3]

    img = resize(img, IMG_HEIGHT, IMG_WIDTH)

    if augment:
        img = random_jitter(img)

    img = normalize(img)

    return img, label_img


def load_train_image(filename):
    return load_image(filename, True)


def load_test_image(filename):
    return load_image(filename, False)


if __name__ == "__main__":
    imgs_paths = glob.glob(DATASET_PATH + '/*/*')
    imgs_paths.sort()

    files = balancer_Data()
    tf.executing_eagerly()
    dataSet = tf.data.Dataset.from_tensor_slices(imgs_paths)
    dataSet = dataSet.map(load_train_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataSet = dataSet.batch(BATCH_SIZE)

    for img in dataSet.take(5):
        plt.imshow(disnormalize(img[0,...]))
        plt.show()

    print('done')
