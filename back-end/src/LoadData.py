"""
Functions to load data image
"""

import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

def loadImage(path) :
    """
    Return numpyArray in RGB - shape (200,200,3)
    :param path:
    :return:
    """
    img = cv2.imread(path)
    return img

def fromDirectory(pathFolder):
    """
    List all files from a folder
    :param pathFolder:
    :return: numpy list of numpy Images
    """
    imgList = []
    imgList = glob.glob(pathFolder+'/*')

    npImg = []
    for imgPath in imgList:
        npImg.append(loadImage(imgPath))

    return np.asarray(npImg)

def makeDataSet(path):
    """
    Get all folder from the path and proccess it.
    Return dataSet of numpy Imgs and their labels
    :return: X and Y
    """

    folderList = glob.glob(path + '/*')
    folderList.sort()

    if not len(folderList):
        raise Exception('None folders. Review path DataSet')

    imgX = []
    imgY = []

    for i, folder in enumerate(folderList):
        X = fromDirectory(folder)
        imgX.append(X)
        imgY.append(np.asarray([i for _ in range(1,X.shape[0])]))

    imgX = np.asarray(imgX)
    imgY = np.asarray(imgY)

    print('---> DATASET <---')
    print('Shape X -> ', imgX.shape)
    print('Shape Y -> ', imgY.shape)

    return imgX, imgY

def getAnalytics(data):
    """
    general purpose analytics
    :param data:
    :return:
    """

    # number img per age
    numElements = []
    for d in data:
        numElements.append(d.shape[0])

    plt.bar(range(1, len(numElements) + 1), numElements)
    plt.title('Number img per age')
    plt.show()

    return

def groupByAge(imgX, imgY):
    """
    Group some range of age a cause the fault of certain images
    :param imgX:
    :param imgY:
    :return:
    """
    # TODO groupByAge
    pass


if __name__ == "__main__":
    DATASET_PATH = '../DataSet/face_age'

    imgX, imgY = makeDataSet(DATASET_PATH)
    getAnalytics(imgX)

    print('done')

