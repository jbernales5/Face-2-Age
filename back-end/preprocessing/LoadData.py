"""
Functions to load data image
"""

import cv2
import numpy as np
import glob


def loadImage(path):
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
    imgList = glob.glob(pathFolder + '/*')

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
        imgY.append(np.asarray([i for _ in range(0, X.shape[0])]))

    imgX = np.asarray(imgX)
    imgY = np.asarray(imgY)

    print('---> DATASET <---')
    print('Shape X -> ', imgX.shape)
    print('Shape Y -> ', imgY.shape)

    return imgX, imgY


if __name__ == "__main__":
    DATASET_PATH = '../DataSet/face_age'

    imgX, imgY = makeDataSet(DATASET_PATH)
    # getAnalytics(imgX)

    print('done')
