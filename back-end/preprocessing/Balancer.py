"""
Balancer functions
"""

import numpy as np
import glob
import os
from Project import Project
from logger import logging


def root_builder():
    """
    build a complete structured dict of files
    :param root:
    :return:
    """
    files = {'ini_value': 0}
    project = Project()
    folder_list = glob.glob(project.DATASET_PATH + '/*')
    folder_list.sort()
    folder_list = [os.path.split(path)[1] for path in folder_list]

    for index, folder in enumerate(folder_list):
        files.update({str(folder): {}})
        img_list = glob.glob(project.DATASET_PATH + '/' + str(folder) + '/*')
        img_list.sort()
        img_list = [os.path.split(img)[1] for img in img_list]
        for img in img_list:
            files[str(folder)].update({str(img): index + 1})

    files.pop('ini_value')

    return files


def replace_tags(files, folder, outpuTag):
    """
    replace the tag of the file, giving a concret folder and the new tag
    :param files:
    :param folder:
    :param outpuTag:
    :return:
    """
    for img in files[folder]:
        files[folder][img] = outpuTag

    return files


def balancer_Data():
    """
    Balance number of elements per age, for not bias the model
    """
    # rules in Project.py

    files = root_builder()
    project = Project()

    for folder in files:
        tag = tag_for_folder(folder, project.RULES_BALANCER)
        files = replace_tags(files, folder, tag)

    return files


def tag_for_folder(folder, rules):
    f = int(folder)
    for rule, tag in rules:
        if rule[0] <= f <= rule[1]:
            return tag

    logging.error("ERROR IN RULES BALANCER: FOLDER " + folder + " IS NOT TAKING INTO ACCOUNT!!!")
    return f  # en caso de fallo, edad de la carpeta a la que corresponde


def groupByAge(imgX, imgY, ini, end):
    """
    Group by range of age from the range ini and end
    :param imgX:
    :param imgY:
    :return:
    """
    rangeLength = (end - ini) + 1  # we want ini and end, both included
    xNew, yNew = [], []
    for i, age in enumerate(imgY):
        if age[0] == ini:
            # age founded
            auxX, auxY = [], []
            for j in range(rangeLength):  # group age
                auxX += list(imgX[i + j])
                auxY += list(imgY[i + j])
                # auxX.append(imgX[i + j])
                # auxY.append(imgY[i + j])

            # concat the rest
            X = np.concatenate((imgX[0][:i], np.asarray(auxX), imgX[0][i:]))
            Y = np.concatenate((imgY[0][:i], np.asarray(auxY), imgY[0][i:]))
            break

        else:  # if it is not the age, go on appending
            xNew.append(imgX[i])
            yNew.append(age)

    return X, Y


if __name__ == "__main__":
    balancer_Data()
    print('done')
