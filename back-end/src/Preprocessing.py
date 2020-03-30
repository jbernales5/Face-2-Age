"""
Preprocessing functions
"""

from LoadData import makeDataSet
from Analytics import getImgPerAgeBar
import matplotlib.pyplot as plt

def groupByAge(imgX, imgY, minRange=1, maxRange=100):
    """
    Group some range of age a cause the fault of certain images
    min and max are the range of people to group by
    :param imgX:
    :param imgY:
    :return:
    """
    ageGroup = getImgPerAgeBar(imgX)
    #minRange = 70
    #maxRange = 80
    countAux = []
    rangeAge = 5
    for i in range(1,90,rangeAge):
        peopleInRange = getNumberOfPeopleRange(ageGroup, i, i+rangeAge)
        print("Personas entre ", i, ' - ', i+rangeAge, ':', peopleInRange)
        countAux.append(peopleInRange)

    # plot
    plt.bar(range(1, len(countAux) + 1), countAux)
    plt.title('Graphic Age group by')
    plt.show()

    return


def balanceData():
    """
    Balance number of elements per age, to not bias the model
    RULES:
        - 80 - ... (840)
        - 70 - 80 (397)

    :return:
    """
    # TODO balanceData
    pass


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


if __name__ == "__main__":
    DATASET_PATH = '../DataSet/face_age'

    imgX, imgY = makeDataSet(DATASET_PATH)
    groupByAge(imgX,imgY)


    print('done')
