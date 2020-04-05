"""
Custom Analytics functions
"""

import matplotlib.pyplot as plt

from Preprocessing import getNumberOfPeopleRange


def getImgPerAgeBar(data, showGraphics=True):
    """
    show a bar plot - number of people per age
    :param data:
    :return:
    """
    # number img per age
    numElements = []
    for d in data:
        numElements.append(d.shape[0])

    if showGraphics:
        plt.bar(range(1, len(numElements) + 1), numElements)
        plt.title('Number img per age')
        plt.show()

    return numElements

def peopleGrouped(imgX):

    ageGroup = getImgPerAgeBar(imgX)
    countAux = []
    rangeAge = 5
    for i in range(1, 90, rangeAge):
        peopleInRange = getNumberOfPeopleRange(ageGroup, i, i + rangeAge)
        print("Personas entre ", i, ' - ', i + rangeAge, ':', peopleInRange)
        countAux.append(peopleInRange)

    # plot
    plt.bar(range(1, len(countAux) + 1), countAux)
    plt.title('Graphic Age group by')
    plt.show()

if __name__ == "__main__":
    print('done')
