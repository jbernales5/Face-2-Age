"""
Custom Analytics functions
"""

import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    print('done')
