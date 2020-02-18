import os
import numpy as np

class TAQLiquid(object):
    """
    Return the names of the upper half liquid traded stocks

    """
    def __init__(self, tradepath = "I:\\R\\trades\\"):
        self._path = tradepath
        sizes = {}
        for dates in os.listdir(self._path):
            for filename in os.listdir(self._path + dates + '\\'):
                if sizes.get(filename.split('_')[0]):
                    sizes[filename.split('_')[0]] += os.path.getsize(self._path + dates + '\\' + filename)
                else:
                    sizes[filename.split('_')[0]] = os.path.getsize(self._path + dates + '\\' + filename)
        sizes = [(i, sizes[i]) for i in sizes.keys()]
        self._sizes = sorted(sizes, key=lambda x: x[1]) # ascending order

    def update(self, names):
        names = set(names)
        self._sizes = [i for i in self._sizes if i[0] in names]
        self._sizes = sorted(self._sizes, key=lambda x: x[1])  # ascending order

    def getLargestHalf(self):
        """
        Get the most liquid half
        :return:
        """
        return np.array([self._sizes[i][0] for i in range(int(len(self._sizes)/2), len(self._sizes))])

    def getSmallestHalf(self):
        """
       Get the least liquid half
       :return:
       """
        return np.array([self._sizes[i][0] for i in range(int(len(self._sizes)/2))])
