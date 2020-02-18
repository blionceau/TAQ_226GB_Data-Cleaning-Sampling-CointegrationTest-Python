class Combination(object):
    """
    This class produces the combinations of pairs of a given number, for example,
    If the number of N = 5, then it should return (start from 0)
    (0, 1), ... (0, 4)
    (1, 2), ...
    (3, 4),
    N*(N-1)/2 pairs
    """

    def __init__(self, N):
        """
        Initialize the class
        :param N: the number of different classes
        """
        self._N = N
        self._pairs = [None] * int(self._N * (self._N - 1) / 2)

        _enum = 0
        for i in range(self._N-1):
            for j in range(i+1, self._N):
                self._pairs[_enum] = (i, j)
                _enum += 1

    def getPairsDict(self):
        _vals = [None] * len(self._pairs)
        return dict(zip(map(lambda x: '{0},{1}'.format(*x), self._pairs), _vals))

    def getSingleDict(self):
        _vals = [None] * self._N
        return dict(enumerate(_vals))

    @property
    def pairs(self):
        return self._pairs