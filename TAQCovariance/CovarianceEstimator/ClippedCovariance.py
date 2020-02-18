from pyRMT import clipped

class ClippedCovariance(object):
    """
    Define the clipped covariance estimator using RMT adjustment
    """

    def __init__(self, _ret, has_mean):
        self._ret = _ret
        if has_mean:
            self._ret -= self._ret.mean(axis=0)
        self._clip_cov = clipped(self._ret, return_covariance=True)


    def cov(self):
        return self._clip_cov