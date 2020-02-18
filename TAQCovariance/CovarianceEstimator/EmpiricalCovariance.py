import numpy as np
from sklearn.covariance import empirical_covariance

class EmpiricalCovariance(object):
    """
    This class utilizes the sklearn package to calculate the empirical covariance estimator
    """

    def __init__(self, ret_mat, has_mean=True):
        """

        :param ret_mat: of shape (nsamples, nfeatures)
        :param has_mean: True if not already demeaned
        """
        self._ret = ret_mat
        if has_mean:
            self._ret -= self._ret.mean(axis=0)
        self._emp_cov = empirical_covariance(self._ret, True)

    def cov(self):
        return self._emp_cov