import numpy as np

class MeanVarianceTCost(object):
    """
    This class utilizes an optimizer to calculate the mean-variance problem
    with T-cost

    In fact, the definition of the problem is,
    max_w w' mu - lambda w' Sigma w - gamma TC(dw)
        s.t. w'e +TC(dw)<=1
    Here dw=w-w0 with given w0
    TC(.) is defined as the transaction cost estimated last time. In fact we are using the following function form,
    gamma * TC(dw) = gamma' * |dw|**(3/5). Here lambda' is a vector which is different for each stock
    """

    def __init__(self, _w0, _gamma, _Sigma, _mu, _lambda):
        """
        Initialize the optimizer
        :param _w0: initial holdings, must has sum(w0) == 1, or otherwise we can just change w0 /= sum(w0)
        :param _gamma: the trading costs coefficient
        :param _Sigma: the covariance matrix
        :param _mu: the mean return
        :param _lambda: risk aversion
        """
        if np.isclose(sum(_w0), 0.):
            _w0 = np.array([1 for i in _w0])
        self._w0 = _w0 / sum(_w0)
        self._gamma = _gamma
        self._Sigma = _Sigma
        self._mu = _mu
        self._lambda = _lambda
        self._w = None

        self._TC = lambda ww: sum([i*abs(j)**.6 for i, j in zip(self._gamma, ww-self._w0)])
        # Gamma is already included inside TC(.)

        self._cons = ({'type': 'ineq', 'fun': lambda w: 1-sum(w)-self._TC(w)})
        self._obj = lambda w: - sum(w* self._mu) + self._lambda * np.dot(w, np.dot(self._Sigma, w)) + self._TC(w)

    def optimize(self):
        from scipy.optimize import minimize
        self._w = minimize(self._obj, self._w0, method='BFGS', tol=1e-6).x

    @property
    def w(self):
        return self._w