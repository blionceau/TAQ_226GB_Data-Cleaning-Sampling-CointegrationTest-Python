import numpy as np


class BlackLitterman(object):
    """
    This class implements the Black-Litterman Model referred to in the lecture notes

    We have the model as
    \pi = \mu + \nu where \nu~N(0, \tau \Sigma)
    q = P\mu+\varepsilon whre \varepsilon~N(0, \Omega)

    stacking them, we have y=X\mu+\varepsilon, here \varepsilon=diag(\tau\Sigma, \Omega)
    X = [I, P]
    """

    def __init__(self, _pi, _q, _P, _gammaSigma, _Omega):
        """
        Initialize the B-L solver
        :param _pi: market equilibrium
        :param _q: investor views of expected return, could be relative
        :param _P: views
        :param _gammaSigma: uncertainty in market equilibrium
        :param _Omega: uncertainty in views
        """
        self._pi = _pi
        self._q = _q
        self._P = _P
        self._gammaSigma = _gammaSigma
        self._Omega = _Omega

        self._N = self._P.shape[1]

        self._y = None
        self._V = None
        self._X = None

        self._ret = None
        self._fit = None

        self.getReturn()

    def getReturn(self):
        self._y = np.append(self._pi, self._q)
        self._V = np.zeros((self._gammaSigma.shape[0] + self._Omega.shape[0],
                      self._gammaSigma.shape[0] + self._Omega.shape[0]))
        self._V[:self._gammaSigma.shape[0],:self._gammaSigma.shape[0]] += self._gammaSigma
        self._V[self._gammaSigma.shape[0]:, self._gammaSigma.shape[0]:] += self._Omega
        self._X = np.vstack([np.identity(self._N), self._P])

        # Below is the statsmodels GLS implementation
        import statsmodels.api as sm
        self._fit = sm.GLS(self._y, self._X, sigma=self._V).fit()
        self._ret = self._fit.params

    @property
    def ret(self):
        return self._ret
    


