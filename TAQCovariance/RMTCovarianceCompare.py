from TAQCovariance import *

class RMTCovarianceCompare(object):
    """
    This class compares the covariance matrix estimator given a
    structure of covariance as well as compare them with different
    covariance matrices and different order of p
    """

    def __init__(self):

        # The sizes to have random matrix effect, and not
        self._p = [int(2**i) + 1 for i in range(8)]
        self._N = 100
        self._T = [int(self._p[i] * self._N) for i in range(8)]

        self._prob = ['Id', 'Toeplitz', 'Dense']
        # Generate three types of matrices as covariance matrices
        self._cov = []
        self._cov.append(np.identity(self._N))

        # Toeplitz, tri-diagonal, with parameter 0.5
        _cov = np.identity(self._N)
        for i in range(self._N-1):
            _cov[i, i+1] = .5
        self._cov.append(_cov)

        # Dense, we suppose the matrix is defined as (1-rho)*identity(N) + rho*ones(N,N)
        # To make the matrix p. d., we need to have the rho within (-1/(N-1), 1).
        # We choose rho to be .02 = 2 * self._N
        self._cov.append((1-.02)*np.identity(self._N) + .02*np.ones((self._N, self._N)))

        # Create a HUGE, one-time random matrix for calculation further
        self._rand_mat = np.random.randn(max(self._T), self._N)

        # Create a matrix to store the results
        print('Calculating the results')
        self._res = []
        for i in range(len(self._cov)):
            # First loop among all of the covariance
            self._res.append([])

            for j in range(len(self._p)):
                self._res[-1].append([])
                # Store the results
                self._res[-1][-1].append(self.score(EmpiricalCovariance(self._rand_mat[:self._T[j], :], False).cov(), i))
                self._res[-1][-1].append(self.score(ClippedCovariance(self._rand_mat[:self._T[j], :], False).cov(), i))
                self._res[-1][-1].append(self.score(ShrinkageCovariance(self._rand_mat[:self._T[j], :], False).cov(), i))
                self._res[-1][-1].append(self.score(BiasedCorrectionCovariance(self._rand_mat[:self._T[j], :], False).cov(), i))

    def score(self, cov, num):
        # We are using the same norm here
        return np.linalg.norm(cov - self._cov[num])

    def plotResult(self):
        import matplotlib.pyplot as plt
        self._methods = ['Empirical', 'Clipped', 'Shrinkage', 'Biased Correct']
        for i in range(len(self._res)):
            plt.figure()
            _res = np.array(self._res[i]).T
            for j in range(_res.shape[0]):
                plt.plot(self._T, _res[j, :], label=r'$Method: {}$'.format(self._methods[j]))
            plt.legend()
            plt.title('Comparisons of the covariance estimators, with {} cov structure'.format(self._prob[i]))
            plt.savefig('./ReturnData/{}.png'.format(self._prob[i]))

    @property
    def res(self):
        return self._res

if __name__ == '__main__':
    rmt = RMTCovarianceCompare()
    rmt.plotResult()
