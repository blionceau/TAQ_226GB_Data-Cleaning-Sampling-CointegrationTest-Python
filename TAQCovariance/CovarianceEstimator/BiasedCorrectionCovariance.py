import numpy as np


class BiasedCorrectionCovariance(object):
    """
    calculate the bias correlated covariance estimator
    """

    def __init__(self, ret_mat, has_mean=True):
        """
        :param ret_mat: of shape (nsamples, nfeatures)
        :param has_mean: True if not already demeaned
        """
        self._ret = ret_mat
        if has_mean:
            self._ret -= self._ret.mean(axis=0)

        self._bias_cov = BiasedCorrectionCovariance.calCov(self._ret)

    @staticmethod
    def calCov(R):
        """
        calculate the Covariance using Goldberg et al paper
        https://arxiv.org/pdf/1711.05360.pdf

        See appendix for the description of this algorithm
        :param R: the return matrix, time by assets, T by N
        :return: the bias corrected PCA
        """
        R = np.matrix(R.T)
        N, T = R.shape
        Z = np.matrix(np.ones(N)/ np.sqrt(N)).T
        S = R * R.T / T
        eigvals, eigvecs = np.linalg.eig(S)
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)
        _n = np.argmax(eigvals)
        max_eigval = eigvals[_n]
        max_eigvecs = eigvecs[:, _n]

        sigma_hat_sqr = max_eigval
        beta_hat = float(np.sign(max_eigvecs.T * Z)) * max_eigvecs
        L_hat = sigma_hat_sqr * beta_hat * beta_hat.T
        Delta_hat = np.diag(np.diag(S - L_hat))
        Delta_hat_2_inv = np.diag(1 / np.sqrt(np.diag(Delta_hat)))
        S_tilde = Delta_hat_2_inv * S * Delta_hat_2_inv
        lambda_hat_tilde = sigma_hat_sqr * \
                           np.linalg.norm(Delta_hat_2_inv * beta_hat, 2)**2

        beta_hat_tilde = Delta_hat_2_inv * beta_hat
        beta_hat_tilde /= np.linalg.norm(beta_hat_tilde, 2)
        z_tilde = Delta_hat_2_inv * Z
        z_tilde /= np.linalg.norm(z_tilde, 2)
        delta_hat_tilde_sqr = (np.trace(S_tilde) - lambda_hat_tilde) / (N - 1 - N/T)
        c1_hat = N / (T * lambda_hat_tilde - N * delta_hat_tilde_sqr)
        Psi_hat = 1+delta_hat_tilde_sqr * c1_hat

        gamma = lambda bet, _z: bet.T * _z
        gamma_beta_z = gamma(beta_hat_tilde, z_tilde)
        rho_hat = float((Psi_hat - 1/Psi_hat) * \
                  (Psi_hat * gamma_beta_z) / \
                  (1 - (Psi_hat * gamma_beta_z)**2))
        beta_hat_rho_hat = (beta_hat + rho_hat * Z) / \
                           np.sqrt(1+2*rho_hat*gamma(beta_hat, Z)+rho_hat**2)
        sigma_hat_rho_hat_sqr = float(gamma(beta_hat, Z) ** 2 / \
                                (gamma(beta_hat_rho_hat, Z)) * sigma_hat_sqr)
        L_hat_rho_hat = sigma_hat_rho_hat_sqr * beta_hat_rho_hat * beta_hat_rho_hat.T
        Delta_hat_rho_hat = np.diag(np.diag(S - L_hat_rho_hat))

        return np.array(L_hat_rho_hat + Delta_hat_rho_hat)

    def cov(self):
        return self._bias_cov