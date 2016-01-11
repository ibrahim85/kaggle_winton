def simulateData(nObs, nFeatures, method='ols'):
    """ generate random data from multivariate normal distribution """
    import numpy as np
    from sklearn.datasets import make_blobs

    if method == 'ols':
        # randomly generate correlation matrix for the distribution. All correlations must
        # be between -1 and 1. 'random' uses the uniform distribution.
        C = np.random.random((nFeatures, nFeatures)) * 2. - 1.
        for i in range(nFeatures):
            C[i, i] = 1.
        # ensure symmetry
        C = (C + C.T) / 2.
        # randomly generate volatilities. All volatilities must be positive.
        v = np.random.rand(nFeatures, 1) * 10.
        # covariance matrix. Convolve volatilites and correlations. (v*v').*(C)
        V = np.dot(v, v.T)
        # alternative way:
        V = np.outer(v, v)
        cov = V * C
        # randomly generate nFeatures
        mu = np.random.rand(nFeatures,) * 2. - 1.
        # randomly generate data
        simulated_data = np.random.multivariate_normal(mu, cov, nObs)
    elif method == 'blobs':
        simulated_data, y = make_blobs(n_samples=nObs, n_features=nFeatures,
                                       centers=max(3, nFeatures / 10.),
                                       random_state=0)
    else:
        raise Exception('method must be ols or blobs')
    return simulated_data


def randomlyNanOutSubsetOfData(data, frac):
    """
    Makes a copy of a dataset with a random fraction frac converted to NaNs
    :param data: dataset
    :param frac: Fraction of the data you want to be NaN (eg 0.29)
    :return: copy of the data (ndarray) with a random fraction NaN'd out
    """
    import random
    import math
    import numpy as np

    # make copy of the data
    datan = np.copy(data)
    (nObs, nFeatures) = datan.shape

    # create a random index for the subsample that will be created to NaNs
    random_index = sorted(random.sample(range(datan.size), int(math.floor(datan.size * frac))))
    datan.shape = (nObs * nFeatures,)
    datan[random_index] = np.nan
    datan.shape = (nObs, nFeatures)

    return datan
