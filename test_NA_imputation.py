#def test_NA_imputation(N_OBS,N_FEATURES):
"""
This is a script that tests our algorithm for imputing missing values (NAs). I
generate data from distributions where the Features
have some relationship to each other (such as multivariate normal), which makes
it worthwhile to do something more intelligent than just using the mean value
for each Feature.

Steps:
1. Create distribution (for multivariate normal, create means & covariances)
2. Generate data from that distribution
3. Convert a randomly selected subset of the data to NAs
4. Apply the algorithm to impute the NAs
5. Calculate how close the algorithm got to the true values
"""
import numpy as np

N_OBS = 100;
N_FEATURES = 3;

# Case 1: Single Multivariate Normal distribution
# randomly generate correlation matrix for the distribution. All correlations must
# be between -1 and 1.
C = np.random.random((N_FEATURES,N_FEATURES))*2. - 1.
for i in range(N_FEATURES):
    C[i,i] = 1.
# randomly generate volatilities. All volatilities must be positive.
v = np.random.rand(N_FEATURES,1)*10.
# covariance matrix. Convolve volatilites and correlations. (v*v').*(C)
V = np.dot(v,v.T)
cov = V * C
# ensure symmetry
cov = (cov + cov.T)/2.
# randomly generate means
mu = np.random.rand(N_FEATURES,)*2. - 1.

# generate data
data = np.random.multivariate_normal(mu,cov,N_OBS)

# choose random fraction of data to convert to NAs
datan = data



