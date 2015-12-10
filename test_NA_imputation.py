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
import random
import math

N_OBS = 100;
N_FEATURES = 3;

# Case 1: Single Multivariate Normal distribution
# randomly generate correlation matrix for the distribution. All correlations must
# be between -1 and 1. 'random' uses the uniform distribution.
C = np.random.random((N_FEATURES,N_FEATURES))*2. - 1.
for i in range(N_FEATURES):
    C[i,i] = 1.
# randomly generate volatilities. All volatilities must be positive.
v = np.random.rand(N_FEATURES,1)*10.
# covariance matrix. Convolve volatilites and correlations. (v*v').*(C)
V = np.dot(v,v.T)
# alternative way:
V = np.outer(v,v)
cov = V * C
# ensure symmetry
cov = (cov + cov.T)/2.
# randomly generate means
mu = np.random.rand(N_FEATURES,)*2. - 1.
# randomly generate data
data = np.random.multivariate_normal(mu,cov,N_OBS)

# choose random fraction of data to convert to NAs
NA_FRACTION = 0.19

# create copy of data. This copy will have data converted to NAs
datan = data
# create a random index for the subsample that will be created to NAs
random_index = sorted(random.sample(range(datan.size),int(math.floor(datan.size*NA_FRACTION))))
# heres a better way to do it! convert to 1D array, convert subset of data
# randomly selected by the index to NA, then return to original shape
datan.shape = (N_OBS*N_FEATURES,)
datan[random_index] = np.nan
datan.shape = (N_OBS,N_FEATURES)
'''
random_ij = [np.array(math.modf(float(i)/100)) for i in random_index] # dude, im smart!
random_i = [int(random_ij[i][0]*100) for i in range(len(random_ij))]
random_j = [int(random_ij[j][1])     for j in range(len(random_ij))]

# set the randomly selected subsample of data to NAs
for k in range(len(random_i)):
    datan[random_i[k],random_j[k]] = np.nan
'''

# apply algorithm to compute NAs
# !!! this is your code !!
# data_imputed = impute_NAs_using_KNN(datan)

# RMSE of difference between imputed and original data
rmse = np.sqrt(((data_imputed[np.isnan(datan)] - data[np.isnan(datan)])**2).sum())

# next: do this many times to get a statistical picture of the method's accuracy
