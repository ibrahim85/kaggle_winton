from __future__ import division
import numpy as np
import simulateData as sd
import fullyConditionalSpecification as fcs
import pandas as pd

# pandas display options
pd.set_option('precision', 2)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
np.set_printoptions(precision=2,suppress=True)

# simulate data
nObs = 10
nFeatures = 4
data_complete = sd.simulateData(nObs, nFeatures)
(n, f) = data_complete.shape
print 'The dataset has %.0f observations and %.0f features' % (n, f)

# NaN out a subset
data = sd.randomlyNanOutSubsetOfData(data_complete, 0.15)
print 'The fraction of the data that is missing is %.2f' % \
      (sum(np.ravel(np.isnan(data))) / data.size)

# test fullyConditionalSpecification module
data_seeded = fcs.seed(data)
yhat, meanscore = fcs.predictOneColumn(data_seeded, 'ols', 1)
print 'The mean xval score is %.2f' % meanscore