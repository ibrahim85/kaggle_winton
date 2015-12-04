#import urllib, requests, zipfile, StringIO
#import numpy, scipy
#import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import ExtraTreesRegressor


# import the training data as a "data frame"
train = pd.read_csv("/home/prinjoh/Kaggle/train.csv")
test  = pd.read_csv("/home/prinjoh/Kaggle/test.csv")

# heatmap of correlations
feature_corr = train.ix[:,'Feature_1':'Feature_25'].corr()
#seaborn.heatmap(feature_corr)

# split the data into X and Y
# note that Y is more than one variable
# ignore the last two variables, 'Weight_Intraday' and 'Weight_Daily'. We dont know what to do with them yet.
# We use them only in the scoring function.
X = train[:'Ret_120']
Y = train['Ret_121':'Ret_PlusTwo']


# Create an Extremely-Randomized-Trees regressor
'''
http://scikit-learn.org/stable/modules/ensemble.html#forest

n_estimators is the number of trees in the forest. The larger the better, but also the longer it will take to compute.
In addition, note that results will stop getting significantly better beyond a critical number of trees.

max_features is the size of the random subsets of features to consider when splitting a node.
The lower the greater the reduction of variance, but also the greater the increase in bias.
Empirical good default values are max_features=n_features for regression problems

Good results are often achieved when setting max_depth=None in combination with min_samples_split=1
(i.e., when fully developing the trees). Bear in mind though that these values are usually not optimal,
and might result in models that consume a lot of ram. The best parameter values should always be cross-validated.
In addition, note that in random forests, bootstrap samples are used by default (bootstrap=True)
while the default strategy for extra-trees is to use the whole dataset (bootstrap=False).

Finally, this module also features the parallel construction of the trees and the parallel computation of
the predictions through the n_jobs parameter. If n_jobs=k then computations are partitioned into k jobs,
and run on k cores of the machine. If n_jobs=-1 then all cores available on the machine are used.
'''
extra = ExtraTreesRegressor(n_estimators=100, criterion='mse', max_depth=None, min_samples_split=1, bootstrap=False, n_jobs=-1)

# Train extra on our data
# use dtype=np.float64 and order='C' for Y, for maximum efficiency
extra = extra.fit(X,Y)

# define success
def wmae(extra,X,Y,W):
    ''' Submissions are evaluated using the Weighted Mean Absolute Error. Each return you predicted is compared with the actual return.
    The formula is then:
    WMAE=(1/n)*∑_i{w_i⋅|y_i−yhat_i|}
    where wi is the weight associated with the return (Weight_Intraday, Weight_Daily for intraday and daily returns),
    yhat_i is the ith predicted return, y_i is the ith actual return, n is the number of predictions.
    '''
    Yhat = extra.predict(X)
    n = len(Y)
    if ((len(Yhat)!=n) or (len(W)!=n)):
        print('y, yhat and w must all be vectors of the same length')
        # ... function not finished yet ...
    return 1

# Measure success:
# Do cross-validation to see how well our method is working
# !! note - i dont know how to pass W to wmae yet !!
# use Winton's scoring function
# use all available CPUs (n_jobs=-1)
W = train['Weight_Intraday':'Weight_Daily']
scores = cross_val_score(extra, X, y=Y, scoring=wmae, cv=100, n_jobs=-1)

# plot histogram of scores
scores.hist(bins=20)

# See how important each feature is to the prediction
fi = extra.feature_importances_

# Now predict the test data, the goal of the project
test_X = test[:'Ret_120']
test_Y = test['Ret_121'::'Ret_PlusTwo']
test_Yhat = extra.predict(test_X)


