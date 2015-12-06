
# coding: utf-8

# In[97]:

#import urllib, requests, zipfile, StringIO
#import numpy, scipy
#import matplotlib.pyplot as plt
#import mlpy
import pandas as pd
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import ExtraTreesRegressor
from time import time
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
#from pylab import *
#import seaborn as sns





# In[ ]:




# In[85]:

# import the training data as a "data frame"
train = pd.read_csv("./data/train.csv")
features=train.ix[:,'Feature_1':'Feature_25']
train_=train.dropna()
#test  = pd.read_csv("./data/test.csv")

# heatmap of correlations
feature_corr = train.ix[:,'Feature_1':'Feature_25'].corr()


# In[110]:

features.isnull().sum()
null_data = features[features.isnull().any(axis=1)]
null_data.head()
null_data['nans']=null_data.apply(lambda x: sum(x.isnull().values), axis = 1)
null_data.head()


# In[99]:

features['feature_1'].dropna()


# In[28]:

train_.info()


# In[25]:

# split the data into X and Y
# note that Y is more than one variable
# ignore the last two variables, 'Weight_Intraday' and 'Weight_Daily'. We dont know what to do with them yet.
# We use them only in the scoring function.
X = train_.ix[:,'Feature_1':'Ret_120']
Y = train_.ix[:,'Ret_121':'Ret_PlusTwo']


# In[83]:

# Create an Extremely-Randomized-Trees regressor
extra = ExtraTreesRegressor(n_estimators=100, criterion='mse', max_depth=None, min_samples_split=1, bootstrap=False, n_jobs=-1)
t0=time()
extra = extra.fit(X,Y)
t1=time()
print "time elapsed %.2f seconds" % (t1-t0)
fi=pd.DataFrame(extra.feature_importances_,index=X.columns.values,columns=['importance'])
# fi.head(10)


# In[80]:

#fi=zip(X.columns.values, extra.feature_importances_)
#type(extra.feature_importances_)
fi.sort_values(by='importance', ascending=False,inplace=True)
#fi.plot(kind='bar')


# In[ ]:

def knn_imputation(df):
    all_data=df.dropna()
    null_data = df[df.isnull().any(axis=1)]
    null_data['nans']=null_data.apply(lambda x: sum(x.isnull().values), axis = 1)
    null_data.sort_values(by='nans', ascending=True, inplace=True)
    


# In[112]:

# null_data = features[features.isnull().any(axis=1)]
# null_data.head()
# null_data['nans']=null_data.apply(lambda x: sum(x.isnull().values), axis = 1)
null_data.sort_values(by='nans', ascending=True, inplace=True)
null_data.head(10)


# In[ ]:

# myknn = KNeighborsClassifier(3).fit(X_train,y_train)
# myknn.predict(X_test)
# knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
#     y_ = knn.fit(X, y).predict(T)

