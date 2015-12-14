
# coding: utf-8

# In[1]:

#import urllib, requests, zipfile, StringIO
#import numpy, scipy
#import matplotlib.pyplot as plt
#import mlpy
import pandas as pd
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import ExtraTreesRegressor
from time import time
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cross_validation import train_test_split
#from pylab import *
#import seaborn as sns





# In[ ]:




# In[25]:

# import the training data as a "data frame"
train = pd.read_csv("./data/train.csv")
features=train.ix[:,'Feature_1':'Feature_25']
train_=train.dropna()
#test  = pd.read_csv("./data/test.csv")

# heatmap of correlations
feature_corr = train.ix[:,'Feature_1':'Feature_25'].corr()


# In[3]:

features.isnull().sum()
null_data = features[features.isnull().any(axis=1)]
null_data.head()
null_data['nans']=null_data.apply(lambda x: sum(x.isnull().values), axis = 1)
null_data.head()


# In[4]:

#features['Feature_1'].dropna()


# In[26]:

train_.info()


# In[29]:

# split the data into X and Y
# note that Y is more than one variable
# ignore the last two variables, 'Weight_Intraday' and 'Weight_Daily'. We dont know what to do with them yet.
# We use them only in the scoring function.
X = train_.ix[:,'Feature_1':'Ret_120']
Y = train_.ix[:,'Ret_121':'Ret_PlusTwo']
#print X, Y


# In[7]:

# Create an Extremely-Randomized-Trees regressor
extra = ExtraTreesRegressor(n_estimators=100, criterion='mse', max_depth=None, min_samples_split=1, bootstrap=False, n_jobs=-1)
t0=time()
extra = extra.fit(X,Y)
t1=time()
print "time elapsed %.2f seconds" % (t1-t0)
fi=pd.DataFrame(extra.feature_importances_,index=X.columns.values,columns=['importance'])
# fi.head(10)


# In[8]:


#fi=zip(X.columns.values, extra.feature_importances_)
#type(extra.feature_importances_)
#fi.sort_values(by='importance', ascending=False,inplace=True)
#fi.plot(kind='bar')


# In[9]:

'''
this will be the function that replaces missing data by KNN'''
def knn_imputation(df):
    df['nans']=df.apply(lambda x: sum(x.isnull().values), axis = 1)
    df.sort_values(by='nans', ascending=True, inplace=True)
    all_data=df[df['nans']==0]
    #null_data = df[df.isnull().any(axis=1)]
    null_data=df[df['nans']!=0]
    #replace missing values for rows with one variable by knn
    
    


# In[187]:

'''
replacing NaNs by KNN. There are many missing values in different columns and rows. Start with a row, let's say row,
with one missing value in column col; then take the Y to be df[0:row-1,col] and X df[0:row-1, all columns except col].
run KNN for X, Y. predict df[row,col]. fill it in. add to the complete data.'''

# null_data = features[features.isnull().any(axis=1)]
# null_data.head()
# null_data['nans']=null_data.apply(lambda x: sum(x.isnull().values), axis = 1)
df=train.ix[:,'Feature_1':'Feature_25'] #keep only the features in df
#df['nans']=df.apply(lambda x: sum(x.isnull().values), axis = 1) #create a column that counts number of NaNs by row
df['nans']=df.apply(lambda x: len(x)-x.count(), axis = 1) #create a column that counts number of NaNs by row
df.sort_values(by='nans', ascending=True, inplace=True)#sort by nr of NaNs. complete cases will be at the top
df=df.reset_index(drop=True) #reset the index after sorting matrix
clean_data=df[df['nans']==0] #not sure if I'll use this
#null_data = df[df.isnull().any(axis=1)]
#null_data=df[df['nans']!=0]
rows_complete=len(df[df.nans==0])
i=0
print df.iloc[rows_complete+1:rows_complete+6,:]
#replace missing values for rows with one variable by knn, have to do it one by one
for row in range(rows_complete,len(df)): #each row in the set of rows that only have one missing value
        if i<10:
            '''code below only works for one missing value currently; should loop through all missing values'''
            col_nan=np.where(df.iloc[row,:].isnull())[0] #gives me the column with NaN for each row --- there's only one to begin with
            print ("working on row ", row, " NaN in column ",df.columns[col_nan])
            data=df.iloc[:row,:]
            data=data.drop(df.columns[col_nan],1)#pick the data frame with complete cases, dropping the column where row has the missing value
            X_na=data.tail(1)
            X=data.iloc[:-1,:]
            Y=df.iloc[:row-1,col_nan]
            myknn = KNeighborsRegressor(n_neighbors=3,weights='distance') 
            myknn.fit(X, Y) #fit KNN to the complete data
            df.iloc[row,col_nan]=myknn.predict(X_na)[0][0] #predict the missing value
            print df.iloc[rows_complete:rows_complete+6,:] 
        i=i+1


# In[178]:

print data.iloc[:-1,:].shape
print myknn.predict(X_na).ravel()[0]


# In[133]:

x=[[1,2,3],[np.NaN,4,np.NaN]]
df=pd.DataFrame(x,columns=['A','B','C'])
print df
n=np.where(df.iloc[1,:].isnull())
print (n)
print (n[0])


# In[21]:

print len(df[df.nans==5])
print df.nans[39000]
# n=df.columns[df.isnull().any()].tolist()
# v=df.apply(lambda x: )

#df.iloc[:,np.where(df.iloc[1,:].isnull())[0]]
#print df.columns[df.loc[1,:].isnull()]


# In[61]:

df.columns[df.loc[39093,:].isnull().any()]


# In[ ]:

sum(df.loc[1,].isnull())
df.loc[1294,]


# In[ ]:

len(df.loc[1,])-df.loc[1,].count()


# In[130]:

X=df.loc[:row-1,:]
X=X.drop('Feature_1',axis=1)
X.head()


# In[151]:

print col_nan
print np.where(df.iloc[row,:].isnull())
print row
print df.loc[row,:]
print df.iloc[1293,0]
print myknn.predict(X_na)


# In[ ]:




# In[ ]:



