#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[2]:


# Importing the dataset
dataset = pd.read_csv('computer_hardware_dataset.csv')


# In[3]:


#this function will provide the descriptive statistics of the dataset.(only int value)
dataset.describe()
dataset.head()


# In[4]:


def data_process():
    df = pd.read_csv('computer_hardware_dataset.csv')
    df.columns = ['vendor', 'Model', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP']
    df = df.drop(['Model'], axis=1)
    cols = ['vendor']

    col_dicts = {
        'vendor': {
            'adviser': 0,
            'amdahl': 1,
            'apollo': 2,
            'basf': 3,
            'bti': 4,
            'burroughs': 5,
            'c.r.d': 6,
            'cdc': 7,
            'cambex': 8,
            'dec': 9,
            'dg': 10,
            'formation': 11,
            'four-phase': 12,
            'gould': 13,
            'hp': 14,
            'harris': 15,
            'honeywell': 16,
            'ibm': 17,
            'ipl': 18,
            'magnuson': 19,
            'microdata': 20,
            'nas': 21,
            'ncr': 22,
            'nixdorf': 23,
            'perkin-elmer': 24,
            'prime': 25,
            'siemens': 26,
            'sperry': 27,
            'sratus': 28,
            'wang': 29
        }
    }
    for col in cols:
        df[col] = df[col].map(col_dicts[col])
    return df;
dataset = data_process()


# In[5]:


#Use vendor, MYCT, MMIN, MMAX, CACH, CHMIN, CHMAX, ERP as feature values, and ERP as label
#determine X and y variables(this values are taken as independent variables)
X = dataset.iloc[:,[0,1,2,3,4,5,6,8]].values
y = dataset.iloc[:, [-2]].values


# In[6]:


# Normalize the dataset

def standard_data(X,y):
    X = preprocessing.MinMaxScaler().fit_transform(X)
    y = preprocessing.MinMaxScaler().fit_transform(y)
    return X,y

X,y = standard_data(X,y)


# In[7]:


#Segment the data set, the ratio of training set and test set is 8:2
X_train, X_test, y_train,y_test = train_test_split(X, y , test_size=0.2, random_state = 0, shuffle = True )


# In[8]:


#Decission tree decision tree, training decision tree regression model
X, y = load_diabetes(return_X_y=True)
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train,y_train)


# In[9]:


print('Accuracy score for the regressor:' + str(regressor.score(X_test,y_test)))


# In[10]:


# Score the decision tree regressor using mean squared error

y_pred = regressor.predict(X_test)
print('mean squared error：' + str(mean_squared_error(y_test, y_pred)))


# In[70]:





# 

# 
