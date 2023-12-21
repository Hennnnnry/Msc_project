#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing


# In[2]:


#get Data
machine_target = ['vendor name','Model Name','MYCT','MMIN','MMAX','CACH','CHMIN','CHMAX','PRP','ERP']
dataset=pd.read_csv('machine.data',names=machine_target, index_col=False)
dataset.head()


# In[3]:


dataset.describe()


# In[4]:


#X = all dataset with features, y = the targets  
x_target = ['MYCT','MMIN','MMAX','CACH','CHMIN','CHMAX','ERP']
X = pd.DataFrame(dataset,columns = x_target)
X = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(X),columns = x_target)
y = pd.DataFrame(dataset,columns = ['PRP'])
y = pd.DataFrame(np.array(preprocessing.MinMaxScaler().fit_transform(y)).flatten(),columns = ['PRP']).PRP


# In[5]:


#divide data into training dataset and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#print(len(X_train), len(X_test), len(y_train), len(y_test))


# In[13]:


#build linear regression model and train it 
model = LinearRegression()
model = model.fit(X_train,y_train)


# In[7]:


#predict result
y_test_predict = model.predict(X_test)


# In[8]:


#predict result 
plt.plot(X.index,model.predict(X),c="blue")
plt.title('Predict Result',size=20,loc = 'center')
plt.show()
#real result 
plt.plot(X.index,y,c="blue")
plt.title('Real Result',size=20,loc = 'center')
plt.show()


# In[9]:


print("train score:",model.score(X_train,y_train))
print("test score:",model.score(X_test,y_test))


# In[10]:


#k-ford cross validation,calculate the mean squared error
kf = KFold(n_splits=5,shuffle=True,random_state=10)
results = []
for (train_index,valid_index) in kf.split(X,y):
    model = LinearRegression()
    model = model.fit(X.iloc[train_index], y.iloc[train_index])
    y_predict = model.predict(X.iloc[valid_index])
    y_real = y.iloc[valid_index]    
    results.append(mean_squared_error(y_real,y_predict))
print("mean squared error is",str(np.mean(results)))


# In[ ]:





# In[ ]:




