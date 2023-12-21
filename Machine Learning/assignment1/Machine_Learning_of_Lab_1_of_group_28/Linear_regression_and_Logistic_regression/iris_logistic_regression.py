#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score


# In[20]:


#get Data
iris_target = ['sepal_length','sepal_width','petal_length','petal_width','class']
dataset=pd.read_csv('iris.data',names=iris_target, index_col=False)
dataset.head()


# In[21]:


#class classfication
print(dataset.iloc[:,-1].unique())


# In[22]:


#target flowers' type，transfer them from strings to numbers
def getFeatureByClass(className):
    try:
        if className == "Iris-setosa":
            return 0
        elif className == "Iris-versicolor":
            return 1
        elif className == "Iris-virginica":
            return 2
    except:
        print("error")
        
dataset['feature'] = dataset['class'].map(lambda x:getFeatureByClass(x))
dataset.drop('class',axis=1,inplace=True)
print(pd.DataFrame(dataset).describe())


# In[23]:


#X = all dataset with features, y = the targets  
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]


# In[24]:


#divide data into training dataset and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(len(X_train), len(X_test), len(y_train), len(y_test))


# In[25]:


#k-ford nested cross validation, I want to find the best hyperparameter C which can acquire model with higher accuracy. 
kf = KFold(n_splits=5,shuffle=True,random_state=10)
C_param = [0.01,0.1,1,10,100]
results = []
for c_param in C_param:
    record_score_lr_kf=[]
    #calculate score with diferent hyperparameter C
    for (train_index,valid_index) in kf.split(X_train,y_train): 
        model = LogisticRegression(C=c_param,max_iter=500)
        model = model.fit(X_train.iloc[train_index], y_train.iloc[train_index])
        model.predict(X.iloc[valid_index])
        record_score_lr_kf.append(model.score(X_train.iloc[valid_index],y_train.iloc[valid_index]))
    results.append(np.mean(record_score_lr_kf))
    print("C_params is",str(c_param),",score is %.4f"%np.mean(record_score_lr_kf))
print("max score is",str(max(results)))


# In[26]:


#build logistic regression model and choose 100 as C
model = LogisticRegression(C=100,max_iter=500)
#train model
model = model.fit(X_train, y_train)


# In[27]:


#predict result
y_test_predict = model.predict(X_test)
print(y_test_predict)


# In[28]:


#predict result  
plt.scatter(X_test.index,y_test_predict,c="blue")
plt.title('Predict Result',size=20,loc = 'center')
plt.show()
#real result 
plt.scatter(X_test.index,y_test,c="red")
plt.title('Real Result',size=20,loc = 'center')
plt.show()


# In[29]:


#model evaluation, calculate the classification of accuracy
print("training datasets' accuracy is ",accuracy_score(y_train , model.predict(X_train)))
print("testing datasets' accuracy is ",accuracy_score(y_test , y_test_predict))


# In[ ]:





# In[ ]:





# In[121]:





# In[ ]:




