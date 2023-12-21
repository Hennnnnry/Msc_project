#!/usr/bin/env python
# coding: utf-8

# In[44]:


#import package
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[45]:


#Process the dataset, get the head and info, remove the id value, pay attention to the file path here
iris = pd.read_csv('Iris.csv')
iris.head()
iris.drop('Id',axis=1,inplace=True)


# In[46]:


#Observe the original data set--the relationship between the three irises and the length of the calyx and the width of the calyx
fig = iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='orange', label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue', label='versicolor',ax=fig)
iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=fig)
fig.set_xlabel("Sepal Length")
fig.set_ylabel("Sepal Width")
fig.set_title("Sepal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()


# In[47]:


#Observe the original data set--the relationship between three irises and petal length and petal width
fig = iris[iris.Species=='Iris-setosa'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='orange', label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='blue', label='versicolor',ax=fig)
iris[iris.Species=='Iris-virginica'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='green', label='virginica', ax=fig)
fig.set_xlabel("Petal Length")
fig.set_ylabel("Petal Width")
fig.set_title(" Petal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()


# In[48]:


#Observe the distribution of calyx length, calyx width, petal length, and petal width in all iris datasets
iris.hist(edgecolor='black', linewidth=1.2)
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.show()


# In[49]:


#Observe the graph of sepal length, sepal width, petal length, petal width in all iris datasets
plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species',y='PetalLengthCm',data=iris)
plt.subplot(2,2,2)
sns.violinplot(x='Species',y='PetalWidthCm',data=iris)
plt.subplot(2,2,3)
sns.violinplot(x='Species',y='SepalLengthCm',data=iris)
plt.subplot(2,2,4)
sns.violinplot(x='Species',y='SepalWidthCm',data=iris)


# In[50]:


#import package
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier


# In[51]:


#About the correlation between features
plt.figure(figsize=(7,4))
sns.heatmap(iris.corr(),annot=True,cmap='cubehelix_r')
plt.show()


# In[52]:


#Divided into validation set and test set, the training set has 105 rows and 5 columns of data, and the test set has 45 rows and 5 columns of data
train, test = train_test_split(iris, test_size = 0.3)
print(train.shape)
print(test.shape)


# In[53]:


#Do data processing, the feature value is ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'], label is Species
train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
train_y=train.Species
test_X= test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
test_y =test.Species


# In[54]:


#Decision tree algorithm training classification model
model=DecisionTreeClassifier()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction,test_y))


# In[55]:


#implement K cross validation
from sklearn.model_selection import cross_val_score
#Parameter transfer: current classifier, training set features, training set lable, cv is the number of classifications, accuracy
scores = cross_val_score(model,train_X,train_y,cv=5,scoring='accuracy')
x1=[1,2,3,4,5]
y1=[]
for item in scores:
    y1.append(item)
    print(item*100)
# Calculate the mean and use the mean to evaluate the results of K cross-validation
print('DecisionTreeClassifier-K cross-validation final average score：' + str(scores.mean()))


# In[56]:



l1=plt.plot(x1,y1,'r--',label='type1')
plt.plot(x1,y1,'ro-')
plt.title('Results of 5 K-fold cross-validation')
plt.xlabel('times')
plt.ylabel('accuracy')
plt.xticks(range(len(x1)+1))
plt.legend()
plt.show()

