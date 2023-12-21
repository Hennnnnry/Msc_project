#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


iris = load_iris()
X, y = iris.data, iris.target
kf = KFold(n_splits=5, shuffle=True, random_state=10)

# In[12]:


train_predict_score = []
test_predict_score = []
for train_index, test_index in kf.split(X, y):
    x_train, x_test = X.take(train_index, axis=0), X.take(test_index, axis=0)
    y_train, y_test = y.take(train_index, axis=0), y.take(test_index, axis=0)
    clf_rbf = SVC(decision_function_shape='ovo', kernel='linear')
    clf_rbf.fit(x_train, y_train)
    y_train_pre_rbf = clf_rbf.predict(x_train)
    y_test_pre_rbf = clf_rbf.predict(x_test)
    train_predict_score.append(accuracy_score(y_train_pre_rbf, y_train))
    test_predict_score.append(accuracy_score(y_test_pre_rbf, y_test))

print('train score:', np.mean(train_predict_score))
print('test score:', np.mean(test_predict_score))


# In[13]:


plt.xlim(0, 6)
plt.ylim(0.8, 1.2)
plt.xlabel('time')
plt.ylabel('score')
# 确定该图的标题
plt.title('classification picture')
time = [1, 2, 3, 4, 5]
plt.scatter(time, test_predict_score, color='r', label='test_score')
plt.scatter(time, train_predict_score, color='y', label='train_score')
plt.legend()
plt.show()


# In[ ]:




