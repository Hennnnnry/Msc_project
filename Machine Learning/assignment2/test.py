from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
from collections import Counter
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SMOTE
from data_process3 import deal_data
# from data_process2 import deal_data
import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from Feature_selection_manual import select_feature
from pandas import DataFrame
from sklearn import preprocessing

# bc = deal_data()
bc = select_feature('trainDataset!.xls')
X = bc.drop(['pCR (outcome)', 'RelapseFreeSurvival (outcome)', 'ID'], axis=1)
y = bc['pCR (outcome)']
print(X.shape)

# 归一化处理
# x_MinMax = preprocessing.MinMaxScaler()
# y_MinMax = preprocessing.MinMaxScaler()
# X_original = np.array(DataFrame(bc, columns=X.columns))
# y_original = np.array(DataFrame(bc, columns=['pCR (outcome)']))
# X = x_MinMax.fit_transform(X_original)
# y = y_MinMax.fit_transform(y_original)

# print('Original dataset shape %s' % Counter(y))
# sm = ADASYN()
# new_X, new_y = sm.fit_resample(X, y)
# print('Resampled dataset shape %s' % Counter(new_y))

# MLP

# kf = KFold(n_splits=5, shuffle=True, random_state=10)
# train_predict_score = []
# test_predict_score = []
# for train_index, test_index in kf.split(new_X, new_y):
#     each_x_train, each_x_test = new_X.take(train_index, axis=0), new_X.take(test_index, axis=0)
#     each_y_train, each_y_test = new_y.take(train_index, axis=0), new_y.take(test_index, axis=0)
#     mlp = MLPClassifier(hidden_layer_sizes=(10, 10), activation='logistic', solver='sgd', learning_rate_init=0.1,
#                         max_iter=1000)
#     mlp.fit(each_x_train, each_y_train)
#     each_y_train_predict = mlp.predict(each_x_train)
#     each_y_test_predict = mlp.predict(each_x_test)
#     train_predict_score.append(accuracy_score(each_y_train, each_y_train_predict))
#     test_predict_score.append(accuracy_score(each_y_test, each_y_test_predict))
#
# print('train score:', np.mean(train_predict_score))
# print('test score:', np.mean(test_predict_score))

# SVM

kf = KFold(n_splits=5, shuffle=True, random_state=10)
train_predict_score = []
test_predict_score = []
clf_rbf = SVC()
for train_index, test_index in kf.split(X, y):
    x_train, x_test = X.take(train_index, axis=0), X.take(test_index, axis=0)
    y_train, y_test = y.take(train_index, axis=0), y.take(test_index, axis=0)
    clf_rbf.fit(x_train, y_train)
    y_train_pre_rbf = clf_rbf.predict(x_train)
    y_test_pre_rbf = clf_rbf.predict(x_test)
    train_predict_score.append(accuracy_score(y_train_pre_rbf, y_train))
    test_predict_score.append(accuracy_score(y_test_pre_rbf, y_test))

print('train score:', np.mean(train_predict_score))
print('test score:', np.mean(test_predict_score))

test_df = deal_data('testDatasetExample.xls')
test_file_predict = clf_rbf.predict(test_df.loc[:, X.columns])

# DT

# kf = KFold(n_splits=5, shuffle=True, random_state=10)
# train_predict_score = []
# test_predict_score = []
# for train_index, test_index in kf.split(new_X, new_y):
#     x_train, x_test = new_X.take(train_index, axis=0), new_X.take(test_index, axis=0)
#     y_train, y_test = new_y.take(train_index, axis=0), new_y.take(test_index, axis=0)
#     dt = DecisionTreeClassifier()
#     dt.fit(x_train, y_train)
#     y_train_pre_rbf = clf_rbf.predict(x_train)
#     y_test_pre_rbf = clf_rbf.predict(x_test)
#     train_predict_score.append(accuracy_score(y_train_pre_rbf, y_train))
#     test_predict_score.append(accuracy_score(y_test_pre_rbf, y_test))
#
# print('train score:', np.mean(train_predict_score))
# print('test score:', np.mean(test_predict_score))



