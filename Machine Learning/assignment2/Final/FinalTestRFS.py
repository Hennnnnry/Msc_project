#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：pythonProject
@File    ：FinalTestRFS.py
@Author  ：Yi Li, Wenxuan Zhu, Xiaofei Li, Qitao Ye, Jinshuai Chang
@Date    ：2022/12/9 8:03 AM
@Description ：predict RFS by mlp
"""
import numpy as np
import statistics
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import AdaBoostRegressor
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error


def deal_data(file_name):
    """
    data_preprocess method:
        1、delete the row that value of pcr is 999
        2、replace other columns that value is 999 using the most common number in the same column.
        3、use graphics to check whether the data is outlier
    Args:
        file_name: input file name
    Returns:
        dataset after dealing the missing data of features and labels
    """
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('max_colwidth', 120)
    dataset = pd.read_excel(file_name)
    dataset = dataset.replace(999, np.nan)
    dataset_ids = dataset['ID'].copy()
    for i in range(len(dataset_ids)):
        dataset_ids[i] = "".join([s for s in dataset_ids[i] if s.isnumeric()])
    dataset['ID'] = dataset_ids
    dataset = get_preprocessed_data(dataset)
    return dataset


def select_features(dataframe):
    """
        select features by SelectFromModel with AdaBoostRegressor as estimator,
        and add the first ten cancer-related features
    Args:
        dataframe: data from data after dealing with missing and outlier
    Returns:
        dataframe with selecting features and Id, RFS
    """
    # 导入数据
    X = dataframe.drop(['ID', 'pCR (outcome)', 'RelapseFreeSurvival (outcome)'], axis=1)
    y = dataframe['RelapseFreeSurvival (outcome)']
    feature_name = ['ID', 'RelapseFreeSurvival (outcome)']
    # choose the first ten cancer-related features manual
    columns = add_cancer_related_feature()
    estimator = AdaBoostRegressor(random_state=0, n_estimators=100)
    model = SelectFromModel(estimator=estimator)
    model.fit(X, y)
    feature_idx = model.get_support()
    s_features = X.columns[feature_idx]
    for t in s_features:
        feature_name.append(t)
    feature_name = feature_name + columns
    feature_name = list(set(feature_name))

    return pd.DataFrame(dataframe, columns=feature_name)


def train_model(dataframe):
    """
    train MLPRegressor model with normalized data, then output the model and selected features
    Args:
        dataframe: data with selected features
    Returns:
        model, names of selected features
    """
    X = dataframe.drop(['ID', 'RelapseFreeSurvival (outcome)'], axis=1)
    y = dataframe['RelapseFreeSurvival (outcome)']
    features = X.columns
    x_MinMax = preprocessing.MinMaxScaler()
    X_original = np.array(pd.DataFrame(dataframe, columns=X.columns))
    X = x_MinMax.fit_transform(X_original)
    # 2、K折交叉划分数据
    kf = KFold(n_splits=5, shuffle=True, random_state=10)
    # 3、划分训练数据和测试数据
    mae_train_arr = []
    mae_test_arr = []

    mlp = MLPRegressor(
        hidden_layer_sizes=(20,), activation='relu', solver='sgd', alpha=0.01, max_iter=10000)

    for train_index, test_index in kf.split(X, y):
        each_x_train, each_x_test = X.take(train_index, axis=0), X.take(test_index, axis=0)
        each_y_train, each_y_test = y.take(train_index, axis=0), y.take(test_index, axis=0)

        mlp.fit(each_x_train, np.array(each_y_train))
        each_y_train_predict = mlp.predict(each_x_train)
        each_y_test_predict = mlp.predict(each_x_test)
        mae_train = mean_absolute_error(each_y_train_predict, each_y_train)
        mae_test = mean_absolute_error(each_y_test_predict, each_y_test)
        mae_train_arr.append(mae_train)
        mae_test_arr.append(mae_test)

    # mae
    #print("MAE of train dataset:" + str(np.mean(mae_train_arr)))
    #print("MAE of test dataset:" + str(np.mean(mae_test_arr)))

    return mlp, features, np.mean(mae_train_arr), np.mean(mae_test_arr)


def record_predict_test_file_result(model, features):
    """
            1、Normalize the train data
            3、predict the test file and get the result
        Args:
            model: the model output by training
            features: by selecting features method
        Returns:
            predict test file value
        """
    x_minmax = preprocessing.MinMaxScaler()
    # 1、deal test data containing data preprocess
    test_df = deal_data('testDatasetExample.xls')
    # 2、predict using selecting features
    test_file_predict = model.predict(x_minmax.fit_transform(test_df.loc[:, features]))
    return test_file_predict


def output_predict_test_file(predict_values):
    """
        1、Normalize the train data
        2、concat value of Id column adding 'TRG00'
        3、predict the test file and get the result
        4、output the file with the predicted result
    Args:
        predict_values: predict values list from models training several times
    Returns:
        predict test file to output predict result file
    """
    # x_minmax = preprocessing.MinMaxScaler()
    # # 1、deal test data containing data preprocess
    # test_df = deal_data('testDatasetExample.xls')
    # # 2、predict using selecting features
    # test_file_predict = model.predict(x_minmax.fit_transform(test_df.loc[:, features]))
    # print(test_file_predict)
    # 3、output predict result to file
    initial_id_prefix = ['TRG00']
    id_values_list = predict_values
    id_values = ['{}_{}'.format(a, b) for b in id_values_list for a in initial_id_prefix]
    first_column_id = pd.DataFrame(columns=['ID'], data=id_values)
    second_column = pd.DataFrame(columns=['PCR'], data=predict_values)
    result = pd.concat([first_column_id, second_column], axis=1)
    result.to_csv('FinalTestRFS.csv', sep=',', index=False, header=True)


def add_cancer_related_feature():
    """
        select ten cancer-related features
    Returns:
        list containing names of ten cancer-related features
    """
    columns = ['Age', 'ER', 'PgR', 'HER2', 'TrippleNegative',
               'ChemoGrade', 'Proliferation', 'HistologyType', 'LNStatus', 'TumourStage']
    return columns


def get_preprocessed_data(dataset):
    """
        deal with missing data
    Args:
        dataset: dataframe read from the input file(train_file or test_file)
    Returns:
        dataset without missing data
    """
    # delete data which the value of 'pCR (outcome)' column is 999
    if {'pCR (outcome)'}.issubset(dataset.columns):
        dataset.dropna(axis=0, how='any', subset=["pCR (outcome)"], inplace=True)
        dataset["pCR (outcome)"] = dataset["pCR (outcome)"].astype(int)

    # deal with missing data(value 999)--return the common data
    dataset["PgR"] = dataset["PgR"].replace(np.NaN, statistics.mode(dataset["PgR"]))
    dataset["PgR"] = dataset["PgR"].astype(int)

    dataset["HER2"] = dataset["HER2"].replace(np.NaN, statistics.mode(dataset["HER2"]))
    dataset["HER2"] = dataset["HER2"].astype(int)

    dataset["TrippleNegative"] = dataset["TrippleNegative"].replace(np.NaN, statistics.mode(dataset["TrippleNegative"]))
    dataset["TrippleNegative"] = dataset["TrippleNegative"].astype(int)

    dataset["ChemoGrade"] = dataset["ChemoGrade"].replace(np.NaN, statistics.mode(dataset["ChemoGrade"]))
    dataset["ChemoGrade"] = dataset["ChemoGrade"].astype(int)

    dataset["Proliferation"] = dataset["Proliferation"].replace(np.NaN, statistics.mode(dataset["Proliferation"]))
    dataset["Proliferation"] = dataset["Proliferation"].astype(int)

    dataset["HistologyType"] = dataset["HistologyType"].replace(np.NaN, statistics.mode(dataset["HistologyType"]))
    dataset["HistologyType"] = dataset["HistologyType"].astype(int)

    dataset["LNStatus"] = dataset["LNStatus"].replace(np.NaN, statistics.mode(dataset["LNStatus"]))
    dataset["LNStatus"] = dataset["LNStatus"].astype(int)

    return dataset


def show_box_plot(dataset):
    """
        check whether the data is outlier in specific column
    Args:
        dataset: dataframe from input file
    Returns:
         graphics
    """
    plt.grid(True)
    labels = ['Age']
    plt.boxplot(dataset[labels].values,
                medianprops={'color': 'red', 'linewidth': '1.5'},
                meanline=True,
                showmeans=True,
                meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.5'},
                flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 10},
                labels=labels)
    plt.yticks(np.arange(20, 81, 10))
    plt.show()


if __name__ == '__main__':
    # train several models to record each result
    predict_result_arr = []
    mae_train_mean_arr = []
    mae_test_mean_arr = []

    # 1、data preprocessing
    data = deal_data('trainDataset.xls')

    # 2、select features from SelectFromModel with
    dataframe = select_features(data)

    # 3、run several times models to get predict_values of mae_train mean value
    # and predict_values of mae_test mean value
    # and record average of predict values
    for k in range(5):
        model, features, mae_train, mae_test = train_model(dataframe)
        mae_train_mean_arr.append(mae_train)
        mae_test_mean_arr.append(mae_test)
        predict_result_arr.append(record_predict_test_file_result(model, features))

    # print mae to console
    print("MAE of train dataset:" + str(np.mean(mae_train_mean_arr)))
    print("MAE of test dataset:" + str(np.mean(mae_test_mean_arr)))
    predict_values = np.average(predict_result_arr, axis=0)
    print(predict_values)
    # 4、output predicted result of test file
    output_predict_test_file(predict_values)