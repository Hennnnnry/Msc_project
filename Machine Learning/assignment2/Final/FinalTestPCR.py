#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：pythonProject
@File    ：FinalTestPCR.py
@Author  ：Yi Li, Wenxuan Zhu, Xiaofei Li, Qitao Ye, Jinshuai Chang
@Description ：predict PCR by random forest model
@Date    ：2022/12/7 8:36 AM
"""
import numpy as np
import statistics
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold


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


def extend_dataset(data):
    """
        Manually complete data with the value of label equal 1 to make sure the classification could be balanced
    Args:
        data: data from data preprocessing
    Returns:
        balanced dataframe
    """
    s_data_1 = data.loc[data['pCR (outcome)'] == 1]
    for i in range(2):
        data = data.append(s_data_1, ignore_index=True)
    return data


def select_features(dataframe):
    """
        training randomforest model with depth 100, then choose features by ranking feature_importances
    Args:
        dataframe: data from balanced data, that is, extend_dataset method
    Returns:
        dataframe with selecting features
    """
    # training RandomForestClassifier for select features
    X = dataframe.drop(['pCR (outcome)', 'RelapseFreeSurvival (outcome)', 'ID'], axis=1)
    y = dataframe['pCR (outcome)']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)
    forest = RandomForestClassifier(n_estimators=100)
    forest.fit(x_train, y_train)

    # calculate importance of features and rank it from top to low
    features = list(x_test.columns)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]

    # finally deal with the features of ID and labels to recover all the features
    final_features = ['ID', 'pCR (outcome)', 'RelapseFreeSurvival (outcome)']
    for i in indices:
        if importances[i] > 0.01:
            final_features.append(features[i])
    return pd.DataFrame(dataframe, columns=final_features)


def train_model(dataframe):
    """
    train random forest model with selected feature data：
        1、K-fold to verify train and test score with selected features
        2、explore the best depth of random forest to find the best param
        3、get the best model
    Args:
        data: get a new dataframe after selecting features
    Returns:
        a good model and selected features
    """
    # choose useful features
    X = dataframe.drop(['pCR (outcome)', 'RelapseFreeSurvival (outcome)', 'ID'], axis=1)
    y = dataframe['pCR (outcome)']
    features = X.columns
    # Normalize data
    x_minmax = preprocessing.MinMaxScaler()
    X_original = np.array(pd.DataFrame(df, columns=X.columns))
    y_original = np.array(pd.DataFrame(df, columns=['pCR (outcome)']))
    X = x_minmax.fit_transform(X_original)
    y = y_original

    kf = KFold(n_splits=5, shuffle=True, random_state=188)
    train_predict_score = []
    test_predict_score = []
    precision_arr = []
    recall_arr = []

    dep_score_train = []
    dep_score_test = []
    # traverse different depths to calculate accuracy, precision, recall
    clf_rbf = RandomForestClassifier(n_estimators=100)
    for dep in range(102, 107):
        for train_index, test_index in kf.split(X, y):
            x_train, x_test = X.take(train_index, axis=0), X.take(test_index, axis=0)
            y_train, y_test = y.take(train_index, axis=0), y.take(test_index, axis=0)
            clf_rbf = RandomForestClassifier(n_estimators=dep)
            clf_rbf.fit(x_train, y_train.ravel())
            y_train_pre_rbf = clf_rbf.predict(x_train)
            y_test_pre_rbf = clf_rbf.predict(x_test)
            train_predict_score.append(accuracy_score(y_train_pre_rbf, y_train))
            test_predict_score.append(accuracy_score(y_test_pre_rbf, y_test))
            precision, recall, F1, _ = precision_recall_fscore_support(y_test, y_test_pre_rbf, average="binary")
            # print("精准率: {0:.2f}. 召回率: {1:.2f}, F1分数: {2:.2f}".format(precision, recall, F1))
            precision_arr.append(precision)
            recall_arr.append(recall)
        dep_score_train.append(np.mean(train_predict_score))
        dep_score_test.append(np.mean(test_predict_score))
        train_predict_score = []
        test_predict_score = []
    print('train score:', max(dep_score_train))
    print('test score:', max(dep_score_test))
    print('precision:', np.mean(precision_arr))
    print('recall:', np.mean(recall_arr))
    return clf_rbf, features


def output_predict_test_file(model, features):
    """
        1、Normalize the train data
        2、concat value of Id column adding 'TRG00'
        3、predict the test file and get the result
        4、output the file with the predicted result
    Args:
        model: the model output by training
        features: by selecting features method
    Returns:
        predict test file to output a predict result file
    """
    x_minmax = preprocessing.MinMaxScaler()
    # 1、deal test data containing data preprocess
    test_df = deal_data('testDatasetExample.xls')
    # 2、predict using selecting features
    test_file_predict = model.predict(x_minmax.fit_transform(test_df.loc[:, features]))
    # 3、output predict result to file
    initial_id_prefix = ['TRG00']
    id_values_list = np.array(test_df['ID'])
    id_values = ['{}_{}'.format(a, b) for b in id_values_list for a in initial_id_prefix]
    first_column_id = pd.DataFrame(columns=['ID'], data=id_values)
    second_column = pd.DataFrame(columns=['PCR'], data=test_file_predict)
    result = pd.concat([first_column_id, second_column], axis=1)
    result.to_csv('FinalTestPCR.csv', sep=',', index=False, header=True)


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
    # 1、data preprocessing
    data = deal_data('trainDataset.xls')
    # 2、balance data
    df = extend_dataset(data)
    # 3、select features by training randomForest
    dataframe = select_features(df)
    # 4、train data and get the model and then predict using randomForest
    model, features = train_model(dataframe)
    # 5、 output result of test file
    output_predict_test_file(model, features)
