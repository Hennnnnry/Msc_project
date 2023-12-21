#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the libraries
from sklearn.model_selection import train_test_split
import numpy as np
import statistics
import pandas as pd
import random


def deal_data(model_type='classification'):
    # 显示所有列
    pd.set_option('display.max_columns', None)
    # 显示所有行
    pd.set_option('display.max_rows', None)
    # 设置value的显示长度为100，默认为50
    pd.set_option('max_colwidth', 100)
    import matplotlib.pyplot as plt
    # 显示所有列
    pd.set_option('display.max_columns', None)
    # 显示所有行
    pd.set_option('display.max_rows', None)
    # 设置value的显示长度为100，默认为50
    pd.set_option('max_colwidth', 100)
    # 来自uci
    dataset = pd.read_excel(r"trainDataset.xls")
    dataset = dataset.replace(999, np.nan)

    #判断数据的用途，进行相应的预处理
    if model_type == "regression":
        dataset = get_regression_preprocessed_data(dataset)
    elif model_type == "classification":
        dataset = get_classfied_preprocessed_data(dataset)
    # 删除过多的标签0数据
    indexNames = dataset[dataset['pCR (outcome)'] == 0].index
    indexNames = random.sample(list(indexNames), 200)
    dataset.drop(indexNames, inplace=True)
    return dataset


def get_classfied_preprocessed_data(dataset):
    # 处理pCR (outcome)缺省值--删除法
    dataset.dropna(axis=0,how='any',subset=["pCR (outcome)"],inplace=True)
    dataset["pCR (outcome)"] = dataset["pCR (outcome)"].astype(int)
    # 处理PgR缺省值--众数插补法
    dataset["PgR"] = dataset["PgR"].replace(np.NaN, statistics.mode(dataset["PgR"]))
    dataset["PgR"] = dataset["PgR"].astype(int)
    # 处理HER2缺省值--众数插补法
    dataset["HER2"] = dataset["HER2"].replace(np.NaN, statistics.mode(dataset["HER2"]))
    dataset["HER2"] = dataset["HER2"].astype(int)
    # 处理TrippleNegative缺省值--众数插补法
    dataset["TrippleNegative"] = dataset["TrippleNegative"].replace(np.NaN, statistics.mode(dataset["TrippleNegative"]))
    dataset["TrippleNegative"] = dataset["TrippleNegative"].astype(int)
    # 处理ChemoGrade缺省值--众数插补法
    dataset["ChemoGrade"] = dataset["ChemoGrade"].replace(np.NaN, statistics.mode(dataset["ChemoGrade"]))
    dataset["ChemoGrade"] = dataset["ChemoGrade"].astype(int)
    # 处理Proliferation缺省值--众数插补法
    dataset["Proliferation"] = dataset["Proliferation"].replace(np.NaN, statistics.mode(dataset["Proliferation"]))
    dataset["Proliferation"] = dataset["Proliferation"].astype(int)
    # 处理HistologyType缺省值--众数插补法
    dataset["HistologyType"] = dataset["HistologyType"].replace(np.NaN, statistics.mode(dataset["HistologyType"]))
    dataset["HistologyType"] = dataset["HistologyType"].astype(int)
    # 处理LNStatus缺省值--众数插补法
    dataset["LNStatus"] = dataset["LNStatus"].replace(np.NaN, statistics.mode(dataset["LNStatus"]))
    dataset["LNStatus"] = dataset["LNStatus"].astype(int)

    return dataset
def get_regression_preprocessed_data(dataset):
    # 处理pCR (outcome)缺省值--众数插补法
    dataset["pCR (outcome)"] = dataset["pCR (outcome)"].replace(np.NaN, statistics.mode(dataset["pCR (outcome)"]))
    dataset["pCR (outcome)"] = dataset["pCR (outcome)"].astype(int)
    # 处理PgR缺省值--众数插补法
    dataset["PgR"] = dataset["PgR"].replace(np.NaN, statistics.mode(dataset["PgR"]))
    dataset["PgR"] = dataset["PgR"].astype(int)
    # 处理HER2缺省值--众数插补法
    dataset["HER2"] = dataset["HER2"].replace(np.NaN, statistics.mode(dataset["HER2"]))
    dataset["HER2"] = dataset["HER2"].astype(int)
    # 处理TrippleNegative缺省值--众数插补法
    dataset["TrippleNegative"] = dataset["TrippleNegative"].replace(np.NaN, statistics.mode(dataset["TrippleNegative"]))
    dataset["TrippleNegative"] = dataset["TrippleNegative"].astype(int)
    # 处理ChemoGrade缺省值--众数插补法
    dataset["ChemoGrade"] = dataset["ChemoGrade"].replace(np.NaN, statistics.mode(dataset["ChemoGrade"]))
    dataset["ChemoGrade"] = dataset["ChemoGrade"].astype(int)
    # 处理Proliferation缺省值--众数插补法
    dataset["Proliferation"] = dataset["Proliferation"].replace(np.NaN, statistics.mode(dataset["Proliferation"]))
    dataset["Proliferation"] = dataset["Proliferation"].astype(int)
    # 处理HistologyType缺省值--众数插补法
    dataset["HistologyType"] = dataset["HistologyType"].replace(np.NaN, statistics.mode(dataset["HistologyType"]))
    dataset["HistologyType"] = dataset["HistologyType"].astype(int)
    # 处理LNStatus缺省值--众数插补法
    dataset["LNStatus"] = dataset["LNStatus"].replace(np.NaN, statistics.mode(dataset["LNStatus"]))
    dataset["LNStatus"] = dataset["LNStatus"].astype(int)
    return dataset
