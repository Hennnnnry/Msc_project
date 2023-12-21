#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the libraries
from sklearn.model_selection import train_test_split
import numpy as np
import statistics
import pandas as pd
import random
from sklearn import preprocessing
import matplotlib.pyplot as plt



def deal_data(file_name=""):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('max_colwidth', 100)

    # read_excel_data
    dataset = pd.read_excel(file_name)
    dataset = dataset.replace(999, np.nan)
    datasetsID = dataset['ID'].copy()
    for i in range(len(datasetsID)):
        datasetsID[i] = "".join([s for s in datasetsID[i] if s.isnumeric()])
    dataset['ID'] = datasetsID

    # deal with missing data
    dataset = get_preprocessed_data(dataset)

    # test outliers,no outliers for age
    # show_box_plot(dataset)

    return dataset

# identify outliers of age, shows no outliers
def show_box_plot(dataset):
    plt.grid(True)
    labels=['Age']
    plt.boxplot(dataset[labels].values,
                medianprops={'color': 'red', 'linewidth': '1.5'},
                meanline=True,
                showmeans=True,
                meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.5'},
                flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 10},
                labels=labels)
    plt.yticks(np.arange(20, 81, 10))
    plt.show()

def get_preprocessed_data(dataset):
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

if __name__ == '__main__':
    print(deal_data())