#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the libraries
from sklearn.model_selection import train_test_split
import numpy as np
import statistics
import pandas as pd
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)
import matplotlib.pyplot as plt
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)
# 来自uci
dataset = pd.read_excel("D:/trainDataset.xls")
dataset = dataset.replace(999,np.nan)

#处理pCR (outcome)缺省值--众数插补法
dataset["pCR (outcome)"] = dataset["pCR (outcome)"].replace(np.NaN, statistics.mode(dataset["pCR (outcome)"]))
dataset["pCR (outcome)"] = dataset["pCR (outcome)"].astype(int)
#处理PgR缺省值--众数插补法
dataset["PgR"] = dataset["PgR"].replace(np.NaN, statistics.mode(dataset["PgR"]))
dataset["PgR"] = dataset["PgR"].astype(int)
#处理HER2缺省值--众数插补法
dataset["HER2"] = dataset["HER2"].replace(np.NaN, statistics.mode(dataset["HER2"]))
dataset["HER2"] = dataset["HER2"].astype(int)
#处理TrippleNegative缺省值--众数插补法
dataset["TrippleNegative"] = dataset["TrippleNegative"].replace(np.NaN, statistics.mode(dataset["TrippleNegative"]))
dataset["TrippleNegative"] = dataset["TrippleNegative"].astype(int)
#处理ChemoGrade缺省值--众数插补法
dataset["ChemoGrade"] = dataset["ChemoGrade"].replace(np.NaN, statistics.mode(dataset["ChemoGrade"]))
dataset["ChemoGrade"] = dataset["ChemoGrade"].astype(int)
#处理Proliferation缺省值--众数插补法
dataset["Proliferation"] = dataset["Proliferation"].replace(np.NaN, statistics.mode(dataset["Proliferation"]))
dataset["Proliferation"] = dataset["Proliferation"].astype(int)
#处理HistologyType缺省值--众数插补法
dataset["HistologyType"] = dataset["HistologyType"].replace(np.NaN, statistics.mode(dataset["HistologyType"]))
dataset["HistologyType"] = dataset["HistologyType"].astype(int)
#处理LNStatus缺省值--众数插补法
dataset["LNStatus"] = dataset["LNStatus"].replace(np.NaN, statistics.mode(dataset["LNStatus"]))
dataset["LNStatus"] = dataset["LNStatus"].astype(int)
print(dataset)


# In[ ]:




