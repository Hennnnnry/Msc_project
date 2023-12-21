#!/usr/bin/env python
# coding: utf-8

# In[70]:


import numpy as np
import pandas as pd
from breast_cancer_classification import classificationResult
from breast_cancer_regression import regressionResult

# export csv
def export_csv():
    testDataSet = read_test_data()
    # get ID
    outputdf1 = pd.DataFrame(columns=['ID'],data=testDataSet['ID'])
    # get classification data form:dataframe
    outputdf2 = classificationResult()
    # get regression data form:dataframe
    outputdf3 = regressionResult()
    result = pd.concat([outputdf1,outputdf2,outputdf3], axis=1)
    print(result)
    outputpath = r"D:\mlassignment\groupwork2\ANNResult.csv"
    result.to_csv(outputpath, sep=',', index=False, header=True)

def read_test_data():
    # get test data
    testDataSet = pd.read_excel(r"D:\mlassignment\groupwork2\testDatasetExample.xls")
    return testDataSet

if __name__ == '__main__':
    export_csv()

