from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from data_process3 import deal_data
import pandas as pd


def read_csv(file_name):
    df = deal_data(file_name)
    X = df.drop(['ID', 'pCR (outcome)', 'RelapseFreeSurvival (outcome)'], axis=1)
    y = df['pCR (outcome)']
    return X, y, df


# select features using selectFromModel
def select_feature(file_name):
    # 导入数据
    X, y, df = read_csv(file_name)
    print(X.shape)
    # X = df.drop(['pCR (outcome)', 'RelapseFreeSurvival (outcome)', 'ID'], axis=1)
    # y = df['pCR (outcome)']
    feature_name = ['ID', 'pCR (outcome)', 'RelapseFreeSurvival (outcome)']
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
    model = SelectFromModel(lsvc, prefit=True)
    feature_idx = model.get_support()
    s_features = X.columns[feature_idx]
    for t in s_features:
        feature_name.append(t)
    return pd.DataFrame(df, columns=feature_name)


if __name__ == '__main__':
    select_feature()
