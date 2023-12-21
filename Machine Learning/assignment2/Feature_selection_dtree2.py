# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from data_process3 import deal_data


def select_feature(file_name):
    # 导入数据
    df = deal_data(file_name)
    # 设置y值
    if {'pCR (outcome)', 'RelapseFreeSurvival (outcome)'}.issubset(df.columns):
        X = df.drop(['pCR (outcome)', 'RelapseFreeSurvival (outcome)', 'ID'], axis=1)
    else:
        X = df.drop(['ID'], axis=1)
    y = df['pCR (outcome)']

    # 训练集和测试集划分
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

    # 训练模型
    dtree = DecisionTreeClassifier(criterion="entropy", random_state=0,
                                   max_depth=4,
                                   min_samples_leaf=5)
    dtree.fit(x_train, y_train)
    # 预测
    pred_train = dtree.predict(x_train)
    pred_test = dtree.predict(x_test)

    # 准确率
    train_acc = accuracy_score(y_train, pred_train)
    test_acc = accuracy_score(y_test, pred_test)
    print("训练集准确率: {0:.2f}, 测试集准确率: {1:.2f}".format(train_acc, test_acc))

    # 其他模型评估指标
    # precision, recall, F1, _ = precision_recall_fscore_support(y_test, pred_test, average="binary")
    # print("精准率: {0:.2f}. 召回率: {1:.2f}, F1分数: {2:.2f}".format(precision, recall, F1))

    # 特征重要度
    features = list(x_test.columns)
    print('feature:', features)
    importances = dtree.feature_importances_
    indices = np.argsort(importances)[::-1]
    num_features = len(importances)

    # 将特征重要度以柱状图展示
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(num_features), importances[indices], color="g", align="center")
    plt.xticks(range(num_features), [features[i] for i in indices], rotation='45')
    plt.xlim([-1, num_features])
    plt.show()

    # 输出各个特征的重要度
    columns = add_manual_feature()
    final_features = ['pCR (outcome)', 'RelapseFreeSurvival (outcome)', 'ID']
    for i in indices:
        # print("{0} - {1:.3f}".format(features[i], importances[i]))
        if importances[i] > 0:
            final_features.append(features[i])
    # print(final_features)
    final_features = final_features + columns
    final_features = list(set(final_features))
    return pd.DataFrame(df, columns=final_features)


def add_manual_feature():
    columns = ['ID', 'pCR (outcome)', 'RelapseFreeSurvival (outcome)', 'Age', 'ER', 'PgR', 'HER2', 'TrippleNegative',
               'ChemoGrade', 'Proliferation', 'HistologyType', 'LNStatus', 'TumourStage']
    return columns


if __name__ == '__main__':
    select_feature()
