from sklearn.model_selection import KFold
import numpy as np
from Feature_selectFromModel2 import select_feature
from sklearn import preprocessing
from data_process3 import deal_data
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR


def process():
    df = select_feature('trainDataset.xls')
    X = df.drop(['pCR (outcome)', 'RelapseFreeSurvival (outcome)', 'ID'], axis=1)
    y = df['RelapseFreeSurvival (outcome)']
    features = X.columns
    # 归一化处理 方案一
    scaler = preprocessing.StandardScaler()
    scaler.fit(np.array(X))
    X = scaler.transform(X)

    kf = KFold(n_splits=5, shuffle=True, random_state=10)
    # 3、划分训练数据和测试数据
    mae_train_arr = []
    mae_test_arr = []

    mlp = SVR(kernel='sigmoid', gamma='auto', coef0=5.0, C=2.0)

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

    # mse 均方误差
    print("训练数据集的MAE:" + str(np.mean(mae_train_arr)))
    print("测试数据集的MAE:" + str(np.mean(mae_test_arr)))

    # 2、预测过程
    test_df = deal_data('testDatasetExample.xls')

    # 方案一的处理方式
    X_test = scaler.transform(test_df.loc[:, features])
    # # 不需要再次做特征选择，只用把特征选出来的列用起来
    test_predict_record = []
    for k in range(10):
        test_predict_record.append(mlp.predict(X_test))
    test_file_predict = np.average(test_predict_record, axis=0)
    print(test_file_predict)
    # 3、预测结果输出到文件（可以做成两个文件）
    return test_file_predict


if __name__ == '__main__':
    process()