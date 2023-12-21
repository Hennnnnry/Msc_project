import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def data_process():
    df = pd.read_csv('machine.data', index_col=False, header=None)
    df.columns = ['vendor', 'Model', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP']
    df = df.drop(['Model'], axis=1)
    cols = ['vendor']

    col_dicts = {
        'vendor': {
            'adviser': 0,
            'amdahl': 1,
            'apollo': 2,
            'basf': 3,
            'bti': 4,
            'burroughs': 5,
            'c.r.d': 6,
            'cdc': 7,
            'cambex': 8,
            'dec': 9,
            'dg': 10,
            'formation': 11,
            'four-phase': 12,
            'gould': 13,
            'hp': 14,
            'harris': 15,
            'honeywell': 16,
            'ibm': 17,
            'ipl': 18,
            'magnuson': 19,
            'microdata': 20,
            'nas': 21,
            'ncr': 22,
            'nixdorf': 23,
            'perkin-elmer': 24,
            'prime': 25,
            'siemens': 26,
            'sperry': 27,
            'sratus': 28,
            'wang': 29
        }
    }

    for col in cols:
        df[col] = df[col].map(col_dicts[col])
    return df


if __name__ == '__main__':
    df = data_process()
    # scaler = StandardScaler()
    # scaler.fit(np.array(df))
    # 1、标准化数据
    x_MinMax = preprocessing.MinMaxScaler()
    y_MinMax = preprocessing.MinMaxScaler()
    # 2、feature 和 label 区分
    train_feature = ['vendor', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'ERP']
    target_feature = ['PRP']
    X = np.array(DataFrame(df, columns=train_feature))
    y = np.array(DataFrame(df, columns=target_feature))
    X = x_MinMax.fit_transform(X)
    y = y_MinMax.fit_transform(y)
    # 2、K折交叉划分数据
    kf = KFold(n_splits=5, shuffle=True, random_state=10)
    # 3、划分训练数据和测试数据
    mse_train_arr = []
    mse_test_arr = []
    test_predict_score = []

    for train_index, test_index in kf.split(X, y):
        each_x_train, each_x_test = X.take(train_index, axis=0), X.take(test_index, axis=0)
        each_y_train, each_y_test = y.take(train_index, axis=0), y.take(test_index, axis=0)

        mlp = MLPRegressor(
            hidden_layer_sizes=(30,), activation='relu', solver='lbfgs', alpha=0.01, max_iter=500)
        mlp.fit(each_x_train, each_y_train.ravel())

        each_y_train_predict = mlp.predict(each_x_train)
        each_y_test_predict = mlp.predict(each_x_test)
        mse_train = mean_squared_error(each_y_train_predict, each_y_train)
        mse_test = mean_squared_error(each_y_test_predict, each_y_test)
        mse_train_arr.append(mse_train)
        mse_test_arr.append(mse_test)

        test_predict_score.append(mlp.score(each_x_test, each_y_test))

    # mse 均方误差
    print("训练数据集的MSE:" + str(np.mean(mse_train_arr)))
    print("测试数据集的MSE:" + str(np.mean(mse_test_arr)))
    # score 准确度
    print('test score:', np.mean(test_predict_score))

    plt.style.use('ggplot')
    # 确定横纵坐标范围
    plt.figure(1)
    plt.xlim(0, 6)
    plt.ylim(0, 0.01)
    plt.xlabel('time')
    plt.ylabel('mes')
    # 确定该图的标题
    plt.title('regressor mse picture')
    time = [1, 2, 3, 4, 5]
    plt.scatter(time, mse_train_arr, color='r', label='train_mse')
    plt.scatter(time, mse_test_arr, color='y', label='test_mse')
    plt.legend()
    plt.plot()
    plt.show()

    plt.figure(2)
    plt.xlim(0, 6)
    plt.ylim(0, 1)
    plt.xlabel('time')
    plt.ylabel('score')
    plt.title('regressor score picture')
    plt.scatter(time, test_predict_score, color='y', label='test_score')
    plt.legend()
    plt.plot()

    plt.show()

