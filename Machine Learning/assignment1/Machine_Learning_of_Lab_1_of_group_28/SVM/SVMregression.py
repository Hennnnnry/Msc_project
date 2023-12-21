import numpy as np
from sklearn.svm import SVR
from sklearn import preprocessing
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


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


# 读取数据
df = data_process()
# 标准化
x_MinMax = preprocessing.MinMaxScaler()
y_MinMax = preprocessing.MinMaxScaler()

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
score_arr = []
for train_index, test_index in kf.split(X):
    x_train, x_test = X.take(train_index, axis=0), X.take(test_index, axis=0)
    y_train, y_test = y.take(train_index, axis=0), y.take(test_index, axis=0)

    mlp = SVR(kernel='linear')
    mlp.fit(x_train, y_train.ravel())

    x_train_predict = mlp.predict(x_train)
    y_test_predict = mlp.predict(x_test)
    mse_train = mean_squared_error(x_train_predict, y_train)
    mse_test = mean_squared_error(y_test_predict, y_test)
    mse_train_arr.append(mse_train)
    mse_test_arr.append(mse_test)
    score_arr.append(mlp.score(x_test, y_test))

print(mse_train_arr)
print(mse_test_arr)
print("训练数据集的MSE:" + str(np.mean(mse_train_arr)))
print("测试数据集的MSE:" + str(np.mean(mse_test_arr)))
print("预测的准确率：" + str(np.mean(score_arr)))
# 确定横纵坐标范围
plt.xlim(0, 6)
plt.ylim(0, 0.1)
plt.xlabel('time')
plt.ylabel('mse')
# 确定该图的标题
plt.title('svm regression mse')
time = [1, 2, 3, 4, 5]
plt.scatter(time, mse_test_arr, color='r', label='mse_test')
plt.scatter(time, mse_train_arr, color='y', label='mset_train')
plt.legend()
plt.show()
