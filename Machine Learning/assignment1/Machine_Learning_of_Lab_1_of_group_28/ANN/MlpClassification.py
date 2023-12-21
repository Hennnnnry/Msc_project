from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    iris = load_iris()
    X, y = iris.data, iris.target
    kf = KFold(n_splits=5, shuffle=True, random_state=10)
    train_predict_score = []
    test_predict_score = []
    for train_index, test_index in kf.split(X, y):
        each_x_train, each_x_test = X.take(train_index, axis=0), X.take(test_index, axis=0)
        each_y_train, each_y_test = y.take(train_index, axis=0), y.take(test_index, axis=0)
        mlp = MLPClassifier(hidden_layer_sizes=(10, 10), activation='logistic', solver='sgd', learning_rate_init=0.1, max_iter=1000)
        mlp.fit(each_x_train, each_y_train)
        each_y_train_predict = mlp.predict(each_x_train)
        each_y_test_predict = mlp.predict(each_x_test)
        train_predict_score.append(accuracy_score(each_y_train, each_y_train_predict))
        test_predict_score.append(accuracy_score(each_y_test, each_y_test_predict))
    print('train score:',  np.mean(train_predict_score))
    print('test score:', np.mean(test_predict_score))
    plt.style.use('ggplot')
    # 确定横纵坐标范围
    plt.xlim(0, 6)
    plt.ylim(0.8, 1.2)
    plt.xlabel('time')
    plt.ylabel('score')
    # 确定该图的标题
    plt.title('classification picture')
    time = [1, 2, 3, 4, 5]
    plt.scatter(time, test_predict_score, color='r', label='test_score')
    plt.scatter(time, train_predict_score, color='y', label='train_score')
    plt.legend()
    plt.show()