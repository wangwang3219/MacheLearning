import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]


if __name__ == "__main__":
    path = 'E:/python/data/iris.data'  # 数据文件路径
    # 路径，浮点型数据，逗号分隔，第4列用函数iris_type单独处理
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
    # 将数据的0-3列组成x，第4列得到y
    x, y = np.split(data, (4,), axis=1)
    # 为了可视化，仅使用前两列特征
    # x = x[:,:2]
    # Logistic回归模型
    logreg = LogisticRegression()
    # 根据数据[x,y]，计算回归参数
    logreg.fit(x, np.ravel(y))    # 画图 ravel将多维降为一维
    # 横纵各采样多少个值
    N, M, P, Q = 100, 100, 100, 100
    # 得到第0列范围
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    # 得到第1列范围
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
    # 得到第2列范围
    x3_min, x3_max = x[:, 2].min(), x[:, 2].max()
    # 得到第3列范围
    x4_min, x4_max = x[:, 3].min(), x[:, 3].max()
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    t3 = np.linspace(x3_min, x3_max, P)
    t4 = np.linspace(x4_min, x4_max, Q)
    # 生成网格采样点
    x1, x2, x3, x4 = np.meshgrid(t1, t2, t3, t4)
    # 测试点
    x_test = np.stack((x1.flat, x2.flat, x3.flat, x4.flat), axis=1) # 增加一维，下标为1
    # 预测值
    y_hat = logreg.predict(x_test)
    # 使之与输入形状相同
    y_hat = y_hat.reshape(x1.shape)
    # 训练集上的预测结果
    y_hat = logreg.predict(x)
    y = y.reshape(-1)
    print(y_hat.shape)
    print(y.shape)
    result = y_hat == y
    print(y_hat)
    print(y)
    print(result)
    c = np.count_nonzero(result)
    print(c)
    print('Accuracy: %.2f%%' % (100 * float(c) / float(len(result))))
