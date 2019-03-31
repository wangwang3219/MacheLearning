import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dataset = np.loadtxt('watermelon_3a .csv', delimiter=",")

# separate the data from the target attributes
x = dataset[:,1:3]
y = dataset[:,3]

# draw scatter diagram to show the raw data
f1 = plt.figure(1)
plt.title('watermelon_3a')
plt.xlabel('density')
plt.ylabel('ratio_sugar')
plt.scatter(x[y == 0,0], x[y == 0,1], marker = '*', color = 'k', s=100, label = 'bad')
plt.scatter(x[y == 1,0], x[y == 1,1], marker = 'o', color = 'g', s=100, label = 'good')
plt.legend(loc = 'upper right')
plt.show()
# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)
# 训练模型
log_model = linear_model.LogisticRegression(solver='liblinear')
log_model.fit(x_train, y_train)
# 预测f(x)
y_pred = log_model.predict(x_test)
# 输出差别
print(sklearn.metrics.confusion_matrix(y_test, y_pred))
print(sklearn.metrics.classification_report(y_test, y_pred))