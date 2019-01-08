import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame,Series
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 读取文件
datafile = u'E:/python/data/dhdhdh.xlsx'  # 文件所在位置，u为防止路径中有中文名称，此处没有，可以省略
data = pd.read_excel(datafile)  # datafile是excel文件，所以用read_excel,如果是csv文件则用read_csv
examDf = DataFrame(data)
examDf = examDf.head(14)
# 绘制散点图,examDf.jt为X轴，examDf.hk为Y轴
plt.scatter(examDf.Connect, examDf.Return, color='darkgreen', label="Exam Data")
plt.legend(loc=1)
# 添加图的标签（x轴，y轴）
plt.xlabel("The Connection amount of the average account")  # 设置X轴标签
plt.ylabel("The ratio of average return amount")  # 设置Y轴标签
plt.show()  # 显示图像
rDf = examDf.corr()#查看数据间的相关系数
print(rDf)
# 拆分训练集和测试集（train_test_split是存在与sklearn中的函数）
X_train, X_test, Y_train, Y_test = train_test_split(examDf.Connect, examDf.Return, test_size=0.2)
# train为训练数据,test为测试数据,examDf为源数据,test_size 规定了训练数据的占比

print("自变量---源数据:", examDf.Connect.shape, "；  训练集:", X_train.shape, "；  测试集:", X_test.shape)
print("因变量---源数据:", examDf.Return.shape, "；  训练集:", Y_train.shape, "；  测试集:", Y_test.shape)

# 散点图
plt.scatter(X_train, Y_train, color="darkgreen", label="train data")  # 训练集为深绿色点
plt.scatter(X_test, Y_test, color="red", label="test data")  # 测试集为红色点

# 添加标签
plt.legend(loc=1)  # 图标位于左上角，即第2象限，类似的，1为右上角，3为左下角，4为右下角
plt.xlabel("The Connection amount of the average account")  # 添加 X 轴名称
plt.ylabel("The ratio of average return amount")  # 添加 Y 轴名称
plt.show()  # 显示散点图
# 调用线性规划包
X_train = X_train.values.reshape(-1,1)
X_test = X_test.values.reshape(-1,1)
model = LinearRegression()

# 线性回归训练
model.fit(X_train, Y_train)  # 调用线性回归包

a = model.intercept_  # 截距
b = model.coef_  # 回归系数

# 训练数据的预测值
y_train_pred = model.predict(X_train)
# 绘制最佳拟合线：标签用的是训练数据的预测值y_train_pred
plt.plot(X_train, y_train_pred, color='blue', linewidth=2, label="best line")

# 测试数据散点图
plt.scatter(X_train, Y_train, color='darkgreen', label="train data")
plt.scatter(X_test, Y_test, color='red', label="test data")

# 添加图标标签
plt.legend(loc=1)  # 图标位于左上角，即第2象限，类似的，1为右上角，3为左下角，4为右下角
plt.xlabel("The Connection amount of the average account")  # 添加 X 轴名称
plt.ylabel("The ratio of average return amount")  # 添加 Y 轴名称
plt.show()  # 显示图像

print("拟合参数:截距", a, ",回归系数：", b)
print("最佳拟合线: Y = ", round(a, 2), "+", round(b[0], 2), "* X")  # 显示线性方程，并限制参数的小数位为两位

