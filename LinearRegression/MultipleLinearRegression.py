import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 读取文件
datafile = u'LinearRegression.xlsx'  # 文件所在位置，u为防止路径中有中文名称，此处没有，可以省略
data = pd.read_excel(datafile)  # datafile是excel文件，所以用read_excel,如果是csv文件则用read_csv
examDf = DataFrame(data)

# 数据清洗,比如第一列有可能是日期，这样的话我们就只需要从第二列开始的数据，
# 这个情况下，把下面中括号中的0改为1就好，要哪些列取哪些列
new_examDf = examDf.iloc[:, 0:]

# 检验数据
print(new_examDf.describe())  # 数据描述，会显示最值，平均数等信息，可以简单判断数据中是否有异常值
print(new_examDf[new_examDf.isnull() == True].count())  # 检验缺失值，若输出为0，说明该列没有缺失值

# 输出相关系数，判断是否值得做线性回归模型
print(new_examDf.corr())  # 0-0.3弱相关；0.3-0.6中相关；0.6-1强相关；

# 通过seaborn添加一条最佳拟合直线和95%的置信带，直观判断相关关系(aspect:图像长宽比例; kind:reg线性关系, scatter散点图)
sns.pairplot(data, x_vars=['Call', 'Connect'], y_vars='Return', height=7, aspect=0.8, kind='reg')
plt.show()

# 拆分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(new_examDf.iloc[:, :2], new_examDf.Return, test_size=0.2)
# new_examDf.ix[:,:2]取了数据中的前两列为自变量，此处与单变量的不同

print("自变量---源数据:", new_examDf.iloc[:, :2].shape, "；  训练集:", X_train.shape, "；  测试集:", X_test.shape)
print("因变量---源数据:", examDf.Return.shape, "；  训练集:", Y_train.shape, "；  测试集:", Y_test.shape)

# 调用线性规划包
model = LinearRegression()

model.fit(X_train, Y_train)  # 线性回归训练

a = model.intercept_  # 截距
b = model.coef_  # 回归系数
print("拟合参数:截距", a, ",回归系数：", b)

# 显示线性方程，并限制参数的小数位为两位
print("最佳拟合线: Y = ", round(a, 2), "+", round(b[0], 2), "* X1 + ", round(b[1], 2), "* X2")

Y_pred = model.predict(X_test)  # 对测试集数据，用predict函数预测

plt.plot(range(len(Y_pred)), Y_pred, 'red', linewidth=2.5, label="predict data")
plt.plot(range(len(Y_test)), Y_test, 'green', label="test data")
plt.legend(loc=2)
plt.show()  # 显示预测值与测试值曲线
