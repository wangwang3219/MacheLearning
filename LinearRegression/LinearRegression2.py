import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

path = "LinearRegression2.csv"
#pandas读入数据
data = pd.read_csv(path)
x = data.iloc[:, :3]
y = data.iloc[:, 3:]

plt.figure(figsize=(9,12))
# figure(num=None, figsize=None, dpi=None, facecolor=None, edgecolor=None, frameon=True)
# num:图像编号或名称，数字为编号 ，字符串为名称
# figsize:指定figure的宽和高，单位为英寸；
# dpi参数指定绘图对象的分辨率，即每英寸多少个像素，缺省值为80  1英寸等于2.5cm,A4纸是21*30cm的纸张 
# facecolor:背景颜色
# edgecolor:边框颜色
# frameon:是否显示边框

plt.subplot(311)     #相当于plt.subplot(3, 1, 1)
# subplot(nrows, ncols, index, **kwargs)
# 分别指定 (行数，列数，位置)
# 其他参数:
# facecolor: 背景色
# polar: 是否用极坐标投影
# projection: 投影

plt.plot(data['TV'],y,'ro')
# r:red  g:green  b:blue c:青色  m:品红  y:黄色  k:黑色  w:白色
# .:点标记  ,:像素标记  o:圆标记  v:三角形下标  ^:三角向上标记  <:三角形左标记  >:三角形右标记  1:三下标记  2:三向上标记  3:三左标记  4:三右标记  s:正方形标记  p:五角大楼标记  *:星标  h:六边形1标记  H:六边形2标记  +:加标记  x:X标记  D:金刚石标记  d:薄金刚石标记  |:Vline标记  _:Hline标记

plt.title('TV')
plt.grid()
# 添加网格

plt.subplot(312)
plt.plot(data['radio'],y,'g^')
plt.title('radio')
plt.grid()
plt.subplot(313)
plt.plot(data['newspaper'],y,'b*')
plt.title('newspaper')
plt.grid()
plt.tight_layout()
# tight_layout会自动调整子图参数，使之填充整个图像区域, 避免重叠

plt.show()
feature_cols = ['TV','radio','newspaper']
X = data[feature_cols]
print(X.head())
print(type(X))
print(X.shape)
y = data['sales']
print(y.head())
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)
# X	待划分的样本特征集合
# y	待划分的样本标签
# test_size	若在0~1之间，为测试集样本数目与原始样本数目之比；若为整数，则是测试集样本的数目。
# random_state	随机数种子
# X_train	划分出的训练集数据（返回值）
# X_test	划分出的测试集数据（返回值）
# y_train	划分出的训练集标签（返回值）
# y_test	划分出的测试集标签（返回值）

linreg = LinearRegression()
model = linreg.fit(X_train,y_train)
print(model)
print(linreg.intercept_)
print(linreg.coef_)
zip(feature_cols,linreg.coef_)
# zip函数有一个参数：从迭代器中依次取一个元组，组成一个元组。
# zip函数有两个参数：zip(a,b)函数分别从a和b中取一个元素组成元组，再次将组成的元组组合成一个新的迭代器。
#                     (1)维数相同：正常组合对应位置的元素。
#                     (2)维数不同：取两者中的最小的行列数

y_pred = linreg.predict(X_test)
print(y_pred)
print(type(y_pred))
print(type(y_pred), type(y_test))
print(len(y_pred), len(y_test))
print(y_pred.shape, y_test.shape)
sum_mean = 0
for i in range(len(y_pred)):
    sum_mean += (y_pred[i] - y_test.values[i]) ** 2

print("均方根误差RMSE by hand:", np.sqrt(sum_mean / len(y_pred)))
plt.figure()
plt.plot(range(len(y_pred)),y_pred,'b',label="predict")#蓝色线表示预测值
plt.plot(range(len(y_pred)),y_test,'r',label="test")#红色线为真实值
plt.legend(loc="upper right")#右上角显示标签
plt.xlabel("the number of sales")
plt.ylabel("value of sales")
plt.show()