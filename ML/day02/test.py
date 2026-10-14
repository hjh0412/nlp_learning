"""
演示： 过拟合 和 欠拟合

欠拟合：
   模型在训练集和测试集上效果都不好
   原因： 模型过于简单
过拟合：
  模型在训练集表现很好， 测试集不好
  原因： 模型过于复杂
"""

#导包
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Lasso,Ridge
import matplotlib.pyplot as plt

#欠拟合演示
def dm01_under_fitting():
    #准备数据
    # 用二次函数生成数据
    x = np.random.uniform(-3,3, size=100)
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(size=100)

    x_train = x.reshape(-1,1)
    #print(x_train[:2])
    #数据预处理 (不需要）

    #特征工程(不需要）

    #模型训练
    #创建模型对象
    model = LinearRegression()
    #训练模型
    model.fit(x_train, y)

    #模型评估
    #预测
    y_pred = model.predict(x_train)
    #绘图查看数据

    plt.scatter(x, y)

    plt.plot(x_train,y_pred, color='red')
    plt.show()


# 正好拟合演示
def dm02_just_fitting():
    # 准备数据
    # 用二次函数生成数据
    x = np.random.uniform(-3, 3, size=100)
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(size=100)

    x_train = x.reshape(-1, 1)
    # print(x_train[:2])
    # 数据预处理 (不需要）
    X = np.hstack([x_train, x_train ** 2])


    # 特征工程(不需要）

    # 模型训练
    # 创建模型对象
    model = LinearRegression()
    # 训练模型
    model.fit(X, y)

    # 模型评估
    # 预测
    y_pred = model.predict(X)
    # 绘图查看数据

    plt.scatter(x, y)

    plt.plot(np.sort(x), y_pred[np.argsort(x)], color='red')
    plt.title("Just Fitting")
    plt.show()

#过拟合演示
def dm03_over_fitting():
    # 准备数据
    # 用二次函数生成数据
    x = np.random.uniform(-3, 3, size=100)
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(size=100)

    x_train = x.reshape(-1, 1)
    # print(x_train[:2])
    # 数据预处理 (不需要）



    # 特征工程(不需要）
    X = np.hstack([x_train, x_train ** 2, x_train ** 3, x_train ** 4, x_train ** 5, x_train ** 6, x_train ** 7 ,   x_train ** 8, x_train ** 9, x_train ** 10])

    # 模型训练
    # 创建模型对象
    model = LinearRegression()
    # 训练模型
    model.fit(X, y)

    # 模型评估
    # 预测
    y_pred = model.predict(X)
    # 绘图查看数据

    plt.scatter(x, y)

    plt.plot(np.sort(x), y_pred[np.argsort(x)], color='red')
    plt.title("Over Fitting")
    plt.show()

#梯度下降算法演示
def dm04_sgd():
    # 准备数据
    # 用二次函数生成数据
    x = np.random.uniform(-3, 3, size=100)
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(size=100)

    x_train = x.reshape(-1, 1)
    # print(x_train[:2])
    # 数据预处理 (不需要）



    # 特征工程
    #X = np.hstack([x_train, x_train ** 2, x_train ** 3, x_train ** 4, x_train ** 5, x_train ** 6, x_train ** 7 ,   x_train ** 8, x_train ** 9, x_train ** 10])
    X = np.hstack([x_train, x_train ** 2])
    # 模型训练
    # 创建模型对象
    model = SGDRegressor(loss="squared_error",learning_rate="constant",eta0=0.01,max_iter=1000)
    # 训练模型
    model.fit(X, y)

    # 模型评估
    # 预测
    y_pred = model.predict(X)
    # 绘图查看数据

    plt.scatter(x, y)

    plt.plot(np.sort(x), y_pred[np.argsort(x)], color='red')
    plt.title("SGD Regressor")
    plt.show()

# 演示L1正则化， 避免过拟合
def dm05_l1_lasso():
    # 准备数据
    # 用二次函数生成数据
    x = np.random.uniform(-3, 3, size=100)
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(size=100)

    x_train = x.reshape(-1, 1)
    # print(x_train[:2])
    # 数据预处理 (不需要）



    # 特征工程(不需要）
    X = np.hstack([x_train, x_train ** 2, x_train ** 3, x_train ** 4, x_train ** 5, x_train ** 6, x_train ** 7 ,   x_train ** 8, x_train ** 9, x_train ** 10])

    # 模型训练
    # 创建模型对象
    model = Lasso(alpha=1)
    # 训练模型
    model.fit(X, y)

    # 模型评估
    # 预测
    y_pred = model.predict(X)
    # 绘图查看数据
    print(y_pred)
    # plt.scatter(x, y)

    # plt.plot(np.sort(x), y_pred[np.argsort(x)], color='red')
    # plt.title("L1 Lasso Regression")
    # plt.show()

# 演示L2正则化， 避免过拟合
def dm06_l2_ridge():
    # 准备数据
    # 用二次函数生成数据
    x = np.random.uniform(-3, 3, size=100)
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(size=100)

    x_train = x.reshape(-1, 1)
    # print(x_train[:2])
    # 数据预处理 (不需要）



    # 特征工程(不需要）
    X = np.hstack([x_train, x_train ** 2, x_train ** 3, x_train ** 4, x_train ** 5, x_train ** 6, x_train ** 7 ,   x_train ** 8, x_train ** 9, x_train ** 10])

    # 模型训练
    # 创建模型对象
    model = Ridge(alpha=1)
    # 训练模型
    model.fit(X, y)

    # 模型评估
    # 预测
    y_pred = model.predict(X)
    # 绘图查看数据

    plt.scatter(x, y)

    plt.plot(np.sort(x), y_pred[np.argsort(x)], color='red')
    plt.title("L2 Ridge Regression")
    plt.show()

#运行正好拟合
dm02_just_fitting()

#运行过拟合
#dm03_over_fitting()

# dm04_sgd()

dm05_l1_lasso()

#dm06_l2_ridge()



