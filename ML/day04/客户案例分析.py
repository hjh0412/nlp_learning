import os
os.environ['OMP_NUM_THREADS'] = "1"
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import seaborn as sns
# 模型评估找到最好的k值 使用肘方法得到
def dm01_肘方法(x):
    # SEE是误差平方和
    mysee = []
    mysscore = []
    for i in range(2,11):
        model = KMeans(n_clusters=i)
        model.fit(x)
        mysee.append(model.inertia_)
        y_pred = model.predict(x)
        mysscore.append(silhouette_score(x,y_pred))

    plt.xlabel('number of culsters')
    plt.ylabel('mysee')    
    plt.plot(range(2,11),mysee,'or-')
    plt.show()
    plt.xlabel('number of culsters')
    plt.ylabel('mysscore')    
    plt.plot(range(2,11),mysscore,'ob-')
    plt.show()
# 使用聚类分析分析用户群体
def dm02_聚类分析(x):
    # 模型训练 同时完成预测
    model = KMeans(n_clusters=5)
    model.fit(x)
    y_pred = model.predict(x)
    print(f"预测的结果是：{y_pred}")
    print(f"质心是：{model.cluster_centers_}")
    # 绘图分析
    # hue参数会把相同类型的值分为一类并上色
    # 绘制聚类分析后的图形
    sns.scatterplot(data=x,x=x.columns[0],y=x.columns[1],hue=y_pred)
    plt.xlabel(x.columns[0])
    plt.ylabel(x.columns[1])
    plt.show()
    # 绘制质心的图
    plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c='black', s=200, alpha=0.6)
    plt.xlabel(x.columns[0])
    plt.ylabel(x.columns[1])
    plt.show()
if __name__=='__main__':
    custom_df = pd.read_csv('D:/itheima/ML/机器学习/Day04/05-资料/customers.csv',encoding='gbk')
    # print(custom_df.info)
    x = custom_df.iloc[:,[3,4]]
    
    # 找到了最佳的K值为5
    dm01_肘方法(x)
    dm02_聚类分析(x)