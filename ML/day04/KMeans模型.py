import os
os.environ['OMP_NUM_THREADS'] = "4"


from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.metrics import calinski_harabasz_score

# 准备数据
# 参数一 样本数量 参数二 特征数量 参数三 簇中心的位置 参数四 随机数种子
x,y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1],[0,0],[1,1],[2,2]],cluster_std=[0.4,0.2,0.2,0.2], random_state=42)
plt.figure()
plt.scatter(x[: ,0],x[: ,1])
plt.show()

model = KMeans(n_clusters=3, random_state=22)
# 模型训练的时候需要用fit.predict方法 得到簇编号
y_pred = model.fit_predict(x)
plt.scatter(x[:,0],x[:,1],c=y_pred)
plt.show()
# 使用CH评估系数来评估模型的好坏
print(calinski_harabasz_score(x,y_pred))