import os
from re import M
import ssl

from sklearn.neighbors.tests.test_neighbors import P
os.environ['OMP_NUM_THREADS'] = "4"


from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.metrics import calinski_harabasz_score
sse_list = []
# 准备数据
# 参数一 样本数量 参数二 特征数量 参数三 簇中心的位置 参数四 随机数种子
x,y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1],[0,0],[1,1],[2,2]],cluster_std=[0.4,0.2,0.2,0.2], random_state=42)
for clo in range(1,20):
    model = KMeans(n_clusters=clo, random_state=0, max_iter=100)
    model.fit(x)
    sse_list.append(model.inertia_)

plt.figure(figsize=(18,8),dpi=100)
plt.xticks(range(0,20),labels=range(0,20))    
plt.grid()
plt.title("sse")
plt.plot(range(1,20),sse_list,'or-')
plt.show()