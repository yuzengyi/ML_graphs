import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def Kmeans_threshold(rank,k=2):
    #k是聚类数量(默认为2) rank是评分列表
    # 使数据为二维
    rank_reshape = np.array(rank).reshape(-1, 1)
    # 使用KMeans进行聚类，输入k与rank
    #显式设置 n_init 参数 n_init 参数指定了 K-means 算法运行的次数，每次都使用不同的质心种子。
    kmeans = KMeans(n_clusters=k, n_init=10).fit(rank_reshape)
    # 找到阈值（聚类中心的平均值）
    threshold = np.mean(kmeans.cluster_centers_)
    return kmeans, threshold

# 读取表格文件
file_path = 'item.xlsx'
df = pd.read_excel(file_path)

# 提取置信度列的数据
rank_column_name = 'rank'
ranks = df[rank_column_name].tolist()

# 使用 Kmeans_threshold 函数处理置信度数据
kmeans_model, threshold_value = Kmeans_threshold(ranks)

# 打印阈值
print("阈值：", threshold_value)

# 绘制直方图
plt.hist(ranks, bins=30, alpha=0.7, label='Rank values', color='gray')
centers = kmeans_model.cluster_centers_

# 绘制聚类中心
for center in centers:
    plt.axvline(x=center[0], color='red', linestyle='--', label=f'Cluster Center {center[0]:.3f}')

# 显示图形
plt.xlabel("Rank values")
plt.ylabel("Frequency")
plt.legend()
plt.show()
