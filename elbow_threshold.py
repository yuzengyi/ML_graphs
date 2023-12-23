import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def elbow_method(rank, max_k=10):
    # 使用肘部法则确定最佳的k值
    sse = []
    rank_reshape = np.array(rank).reshape(-1, 1)
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, n_init=10).fit(rank_reshape)
        sse.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), sse, 'bx-')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.title('Elbow Method For Optimal k')
    plt.show()

    # 返回最佳的k值
    return np.argmin(np.diff(sse[-3:])) + (max_k - 2)


def Kmeans_threshold(rank, k):
    # 使用确定的k值进行KMeans聚类
    rank_reshape = np.array(rank).reshape(-1, 1)
    kmeans = KMeans(n_clusters=k, n_init=10).fit(rank_reshape)
    threshold = np.mean(kmeans.cluster_centers_)
    return kmeans, threshold


# 读取表格文件
file_path = 'item.xlsx'
df = pd.read_excel(file_path)

# 提取置信度列的数据
rank_column_name = 'rank'
ranks = df[rank_column_name].tolist()

# 使用肘部法则确定最佳的k值
optimal_k = elbow_method(ranks)

# 使用 Kmeans_threshold 函数处理置信度数据
kmeans_model, threshold_value = Kmeans_threshold(ranks, optimal_k)

# 打印阈值和最佳的k值
print("最佳的k值：", optimal_k)
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
