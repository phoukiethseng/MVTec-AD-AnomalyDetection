import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# 设置字体为 SimHei（黑体），确保中文字符可以显示
rcParams['font.sans-serif'] = ['SimHei']  # 或者使用 ['Microsoft YaHei']
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 从Excel文件中指定的sheet加载特征
def load_features(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name, index_col=0)
    return df.values

def visualize_with_tsne(good_feature_path, bad_feature_path, sheet_name_good, sheet_name_bad):
    # 检查文件是否存在
    if not os.path.exists(good_feature_path):
        print(f"Error: {good_feature_path} does not exist.")
        return
    if not os.path.exists(bad_feature_path):
        print(f"Error: {bad_feature_path} does not exist.")
        return

    # 加载特定 sheet 的特征
    X_good = load_features(good_feature_path, sheet_name_good)
    X_bad = load_features(bad_feature_path, sheet_name_bad)

    # 合并特征并创建标签
    X = np.vstack((X_good, X_bad))
    y = np.array([0] * len(X_good) + [1] * len(X_bad))  # 标签0代表“好”，1代表“坏”

    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # 绘制t-SNE结果
    plt.figure(figsize=(10, 8))
    plt.scatter(X_tsne[y == 0, 0], X_tsne[y == 0, 1], label="Good", alpha=0.7, s=40)
    plt.scatter(X_tsne[y == 1, 0], X_tsne[y == 1, 1], label="Bad", alpha=0.7, s=40)
    plt.title("HOG特征的t-SNE可视化 category")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.show()

# 定义特征文件的路径和指定的sheet名称
good_feature_path = r"good_feature_path"
bad_feature_path = r"bad_feature_path"
sheet_name_good = "category"  # 替换为实际的 "Good" 数据的 sheet 名称
sheet_name_bad = "category"   # 替换为实际的 "Bad" 数据的 sheet 名称

# 执行可视化
visualize_with_tsne(good_feature_path, bad_feature_path, sheet_name_good, sheet_name_bad)

