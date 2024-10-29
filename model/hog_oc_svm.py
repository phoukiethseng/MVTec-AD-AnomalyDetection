import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np


# 加载特征
def load_features(file_path, sheet_names):
    features = {}
    for sheet in sheet_names:
        data = pd.read_excel(file_path, sheet_name=sheet)
        if not data.empty:
            features[sheet] = data.values
    return features


def get_feature_from_excel(file_path, feature_size, category):
    """
    Get features from saved Excel file
    :param data_class: class of data ("good" or "bad"). Default is "good".
    :param category: Specify which category to get features from (eg: "bottle", "cable" ...etc.). Default: "bottle"
    :param feature_size: number of feature or dimension of feature descriptor
    :param file_path: path to saved feature descriptor
    """
    select_columns = [col_idx for col_idx in range(feature_size)]
    df = pd.read_excel(file_path, sheet_name=category, header=None, index_col=None, usecols=select_columns)

    X = df.to_numpy()
    return X

# 训练SVDD模型
def train_svdd(normal_features):
    # 数据标准化
    scaler = StandardScaler()
    normal_features = scaler.fit_transform(normal_features)

    # 参数网格
    param_grid = {
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto', 0.1, 1, 10],
        'nu': [0.05, 0.1, 0.2]
    }

    # 创建OneClassSVM模型
    model = OneClassSVM()

    # 网格搜索
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(normal_features)

    print("Best parameters found: ", grid_search.best_params_)
    return grid_search.best_estimator_, scaler


# 预测
def predict(model, test_features, scaler):

    test_features = scaler.transform(test_features)  # 使用训练时的Scaler
    predictions = model.predict(test_features)
    return np.where(predictions == 1, 1, -1)  # 1为正常，-1为异常



# 保存分类报告
def save_classification_report(y_true, y_pred, output_file, category):
    report = classification_report(y_true, y_pred, target_names=['Abnormal', 'Normal'], output_dict=True,zero_division=1)
    df_report = pd.DataFrame(report).transpose()
    excel_writer = pd.ExcelWriter(output_file, mode="a", if_sheet_exists="replace")
    df_report.to_excel(excel_writer,sheet_name=category)
    excel_writer.close()

# 主程序
if __name__ == "__main__":
    # 文件路径
    excel_file = './hog_descriptor_good.xlsx'
    excel_file2 = './hog_descriptor_bad.xlsx'
    excel_file3 = './hog_descriptor_good_(test_set).xlsx'
    output_file = './classification_reports3.xlsx'
    sheet_names = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
                   'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile',
                   'toothbrush', 'transistor', 'wood', 'zipper']  # 你需要的工作表名称

    # 加载特征
    features = load_features(excel_file, sheet_names)

    # 逐个类别训练模型并进行预测
    for category, normal_features in features.items():
        if len(normal_features) > 1:  # 确保有足够的数据进行训练和测试
            # 训练模型
            model, scaler = train_svdd(normal_features)

            # # 为了演示，假设有损坏样本特征，我们这里用正常样本作为测试，实际使用时替换为真实损坏样本特征
            # test_features = normal_features  # 替换为损坏样本特征
            # test_labels = np.array([1] * len(normal_features))  # 假设测试样本都是正常的

            damaged_features = get_feature_from_excel(excel_file2, feature_size=8100, category=category)
            normal_features1 = get_feature_from_excel(excel_file3, feature_size=8100, category=category)
            test_features = np.vstack((normal_features1, damaged_features))
            # test_features = damaged_features
            # test_labels = np.repeat([-1], test_features.shape[0])
            test_labels = np.array([1] * len(normal_features1) + [-1] * len(damaged_features))

            # 进行预测
            predictions = predict(model, test_features, scaler)


            # 打印分类报告
            print(f"Classification report for {category}:")
            print(classification_report(test_labels, predictions))

            # 保存分类报告
            save_classification_report(test_labels, predictions, output_file, category)

            # # 可视化结果（可选）
            # plt.figure(figsize=(8, 6))
            # plt.scatter(normal_features[:, 0], normal_features[:, 1], color='blue', label='Normal', alpha=0.5)
            # plt.scatter(damaged_features[:, 0], damaged_features[:, 1], color='red', label='Damaged', alpha=0.5)
            # plt.title(f'Feature Visualization for {category}')
            # plt.xlabel('Feature 1')
            # plt.ylabel('Feature 2')
            # plt.legend()
            # plt.show()
        else:
            print(f"Not enough data for {category} to perform training.")