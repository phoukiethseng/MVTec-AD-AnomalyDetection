import os
import cv2
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# 定义图片文件夹路径
root_dir = './pic/mvtec_anomaly_detection/'

# 创建一个空的DataFrame来存储特征数据
columns = ['filename', 'keypoint_id'] + [f'feature_{i}' for i in range(128)]
df = pd.DataFrame(columns=columns)

# 初始化SIFT
sift = cv2.SIFT_create()


def process_image(image_path):
    try:
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 提取SIFT特征
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)

        rows = []
        if descriptors is not None:
            # 仅取前63个描述子
            descriptors = descriptors[:63, :]
            for idx, descriptor in enumerate(descriptors):
                descriptor_data = {'filename': image_path, 'keypoint_id': idx}
                for i, feature in enumerate(descriptor):
                    descriptor_data[f'feature_{i}'] = feature
                rows.append(descriptor_data)
        return rows
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return []


# 分批次处理图片
batch_size = 100  # 根据系统性能调整批次大小
image_paths = [os.path.join(dirpath, filename)
               for dirpath, _, filenames in os.walk(root_dir)
               for filename in filenames if filename.endswith('.jpg') or filename.endswith('.png')]

for i in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[i:i + batch_size]
    with ThreadPoolExecutor(max_workers=4) as executor:  # 限制并行任务数量
        for result in executor.map(process_image, batch_paths):
            if result:
                df = pd.concat([df, pd.DataFrame(result)], ignore_index=True)

# 保存特征数据到CSV文件
df.to_csv('sift_features.csv', index=False)

print("特征提取完成，数据已保存到 sift_features.csv 文件中。")
