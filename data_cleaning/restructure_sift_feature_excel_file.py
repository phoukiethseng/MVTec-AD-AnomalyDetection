import pandas as pd
import numpy as np

from utils.constants import CATEGORIES
from utils.constants import ABNORMAL_TYPE

def main():
    # Load sift feature raw data
    file = '../../output/sift_features.csv'
    df = pd.read_csv(file, header=0, index_col=0)
    # print(df.loc[filename(3, 'carpet', True)] # Select specific good class sample of a category
    #       .loc[:, 'feature_0':'feature_127'] # Select only sift feature columns
    #       .iloc[:60] # Select the first 60 keypoints
    #       )

    NUM_KEYPOINTS = 20

    # Good feature (Test set)
    for category in CATEGORIES:
        index = 0
        finished = False
        feature_list = []
        while not finished:
            file = filename(index, category, is_train_dataset=False, abnormal_type='good')
            feature = df.loc[file].loc[:, 'feature_0':'feature_127'].iloc[:NUM_KEYPOINTS].to_numpy()
            index += 1
            if feature.shape[0] >= NUM_KEYPOINTS:  # Skip sample that has num of keypoint lower than NUM_KEYPOINTS
                feature_list.append(feature.flatten())
            try:
                df.loc[filename(index, category, is_train_dataset=False, abnormal_type='good')]
            except KeyError:
                finished = True

        X = np.array(feature_list)
        print(category, X, X.shape)

        save_path = '../output/sift_descriptor_good_(test_set).xlsx'
        excel_writer = pd.ExcelWriter(save_path, mode='a', if_sheet_exists='replace')
        processed_df = pd.DataFrame(X)
        processed_df.to_excel(excel_writer, sheet_name=category, index=False, header=False)
        excel_writer.close()

    # Bad feature
    for category in CATEGORIES:
        feature_list = []
        print(category)
        for abnormal_type in ABNORMAL_TYPE[category]:
            print(abnormal_type)
            index = 0
            finished = False
            while not finished:
                file = filename(index, category, is_train_dataset=False, abnormal_type=abnormal_type)
                feature = df.loc[file].loc[:, 'feature_0':'feature_127'].iloc[:NUM_KEYPOINTS].to_numpy()
                index += 1
                if feature.shape[0] >= NUM_KEYPOINTS:  # Skip sample that has num of keypoint lower than NUM_KEYPOINTS
                    feature_list.append(feature.flatten())
                try:
                    df.loc[filename(index, category, is_train_dataset=False, abnormal_type=abnormal_type)]
                except KeyError:
                    finished = True

        X = np.array(feature_list)
        print(X.shape, X)

        save_path = '../../output/sift_descriptor_bad.xlsx'
        excel_writer = pd.ExcelWriter(save_path, mode='a', if_sheet_exists='replace')
        processed_df = pd.DataFrame(X)
        processed_df.to_excel(excel_writer, sheet_name=category, index=False, header=False)
        excel_writer.close()

    # Good feature
    for category in CATEGORIES:
        index = 0
        finished = False
        feature_list = []
        while not finished:
            file = filename(index, category, is_train_dataset=True)
            feature = df.loc[file].loc[:, 'feature_0':'feature_127'].iloc[:NUM_KEYPOINTS].to_numpy()
            index += 1
            if feature.shape[0] >= NUM_KEYPOINTS: # Skip sample that has num of keypoint lower than NUM_KEYPOINTS
                feature_list.append(feature.flatten())
            try:
                df.loc[filename(index, category, is_train_dataset=True)]
            except KeyError:
                finished = True

        X = np.array(feature_list)
        print(category, X, X.shape)

        save_path = '../../output/sift_descriptor_good.xlsx'
        excel_writer = pd.ExcelWriter(save_path, mode='a', if_sheet_exists='replace')
        processed_df = pd.DataFrame(X)
        processed_df.to_excel(excel_writer, sheet_name=category, index=False, header=False)
        excel_writer.close()

    return None

def filename(id, category, is_train_dataset, abnormal_type=None):
    assert id is not None
    assert category in CATEGORIES
    if is_train_dataset is not True:
        assert abnormal_type is not None
        if abnormal_type != 'good':
            assert abnormal_type in ABNORMAL_TYPE[category]

    prefix = './pic/mvtec_anomaly_detection'

    if is_train_dataset:
        # Train dataset
        return f"{prefix}/{category}\\train\\good\\{id:03}.png"
    else:
        # Test dataset
        return f"{prefix}/{category}\\test\\{abnormal_type}\\{id:03}.png"

if __name__ == '__main__':
    main()