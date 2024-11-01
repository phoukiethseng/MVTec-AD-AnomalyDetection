import os
from pathlib import Path
import cv2 as cv
import numpy as np

from utils.feature import save_feature_to_excel
from utils.constants import CATEGORIES

def main():
    for category in CATEGORIES:
        dataset_path = "../dataset"
        # compute_good_dataset_hog_feature(dataset_path=good_dataset_path, category=category)
        # compute_bad_dataset_hog_feature(dataset_path, category)
        compute_good_dataset_for_testing(dataset_path, category)
    return None


def compute_good_dataset_for_testing(dataset_path, category):
    assert dataset_path is not None, "dataset_path cannot be empty"
    assert category in CATEGORIES, "invalid data category"

    file_list = list(Path(dataset_path).glob(f'{category}/test/good/*.png'))
    hog = get_hog_descriptor()
    num_sample, feature_size = len(file_list), hog.getDescriptorSize()
    X = np.zeros((num_sample, feature_size))

    for index, file in enumerate(file_list):
        img = cv.imread(file)
        img = cv.resize(img, (128,128))
        hist = hog.compute(img)
        X[index, :] = hist

    save_path = '../../output/hog_descriptor_good_(test_set).xlsx'
    save_feature_to_excel(save_path, X, category)

def compute_bad_dataset_hog_feature(dataset_path, category):
    assert dataset_path is not None, "dataset_path cannot be empty"
    assert category in CATEGORIES, "invalid data category"

    exclude_path = Path(os.path.join(dataset_path, f"{category}/test/good"))
    file_list = list(Path(dataset_path).glob(f'{category}/test/**/*.png'))
    hog = get_hog_descriptor()
    num_sample, feature_size = len(file_list), hog.getDescriptorSize()
    X = np.zeros((num_sample, feature_size))

    for index, file in enumerate(file_list):
        if file.is_relative_to(exclude_path):
            continue
        img = cv.imread(file)
        img = cv.resize(img, (128,128))
        hist = hog.compute(img)
        X[index, :] = hist

    save_path = '../../output/hog_descriptor_bad.xlsx'
    save_feature_to_excel(save_path, X, category)

def compute_good_dataset_hog_feature(dataset_path, category):
    assert dataset_path is not None, "dataset_path cannot be empty"
    assert category in CATEGORIES, "invalid data category"

    img_files = list(Path(dataset_path).glob(f"{category}/train/good/*.png"))
    num_img = len(img_files)

    hog = get_hog_descriptor()

    num_sample, num_feature = num_img, hog.getDescriptorSize()
    X = np.zeros((num_sample, num_feature))
    print(f"Computing HOG descriptor for {category} category...")
    for index, file_path in enumerate(img_files):
        img = cv.imread(file_path)
        # All images have to be resized to 128x128
        img = cv.resize(img, (128, 128))
        hist = hog.compute(img)
        X[index, :] = hist
    print("Done Computing HOG descriptor.")
    save_path = f"./output/hog_descriptor_good.xlsx"
    save_feature_to_excel(save_path, X, category=category)

def get_hog_descriptor():
    winSize = (128, 128)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                           histogramNormType, L2HysThreshold, gammaCorrection, nlevels)

    return hog

if __name__ == '__main__':
    main()