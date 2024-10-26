from pathlib import Path
import cv2 as cv
import numpy as np

from utils.feature import save_feature_to_excel
from utils.constants import CATEGORIES

def main():
    for category in CATEGORIES:
        good_dataset_path = f"../dataset/{category}/train/good"
        compute_hog_feature(dataset_path=good_dataset_path, category=category, data_class="good")
    return None

def compute_hog_feature(dataset_path, category, data_class):

    assert dataset_path is not None, "dataset_path cannot be empty"
    assert category in CATEGORIES, "invalid data category"
    assert data_class in ["good", "bad"], "invalid data class, should be ['good', 'bad']"

    # Get all good image of bottle
    img_files = list(Path(dataset_path).glob("*.png"))
    num_img = len(img_files)

    hog = get_hog_descriptor()

    num_sample, num_feature = num_img, hog.getDescriptorSize()
    X = np.zeros((num_sample, num_feature))
    print(f"Computing HOG descriptor for {category} category...")
    for index, file_path in enumerate(img_files):
        img = cv.imread(file_path)
        # Bottle image size is 900x900 and we have to resize it to 128x128
        img = cv.resize(img, (128, 128))
        hist = hog.compute(img)
        X[index, :] = hist
    print("Done Computing HOG descriptor.")
    save_path = f"./output/hog_descriptor_{data_class}.xlsx"
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