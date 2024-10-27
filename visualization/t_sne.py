from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from unicodedata import category

from utils.feature import get_feature_from_excel

def main():

    category = 'toothbrush'

    X_good = get_hog_features(category, 'good')
    X_bad = get_hog_features(category, 'bad')

    perplexity = 30
    max_iteration = 5000
    t_sne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=max_iteration)
    Y_good = t_sne.fit_transform(X_good)
    Y_bad = t_sne.fit_transform(X_bad)

    # Plot
    plt.figure(figsize=(10,10))
    plt.scatter(Y_good[:, 0], Y_good[:, 1], c='blue', label='good' )
    plt.scatter(Y_bad[:, 0], Y_bad[:, 1], c='red', label='bad' )
    plt.title(f"{category} dataset (perplexity={perplexity}, max_iteration={max_iteration})")
    plt.legend()
    plt.show()

    return None

def get_hog_features(category, data_class):
    file_path = f"../../output/hog_descriptor_{data_class}.xlsx"
    X = get_feature_from_excel(file_path, feature_size=8100, category=category)
    return X

if __name__ == '__main__':
    main()
