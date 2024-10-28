import math

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

from utils.constants import CATEGORIES
from utils.feature import get_feature_from_excel

def main():

    for category in CATEGORIES:
        X_good = get_hog_features(category, 'good')
        X_bad = get_hog_features(category, 'bad')

        num_good, num_bad = X_good.shape[0], X_bad.shape[0]

        X = np.vstack((X_good, X_bad))

        perplexity = math.floor(math.sqrt(min(num_bad, num_good)))
        t_sne = TSNE(n_components=2, perplexity=perplexity, random_state=78, method='exact')
        Y = t_sne.fit_transform(X)

        # Plot
        plt.figure(figsize=(10,10))
        plt.scatter(Y[:num_good, 0], Y[:num_good, 1], c='blue', label='good')
        plt.scatter(Y[num_good:, 0], Y[num_good:, 1], c='red', label='bad' )
        plt.title(f"{category} dataset (perplexity={perplexity})")
        plt.legend()
        plt.show()

    return None

def get_hog_features(category, data_class):
    file_path = f"../../output/hog_descriptor_{data_class}.xlsx"
    X = get_feature_from_excel(file_path, feature_size=8100, category=category)
    return X

if __name__ == '__main__':
    main()
