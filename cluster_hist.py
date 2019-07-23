import glob
import json
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from data import get_data


def extract_color_histogram(image, bins=(8, 8, 8)):

    rgb = cv2.imread(image)
    hist = cv2.calcHist([rgb], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)

    return hist.flatten()


def hist_pca(fs, calc_hist=False):

    if calc_hist:
        hist = []
        for img in fs:
            hist.append(extract_color_histogram(img))

        hist = np.asarray(hist)
        np.save("saved_data/histogram.npy", hist)
    else:
        hist = np.load("saved_data/histogram.npy")

    cluster = KMeans(3).fit_predict(hist)
    viz = PCA(n_components=2).fit_transform(hist)

    df = pd.DataFrame(
        data={"name": fs, "cluster": cluster, "x": viz[:, 0], "y": viz[:, 1]}
    )
    df.to_csv("saved_data/hist_pca.csv", index=False)


if __name__ == "__main__":

    fs = get_data()
    hist_pca(fs, calc_hist=True)
