import json
import cv2
import numpy as np
import pandas as pd
import scipy
from scipy import cluster  # weird scipy issue
from sklearn.manifold import TSNE
from data import get_data


def hex2rgb(h):

    h = h.lstrip("#")
    c = tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))

    return tuple(map(float, c))


def top_rgb(f):

    im = cv2.imread(f)
    im = cv2.resize(im, (200, 200))
    ar = np.asarray(im)
    shape = ar.shape
    ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

    codes, dist = scipy.cluster.vq.vq(ar, colours)

    return np.bincount(codes, minlength=len(colours))


def rgb_tsne(fs):

    hist = np.zeros((len(fs), len(colours)))
    for i, img in enumerate(fs):
        hist[i] = top_rgb(img)

    hist = hist / hist.mean(0)  # remove background
    viz = TSNE(n_components=2, verbose=3).fit_transform(hist)

    c = [hexes[i] for i in np.argmax(hist, 1)]
    df = pd.DataFrame(
        data={
            "fname": fs,
            "cluster": np.argmax(hist, 1),
            "x": viz[:, 0],
            "y": viz[:, 1],
            "col": c,
        }
    )

    df.to_csv("saved_data/rgb_tsne.csv", index=False)


if __name__ == "__main__":

    hexes = ["#e70000", "#00811f", "#0044ff", "#000000", "#ffffff", "#fffa00"]
    col_names = ["red", "green", "blue", "black", "white", "yellow"]
    colours = np.asarray([hex2rgb(c) for c in hexes])

    fs = get_data()
    rgb_tsne(fs)
