import glob
import os
import json
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers
from keras.utils import to_categorical
from keras import regularizers
from keras.applications import VGG16
from sklearn.metrics import roc_auc_score

np.random.seed(123)


def extract_features(directory, sample_count, class_mode=None):

    datagen = ImageDataGenerator(rescale=1. / 255)
    batch_size = 20

    # Base of the VGG16 net
    conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(150, 150, 3))

    # make arrays
    # generator will give out the images one by one along with labels
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    if class_mode is not None:
        labels = np.zeros(shape=sample_count)
    else:
        labels = None

    generator = datagen.flow_from_directory(
        directory, target_size=(150, 150), batch_size=batch_size, class_mode=class_mode
    )

    i = 0
    # iterate through images and hand them to conv net to get prediction
    for batch in generator:

        if class_mode is not None:
            batch, labels_batch = batch
            labels[i * batch_size : (i + 1) * batch_size] = labels_batch

        features_batch = conv_base.predict(batch, verbose=1)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        i += 1
        # when we've done all of the images stop
        if i * batch_size >= sample_count:
            break

    # if there are classes we make a dictionary of them, otherwise None
    classes = generator.class_indices if class_mode is not None else None

    return features, labels, classes

    # number of test and train images


def train_test_image(train_dir, test_dir):

    ntrain = len(os.listdir(train_dir + "/non_maxi")) + len(
        os.listdir(train_dir + "/maxi")
    )
    ntest = len(os.listdir(test_dir + "/non_maxi")) + len(
        os.listdir(test_dir + "/maxi")
    )

    train_features, train_labels, classes = extract_features(
        train_dir, ntrain, class_mode="binary"
    )
    test_features, test_labels, classes = extract_features(
        test_dir, ntest, class_mode="binary"
    )

    train_features = np.reshape(train_features, (-1, 4 * 4 * 512))
    test_features = np.reshape(test_features, (-1, 4 * 4 * 512))

    np.save("saved_data/train_features.npy", train_features)
    np.save("saved_data/train_labels.npy", train_labels)
    np.save("saved_data/test_features.npy", test_features)
    np.save("saved_data/test_labels.npy", test_labels)

    return classes


def net():

    model = models.Sequential()
    # 256 hidden units
    model.add(layers.Dense(256, activation="relu", input_dim=4 * 4 * 512))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=optimizers.RMSprop(lr=1e-4),
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )

    return model


if __name__ == "__main__":

    # define directories
    base_dir = "data/"
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")

    classes = train_test_image(train_dir, test_dir)

    train_features = np.load("saved_data/train_features.npy")
    train_labels = np.load("saved_data/train_labels.npy")
    test_features = np.load("saved_data/test_features.npy")
    test_labels = np.load("saved_data/test_labels.npy")

    model = net()
    history = model.fit(
        train_features,
        train_labels,
        epochs=10,
        batch_size=64,
        validation_data=(test_features, test_labels),
    )

    preds = model.predict(test_features)
    print("ROC AUC: ", roc_auc_score(test_labels, preds))

    # label is a probability between 0 and 1, apply a cut off to it
    label = (preds > 0.5).astype(int)

    test_fs = datagen.flow_from_directory(test_dir).filenames
    test_fs = ["".join(x) for x in zip(["data/test/"] * len(test_fs), test_fs)]
    df = pd.read_csv("saved_data/rgb_tsne.csv")
    df = df.set_index("fname").loc[test_fs].reset_index(inplace=False)
    long = classes["maxi"]
    df["length"] = ["maxi" if i == long else "not maxi" for i in label]
    df.to_csv("saved_data/cnn_rgb_tsne.csv", index=False)
