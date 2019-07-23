import glob
import json


def read_data(f="data/**/**/*.jpg"):

    fs = glob.glob(f)
    json.dump(fs, open("saved_data/files.json", "w"))


def get_data():

    with open("saved_data/files.json") as f:
        fs = json.load(f)

    return fs


if __name__ == "__main__":

    read_data()
    fs = get_data()
