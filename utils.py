import numpy as np
from pandas import read_csv
import json

def read_rating(file_path, has_header=True):
    rating_mat = list()
    with open(file_path) as fp:
        if has_header is True:
            fp.readline()
        for line in fp:
            line = line.split(',')
            user, item, rating = line[0], line[1], line[2]
            rating_mat.append([user, item, rating])
    return np.array(rating_mat).astype('float32')


def read_feature(file_path):
    with open(file_path, "r") as feat_file:
        feat_mat = json.load(feat_file)
    return feat_mat
