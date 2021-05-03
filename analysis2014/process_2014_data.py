import gzip
import json

import numpy as np


def get_reviews_in_json_file():
    all_reviews = []
    with gzip.open("baby_5_core.json.gz", "r") as f:
        for line in f:
            review = {}
            r = eval(line)
            review["reviewer_id"] = r["reviewerID"]
            review["asin"] = r["asin"]
            review["rating"] = r["overall"]
            all_reviews.append(review)

    with open("baby_5_core_reviews.json", "w") as b:
        json.dump(all_reviews, b)


def get_image_vec_list():
    with open("analysis2014/reviews_products.json", "r") as rp_file:
        rp = json.load(rp_file)

    with open("analysis2014/baby_image_features.b", "rb") as baby_img:
        x = {}
        while True:
            asin = baby_img.read(10)
            if not asin:
                break
            asin = asin.decode()
            a = np.fromfile(baby_img, "f", 4096).tolist()
            if rp.get(asin) is None:
                continue
            x[asin] = a
            if len(x) % 1000 == 0:
                print("processed {} asins".format(len(x)))
    return x


def get_mapped_ratings(x):
    with open("analysis2014/baby_5_core_reviews.json", "r") as rf:
        reviews = json.load(rf)
    for i, r in enumerate(reviews):
        if i % 10000 == 0:
            print("Done with {} reviews".format(i))
        r["img_vector"] = x.get(r["asin"])
    return reviews


if __name__ == '__main__':
    get_image_vec_list()
