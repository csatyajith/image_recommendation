import gzip
import json
from urllib.request import urlopen

import pandas as pd


# load the metadata
def load_cdl_data():
    data = []
    link = 'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_AMAZON_FASHION.json.gz'
    f = urlopen(link)
    with gzip.open(f) as f:
        for l in f:
            data.append(json.loads(l.strip()))
    # total length of list, this number equals total number of products
    print(len(data))
    # first row of the list
    print(data[0])
    # convert list into pandas dataframe
    df = pd.DataFrame.from_dict(data)
    print(len(df))

    return df


def main():
    data = load_cdl_data()
    print(len(data))


if __name__ == '__main__':
    main()
