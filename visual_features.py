import ast
from collections import namedtuple

import cv2
import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
import pandas as pd


class VisualFeatures:
    def __init__(self):
        self.batch = namedtuple('Batch', ['data'])

    @staticmethod
    def create_featurizer():
        num_layers = 50
        resnet = 'resnet-50'

        path = 'http://data.mxnet.io/models/imagenet/'
        mx.test_utils.download(path + 'resnet/' + str(num_layers) + '-layers/' + resnet + '-0000.params')
        mx.test_utils.download(path + 'resnet/' + str(num_layers) + '-layers/' + resnet + '-symbol.json')

        sym, arg_params, aux_params = mx.model.load_checkpoint(resnet, 0)
        mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
        mod.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))], label_shapes=mod._label_shapes)
        mod.set_params(arg_params, aux_params, allow_missing=True)

        all_layers = sym.get_internals()
        fe_sym = all_layers['flatten0_output']
        fe_mod = mx.mod.Module(symbol=fe_sym, context=mx.cpu(), label_names=None)
        fe_mod.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))])
        fe_mod.set_params(arg_params, aux_params)
        fe_mod.save_checkpoint('featurizer-v1', 0)

    def load_featurizer(self, featurizer_path):
        sym, arg_params, aux_params = mx.model.load_checkpoint(featurizer_path, 0)
        fe_mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
        fe_mod.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))])
        fe_mod.set_params(arg_params, aux_params)
        return fe_mod

    @staticmethod
    def get_image(fname, show=False):
        img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
        if img is None:
            return None
        if show:
            plt.imshow(img)
            plt.axis('off')
        # convert into format (batch, RGB, width, height)
        img = cv2.resize(img, (224, 224))
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        img = img[np.newaxis, :]

        return img

    def process_file(self, file_name, fe_mod, start_index=0):
        products = []
        dim = 2048
        xb = np.empty(shape=[0, dim])
        index = start_index
        data = pd.read_csv(file_name)
        print(data.shape)
        data = data.iloc[:, 1:3]
        for i, row in data.iterrows():
            if type(row['image']) != str:
                continue
            print(row["image"])
            if i > 3:
                break
            product = {'id': str(index), 'productTitle': row['title'], 'imageUrl': row['image']}
            # download image to be featurized and preprocess it
            product['imageUrl'] = ast.literal_eval(product['imageUrl'])
            file = mx.test_utils.download(product['imageUrl'][0])
            product['imageFileName'] = file
            img = self.get_image(file)
            fe_mod.forward(self.batch([mx.nd.array(img)]))
            features = fe_mod.get_outputs()[0].asnumpy()
            xb = np.append(xb, features, axis=0)
            products.append(product)
            index += 1
        return products, xb


# role = sm.get_execution_role()


#
def main():
    visual_features = VisualFeatures()
    fe_mod = visual_features.load_featurizer("featurizer_checkpoints/featurizer-v1")
    products_1, train_features = visual_features.process_file('fashion_data.csv', fe_mod)
    print(len(train_features))
    print(products_1)


def test_load_featurizer():
    visual_features = VisualFeatures()
    fe_mod = visual_features.load_featurizer("featurizer_checkpoints/featurizer-v1")


if __name__ == '__main__':
    main()
