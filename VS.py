import matplotlib.pyplot as plt
import cv2
import numpy as np
from collections import namedtuple
import mxnet as mx
import sagemaker as sm

role = sm.get_execution_role()

Batch = namedtuple('Batch', ['data'])
num_layers = 50
resnet = 'resnet-50'

path = 'http://data.mxnet.io/models/imagenet/'
[mx.test_utils.download(path + 'resnet/' + str(num_layers) + '-layers/' + resnet + '-0000.params'),
 mx.test_utils.download(path + 'resnet/' + str(num_layers) + '-layers/' + resnet + '-symbol.json')]

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

# add comm

def process_file(file_name, start_index=0):
    products = []
    index = start_index
    dim = 2048
    xb = np.empty(shape=[0, dim], dtype=np.float32)

    with open(file_name) as file:
        for line in file:
            product = {}
            fields = line.split(',')
            product['id'] = str(index)
            product['productTitle'] = fields[0].strip()
            product['imageUrl'] = fields[1].strip()
            # download image to be featurized and preprocess it
            file = mx.test_utils.download(product['imageUrl'])
            product['imageFileName'] = file

            img = get_image(file)
            # extract features
            fe_mod.forward(Batch([mx.nd.array(img)]))
            features = fe_mod.get_outputs()[0].asnumpy()
            # the Knn algorithm we'll use requires float32 rather than the default float64
            xb = np.append(xb, features.astype(np.float32), axis=0)
            products.append(product)
            index += 1

    return (products, xb)


products, train_features = process_file('fashion_data.txt')
