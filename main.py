import logging

from cdl_model import CollaborativeDeepLearning
from utils import read_rating, read_feature


def main():
    train_mat = read_rating('data/ml-1m/normalTrain.csv')
    test_mat = read_rating('data/ml-1m/test.csv')
    item_mat = read_feature('2014_analysis/sub_file.json')
    num_item_feat = len(item_mat[0])

    model = CollaborativeDeepLearning(item_mat, [num_item_feat, 512, 64])
    model.pretrain(lamda_w=0.001, encoder_noise=0.3, epochs=10)
    # model_history = model.fineture(train_mat, test_mat, lamda_u=0.01, lamda_v=0.1, lamda_n=0.1, lr=0.01, epochs=3)
    # testing_rmse = model.getRMSE(test_mat)
    # print('Testing RMSE = {}'.format(testing_rmse))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    main()
