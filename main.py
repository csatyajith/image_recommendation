import logging

from cdl_model import CDL
from utils import read_rating, read_feature
from analysis2014.process_2014_data import get_image_vec_list


def main():
    # train_mat = read_rating('data/ml-1m/normalTrain.csv')
    # test_mat = read_rating('data/ml-1m/test.csv')
    item_mat = get_image_vec_list()
    num_item_feat = len(item_mat[0])

    model = CDL(item_mat, [num_item_feat, 512, 64])
    model.pre_train_item_representation(lambda_w=0.001, encoder_noise=0.3, epochs=10)
    # model_history = model.fineture(train_mat, test_mat, lamda_u=0.01, lamda_v=0.1, lamda_n=0.1, lr=0.01, epochs=3)
    # testing_rmse = model.getRMSE(test_mat)
    # print('Testing RMSE = {}'.format(testing_rmse))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    main()

#%%