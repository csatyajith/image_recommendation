import logging

import numpy as np
from keras.layers import Input, Embedding, Dot, Flatten, Dense, Dropout, Add
from keras.layers.noise import GaussianNoise
from keras.models import Model
from keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal


class CDL:
    def __init__(self, item_mat, hidden_layers, item_dict):
        """
        hidden_layers = a list of three integer indicating the embedding dimension of autoencoder
        item_mat = item feature matrix with shape (# of item, # of item features)
        """
        assert (len(hidden_layers) == 3)
        self.item_mat = item_mat
        self.item_dict = item_dict
        self.hidden_layers = hidden_layers
        self.item_dim = hidden_layers[0]
        self.embedding_dim = hidden_layers[-1]
        self.trained_encoders = []
        self.trained_decoders = []
        self.cdl_model = None

    def pre_train_item_representation(self, lambda_w=0.1, encoder_noise=0.1, dropout_rate=0.1, activation='sigmoid',
                                      batch_size=64,
                                      epochs=10):
        """
        Pretraining the Stacked de-noising auto encoder (SDAE)
        :param lambda_w: Regularization parameter
        :param encoder_noise: Standard deviation of gaussian noise
        :param dropout_rate: Dropout rate for the encoded layer
        :param activation: Activation function to be used at the encoder and decoder points
        :param batch_size: Batch size to be used while training
        :param epochs: Number of epochs to train for
        :return:
        """
        x_train = self.item_mat
        for input_dim, hidden_dim in zip(self.hidden_layers[:-1], self.hidden_layers[1:]):
            logging.info('Pretraining the layer: Input dim {} -> Output dim {}'.format(input_dim, hidden_dim))
            pretrain_input = Input(shape=(input_dim,))

            encoded = GaussianNoise(stddev=encoder_noise)(pretrain_input)
            encoded = Dropout(dropout_rate)(encoded)
            encoder = Dense(hidden_dim, activation=activation, kernel_regularizer=l2(lambda_w),
                            bias_regularizer=l2(lambda_w))(encoded)
            decoder = Dense(input_dim, activation=activation, kernel_regularizer=l2(lambda_w),
                            bias_regularizer=l2(lambda_w))(encoder)

            ae = Model(inputs=pretrain_input, outputs=decoder)
            ae_encoder = Model(inputs=pretrain_input, outputs=encoder)
            encoded_input = Input(shape=(hidden_dim,))
            decoder_layer = ae.layers[-1]
            ae_decoder = Model(encoded_input, decoder_layer(encoded_input))

            ae.compile(loss='mse', optimizer='rmsprop')
            ae.fit(x_train, x_train, batch_size=batch_size, epochs=epochs, verbose=2)  # , callbacks=[enc_chkpoint])

            self.trained_encoders.append(ae_encoder)
            self.trained_decoders.append(ae_decoder)
            x_train = ae_encoder.predict(x_train)

    def train_cdl_model(self, train_mat, test_mat, lambda_u=0.1, lambda_v=0.1, lambda_n=0.1,
                        batch_size=64, epochs=10):
        """
        Train CDL model using the pretrained model, user embeddings, item embeddings. Predict the rating for a product.
        :param train_mat: The training matrix as a dataframe consisting of ["asin", "reviewer_id", "item_id", "rating"]
        :param test_mat: Test matrix in the same structure as the train matrix above
        :param lambda_u: Regularization parameter
        :param lambda_v: Regularization parameter
        :param lambda_n: Regularization parameter
        :param batch_size: Batch size to be used while training
        :param epochs: Number of epochs to train for
        :return: The trained CDL model that can recommend products.
        """
        num_user = int(max(train_mat["reviewer_id"].max(), test_mat["reviewer_id"].max()) + 1)
        num_item = int(max(train_mat["item_id"].max(), test_mat["item_id"].max()) + 1)

        item_feat_input_layer = Input(shape=(self.item_dim,), name='item_feat_input')
        encoded = self.trained_encoders[0](item_feat_input_layer)
        encoded = self.trained_encoders[1](encoded)
        decoded = self.trained_decoders[1](encoded)
        decoded = self.trained_decoders[0](decoded)

        # From the integer user ids, we first intend to get the user embedding layer
        user_input_layer = Input(shape=(1,), dtype='int32', name='user_input')
        user_embedding_layer = Embedding(input_dim=num_user, output_dim=self.embedding_dim, input_length=1,
                                         name='user_embedding', embeddings_regularizer=l2(lambda_u),
                                         embeddings_initializer=RandomNormal(mean=0, stddev=1))(user_input_layer)
        user_embedding_layer = Flatten(name='user_flatten')(user_embedding_layer)

        # From the integer item ids and the product representation vectors, we get the item embedding layer
        item_input_layer = Input(shape=(1,), dtype='int32', name='item_input')
        item_offset_vector = Embedding(input_dim=num_item, output_dim=self.embedding_dim, input_length=1,
                                       name='item_offset_vector', embeddings_regularizer=l2(lambda_v),
                                       embeddings_initializer=RandomNormal(mean=0, stddev=1))(item_input_layer)
        item_offset_vector = Flatten(name='item_flatten')(item_offset_vector)
        item_embedding_layer = Add()([encoded, item_offset_vector])

        dot_layer = Dot(axes=-1, name='dot_layer')([user_embedding_layer, item_embedding_layer])

        self.cdl_model = Model(inputs=[user_input_layer, item_input_layer, item_feat_input_layer],
                               outputs=[dot_layer, decoded])
        self.cdl_model.compile(optimizer='rmsprop', loss=['mse', 'mse'], loss_weights=[1, lambda_n])

        train_user, train_item, train_item_feat, train_label = self.get_input_and_labels(train_mat)
        test_user, test_item, test_item_feat, test_label = self.get_input_and_labels(test_mat)

        model_history = self.cdl_model.fit([train_user, train_item, train_item_feat], [train_label, train_item_feat],
                                           epochs=epochs, batch_size=batch_size, validation_data=(
                [test_user, test_item, test_item_feat], [test_label, test_item_feat]))
        return model_history

    def get_input_and_labels(self, rating_mat):
        """
        This function takes a train or test matrix as input and returns the reviewer_id, item_id, item_features and
        the labels in Keras usable formats
        :param rating_mat: A ratings matrix as a dataframe consisting of ["asin", "reviewer_id", "item_id", "rating"]
        :return: user_id, item_ids, item_features and labels
        """
        user_ids = rating_mat[["reviewer_id"]]
        asins = rating_mat["asin"].tolist()
        item_ids = rating_mat[["item_id"]]
        item_features = []
        for x in range(len(asins)):
            one_feat = self.item_dict.get(asins[x])
            if one_feat is None:
                one_feat = [0 for _ in range(4096)]
            item_features.append(np.array(one_feat))
        labels = np.array(rating_mat["rating"].values.tolist()) / 5
        return user_ids, item_ids, np.array(item_features), labels

    def get_rmse(self, test_mat):
        """
        Takes the test matrix as input and returns the root mean squared error for it.
        :param test_mat: Test matrix as the same format as train matrix
        :return: Mean squared error between predictions and results
        """
        test_user, test_item, test_item_feat, test_label = self.get_input_and_labels(test_mat)
        pred_out = self.cdl_model.predict([test_user, test_item, test_item_feat])
        return np.sqrt(np.mean(np.square(test_label.flatten() - pred_out[0].flatten())))

