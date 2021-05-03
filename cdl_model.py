import logging

import numpy as np
from keras.layers import Input, Embedding, Dot, Flatten, Dense, Dropout, Add
from keras.layers.noise import GaussianNoise
from keras.models import Model
from keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal


class CollaborativeDeepLearning:
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

    def pretrain(self, lamda_w=0.1, encoder_noise=0.1, dropout_rate=0.1, activation='sigmoid', batch_size=64,
                 epochs=10):
        '''
        layer-wise pretraining on item features (item_mat)
        '''
        self.trained_encoders = []
        self.trained_decoders = []
        X_train = self.item_mat
        for input_dim, hidden_dim in zip(self.hidden_layers[:-1], self.hidden_layers[1:]):
            logging.info('Pretraining the layer: Input dim {} -> Output dim {}'.format(input_dim, hidden_dim))
            pretrain_input = Input(shape=(input_dim,))
            encoded = GaussianNoise(stddev=encoder_noise)(pretrain_input)
            encoded = Dropout(dropout_rate)(encoded)
            encoder = Dense(hidden_dim, activation=activation, kernel_regularizer=l2(lamda_w),
                            bias_regularizer=l2(lamda_w))(encoded)
            decoder = Dense(input_dim, activation=activation, kernel_regularizer=l2(lamda_w),
                            bias_regularizer=l2(lamda_w))(encoder)
            # autoencoder

            # enc_chkpoint = ModelCheckpoint(filepath="featurizer_checkpoints")
            ae = Model(inputs=pretrain_input, outputs=decoder)
            # encoder
            ae_encoder = Model(inputs=pretrain_input, outputs=encoder)
            # decoder
            encoded_input = Input(shape=(hidden_dim,))
            decoder_layer = ae.layers[-1]  # the last layer
            ae_decoder = Model(encoded_input, decoder_layer(encoded_input))

            ae.compile(loss='mse', optimizer='rmsprop')
            ae.fit(X_train, X_train, batch_size=batch_size, epochs=epochs, verbose=2)  # , callbacks=[enc_chkpoint])

            self.trained_encoders.append(ae_encoder)
            self.trained_decoders.append(ae_decoder)
            X_train = ae_encoder.predict(X_train)

    def fineture(self, train_mat, test_mat, lamda_u=0.1, lamda_v=0.1, lamda_n=0.1, lr=0.001,
                 batch_size=64, epochs=10):
        '''
        Fine-tuning with rating prediction
        '''
        num_user = int(max(train_mat["reviewer_id"].max(), test_mat["reviewer_id"].max()) + 1)
        num_item = int(max(train_mat["item_id"].max(), test_mat["item_id"].max()) + 1)

        # item autoencoder
        itemfeat_InputLayer = Input(shape=(self.item_dim,), name='item_feat_input')
        encoded = self.trained_encoders[0](itemfeat_InputLayer)
        encoded = self.trained_encoders[1](encoded)
        decoded = self.trained_decoders[1](encoded)
        decoded = self.trained_decoders[0](decoded)

        # user embedding
        user_InputLayer = Input(shape=(1,), dtype='int32', name='user_input')
        user_EmbeddingLayer = Embedding(input_dim=num_user, output_dim=self.embedding_dim, input_length=1,
                                        name='user_embedding', embeddings_regularizer=l2(lamda_u),
                                        embeddings_initializer=RandomNormal(mean=0, stddev=1))(user_InputLayer)
        user_EmbeddingLayer = Flatten(name='user_flatten')(user_EmbeddingLayer)

        # item embedding
        item_InputLayer = Input(shape=(1,), dtype='int32', name='item_input')
        item_OffsetVector = Embedding(input_dim=num_item, output_dim=self.embedding_dim, input_length=1,
                                      name='item_offset_vector', embeddings_regularizer=l2(lamda_v),
                                      embeddings_initializer=RandomNormal(mean=0, stddev=1))(item_InputLayer)
        item_OffsetVector = Flatten(name='item_flatten')(item_OffsetVector)
        item_EmbeddingLayer = Add()([encoded, item_OffsetVector])

        # rating prediction
        dotLayer = Dot(axes=-1, name='dot_layer')([user_EmbeddingLayer, item_EmbeddingLayer])

        self.cdl_model = Model(inputs=[user_InputLayer, item_InputLayer, itemfeat_InputLayer],
                               outputs=[dotLayer, decoded])
        self.cdl_model.compile(optimizer='rmsprop', loss=['mse', 'mse'], loss_weights=[1, lamda_n])

        train_user, train_item, train_item_feat, train_label = self.matrix2input(train_mat)
        test_user, test_item, test_item_feat, test_label = self.matrix2input(test_mat)

        model_history = self.cdl_model.fit([train_user, train_item, train_item_feat], [train_label, train_item_feat],
                                           epochs=epochs, batch_size=batch_size, validation_data=(
                [test_user, test_item, test_item_feat], [test_label, test_item_feat]))
        return model_history

    def matrix2input(self, rating_mat):
        train_user = rating_mat[["reviewer_id"]]
        asins = rating_mat["asin"].tolist()
        train_item = rating_mat[["item_id"]]
        train_item_feat = []
        for x in range(len(asins)):
            one_feat = self.item_dict.get(asins[x])
            if one_feat is None:
                one_feat = [0 for _ in range(4096)]
            train_item_feat.append(np.array(one_feat))
        train_label = np.array(rating_mat["rating"].values.tolist())/5
        return train_user, train_item, np.array(train_item_feat), train_label

    def build(self, train_mat, test_mat, lamda_u=0.1, lamda_v=0.1, lamda_n=0.1, lr=0.001, batch_size=64, epochs=10):
        # rating prediction
        num_user = int(max(train_mat[:, 0].max(), test_mat[:, 0].max()) + 1)
        num_item = int(max(train_mat[:, 1].max(), test_mat[:, 1].max()) + 1)

        # item autoencoder
        itemfeat_InputLayer = Input(shape=(self.item_dim,), name='item_feat_input')
        encoded = self.trained_encoders[0](itemfeat_InputLayer)
        encoded = self.trained_encoders[1](encoded)
        decoded = self.trained_decoders[1](encoded)
        decoded = self.trained_decoders[0](decoded)

        # user embedding
        user_InputLayer = Input(shape=(1,), dtype='int32', name='user_input')
        user_EmbeddingLayer = Embedding(input_dim=num_user, output_dim=self.embedding_dim, input_length=1,
                                        name='user_embedding', embeddings_regularizer=l2(lamda_u),
                                        embeddings_initializer=RandomNormal(mean=0, stddev=1))(user_InputLayer)
        user_EmbeddingLayer = Flatten(name='user_flatten')(user_EmbeddingLayer)

        # item embedding
        item_InputLayer = Input(shape=(1,), dtype='int32', name='item_input')
        item_OffsetVector = Embedding(input_dim=num_item, output_dim=self.embedding_dim, input_length=1,
                                      name='item_offset_vector', embeddings_regularizer=l2(lamda_v),
                                      embeddings_initializer=RandomNormal(mean=0, stddev=1))(item_InputLayer)
        item_OffsetVector = Flatten(name='item_flatten')(item_OffsetVector)
        item_EmbeddingLayer = Add()([encoded, item_OffsetVector])

        # rating prediction
        dotLayer = Dot(axes=-1, name='dot_layer')([user_EmbeddingLayer, item_EmbeddingLayer])

        prediction_layer = Dot(axes=-1, name='prediction_layer')([user_EmbeddingLayer, encoded])
        self.model = Model(inputs=[user_InputLayer, itemfeat_InputLayer], outputs=[prediction_layer])

    def getRMSE(self, test_mat):
        test_user, test_item, test_item_feat, test_label = self.matrix2input(test_mat)
        pred_out = self.cdl_model.predict([test_user, test_item, test_item_feat])
        # pred_out = self.cdl_model.predict([test_user, test_item, test_item_feat])
        return np.sqrt(np.mean(np.square(test_label.flatten() - pred_out[0].flatten())))


    def get_sample_labels_and_preds(self, test_mat):
        test_user, test_item, test_item_feat, test_label = self.matrix2input(test_mat)
        pred_out = self.cdl_model.predict([test_user, test_item, test_item_feat])
        # pred_out = self.cdl_model.predict([test_user, test_item, test_item_feat])
        return test_label.flatten(), pred_out[0].flatten()