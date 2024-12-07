import sys
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.layers import Dropout, Dense, Flatten, Activation, Conv1D, LeakyReLU, Bidirectional, LSTM
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tools.tools import split_seq_df, get_label

sys.setrecursionlimit(15000)
import numpy as np


def one_hot(seq):
    nrows = len(seq)
    seq_len = len(seq[0])
    dict_base = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "U": [0, 0, 0, 1],
        "T": [0, 0, 0, 1]
    }
    result = np.zeros((nrows, seq_len, 4), dtype='int')
    for i in range(nrows):
        one_seq = seq[i]
        for j in range(seq_len):
            result[i, j, :] = dict_base[one_seq[j]]
    return result



def create_model(config):
    print('building model...............')
    model = Sequential()
    params = {
        "filters_1": 250,
        "kernel_size_1": 2,
        "filters_2": 100,
        "kernel_size_2": 2,
        "filters_3": 100,
        "kernel_size_3": 3,
        "filters_4": 250,
        "kernel_size_4": 2,
        "filters_5": 250,
        "kernel_size_5": 10,
        "dense_1": 320
    }

    #************************* convolutional layer  ***********************
    model.add(Conv1D(
                            input_shape=(41,4),
                            filters=params["filters_1"],
                            kernel_size=params["kernel_size_1"],
                            padding="valid",
                            activation="linear",
                            strides=1,
                            #W_regularizer = l2(0.01),
                            kernel_initializer='he_normal',
                            name="cov1"))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dropout(0.2))


    model.add(Conv1D(filters=params["filters_2"],
                            kernel_size=params["kernel_size_2"],
                            padding="valid",
                            activation="linear",
                            strides=1, kernel_initializer='he_normal',
                            name = "cov2"))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dropout(0.2))



    model.add(Conv1D(filters=params["filters_3"],
                            kernel_size=params["kernel_size_3"],
                            padding="valid",
                            activation="linear",
                            strides=1, kernel_initializer='he_normal',
                            name="cov3"))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dropout(0.2))


    model.add(Conv1D(filters=params["filters_4"],
                            kernel_size=params["kernel_size_4"],
                            padding="valid",
                            activation="linear",
                            strides=1, kernel_initializer='he_normal',
                            name = "cov4"))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dropout(0.2))



    model.add(Conv1D(filters=params["filters_5"],
                            kernel_size=params["kernel_size_5"],
                            padding="valid",
                            activation="linear",
                            strides=1,
                            kernel_initializer='he_normal',
                            name = "cov5"))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dropout(0.5))


    model.add(Flatten())
    model.add(Dense(units=params["dense_1"], kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dropout(0.5))
    model.add(Dense(units=1))
    model.add(Activation('sigmoid'))
    # print(model.summary())
    return model

class TA_keras():
    def __init__(self,config):
        self.config = config
        self.best_model = None
        self.temp_model = None
        self.global_model_save_path = config.global_model_save_path
        self.bestmodel_path = config.model_save_path+config.model_name+"_.h5"

    def fit(self,X,y,X_train=None,X_dev=None):
        print("*"*100)
        print(X)
        if X_train is None:
            X_train, X_dev = split_seq_df(X)
        y_train = get_label(X_train)
        y_dev = get_label(X_dev)
        
        X_train = one_hot(X_train.iloc[:,1].values)
        X_dev = one_hot(X_dev.iloc[:,1].values)

        if self.config.load_global_pretrain_model:
            print("loading global pretrain model ...")
            model = load_model(self.config.global_model_save_path)
        else:
            model = create_model(self.config)

        sgd = SGD(lr=self.config.learning_rate, momentum=0.9, decay=1e-6, nesterov=True)
        checkpointer = ModelCheckpoint(filepath=self.bestmodel_path, verbose=0, save_best_only=True)
        earlystopper = EarlyStopping(monitor='val_loss', patience=self.config.patience, verbose=0)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
        model.fit(X_train,
                  y_train,
                  batch_size=self.config.batch_size,
                  epochs=self.config.num_epochs,
                  shuffle=True,
                  verbose=0,
                  validation_data=(X_dev, y_dev),
                  callbacks=[checkpointer, earlystopper])
        self.best_model = load_model(self.bestmodel_path)
        print('training done!')

    def predict_proba(self,X):
        #test
        model = self.best_model
        X = one_hot(X.iloc[:, 1].values)
        pred_prob_test = model.predict(X, verbose=0)
        pred_prob_test = np.squeeze(pred_prob_test)
        return pred_prob_test


