from ast import mod
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import lightgbm


def get_new_model(model_name, fea_num):
    if model_name == "RF":
        return RandomForestClassifier(criterion="entropy")
    if model_name == "MNB":
        return MultinomialNB()
    if model_name == "AB":
        return AdaBoostClassifier()
    if model_name == "GNB":
        return GaussianNB()
    if model_name == "LD":
        return LinearDiscriminantAnalysis()
    if model_name == "ET":
        return ExtraTreesClassifier(random_state=0, criterion="entropy")
    if model_name == "GB":
        return GradientBoostingClassifier(random_state=0)
    if model_name == "XGB":
        # to remove "warning" by use "eval_metric='logloss'"
        return XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    if model_name == "LR":
        return LogisticRegression(max_iter=10000)
    if model_name == "KNN":
        return KNeighborsClassifier()
    if model_name == "SVM":
        return SVC(kernel='rbf', probability=True)
    if model_name == "LGBM":
        return lightgbm.LGBMClassifier()
    if model_name == "CNN1":
        return CNNModel(fea_num)
    if model_name == "DNN1":
        return DNNModel(fea_num)
    if model_name == "LSTM1":
        return LSTMModel(fea_num)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, LSTM, Bidirectional, Dense, add, Activation, Permute, Multiply, Dropout, BatchNormalization,LeakyReLU
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense,RepeatVector,Permute, Layer,add, Activation, Multiply
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Layer, LeakyReLU, Dropout
from keras.layers import add, Activation


class ResidualBlock(Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv1 = Conv1D(filters, kernel_size=kernel_size, padding='same')
        self.conv2 = Conv1D(filters, kernel_size=kernel_size, padding='same')
        self.activation = LeakyReLU()

    def call(self, inputs):
        shortcut = inputs
        x = self.conv1(inputs)
        x = self.activation(x)
        x = self.conv2(x)
        x = add([shortcut, x])
        x = self.activation(x)
        return x

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.attention = Dense(1)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        attention = self.attention(inputs)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(inputs.shape[-1])(attention)
        attention = Permute([2, 1])(attention)
        output = Multiply()([inputs, attention])
        return output

def create_cnn_model(input_shape):
    print('building model...............')
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3,input_shape=input_shape))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))
    model.add(ResidualBlock(32, 3))
    model.add(MaxPooling1D(pool_size=2))
    model.add(AttentionLayer())
    model.add(Flatten())
    model.add(Dense(16))
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model



class CNNModel:
    def __init__(self, input_shape):
        self.model = KerasClassifier(build_fn=lambda: create_cnn_model((input_shape,1)), epochs=500, batch_size=512, verbose=0)

    def fit(self, X, y):
        X = np.expand_dims(X, axis=-1)
#         print(X.shape)
#         return self.model.fit(X, y)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)

        return self.model.fit(X, y, validation_split=0.2, callbacks=[early_stopping])

    def predict(self, X):
        X = np.expand_dims(X, axis=-1)
        return self.model.predict(X)

    def predict_proba(self, X):
        X = np.expand_dims(X, axis=-1)
        return self.model.predict_proba(X)
    
    
    
def create_dnn_model(input_dim, dense_1=250, dense_2=100, dense_3=50, dropout_1=0.2, dropout_2=0.2, dropout_3=0.2):
    print('building DNN model...............')
    model = Sequential()

    model.add(Dense(units=dense_1, activation='tanh', kernel_initializer='he_normal', input_dim=input_dim))
    model.add(Dropout(dropout_1))

    model.add(Dense(units=dense_2, activation='tanh', kernel_initializer='he_normal'))
    model.add(Dropout(dropout_2))

    model.add(Dense(units=dense_3, activation='tanh', kernel_initializer='he_normal'))
    model.add(Dropout(dropout_3))

    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])
    return model

class DNNModel:
    def __init__(self,input_dim, dense_1=250, dense_2=100, dense_3=50, dropout_1=0.2, dropout_2=0.2, dropout_3=0.2):
        
        self.model = KerasClassifier(build_fn=lambda: create_dnn_model(input_dim,dense_1, dense_2, dense_3,
                                                               dropout_1, dropout_2, dropout_3),
                                     epochs=30,
                                     batch_size=32,
                                     verbose=0)

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    def predict_proba(self, X):
        return self.model.predict_proba(X)


class ResidualLSTM(Layer):
    def __init__(self, units, **kwargs):
        super(ResidualLSTM, self).__init__(**kwargs)
        self.lstm1 = Bidirectional(LSTM(units, return_sequences=True))
        self.lstm2 = Bidirectional(LSTM(units, return_sequences=True))

    def call(self, inputs):
        shortcut = inputs
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        x = add([shortcut, x])
        x = Activation('tanh')(x)
        return x

class ScaledDotProductAttention(Layer):
    def __init__(self, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.d_model = input_shape[-1]
        super(ScaledDotProductAttention, self).build(input_shape)

    def call(self, inputs):
        q = inputs
        k = inputs
        v = inputs
        matmul_qk = K.batch_dot(q,k,axes=[2,2])
        dk = K.cast(K.shape(k)[-1], 'float32')
        scaled_attention_logits = matmul_qk / K.sqrt(dk)
        attention_weights = K.softmax(scaled_attention_logits,axis=-1)
        output = K.batch_dot(attention_weights,v,axes=[2,1])
        return output

def create_lstm_model(input_shape):
    print('building model...............')
    model = Sequential()
    model.add(Bidirectional(LSTM(32, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(ResidualLSTM(32))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(ScaledDotProductAttention())
    model.add(Dense(16))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model



class LSTMModel:
    def __init__(self, input_shape):
        self.model = KerasClassifier(build_fn=lambda: create_lstm_model((1,input_shape)), epochs=100, batch_size=256, verbose=0)

    def fit(self, X, y):
        X = np.expand_dims(X, axis=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        return self.model.fit(X, y, validation_split=0.2, callbacks=[early_stopping])

    def predict(self, X):
        X = np.expand_dims(X, axis=1)
        return self.model.predict(X).reshape(-1,)

    def predict_proba(self, X):
        X = np.expand_dims(X, axis=1)
        return self.model.predict_proba(X).reshape(-1,2)