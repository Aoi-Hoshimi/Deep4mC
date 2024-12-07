from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

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
    

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import numpy as np

def create_cnn_model(input_shape):
    print('building CNN model...............')
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, activation='tanh', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

class CNNModel:
    def __init__(self, input_shape):
        self.model = KerasClassifier(build_fn=lambda: create_cnn_model((input_shape,1)), epochs=10, batch_size=32, verbose=0)

    def fit(self, X, y):
        X = np.expand_dims(X, axis=-1)
        print(X.shape)
        return self.model.fit(X, y)

    def predict(self, X):
        X = np.expand_dims(X, axis=-1)
        return self.model.predict(X)

    def get_prob(self, X):
        X = np.expand_dims(X, axis=-1)
        return self.model.predict_proba(X)
    

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import numpy as np

def create_lstm_model(input_shape):
    print('building LSTM model...............')
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(64))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(16, activation='tanh'))
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
        return self.model.predict_proba(X)