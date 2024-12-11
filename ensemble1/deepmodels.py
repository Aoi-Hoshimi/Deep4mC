from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


# 定义基础模型类，继承自BaseEstimator和ClassifierMixin，使其更符合sklearn风格
class BaseDeepModel(BaseEstimator, ClassifierMixin):
    def __init__(self, epochs=10, batch_size=32, verbose=0):
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, X, y):
        raise NotImplementedError("Subclasses must implement fit method")

    def predict(self, X):
        raise NotImplementedError("Subclasses must implement predict method")

    def predict_proba(self, X):
        raise NotImplementedError("Subclasses must implement predict_proba method")


# 重新定义CNN模型类，继承自BaseDeepModel
class CNNModel(BaseDeepModel):
    def __init__(self, input_shape, epochs=10, batch_size=32, verbose=0):
        super(CNNModel, self).__init__(epochs, batch_size, verbose)
        self.input_shape = input_shape
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Conv1D(32, kernel_size=3, activation='tanh', input_shape=self.input_shape))
        model.add(MaxPooling1D(pool_size=1, strides=1))
        model.add(Flatten())
        model.add(Dense(64, activation='tanh'))
        model.add(Dense(32, activation='tanh'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        return model

    def fit(self, X, y):
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        return self

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype(int).reshape(-1,)

    def predict_proba(self, X):
        return self.model.predict(X)