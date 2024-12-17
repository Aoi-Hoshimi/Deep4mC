from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Bidirectional, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


# 基础的模型类，继承自BaseEstimator和ClassifierMixin，用于规范模型的基本行为，使其更符合sklearn风格
class BaseDeepModel(BaseEstimator, ClassifierMixin):
    def __init__(self, epochs=10, batch_size=1024, verbose=0):
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, X, y):
        raise NotImplementedError("Subclasses must implement fit method")

    def predict(self, X):
        raise NotImplementedError("Subclasses must implement predict method")

    def predict_proba(self, X):
        raise NotImplementedError("Subclasses must implement predict_proba method")


# BLSTM模型类，继承自BaseDeepModel
class BLSTMModel(BaseDeepModel):
    def __init__(self, input_shape, epochs=10, batch_size=1024, verbose=0):
        super(BLSTMModel, self).__init__(epochs, batch_size, verbose)
        self.input_shape = input_shape
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        # 使用双向LSTM层，这里可以根据实际情况调整参数，如units数量等
        model.add(Bidirectional(LSTM(16, return_sequences=True), input_shape=self.input_shape))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(8)))
        model.add(Dropout(0.2))
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