from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM
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


# CNN模型类，继承自BaseDeepModel
class CNNModel(BaseDeepModel):
    def __init__(self, input_shape, epochs=10, batch_size=1024, verbose=0, dropout_rate=0.3):
        super(CNNModel, self).__init__(epochs, batch_size, verbose)
        self.input_shape = input_shape
        self.dropout_rate = dropout_rate
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        # 添加卷积层和池化层
        model.add(Conv1D(8, kernel_size=3, activation='relu', input_shape=self.input_shape, padding='same'))
        model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
        model.add(Dropout(self.dropout_rate))  # 添加Dropout以减少过拟合

        # 再添加一组卷积层和池化层
        model.add(Conv1D(16, kernel_size=3, activation='relu', padding='same'))
        model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
        model.add(Dropout(self.dropout_rate))

        # 展平
        model.add(Flatten())

        # 添加全连接层
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(1, activation='sigmoid'))  # 输出层

        # 编译模型
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=0.001),
            metrics=['accuracy']
        )
        return model


    def fit(self, X, y):
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        return self

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype(int).reshape(-1,)

    def predict_proba(self, X):
        return self.model.predict(X)
