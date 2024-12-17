import numpy as np
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
import os
import sys

# 获取当前脚本所在的目录（prepare）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取当前脚本所在目录的上一级目录
project_dir = os.path.dirname(current_dir)
# 将项目目录添加到模块搜索路径中
sys.path.append(project_dir)

from fs.encode1 import ENAC2, binary, NCP, EIIP
from prepare.prepare_ml import load_data, ml_code, read_fasta_data
from ensemble1.deepmodels import CNNModel
from ensemble1.BLSTM import BLSTMModel
from prepare.pretrain_blstm import load_and_encode_fasta_data as blstm_load_and_encode_fasta_data
from prepare.pretrain_cnn import load_and_encode_fasta_data as cnn_load_and_encode_fasta_data


# 基础的模型类，继承自BaseEstimator和ClassifierMixin，用于规范模型的基本行为，使其更符合sklearn风格
class BaseDeepModel:
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


# Stacking集成CNN和BLSTM
class StackingCNNBLSTMModel:
    def __init__(self):
        self.cnn_model = None
        self.blstm_model = None
        self.meta_model = None
        self.test_real_label = None  # 存储测试集真实标签
        self.train_accuracies = []  # 记录每次交叉验证折叠训练的训练准确率
        self.test_accuracies = []  # 记录每次交叉验证折叠在测试集上的准确率
        self.mean_train_accuracy = None  # 记录平均训练准确率
        self.mean_test_accuracy = None  # 记录平均测试准确率

    def load_pretrained_models(self, dataset_path):
        base_dir = '/root/autodl-tmp/Deep4mC-V2/pretrained_models'
        cnn_model_path = os.path.join(base_dir, f'cnn_{os.path.basename(dataset_path)}.h5')
        blstm_model_path = os.path.join(base_dir, f'blstm_{os.path.basename(dataset_path)}.h5')

        self.cnn_model = load_model(cnn_model_path)
        self.blstm_model = load_model(blstm_model_path)

        # 编译模型
        self.cnn_model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
        self.blstm_model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

    def fit(self, dataset_path, feature_extract_methods):
        self.load_pretrained_models(dataset_path)

        X_train, _, X_test, y_train, self.test_real_label = self._load_and_encode_data(dataset_path, feature_extract_methods)

        # 获取CNN和BLSTM模型对训练集的预测结果作为元模型的新特征
        cnn_train_pred = self.cnn_model.predict(X_train).reshape(-1, 1)
        blstm_train_pred = self.blstm_model.predict(X_train).reshape(-1, 1)
        meta_train_features = np.concatenate([cnn_train_pred, blstm_train_pred], axis=1)

        # 配置SVM作为元模型，可调整SVM的参数，比如核函数、C值等
        self.meta_model = SVC(kernel='rbf', C=1.0)

        # 使用分层交叉验证训练SVM元模型，并记录每次折叠训练的训练准确率和在测试集上的准确率
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # 可调整交叉验证的折数和随机种子等参数
        for fold, (train_index, val_index) in enumerate(skf.split(meta_train_features, y_train)):
            X_train_fold, X_val_fold = meta_train_features[train_index], meta_train_features[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

            self.meta_model.fit(X_train_fold, y_train_fold)
            train_pred = self.meta_model.predict(X_val_fold)
            train_accuracy = accuracy_score(y_val_fold, train_pred)
            self.train_accuracies.append(train_accuracy)

            # 在整个测试集上评估当前折叠训练的模型
            test_pred = self.meta_model.predict(meta_train_features)
            test_accuracy = accuracy_score(y_train, test_pred)
            self.test_accuracies.append(test_accuracy)

            print(f'Fold {fold + 1}: Train Accuracy = {train_accuracy}, Test Accuracy = {test_accuracy}')

        self.mean_train_accuracy = np.mean(self.train_accuracies)
        self.mean_test_accuracy = np.mean(self.test_accuracies)

    def predict(self, dataset_path, feature_extract_methods):
        _, _, X_test, _, _ = self._load_and_encode_data(dataset_path, feature_extract_methods)

        # 获取CNN和BLSTM模型对测试集的预测结果作为元模型的输入特征
        cnn_test_pred = self.cnn_model.predict(X_test).reshape(-1, 1)
        blstm_test_pred = self.blstm_model.predict(X_test).reshape(-1, 1)
        meta_test_features = np.concatenate([cnn_test_pred, blstm_test_pred], axis=1)

        # 用元模型进行最终预测
        self.test_prediction = self.meta_model.predict(meta_test_features)
        self.test_accuracy = accuracy_score(self.test_real_label, self.test_prediction)
        return self.test_prediction

    def _load_and_encode_data(self, dataset_path, feature_extract_methods):
        X_train, _, X_test, y_train, y_test = cnn_load_and_encode_fasta_data(dataset_path)
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)

        # 为数据添加表示序列长度的维度（将序列长度设为3，以满足卷积核大小为3时的维度计算要求）
        X_train = np.expand_dims(X_train, axis=1)
        X_train = np.repeat(X_train, 3, axis=1)
        X_test = np.expand_dims(X_test, axis=1)
        X_test = np.repeat(X_test, 3, axis=1)

        return X_train, X_train.shape[1], X_test, y_train, y_test


if __name__ == "__main__":
    #dataset_paths = ['4mC_E.coli']
    dataset_paths = ['4mC_A.thaliana', '4mC_C.elegans', '4mC_D.melanogaster', '4mC_E.coli', '4mC_G.subterraneus',
                     '4mC_G.pickeringii']
    feature_extract_methods = {
        "ENAC": ENAC2,
        "binary": binary,
        "NCP": NCP,
        "EIIP": EIIP
    }
    summary_data = []
    for dataset_path in dataset_paths:
        print(f"开始训练数据集: {dataset_path}")
        stacking_model = StackingCNNBLSTMModel()
        stacking_model.fit(dataset_path, feature_extract_methods)
        y_pred = stacking_model.predict(dataset_path, feature_extract_methods)

        dataset_info = {
            "Dataset": dataset_path,
            "SVM_Mean_Train_Accuracy": stacking_model.mean_train_accuracy,
            "SVM_Mean_Test_Accuracy": stacking_model.mean_test_accuracy,
            "Stacking_Accuracy": stacking_model.test_accuracy
        }
        summary_data.append(dataset_info)

    # 用pandas创建DataFrame汇总表
    summary_df = pd.DataFrame(summary_data)
    print(summary_df)