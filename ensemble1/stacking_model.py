import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import os
import sys

# 获取当前脚本所在的目录（prepare）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取当前脚本所在目录的上一级目录
project_dir = os.path.dirname(current_dir)
# 将项目目录添加到模块搜索路径中
sys.path.append(project_dir)

from ensemble1.deepmodels import CNNModel
from ensemble1.BLSTM import BLSTMModel
from prepare.pretrain_blstm import load_and_encode_fasta_data as blstm_load_and_encode_data
from prepare.pretrain_cnn import load_and_encode_fasta_data as cnn_load_and_encode_data
from fs.encode1 import ENAC2, binary, NCP, EIIP


class StackingCNNBLSTMModel():
    def __init__(self, n_folds=5):
        self.n_folds = n_folds
        self.cnn_model = None
        self.blstm_model = None
        self.meta_model = Sequential([
            Dense(64, activation='relu', input_dim=2, kernel_regularizer=tf.keras.regularizers.l2(0.001)),  # 添加L2正则化，系数可调整
            Dense(1, activation='sigmoid')
        ])
        self.validate_prediction = None
        self.validate_real_label = None
        self.test_prediction = None
        self.test_real_label = None
        self.mlp_mean_accuracy = None
        self.mlp_mean_loss = None

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
        """
        整体训练逻辑，包括加载预训练的CNN和BLSTM模型、收集预测结果并训练元模型。
        """
        self.load_pretrained_models(dataset_path)

        # 加载并编码CNN模型所需数据，获取数据形状等信息以及标签
        cnn_X_train, _, cnn_X_test, cnn_y_train, cnn_y_test = cnn_load_and_encode_data(dataset_path)
        cnn_X_train = cnn_X_train.astype(np.float32)
        cnn_X_test = cnn_X_test.astype(np.float32)
        # 为数据添加表示序列长度的维度，以适配CNN卷积核计算要求
        cnn_X_train = np.expand_dims(cnn_X_train, axis=1)
        cnn_X_train = np.repeat(cnn_X_train, 3, axis=1)
        cnn_X_test = np.expand_dims(cnn_X_test, axis=1)
        cnn_X_test = np.repeat(cnn_X_test, 3, axis=1)

        # 加载并编码BLSTM模型所需数据，获取数据形状等信息以及标签
        blstm_X_train, _, blstm_X_test, blstm_y_train, blstm_y_test = blstm_load_and_encode_data(dataset_path)
        blstm_X_train = blstm_X_train.astype(np.float32)
        blstm_X_test = blstm_X_test.astype(np.float32)
        # 添加表示序列长度的维度，以适配BLSTM输入要求
        blstm_X_train = np.expand_dims(blstm_X_train, axis=1)
        blstm_X_train = np.repeat(blstm_X_train, 3, axis=1)
        blstm_X_test = np.expand_dims(blstm_X_test, axis=1)
        blstm_X_test = np.repeat(blstm_X_test, 3, axis=1)

        # 存储每次fold中CNN和BLSTM模型在验证集上的预测结果
        base_models_5fold_prediction = np.zeros((cnn_X_train.shape[0], 2))

        skf = list(StratifiedKFold(n_splits=self.n_folds).split(cnn_X_train, cnn_y_train))
        for i, (train_index, val_index) in enumerate(skf):
            cnn_pred_val = self.cnn_model.predict(cnn_X_train[val_index])
            cnn_acc_val = accuracy_score(cnn_y_train[val_index], (cnn_pred_val.reshape(-1) > 0.5).astype(int))
            cnn_mse = mean_squared_error(cnn_y_train[val_index], cnn_pred_val.reshape(-1))

            blstm_pred_val = self.blstm_model.predict(blstm_X_train[val_index])
            blstm_acc_val = accuracy_score(cnn_y_train[val_index], (blstm_pred_val.reshape(-1) > 0.5).astype(int))
            blstm_mse = mean_squared_error(cnn_y_train[val_index], blstm_pred_val.reshape(-1))

            # 对准确率进行归一化
            cnn_acc_norm = (cnn_acc_val - np.min([cnn_acc_val, blstm_acc_val])) / (np.max([cnn_acc_val, blstm_acc_val]) - np.min([cnn_acc_val, blstm_acc_val]))
            blstm_acc_norm = (blstm_acc_val - np.min([cnn_acc_val, blstm_acc_val])) / (np.max([cnn_acc_val, blstm_acc_val]) - np.min([cnn_acc_val, blstm_acc_val]))

            # 对均方误差进行归一化（映射到0到1之间，值越小越好，用最大值减去当前值）
            cnn_mse_norm = 1 - (cnn_mse - np.min([cnn_mse, blstm_mse])) / (np.max([cnn_mse, blstm_mse]) - np.min([cnn_mse, blstm_mse]))
            blstm_mse_norm = 1 - (blstm_mse - np.min([cnn_mse, blstm_mse])) / (np.max([cnn_mse, blstm_mse]) - np.min([cnn_mse, blstm_mse]))

            # 通过交叉验证等确定了准确率和损失值的权重分配比例，假设准确率权重为0.6，损失值权重为0.4
            alpha = 0.6
            beta = 0.4
            # 综合计算权重
            cnn_weight = alpha * cnn_acc_norm + beta * cnn_mse_norm
            blstm_weight = alpha * blstm_acc_norm + beta * blstm_mse_norm

            # 归一化权重，确保两者权重和为1
            total_weight = cnn_weight + blstm_weight
            cnn_weight = cnn_weight / total_weight
            blstm_weight = blstm_weight / total_weight

            base_models_5fold_prediction[val_index, 0] = cnn_pred_val.reshape(-1) * cnn_weight
            base_models_5fold_prediction[val_index, 1] = blstm_pred_val.reshape(-1) * blstm_weight

        # 增加模型输出特征处理，归一化
        scaler = MinMaxScaler()
        base_models_5fold_prediction = scaler.fit_transform(base_models_5fold_prediction)

        # 训练元模型MLP，添加编译步骤，指定损失函数和优化器等，可以进一步调整优化训练轮数、批次大小等超参数
        self.meta_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        epoch_losses = []
        epoch_accuracies = []
        for epoch in range(100):
            history = self.meta_model.fit(base_models_5fold_prediction, cnn_y_train, epochs=1, batch_size=1024, verbose=0)
            epoch_loss = history.history['loss'][0]
            epoch_accuracy = history.history['accuracy'][0]
            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_accuracy)
            print(f'Epoch {epoch + 1}: Loss = {epoch_loss}, Accuracy = {epoch_accuracy}')
        self.mlp_mean_loss = np.mean(epoch_losses)
        self.mlp_mean_accuracy = np.mean(epoch_accuracies)
        print("Meta - Mean Loss:", self.mlp_mean_loss)
        print("Meta - Mean Accuracy:", self.mlp_mean_accuracy)
        self.validate_prediction = base_models_5fold_prediction
        self.validate_real_label = cnn_y_train
        return self

    def predict(self, dataset_path, feature_extract_methods):
        """
        使用训练好的模型（这里主要是指加载的预训练模型以及训练好的元模型）进行预测，
        包括先让CNN和BLSTM模型分别预测，然后用元模型整合结果进行最终预测。
        同时获取测试集的真实标签用于后续准确率计算。
        """
        # 加载并编码测试集数据（CNN部分）
        cnn_X_train, _, cnn_X_test, cnn_y_train, cnn_y_test = cnn_load_and_encode_data(dataset_path)
        cnn_X_test = cnn_X_test.astype(np.float32)
        cnn_X_test = np.expand_dims(cnn_X_test, axis=1)
        cnn_X_test = np.repeat(cnn_X_test, 3, axis=1)

        # 加载并编码测试集数据（BLSTM部分）
        blstm_X_train, _, blstm_X_test, blstm_y_train, blstm_y_test = blstm_load_and_encode_data(dataset_path)
        blstm_X_test = blstm_X_test.astype(np.float32)
        blstm_X_test = np.expand_dims(blstm_X_test, axis=1)
        blstm_X_test = np.repeat(blstm_X_test, 3, axis=1)

        base_models_ind_prediction = np.zeros((cnn_X_test.shape[0], 2))

        cnn_pred_test = self.cnn_model.predict(cnn_X_test)
        base_models_ind_prediction[:, 0] = cnn_pred_test.reshape(-1)

        blstm_pred_test = self.blstm_model.predict(blstm_X_test)
        base_models_ind_prediction[:, 1] = blstm_pred_test.reshape(-1)

        scaler = MinMaxScaler()
        base_models_ind_prediction = scaler.fit_transform(base_models_ind_prediction)

        y_pred_prob = self.meta_model.predict(base_models_ind_prediction)  # MLP预测输出，注意这里和逻辑回归用法不同，直接得到预测概率
        y_pred = (y_pred_prob > 0.5).astype(int)  # 将概率转换为类别标签，以0.5为阈值

        # 获取测试集真实标签，统一使用CNN数据加载中的标签
        self.test_real_label = cnn_y_test
        self.test_prediction = y_pred
        self.test_prediction = self.test_prediction.squeeze()

        accuracy = accuracy_score(self.test_real_label, self.test_prediction)
        print(f"Overall accuracy for {dataset_path}: {accuracy}")
        return y_pred


if __name__ == "__main__":
    dataset_paths = ['4mC_A.thaliana', '4mC_C.elegans', '4mC_D.melanogaster', '4mC_E.coli', '4mC_G.subterraneus',
                       '4mC_G.pickeringii']
    #dataset_paths = ['4mC_G.subterraneus']
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

        correct_count = np.sum(stacking_model.test_real_label == stacking_model.test_prediction)
        accuracy = correct_count / len(stacking_model.test_real_label)
        dataset_info = {
            'Dataset': dataset_path,
            'MLP_Mean_Accuracy': stacking_model.mlp_mean_accuracy,
            'MLP_Mean_Loss': stacking_model.mlp_mean_loss,
            'Stacking_Accuracy': accuracy
        }
        summary_data.append(dataset_info)

    # 用pandas创建DataFrame汇总表
    summary_df = pd.DataFrame(summary_data)
    print(summary_df)