import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from ensemble1.deepmodels import CNNModel
from prepare.pretrain_cnn import load_and_encode_data as cnn_load_and_encode_data
from prepare.pretrain_transformer import load_and_encode_data as transformer_load_and_encode_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from fs.encode1 import ENAC2, binary, NCP, EIIP
from tensorflow.keras.regularizers import l2
import os


class StackingCNNTransformerModel():
    def __init__(self, n_folds=5):
        self.n_folds = n_folds
        self.cnn_models = []# 存储CNN模型实例
        self.transformer_models = []# 存储Transformer模型实例
        # 元模型使用MLP，简单定义一个两层结构，可进一步调整结构、神经元数量
        self.meta_model = Sequential([
            Dense(64, activation='relu', input_dim=2, kernel_regularizer=l2(0.001)),  # 添加L2正则化，系数可调整
            Dense(1, activation='sigmoid')
        ])
        self.validate_prediction = None
        self.validate_real_label = None
        self.test_prediction = None
        self.test_real_label = None

    def fit(self, dataset_path, feature_extract_methods):
        """
        整体训练逻辑，包括加载预训练的CNN和Transformer模型、收集预测结果并训练元模型。
        """
        # 加载并编码CNN模型所需数据，获取数据形状等信息以及标签
        cnn_X_train, _, cnn_X_test, cnn_y_train, cnn_y_test = cnn_load_and_encode_data(dataset_path, feature_extract_methods)
        cnn_X_train = cnn_X_train.astype(np.float32)
        cnn_X_test = cnn_X_test.astype(np.float32)
        # 为数据添加表示序列长度的维度，以适配CNN卷积核计算要求
        cnn_X_train = np.expand_dims(cnn_X_train, axis=1)
        cnn_X_train = np.repeat(cnn_X_train, 3, axis=1)
        cnn_X_test = np.expand_dims(cnn_X_test, axis=1)
        cnn_X_test = np.repeat(cnn_X_test, 3, axis=1)

        # 加载并编码Transformer模型所需数据，获取数据形状等信息以及标签
        transformer_X_train, _, transformer_X_test, transformer_y_train, transformer_y_test = transformer_load_and_encode_data(dataset_path)
        transformer_X_train = transformer_X_train.astype(np.float32)
        transformer_X_test = transformer_X_test.astype(np.float32)

        # 存储每次fold中CNN和Transformer模型在验证集上的预测结果
        base_models_5fold_prediction = np.zeros((cnn_X_train.shape[0], 2))

        skf = list(StratifiedKFold(n_splits=self.n_folds).split(cnn_X_train, cnn_y_train))
        base_dir = r'E:\4mC\DeepSF-4mC-V2\prepare\pretrained_models' #我设备上的项目文件基目录，需要根据实际调整
        for i, (train_index, val_index) in enumerate(skf):
            # 加载预训练的CNN模型
            cnn_model_path = os.path.join(base_dir, f'cnn_{os.path.basename(dataset_path)}.h5')
            cnn_model = CNNModel(input_shape=(3, cnn_X_train.shape[2]))
            cnn_model.model.load_weights(cnn_model_path)
            self.cnn_models.append(cnn_model)

            cnn_pred_val = cnn_model.predict_proba(cnn_X_train[val_index])
            cnn_acc_val = np.mean((cnn_pred_val.reshape(-1) > 0.5).astype(int) == cnn_y_train[val_index])  # 计算CNN模型在验证集上的准确率

            # 加载预训练的Transformer模型
            transformer_model_path = os.path.join(base_dir, f'transformer_{os.path.basename(dataset_path)}.h5')
            transformer_model = tf.keras.models.load_model(transformer_model_path)
            transformer_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, decay=0.01),
                                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
            self.transformer_models.append(transformer_model)

            transformer_pred_val = transformer_model(
                tf.expand_dims(tf.convert_to_tensor(transformer_X_train[val_index], dtype=tf.float32), axis=0)).numpy()
            transformer_pred_val = transformer_pred_val.squeeze(0)
            transformer_acc_val = np.mean((transformer_pred_val.mean(axis=1) > 0.5).astype(int) == cnn_y_train[val_index])  # 计算Transformer模型在验证集上的准确率

            # 根据准确率动态计算权重，这里简单使用准确率占比作为权重
            total_acc = cnn_acc_val + transformer_acc_val
            cnn_weight = cnn_acc_val / total_acc if total_acc > 0 else 0.5  # 避免除0情况，设置默认权重
            transformer_weight = transformer_acc_val / total_acc if total_acc > 0 else 0.5

            base_models_5fold_prediction[val_index, 0] = cnn_pred_val.reshape(-1) * cnn_weight
            base_models_5fold_prediction[val_index, 1] = transformer_pred_val.mean(axis=1) * transformer_weight

        # 增加模型输出特征处理，归一化
        scaler = MinMaxScaler()
        base_models_5fold_prediction = scaler.fit_transform(base_models_5fold_prediction)

        # 训练元模型MLP，添加编译步骤，指定损失函数和优化器等，可以进一步调整优化训练轮数、批次大小等超参数
        self.meta_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        self.meta_model.fit(base_models_5fold_prediction, cnn_y_train, epochs=100, batch_size=32, verbose=1)  # 可调整训练轮数、批次大小等超参数

        self.validate_prediction = base_models_5fold_prediction
        self.validate_real_label = cnn_y_train
        return self

    def predict(self, dataset_path, feature_extract_methods):
        """
        使用训练好的模型（这里主要是指加载的预训练模型以及训练好的元模型）进行预测，
        包括先让CNN和Transformer模型分别预测，然后用元模型整合结果进行最终预测。
        同时获取测试集的真实标签用于后续准确率计算。
        """
        # 加载并编码测试集数据（CNN部分）
        cnn_X_train, _, cnn_X_test, cnn_y_train, cnn_y_test = cnn_load_and_encode_data(dataset_path, feature_extract_methods)
        cnn_X_test = cnn_X_test.astype(np.float32)
        cnn_X_test = np.expand_dims(cnn_X_test, axis=1)
        cnn_X_test = np.repeat(cnn_X_test, 3, axis=1)

        # 加载并编码测试集数据（Transformer部分）
        transformer_X_train, _, transformer_X_test, transformer_y_train, transformer_y_test = transformer_load_and_encode_data(dataset_path)
        transformer_X_test = transformer_X_test.astype(np.float32)

        base_models_ind_prediction = np.zeros((cnn_X_test.shape[0], 2))
        for i in range(len(self.cnn_models)):
            cnn_pred_test = self.cnn_models[i].predict_proba(cnn_X_test)
            base_models_ind_prediction[:, 0] += cnn_pred_test.reshape(-1)

            transformer_pred_test = self.transformer_models[i](tf.expand_dims(tf.convert_to_tensor(transformer_X_test, dtype=tf.float32), axis=0)).numpy()
            transformer_pred_test = transformer_pred_test.squeeze(0)
            base_models_ind_prediction[:, 1] += transformer_pred_test.mean(axis=1)

        scaler = MinMaxScaler()
        base_models_ind_prediction = scaler.fit_transform(base_models_ind_prediction)

        y_pred_prob = self.meta_model.predict(base_models_ind_prediction)  # MLP预测输出，注意这里和逻辑回归用法不同，直接得到预测概率
        y_pred = (y_pred_prob > 0.5).astype(int)  # 将概率转换为类别标签，以0.5为阈值

        # 获取测试集真实标签，统一使用CNN数据加载中的标签
        self.test_real_label = cnn_y_test
        self.test_prediction = y_pred
        self.test_prediction = self.test_prediction.squeeze()

        return y_pred


if __name__ == "__main__":
    dataset_paths = ['4mC_C.equisetifolia', '4mC_S.cerevisiae']
    feature_extract_methods = {
        "ENAC": ENAC2,
        "binary": binary,
        "NCP": NCP,
        "EIIP": EIIP
    }
    summary_data = []
    for dataset_path in dataset_paths:
        print(f"开始训练数据集: {dataset_path}")
        stacking_model = StackingCNNTransformerModel()
        stacking_model.fit(dataset_path, feature_extract_methods)
        y_pred = stacking_model.predict(dataset_path, feature_extract_methods)

        correct_count = np.sum(stacking_model.test_real_label == stacking_model.test_prediction)
        accuracy = correct_count / len(stacking_model.test_real_label)
        dataset_info = {
            'Dataset': dataset_path,
            'Stacking_Accuracy': accuracy
        }
        summary_data.append(dataset_info)

    # 用pandas创建DataFrame展示汇总表
    summary_df = pd.DataFrame(summary_data)
    print(summary_df)