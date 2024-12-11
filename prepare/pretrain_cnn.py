import numpy as np
import catboost as cb
from sklearn.metrics import accuracy_score
from fs.encode1 import ENAC2, binary, NCP, EIIP
from prepare.prepare_ml import load_data, ml_code
from ensemble1.deepmodels import CNNModel
import pandas as pd


# 加载数据并进行特征编码，整合为训练集和测试集特征矩阵以及对应的标签
def load_and_encode_data(dataset_name):
    """
    根据给定的数据集名称（物种名称）加载对应文件夹下的数据文件，
    并使用指定的特征编码方法（ENAC、binary、NCP、EIIP）进行编码，
    返回编码后的训练集特征、测试集特征以及对应的标签。
    """
    # 使用load_data函数获取训练集和测试集数据
    data_train, data_test = load_data(dataset_name)

    feature_extract_method = {
        "ENAC": ENAC2,
        "binary": binary,
        "NCP": NCP,
        "EIIP": EIIP
    }

    X_train, y_train, _ = ml_code(data_train, feature_extract_method=feature_extract_method)
    X_test, y_test, _ = ml_code(data_test, feature_extract_method=feature_extract_method)

    return X_train, X_train.shape[1], X_test, y_train, y_test


# 训练CNN模型
def train_cnn_model(cnn_model, X_train, X_test, y_train, y_test, num_epochs):
    history = cnn_model.model.fit(X_train, y_train, epochs=num_epochs, batch_size=cnn_model.batch_size,
                                  verbose=cnn_model.verbose, validation_data=(X_test, y_test))
    epoch_accuracies = []
    epoch_losses = []
    for epoch in range(num_epochs):
        epoch_accuracies.append(history.history['accuracy'][epoch])  # 获取训练集上每一轮的准确率
        epoch_losses.append(history.history['loss'][epoch])  # 获取训练集上每一轮的损失值
        print(
            f'Epoch {epoch + 1}/{num_epochs}, Accuracy: {history.history["accuracy"][epoch]}, Loss: {history.history["loss"][epoch]}')
    return cnn_model, epoch_accuracies, epoch_losses  # 返回模型以及收集的准确率和损失值列表


# 使用CatBoost进行进一步训练并评估，输入变为CNN模型的输出特征，并收集更多指标
def train_catboost_model(model, X_train, X_test, y_train, y_test):
    # 获取CNN模型输出的特征作为CatBoost的输入特征
    model_output_train = model.predict(X_train).reshape(-1, 1)
    model_output_test = model.predict(X_test).reshape(-1, 1)

    # 配置CatBoost分类器
    catboost_model = cb.CatBoostClassifier(iterations=50, depth=6, learning_rate=0.1, loss_function='Logloss')

    # 使用训练数据训练CatBoost模型，并记录训练过程信息
    catboost_model.fit(model_output_train, y_train, eval_set=(model_output_test, y_test), plot=False,
                       verbose=False)  # 显示训练过程是否可视化等信息
    # 获取训练后的Logloss值
    logloss_value = catboost_model.get_best_score()['validation']['Logloss']
    # 在测试集上评估模型性能（准确率）
    y_pred = catboost_model.predict(model_output_test)
    accuracy = accuracy_score(y_pred, y_test)
    print(f'Accuracy on test set: {accuracy}, Logloss: {logloss_value}')

    return catboost_model, accuracy, logloss_value  # 返回模型以及准确率和Logloss值


if __name__ == "__main__":
    dataset_names = ['4mC_C.equisetifolia', '4mC_F.vesca', '4mC_S.cerevisiae']
    summary_data = []
    for dataset_name in dataset_names:
        print(f"开始训练物种: {dataset_name}")
        # 加载并编码数据
        X_train, _, X_test, y_train, y_test = load_and_encode_data(dataset_name)
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)

        # 为数据添加表示序列长度的维度，序列长度=卷积核大小=3
        X_train = np.expand_dims(X_train, axis=1)
        X_train = np.repeat(X_train, 3, axis=1)
        X_test = np.expand_dims(X_test, axis=1)
        X_test = np.repeat(X_test, 3, axis=1)

        # 创建CNN模型实例，传递正确的输入形状参数
        cnn_model = CNNModel(input_shape=(3, X_train.shape[2]))
        num_epochs_cnn = 10  # CNN模型训练的轮数，初步设为10

        # 训练CNN模型以及收集更多指标信息
        trained_cnn_model, cnn_epoch_accuracies, cnn_epoch_losses = train_cnn_model(cnn_model, X_train, X_test,
                                                                                    y_train, y_test, num_epochs_cnn)

        # 使用CatBoost进行进一步训练并收集更多指标信息
        trained_catboost_model, catboost_accuracy, catboost_logloss = train_catboost_model(trained_cnn_model,
                                                                                            X_train, X_test,
                                                                                            y_train, y_test)
        # 收集每个物种的相关性能指标信息到字典中
        species_info = {
            'Species': dataset_name,
            'CNN_Mean_Accuracy': np.mean(cnn_epoch_accuracies),  # 计算CNN模型训练各轮次准确率的平均值
            'CNN_Mean_Loss': np.mean(cnn_epoch_losses),  # 计算CNN模型训练各轮次损失值的平均值
            'CatBoost_Accuracy': catboost_accuracy,
            'CatBoost_Logloss': catboost_logloss
        }
        summary_data.append(species_info)

    # 使用pandas创建DataFrame并展示汇总表
    summary_df = pd.DataFrame(summary_data)
    print(summary_df)