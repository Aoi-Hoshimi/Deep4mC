import numpy as np
import catboost as cb
import tensorflow as tf
from sklearn.metrics import accuracy_score
import pandas as pd
import time
import os
import sys
import gc

# 获取当前脚本所在的目录（即 prepare 目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取当前脚本所在目录的上一级目录（也就是包含 fs 的 Deep4mC-V2 目录）
project_dir = os.path.dirname(current_dir)
# 将项目目录添加到模块搜索路径中
sys.path.append(project_dir)

from fs.encode1 import ENAC2, binary, NCP, EIIP
from prepare.prepare_ml import load_data, ml_code, read_fasta_data
from ensemble1.deepmodels import CNNModel


# 封装数据加载和编码函数，使其更通用，方便替换不同数据集
def load_and_encode_data(dataset_path, feature_extract_methods):
    """
    根据给定的数据集路径和特征提取方法字典加载并编码数据，
    返回编码后的训练集特征、测试集特征以及对应的标签。
    """
    data_train, data_test = load_data(dataset_path)
    X_train, y_train, _ = ml_code(data_train, feature_extract_method=feature_extract_methods)
    X_test, y_test, _ = ml_code(data_test, feature_extract_method=feature_extract_methods)
    return X_train, X_train.shape[1], X_test, y_train, y_test


def load_and_encode_fasta_data(dataset):
    """
    根据给定的数据集名称（对应文件夹名称，其下包含划分好的FASTA格式训练集和测试集文件）加载并编码数据，
    返回编码后的训练集特征、测试集特征以及对应的标签。

    参数:
    dataset: 数据集名称，对应着存放FASTA格式数据文件的文件夹名称，该文件夹下需包含train_pos.txt、train_neg.txt、test_pos.txt、test_neg.txt文件

    返回:
    X_train: 编码后的训练集特征
    X_train.shape[1]: 特征数量（用于后续模型输入形状等设置）
    X_test: 编码后的测试集特征
    y_train: 训练集标签
    y_test: 测试集标签
    """
    base_dir = '/root/autodl-tmp/Deep4mC-V2/data/4mC'  # 数据集所在的绝对路径
    train_pos = os.path.join(base_dir, dataset, "train_pos.txt")
    train_neg = os.path.join(base_dir, dataset, "train_neg.txt")
    test_pos = os.path.join(base_dir, dataset, "test_pos.txt")
    test_neg = os.path.join(base_dir, dataset, "test_neg.txt")

    # 读取训练集正样本数据
    data_train_pos = read_fasta_data(train_pos)
    # 读取训练集负样本数据
    data_train_neg = read_fasta_data(train_neg)
    # 读取测试集正样本数据
    data_test_pos = read_fasta_data(test_pos)
    # 读取测试集负样本数据
    data_test_neg = read_fasta_data(test_neg)

    train_seq = data_train_pos + data_train_neg
    train_seq_id = [1] * len(data_train_pos) + [0] * len(data_train_neg)
    data_train = pd.DataFrame({"label": train_seq_id, "seq": train_seq})

    test_seq = data_test_pos + data_test_neg
    test_seq_id = [1] * len(data_test_pos) + [0] * len(data_test_neg)
    data_test = pd.DataFrame({"label": test_seq_id, "seq": test_seq})

    # 使用ml_code函数将训练集和测试集数据转换为特征矩阵
    X_train, y_train, _ = ml_code(data_train)
    X_test, y_test, _ = ml_code(data_test)

    return X_train, X_train.shape[1], X_test, y_train, y_test


# 训练CNN模型
def train_cnn_model(cnn_model, X_train, X_test, y_train, y_test, num_epochs, gradient_accumulation_steps=4):
    epoch_accuracies = []
    epoch_losses = []
    val_losses = []
    epoch_times = []

    optimizer = cnn_model.model.optimizer
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    for epoch in range(num_epochs):
        start_time = time.time()

        # 初始化累积梯度
        accumulated_gradients = [
            tf.Variable(tf.zeros_like(var), trainable=False) for var in cnn_model.model.trainable_variables
        ]
        epoch_loss = 0.0
        epoch_accuracy = tf.keras.metrics.BinaryAccuracy()

        # 训练循环
        with tf.GradientTape() as tape:
            predictions = cnn_model.model(X_train, training=True)
            loss = loss_fn(tf.expand_dims(y_train, axis=-1), predictions)
            epoch_loss = loss.numpy()

        gradients = tape.gradient(loss, cnn_model.model.trainable_variables)

        # 累积梯度（梯度直接应用到整个数据集）
        for i in range(len(gradients)):
            accumulated_gradients[i].assign_add(gradients[i])

        optimizer.apply_gradients(
            zip([grad / gradient_accumulation_steps for grad in accumulated_gradients], cnn_model.model.trainable_variables)
        )

        # 清零累积梯度
        for grad in accumulated_gradients:
            grad.assign(tf.zeros_like(grad))

        # 更新训练准确率
        epoch_accuracy.update_state(tf.expand_dims(y_train, axis=-1), predictions)

        # 验证阶段
        val_predictions = cnn_model.model(X_test, training=False)
        val_loss = loss_fn(tf.expand_dims(y_test, axis=-1), val_predictions).numpy()
        val_accuracy = tf.keras.metrics.BinaryAccuracy()
        val_accuracy.update_state(tf.expand_dims(y_test, axis=-1), val_predictions)

        # 记录每个epoch的训练时间
        end_time = time.time()
        epoch_time = end_time - start_time
        epoch_times.append(epoch_time)

        # 记录指标
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy.result().numpy())
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Time: {epoch_time:.2f}s, "
            f"Train Loss: {epoch_losses[-1]:.4f}, Train Accuracy: {epoch_accuracies[-1]:.4f}, "
            f"Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accuracy.result().numpy():.4f}"
        )

    return cnn_model, epoch_accuracies, epoch_losses, val_losses, epoch_times



# 使用CatBoost进行进一步训练并评估
def train_catboost_model(model, X_train, X_test, y_train, y_test):
    # 获取CNN模型输出的特征作为CatBoost的输入特征
    model_output_train = model.predict(X_train).reshape(-1, 1)
    model_output_test = model.predict(X_test).reshape(-1, 1)

    # 使用完后将临时变量删除，释放内存
    del X_train
    del X_test

    # 配置CatBoost分类器
    catboost_model = cb.CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, loss_function='Logloss')

    # 使用训练数据训练CatBoost模型，并记录训练过程信息
    catboost_model.fit(model_output_train, y_train, eval_set=(model_output_test, y_test), plot=False,
                       verbose=False)  # 显示训练过程是否可视化等信息
    # 获取训练后的Logloss值
    logloss_value = catboost_model.get_best_score()['validation']['Logloss']
    # 在测试集上评估模型性能（准确率）
    y_pred = catboost_model.predict(model_output_test)
    accuracy = accuracy_score(y_pred, y_test)
    print(f'Accuracy on test set: {accuracy}, Logloss: {logloss_value})')

    # 使用完后将临时变量删除，释放内存
    del model_output_train
    del model_output_test

    return catboost_model, accuracy, logloss_value  # 返回模型以及准确率和Logloss值


# 保存预训练模型
def save_pretrained_model(model, save_path):
    model.model.save(save_path)


if __name__ == "__main__":
    dataset_paths = [] # 纯序列数据集
    fasta_dataset_names = ['4mC_A.thaliana', '4mC_C.elegans', '4mC_D.melanogaster', '4mC_E.coli', '4mC_G.subterraneus',
                           '4mC_G.pickeringii']  # FASTA格式数据集
    #fasta_dataset_names = ['4mC_E.coli']
    feature_extract_methods = {
        "ENAC": ENAC2,
        "binary": binary,
        "NCP": NCP,
        "EIIP": EIIP
    }
    summary_data = []
    # 处理纯序列数据
    for dataset_path in dataset_paths:
        print(f"开始训练数据集: {dataset_path}")
        X_train, _, X_test, y_train, y_test = load_and_encode_data(dataset_path, feature_extract_methods)
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)

        # 为数据添加表示序列长度的维度（将序列长度设为3，以满足卷积核大小为3时的维度计算要求）
        X_train = np.expand_dims(X_train, axis=1)
        X_train = np.repeat(X_train, 3, axis=1)
        X_test = np.expand_dims(X_test, axis=1)
        X_test = np.repeat(X_test, 3, axis=1)

        # 创建CNN模型实例，传递正确的输入形状参数，sequence_length设为3
        cnn_model = CNNModel(input_shape=(100, 1), epochs=10, batch_size=1024, verbose=1)
        num_epochs_cnn = 100  # CNN模型训练的轮数

        # 训练CNN模型以及收集更多指标信息
        trained_cnn_model, cnn_epoch_accuracies, cnn_epoch_losses, val_losses, epoch_times = train_cnn_model(
            cnn_model, X_train, X_test, y_train, y_test, num_epochs_cnn)

        # 使用CatBoost进行进一步训练并收集更多指标信息
        trained_catboost_model, catboost_accuracy, catboost_logloss = train_catboost_model(trained_cnn_model,
                                                                                           X_train, X_test,
                                                                                           y_train, y_test)
        # 收集每个数据集的相关性能指标信息到字典中
        dataset_info = {
            'Dataset': dataset_path,
            'CNN_Mean_Accuracy': np.mean(cnn_epoch_accuracies),  # 计算CNN模型训练各轮次准确率的平均值
            'CNN_Mean_Loss': np.mean(cnn_epoch_losses),  # 计算CNN模型训练各轮次损失值的平均值
            'CNN_Mean_Val_Loss': np.mean(val_losses),  # 新增计算验证集平均损失值
            'CNN_Mean_Epoch_Time': np.mean(epoch_times),  # 新增计算平均每轮训练时间
            'CatBoost_Accuracy': catboost_accuracy,
            'CatBoost_Logloss': catboost_logloss
        }
        summary_data.append(dataset_info)

        # 保存预训练的CNN模型
        save_path = os.path.join('pretrained_models', f'cnn_{os.path.basename(dataset_path)}.h5')
        save_pretrained_model(trained_cnn_model, save_path)

    # 处理FASTA格式的数据
    for fasta_dataset_name in fasta_dataset_names:
        print(f"开始训练FASTA数据集: {fasta_dataset_name}")
        X_train, _, X_test, y_train, y_test = load_and_encode_fasta_data(fasta_dataset_name)
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)

        # 为数据添加表示序列长度的维度
        X_train = np.expand_dims(X_train, axis=1)
        X_train = np.repeat(X_train, 3, axis=1)
        X_test = np.expand_dims(X_test, axis=1)
        X_test = np.repeat(X_test, 3, axis=1)

        # 创建CNN模型实例，传递正确的输入形状参数
        cnn_model = CNNModel(input_shape=(3, X_train.shape[2]))
        num_epochs_cnn = 100

        # 训练CNN模型以及收集更多指标信息
        trained_cnn_model, cnn_epoch_accuracies, cnn_epoch_losses, val_losses, epoch_times = train_cnn_model(
            cnn_model, X_train, X_test, y_train, y_test, num_epochs_cnn)

        # 使用CatBoost进行进一步训练并收集更多指标信息
        trained_catboost_model, catboost_accuracy, catboost_logloss = train_catboost_model(trained_cnn_model,
                                                                                           X_train, X_test,
                                                                                           y_train, y_test)

        # 收集性能指标信息到字典
        dataset_info = {
            'Dataset': fasta_dataset_name,
            'CNN_Mean_Accuracy': np.mean(cnn_epoch_accuracies),
            'CNN_Mean_Loss': np.mean(cnn_epoch_losses),
            'CNN_Mean_Val_Loss': np.mean(val_losses),
            'CNN_Mean_Epoch_Time': np.mean(epoch_times),
            'CatBoost_Accuracy': catboost_accuracy,
            'CatBoost_Logloss': catboost_logloss
        }
        summary_data.append(dataset_info)

        # 保存预训练的CNN模型
        save_path = os.path.join('pretrained_models', f'cnn_{os.path.basename(fasta_dataset_name)}.h5')
        save_pretrained_model(trained_cnn_model, save_path)

    # 使用pandas创建DataFrame汇总表
    summary_df = pd.DataFrame(summary_data)
    print(summary_df)