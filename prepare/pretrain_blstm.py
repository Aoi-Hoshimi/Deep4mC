import numpy as np
import tensorflow as tf
import catboost as cb
from sklearn.metrics import accuracy_score
import pandas as pd
import time
import os
import sys
import gc

"""
# 设置GPU显存动态分配
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
"""

# 开启XLA编译优化
tf.config.optimizer.set_jit(True)

# 获取当前脚本所在的目录（prepare）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取当前脚本所在目录的上一级目录
project_dir = os.path.dirname(current_dir)
# 将项目目录添加到模块搜索路径中
sys.path.append(project_dir)

from fs.encode1 import ENAC2, binary, NCP, EIIP
from prepare.prepare_ml import load_data, ml_code, read_fasta_data
from ensemble1.BLSTM import BLSTMModel



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


# 训练BLSTM模型
def train_blstm_model(blstm_model, X_train, X_test, y_train, y_test, num_epochs):
    epoch_accuracies = []
    epoch_losses = []
    val_losses = []  # 用于记录验证集损失值的列表
    epoch_times = []  # 用于记录每轮训练时间的列表
    gradient_accumulation_steps = 4  # 设置梯度累积的步数，这里表示每4个小批次累积一次梯度更新
    # 将列表推导式中的tf.zeros_like(var)改为tf.Variable(tf.zeros_like(var))，创建Variable类型对象
    accumulated_gradients = [tf.Variable(tf.zeros_like(var)) for var in blstm_model.model.trainable_variables]

    for epoch in range(num_epochs):
        start_time = time.time()  # 记录每轮训练开始时间
        for step in range(len(X_train) // blstm_model.batch_size):
            current_batch_x = X_train[step * blstm_model.batch_size: (step + 1) * blstm_model.batch_size]
            current_batch_y = y_train[step * blstm_model.batch_size: (step + 1) * blstm_model.batch_size]
            current_batch_y = np.expand_dims(current_batch_y, axis=-1)  # 将一维标签数据扩展为二维，列向量形式

            with tf.GradientTape() as tape:
                predictions = blstm_model.model(current_batch_x)
                loss = tf.keras.losses.binary_crossentropy(current_batch_y, predictions)
                loss = tf.reduce_mean(loss)

            gradients = tape.gradient(loss, blstm_model.model.trainable_variables)

            for i in range(len(gradients)):
                accumulated_gradients[i].assign_add(gradients[i])  # 使用assign_add方法来累加梯度

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer = blstm_model.model.optimizer
                optimizer.apply_gradients(zip([grad for grad in accumulated_gradients], blstm_model.model.trainable_variables))
                for i in range(len(accumulated_gradients)):
                    accumulated_gradients[i].assign(tf.zeros_like(accumulated_gradients[i]))  # 重置为零

            # 每一个小批次结束后记录相关指标（可调整记录频率）
            if step % (len(X_train) // blstm_model.batch_size // 10) == 0:  # 设置每1/10个完整轮次记录一次
                history = blstm_model.model.fit(X_train, y_train, epochs=epoch + 1, initial_epoch=epoch,
                                                batch_size=blstm_model.batch_size,
                                                verbose=blstm_model.verbose, validation_data=(X_test, y_test))
                epoch_accuracies.append(history.history['accuracy'][0])
                epoch_losses.append(history.history['loss'][0])
                val_losses.append(history.history['val_loss'][0])

        end_time = time.time()  # 记录每轮训练结束时间
        epoch_time = end_time - start_time  # 计算每轮训练耗时
        epoch_times.append(epoch_time)
        print(
            f'Epoch {epoch + 1}/{num_epochs}, Training Time: {epoch_time:.2f}s, Loss: {history.history["loss"][0]:.4f}, Val Loss: {history.history["val_loss"][0]:.4f}')

        # 手动触发垃圾回收
        gc.collect()

    return blstm_model, epoch_accuracies, epoch_losses, val_losses, epoch_times  # 返回模型以及收集的准确率、损失值、验证集损失值和训练时间列表


# 使用CatBoost进行进一步训练并评估
def train_catboost_model(model, X_train, X_test, y_train, y_test):
    # 获取BLSTM模型输出的特征作为CatBoost的输入特征
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
    print(f'Accuracy on test set: {accuracy}, Logloss: {logloss_value}')

    # 使用完后将临时变量删除，释放内存
    del model_output_train
    del model_output_test

    return catboost_model, accuracy, logloss_value  # 返回模型以及准确率和Logloss值


# 保存预训练模型
def save_pretrained_model(model, save_path):
    model.model.save(save_path)


if __name__ == "__main__":

    # 既有纯序列格式，又有FASTA格式的数据，分别列出对应数据集名称（对应文件夹名称）
    dataset_paths = []
    fasta_dataset_names = ['4mC_A.thaliana','4mC_C.elegans','4mC_D.melanogaster','4mC_E.coli','4mC_G.subterraneus','4mC_G.pickeringii']
    feature_extract_methods = {
        "ENAC": ENAC2,
        "binary": binary,
        "NCP": NCP,
        "EIIP": EIIP
    }
    summary_data = []
    # 处理纯序列格式的数据
    for dataset_path in dataset_paths:
        print(f"开始训练数据集: {dataset_path}")
        X_train, _, X_test, y_train, y_test = load_and_encode_data(dataset_path, feature_extract_methods)
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)

        # 为数据添加表示序列长度的维度
        X_train = np.expand_dims(X_train, axis=1)
        X_train = np.repeat(X_train, 3, axis=1)
        X_test = np.expand_dims(X_test, axis=1)
        X_test = np.repeat(X_test, 3, axis=1)

        # 创建BLSTM模型实例，传递正确的输入形状参数，sequence_length设为3
        blstm_model = BLSTMModel(input_shape=(3, X_train.shape[2]))
        num_epochs_blstm = 100  # BLSTM模型训练的轮数

        # 训练BLSTM模型以及收集更多指标信息
        trained_blstm_model, blstm_epoch_accuracies, blstm_epoch_losses, val_losses, epoch_times = train_blstm_model(
            blstm_model, X_train, X_test, y_train, y_test, num_epochs_blstm)

        # 使用CatBoost进行进一步训练并收集更多指标信息
        trained_catboost_model, catboost_accuracy, catboost_logloss = train_catboost_model(trained_blstm_model,
                                                                                           X_train, X_test,
                                                                                           y_train, y_test)
        # 收集每个数据集的相关性能指标信息到字典中
        dataset_info = {
            'Dataset': dataset_path,
            'BLSTM_Mean_Accuracy': np.mean(blstm_epoch_accuracies),  # 计算BLSTM模型训练各轮次准确率的平均值
            'BLSTM_Mean_Loss': np.mean(blstm_epoch_losses),  # 计算BLSTM模型训练各轮次损失值的平均值
            'BLSTM_Mean_Val_Loss': np.mean(val_losses),  # 计算验证集平均损失值
            'BLSTM_Mean_Epoch_Time': np.mean(epoch_times),  # 计算平均每轮训练时间
            'CatBoost_Accuracy': catboost_accuracy,
            'CatBoost_Logloss': catboost_logloss
        }
        summary_data.append(dataset_info)

        # 保存预训练的BLSTM模型
        save_path = os.path.join('pretrained_models', f'blstm_{os.path.basename(dataset_path)}.h5')
        save_pretrained_model(trained_blstm_model, save_path)

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

        # 创建BLSTM模型实例，传递正确的输入形状参数
        blstm_model = BLSTMModel(input_shape=(3, X_train.shape[2]))
        num_epochs_blstm = 100

        # 训练BLSTM模型以及收集更多指标信息
        trained_blstm_model, blstm_epoch_accuracies, blstm_epoch_losses, val_losses, epoch_times = train_blstm_model(
            blstm_model, X_train, X_test, y_train, y_test, num_epochs_blstm)

        # 使用CatBoost进行进一步训练并收集更多指标信息
        trained_catboost_model, catboost_accuracy, catboost_logloss = train_catboost_model(trained_blstm_model,
                                                                                           X_train, X_test,
                                                                                           y_train, y_test)

        # 收集性能指标信息到字典
        dataset_info = {
            'Dataset': fasta_dataset_name,
            'BLSTM_Mean_Accuracy': np.mean(blstm_epoch_accuracies),
            'BLSTM_Mean_Loss': np.mean(blstm_epoch_losses),
            'BLSTM_Mean_Val_Loss': np.mean(val_losses),
            'BLSTM_Mean_Epoch_Time': np.mean(epoch_times),
            'CatBoost_Accuracy': catboost_accuracy,
            'CatBoost_Logloss': catboost_logloss
        }
        summary_data.append(dataset_info)

        # 保存预训练的BLSTM模型
        save_path = os.path.join('pretrained_models', f'blstm_{os.path.basename(fasta_dataset_name)}.h5')
        save_pretrained_model(trained_blstm_model, save_path)

    # 使用pandas创建DataFrame汇总表
    summary_df = pd.DataFrame(summary_data)
    print(summary_df)
