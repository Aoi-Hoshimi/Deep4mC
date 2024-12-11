import os
import pandas as pd
import tensorflow as tf
import numpy as np
import catboost as cb
from sklearn.metrics import accuracy_score
from ensemble1.Transformer import Transformer
from prepare.prepare_dl import build_vocab, tokenizer, trans, pretrain_one_hot
from prepare.prepare_ml import load_data
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping


# 封装数据加载和编码函数，使其更通用，方便替换不同数据集
def load_and_encode_data(dataset_path):
    """
    根据给定的数据集路径加载对应文件夹下的数据文件，
    进行必要的特征编码等预处理操作，返回编码后的训练集特征、测试集特征以及对应的标签。
    """
    data_train, data_test = load_data(dataset_path)
    train_seqs = data_train['seq'].tolist()
    test_seqs = data_test['seq'].tolist()
    k = 4
    vocab = build_vocab(train_seqs + test_seqs, k)
    print("Vocabulary size:", len(vocab))
    data_train = tokenizer(data_train, k)
    data_test = tokenizer(data_test, k)
    train_data = trans(data_train, vocab)
    test_data = trans(data_test, vocab)
    X_train = np.array([x[0] for x in train_data])
    y_train = np.array([x[1] for x in train_data])
    X_test = np.array([x[0] for x in test_data])
    y_test = np.array([x[1] for x in test_data])
    vocab_dic = vocab
    X_train_encoded = pretrain_one_hot(vocab_dic)[X_train]
    X_test_encoded = pretrain_one_hot(vocab_dic)[X_test]
    X_train_encoded = X_train_encoded.reshape(X_train_encoded.shape[0], -1)
    X_test_encoded = X_test_encoded.reshape(X_test_encoded.shape[0], -1)
    scaler = MinMaxScaler()
    X_train_encoded = scaler.fit_transform(X_train_encoded)
    X_test_encoded = scaler.transform(X_test_encoded)
    return X_train_encoded, X_train_encoded.shape[1], X_test_encoded, y_train, y_test


# 训练Transformer模型，使用tf.data.Dataset处理批次划分
def train_transformer_model(transformer_model, X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, num_epochs,
                            optimizer, criterion, callbacks=None):
    """
    训练Transformer模型，添加了callbacks参数用于接收回调函数列表

    参数:
    transformer_model: 要训练的Transformer模型实例
    X_train_tensor: 训练集数据张量
    X_test_tensor: 测试集数据张量
    y_train_tensor: 训练集标签张量
    y_test_tensor: 测试集标签张量
    num_epochs: 训练轮数
    optimizer: 优化器实例
    criterion: 损失函数
    callbacks: 回调函数列表，默认为None

    返回:
    训练后的Transformer模型实例和最终损失值
    """
    batch_size = 32  # 可根据实际情况调整批次大小
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_tensor, y_train_tensor)).batch(batch_size)
    final_loss = None
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}")
        for X_batch_train, y_batch_train in train_dataset:
            with tf.GradientTape() as tape:
                outputs = transformer_model(X_batch_train)
                loss = criterion(y_batch_train, outputs)
            trainable_vars = transformer_model.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            optimizer.apply_gradients(zip(gradients, trainable_vars))
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.numpy()}')
        final_loss = loss.numpy()  # 记录最后一轮的损失值作为最终损失值
    return transformer_model, final_loss


# 使用CatBoost进行进一步训练并评估
def train_catboost_model(transformer_model, X_train, X_test, y_train, y_test):
    transformer_output_train = transformer_model(tf.expand_dims(tf.convert_to_tensor(X_train, dtype=tf.float32), axis=0)).numpy()
    transformer_output_test = transformer_model(tf.expand_dims(tf.convert_to_tensor(X_test, dtype=tf.float32), axis=0)).numpy()
    transformer_output_train = transformer_output_train.squeeze(0)
    transformer_output_test = transformer_output_test.squeeze(0)
    catboost_model = cb.CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, loss_function='Logloss')
    catboost_model.fit(transformer_output_train, y_train, eval_set=(transformer_output_test, y_test))
    y_pred = catboost_model.predict(transformer_output_test)
    accuracy = accuracy_score(y_pred, y_test)
    print(f'Accuracy on test set: {accuracy}')
    return catboost_model, accuracy  # 返回准确率值


# 新增函数用于保存预训练好的Transformer模型
def save_pretrained_model(model, save_path):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, decay=0.01),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model.save(save_path)


if __name__ == "__main__":
    # 假设不同数据集路径存放在一个列表中，可以按需修改
    #dataset_paths = ['4mC_C.equisetifolia', '4mC_S.cerevisiae']
    dataset_paths = ['4mC_S.cerevisiae']
    summary_data = []
    for dataset_path in dataset_paths:
        print(f"开始训练数据集: {dataset_path}")
        # 加载并编码数据
        X_train_encoded, input_dim, X_test_encoded, y_train, y_test = load_and_encode_data(dataset_path)
        X_train_encoded = X_train_encoded.astype(np.float32)
        X_test_encoded = X_test_encoded.astype(np.float32)
        X_train_tensor = tf.expand_dims(tf.convert_to_tensor(X_train_encoded, dtype=tf.float32), axis=0)
        X_test_tensor = tf.expand_dims(tf.convert_to_tensor(X_test_encoded, dtype=tf.float32), axis=0)
        y_train_tensor = tf.expand_dims(tf.convert_to_tensor(y_train, dtype=tf.int32), axis=0)
        y_test_tensor = tf.expand_dims(tf.convert_to_tensor(y_test, dtype=tf.int32), axis=0)

        # 手动创建Transformer实例，参数可根据具体需求传入
        transformer_model = Transformer(input_dim=input_dim, depth=3, heads=4, dim_head=32, mlp_dim=64, dropout=0.1)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, decay=0.01)
        criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        num_epochs = 100

        early_stopping = EarlyStopping(monitor='val_loss', patience=10,
                                       restore_best_weights=True)  # 监控验证集损失，耐心值设为10，即连续10轮不下降就停止并恢复最佳权重

        # 训练Transformer模型时传入早停回调函数
        trained_transformer_model, final_loss = train_transformer_model(transformer_model, X_train_tensor,
                                                                        X_test_tensor,
                                                                        y_train_tensor, y_test_tensor, num_epochs,
                                                                        optimizer, criterion,
                                                                        callbacks=[early_stopping])

        # 使用CatBoost进行进一步训练并评估，接收返回的模型和准确率值
        trained_catboost_model, catboost_accuracy = train_catboost_model(trained_transformer_model, X_train_encoded, X_test_encoded,
                                                                         y_train, y_test)

        # 收集每个数据集的相关性能指标信息到字典中（可按需扩展记录的指标内容）
        dataset_info = {
            'Dataset': dataset_path,
            'Transformer_Loss': final_loss,  # 假设此处能获取最终的损失值，可根据实际调整
            'CatBoost_Accuracy': catboost_accuracy
        }
        summary_data.append(dataset_info)

        # 保存预训练的Transformer模型，保存路径可按需修改
        save_path = os.path.join('pretrained_models', f'transformer_{os.path.basename(dataset_path)}.h5')
        save_pretrained_model(trained_transformer_model, save_path)

    # 使用pandas创建DataFrame并展示汇总表（如果需要展示汇总结果的话）
    summary_df = pd.DataFrame(summary_data)
    print(summary_df)