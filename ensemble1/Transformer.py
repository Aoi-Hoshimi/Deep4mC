import tensorflow as tf
from tensorflow.keras import layers as nn


def Transformer(input_dim, depth, heads, dim_head, mlp_dim, dropout=0.1):
    """
    构建函数式的Transformer模型

    参数:
    input_dim: 输入维度
    depth: Transformer的层数
    heads: 多头注意力机制的头数
    dim_head: 每个头的维度
    mlp_dim: 前馈网络中间层维度
    dropout: dropout概率

    返回:
    构建好的函数式Transformer模型实例
    """
    inputs = tf.keras.Input(shape=(None, input_dim))  # 定义输入，假设序列长度维度设为None，可根据实际调整

    x = inputs
    for _ in range(depth):
        attn = tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim=dim_head, dropout=dropout)
        attn_output = attn(query=x, key=x, value=x)
        x = attn_output + x

        ff = FeedForward(input_dim, mlp_dim, dropout)
        x = ff(x) + x

    outputs = nn.Dense(2)(x)  # 二分类任务，输出维度为2

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def FeedForward(input_dim, mlp_dim, dropout=0.1):
    """
    构建前馈网络模块

    参数:
    input_dim: 输入维度
    mlp_dim: 中间层维度
    dropout: dropout概率

    返回:
    构建好的前馈网络函数式模块
    """
    return tf.keras.Sequential([
        nn.Dense(mlp_dim),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(dropout),
        nn.Dense(input_dim)
    ])