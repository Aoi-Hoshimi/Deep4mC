import random
import numpy as np
import pandas as pd
import os
from random import seed
import tensorflow as tf
def set_seeds(seed1=0):
    os.environ['PYTHONHASHSEED'] = str(seed1)
    seed(seed1)
    tf.random.set_seed(seed1)
    np.random.seed(seed1)


def set_global_determinism(seed1=0):
    set_seeds(seed1)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
# def sample_h(dataset,label,sample_num):
def sample_h(seqs_df, pos_num_choose):
    data_size = seqs_df.shape[0]
    labels = [(0 if x.find('+') == -1 else 1) for x in seqs_df["seq_id"]]
    labels_index = np.array(range(0, data_size))
    labels_p = [(True if value == 1 else False) for value in labels]
    labels_n = [(True if value == 0 else False) for value in labels]
    labels_p = set(labels_index[labels_p])  # 正样本下标
    labels_n = set(labels_index[labels_n])  # 负样本下标

    indexes_shuffle_p = np.random.choice(list(labels_p), len(labels_p), replace=False) 
    indexes_shuffle_n = np.random.choice(list(labels_n), len(labels_n), replace=False)

    indexes_choose_p, indexes_choose_n = list(indexes_shuffle_p[:pos_num_choose]), list(
        indexes_shuffle_n[:pos_num_choose])
    indexes_remain_p, indexes_remain_n = list(indexes_shuffle_p[pos_num_choose:]), list(
        indexes_shuffle_n[pos_num_choose:])

    indexes_choose_p.extend(indexes_choose_n)
    indexes_remain_p.extend(indexes_remain_n)

    random.shuffle(indexes_choose_p)
    random.shuffle(indexes_remain_p)

    return seqs_df.iloc[indexes_choose_p, :], seqs_df.iloc[indexes_remain_p, :]


def split_seq_df(data,freq=0.2):
    labels = get_label(data)
    # get pos and neg sample index
    labels_index = np.array(range(0, data.shape[0]))
    labels_p = [(True if value == 1 else False) for value in labels]
    labels_n = [(True if value == 0 else False) for value in labels]
    labels_p = set(labels_index[labels_p])
    labels_n = set(labels_index[labels_n])

    labels_p_test = set(random.sample(labels_p, int(len(labels_p) * freq)))
    labels_n_test = set(random.sample(labels_n, int(len(labels_n) * freq)))
    labels_p_train = labels_p.difference(labels_p_test)
    labels_n_train = labels_n.difference(labels_n_test)
    labels_train = list(labels_p_train.union(labels_n_train))
    labels_test = list(labels_p_test.union(labels_n_test))

    random.shuffle(labels_train)
    random.shuffle(labels_test)

    data_train = data.iloc[labels_train,:]
    data_test = data.iloc[labels_test, :]
    return data_train, data_test



def split_h(data, label1=None, freq=0.2):
    data_size = len(data)
    data1 = pd.DataFrame(data)
    if label1 is None:
        labels = np.array(data1.iloc[:, 1])
    else:
        labels = np.array(label1)
    labels_index = np.array(range(0, data_size))
    labels_p = [(True if value == 1 else False) for value in labels]
    labels_n = [(True if value == 0 else False) for value in labels]
    labels_p = set(labels_index[labels_p])
    labels_n = set(labels_index[labels_n])

    labels_p_test = set(random.sample(labels_p, int(len(labels_p) * freq)))
    labels_n_test = set(random.sample(labels_n, int(len(labels_n) * freq)))
    labels_p_train = labels_p.difference(labels_p_test)
    labels_n_train = labels_n.difference(labels_n_test)
    labels_train = list(labels_p_train.union(labels_n_train))
    labels_test = list(labels_p_test.union(labels_n_test))

    random.shuffle(labels_train)
    random.shuffle(labels_test)

    if type(data)==pd.core.frame.DataFrame:

        data_train = data.iloc[labels_train,:]
        data_test = data.iloc[labels_test,:]
        print(data_test.shape)
    else:
        data_train = np.array([data[x] for x in labels_train])
        data_test = np.array([data[x] for x in labels_test])

    if label1 is None:
        return data_train, data_test
    else:
        return data_train, data_test, labels[labels_train], labels[labels_test]


def split_h2(data, freq=0.2):
    data_size = len(data)
    data1 = pd.DataFrame(data)
    labels = np.array(data1.iloc[:, 1])
    labels_index = np.array(range(0, data_size))
    labels_p = [(True if value == 1 else False) for value in labels]
    labels_n = [(True if value == 0 else False) for value in labels]
    labels_p = set(labels_index[labels_p])
    labels_n = set(labels_index[labels_n])

    labels_p_test = set(random.sample(labels_p, int(len(labels_p) * freq)))
    labels_n_test = set(random.sample(labels_n, int(len(labels_n) * freq)))
    labels_p_train = labels_p.difference(labels_p_test)
    labels_n_train = labels_n.difference(labels_n_test)
    labels_train = list(labels_p_train.union(labels_n_train))
    labels_test = list(labels_p_test.union(labels_n_test))

    random.shuffle(labels_train)
    random.shuffle(labels_test)

    data_train = [data[x] for x in labels_train]
    data_test = [data[x] for x in labels_test]

    return data_train, data_test


def sample_h(seqs_df, sample_num_half):
    data_size = seqs_df.shape[0]
    # labels = [(0 if x.find('+') == -1 else 1) for x in seqs_df["seq_id"]]
    labels = seqs_df["label"].values
    labels_index = np.array(range(0, data_size))
    labels_p = [(True if value == 1 else False) for value in labels]
    labels_n = [(True if value == 0 else False) for value in labels]
    labels_p = set(labels_index[labels_p])  # 正样本下标
    labels_n = set(labels_index[labels_n])  # 负样本下标

    indexes_shuffle_p = np.random.choice(list(labels_p), len(labels_p), replace=False)  # 打乱顺序
    indexes_shuffle_n = np.random.choice(list(labels_n), len(labels_n), replace=False)  # 打乱顺序

    indexes_choose_p, indexes_choose_n = list(indexes_shuffle_p[:sample_num_half]), list(
        indexes_shuffle_n[:sample_num_half])
    indexes_remain_p, indexes_remain_n = list(indexes_shuffle_p[sample_num_half:]), list(
        indexes_shuffle_n[sample_num_half:])

    indexes_choose_p.extend(indexes_choose_n)
    indexes_remain_p.extend(indexes_remain_n)

    random.shuffle(indexes_choose_p)
    random.shuffle(indexes_remain_p)
    data_choose = seqs_df.iloc[indexes_choose_p, :]
    data_remain = seqs_df.iloc[indexes_remain_p, :]
    data_choose.index = range(data_choose.shape[0])
    data_remain.index = range(data_remain.shape[0])
    return data_choose,data_remain

from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef, f1_score


def score2(y_real, pre_label):
    auc1 = roc_auc_score(y_real, pre_label)
    pre_label_acc = [(1 if x > 0.5 else 0) for x in pre_label]
    # print(pre_label_acc)
    cm = confusion_matrix(y_real, pre_label_acc)
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]
    acc = (tp + tn) / (tp + fp + fn + tn)
    #     pre = tp/(tp+fp)
    #     rec = tp/(tp+fn)
    spec = tn / (tn + fp)  # SP
    sens = tp / (tp + fn)  # SN
    mcc = matthews_corrcoef(y_real, pre_label_acc)  # 根据CPU中的，计算ACC
    f1score = f1_score(y_real, pre_label_acc)
    return acc, sens, spec, mcc, auc1, f1score

def score3(y_real, pre_label,pre_prob):
    auc1 = roc_auc_score(y_real, pre_prob)
    cm = confusion_matrix(y_real, pre_label)
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]
    acc = (tp + tn) / (tp + fp + fn + tn)
    #     pre = tp/(tp+fp)
    #     rec = tp/(tp+fn)
    spec = tn / (tn + fp)  # SP
    sens = tp / (tp + fn)  # SN
    mcc = matthews_corrcoef(y_real, pre_label)  # 根据CPU中的，计算ACC
    f1score = f1_score(y_real, pre_label)
    return acc, sens, spec, mcc, auc1, f1score

def get_label(seqs_df):
    label = seqs_df["label"].values
    return label


import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from matplotlib.colors import ListedColormap

def plot_tsne(X,y,name1=""):
    '''t-SNE'''
    # tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    tsne = manifold.TSNE(n_components=2, random_state=501)
    X_tsne = tsne.fit_transform(X)
    colours = ListedColormap(['r', 'b', 'g', 'y', 'm'])
    print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        # plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),fontdict={'weight': 'bold', 'size': 9})
        plt.text(X_norm[i, 0], X_norm[i, 1], str('o'), color=plt.cm.Set1(y[i]),fontdict={'weight': 'bold', 'size': 9})

    plt.title("t-SNE")
    plt.xticks([])
    plt.yticks([])
    plt.show()

def plot_tsne2(X,y,save_path,name1="",model_name=""):
    '''t-SNE'''
    # tsne = manifold.TSNE(n_components=2, random_state=0)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    labels_p = [(True if value == 1 else False) for value in y]
    labels_n = [(True if value == 0 else False) for value in y]


    X_tsne = tsne.fit_transform(X)
    colours = ListedColormap(['r', 'b', 'g', 'y', 'm'])
    print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
    color1 = ['red','blue']
    '''嵌入空间可视化'''
    # x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    # X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(8, 8))
    ps1 = []
    ps2 = []

    X_p = X_tsne[labels_p]
    X_n = X_tsne[labels_n]
    # for i in range(X_p.shape[0]):
    p1 = plt.scatter(X_p[:, 0], X_p[:, 1], color=color1[0], alpha=0.5, s=7,label="pos")
    p2 = plt.scatter(X_n[:, 0], X_n[:, 1], color=color1[1], alpha=0.5, s=7,label="neg")
        # ps1.append(p1)
        # ps2.append(p2)

    plt.title("t-SNE "+model_name+" "+name1)
    plt.xticks([])
    plt.yticks([])

    plt.legend()
    plt.savefig(save_path + name1 + "_" + model_name + ".pdf")
    plt.show()

def plot_tsne3(X,y,save_path,name1="",model_name=""):
    '''t-SNE'''
    # tsne = manifold.TSNE(n_components=2, random_state=0)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    labels_p = [(True if value == 1 else False) for value in y]
    labels_n = [(True if value == 0 else False) for value in y]


    X_tsne = tsne.fit_transform(X)
    colours = ListedColormap(['r', 'b', 'g', 'y', 'm'])
    print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
    color1 = ['red','blue']
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(8, 8))

    X_p = X_norm[labels_p]
    X_n = X_norm[labels_n]
    # for i in range(X_p.shape[0]):
    p1 = plt.scatter(X_p[:, 0], X_p[:, 1], color=color1[0], alpha=0.6, s=8,label="pos")
    p2 = plt.scatter(X_n[:, 0], X_n[:, 1], color=color1[1], alpha=0.6, s=8,label="neg")
        # ps1.append(p1)
        # ps2.append(p2)

    plt.title("t-SNE "+model_name+" "+name1)
    plt.xticks([])
    plt.yticks([])

    plt.legend()
    # plt.savefig(save_path + name1 + "_" + model_name + ".pdf")
    plt.show()

def plot_tsne4(X,y,save_path,name1="",model_name=""):
    '''t-SNE'''
    # tsne = manifold.TSNE(n_components=2, random_state=0)
    labels_p = [(True if value == 1 else False) for value in y]
    labels_n = [(True if value == 0 else False) for value in y]

    color1 = ['red','blue']
    plt.figure(figsize=(8, 8))
    ps1 = []
    ps2 = []

    X_p = X[labels_p]
    X_n = X[labels_n]
    # for i in range(X_p.shape[0]):
    p1 = plt.scatter(X_p[:, 0], X_p[:, 1], color=color1[0], alpha=0.6, s=8,label="pos")
    p2 = plt.scatter(X_n[:, 0], X_n[:, 1], color=color1[1], alpha=0.6, s=8,label="neg")
        # ps1.append(p1)
        # ps2.append(p2)

    plt.title(" "+model_name+" "+name1)
    plt.xticks([])
    plt.yticks([])

    plt.legend()
    # plt.savefig(save_path + name1 + "_" + model_name + ".pdf")
    plt.show()