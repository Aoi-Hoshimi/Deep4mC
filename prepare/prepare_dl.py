import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import itertools


def remove_number(ss):
    ss1 = ss
    for i in range(10):
        ss1 = ss1.replace(str(i), '')
    return ss1


def read_seq_data(file_path):
    '''
    Arguments:
    file_path -- （"../data/test.txt"）

    Return:
    seq_data -- DataFrame，（label，seq）
    '''
    m = []
    with open(file_path, "r") as f:
        lines = ""
        for line in f.readlines():
            line = line.upper().strip()
            if line.startswith(">"):
                if lines != "":
                    m.append([key, lines])
                    lines = ""
                key = line[1:]
            else:
                line = remove_number(line)
                line = line.replace("T", "U")
                lines += line
    m.append([key, lines])
    seq_data = pd.DataFrame(m, columns=["label", "seq"])
    seq_data["label"] = seq_data["label"].apply(lambda x: (0 if x.find('+') == -1 else 1)).tolist()
    # seq_data["label"] = seq_data["label"].apply(lambda x: (1 if x.find('NEGATIVE') == -1 else 0)).tolist()
    return seq_data

#bases='AUGC'
def all_k_tuple(k,bases="AUGC"):
    all_kmer = []
    for i in itertools.product(bases, repeat=k):
        all_kmer.append(''.join(i))
    return all_kmer


def build_vocab(k):
    words = all_k_tuple(k)
    vocab_dic = {word: idx for idx, word in enumerate(words)}
    return vocab_dic

def tokenizer(seqs_df1, k):
    seqs_df = seqs_df1.copy()
    seqs_df["seq"] = seqs_df["seq"].apply(lambda x: [x[i:i + k] for i in range(0, len(x) - k + 1)])
    seqs_df["len_original"] = seqs_df["seq"].apply(lambda x: len(x))
    return seqs_df

def trans(seqs_df1,vocab):
    seqs_df = seqs_df1.copy()
    seqs_df["seq"] = seqs_df["seq"].apply(lambda x: [vocab[i] for i in x])
    x = [tuple(x) for x in seqs_df.loc[:, ["seq", "label", "len_original"]].values]
    return x


def pretrain(seqs_df, vocab,embed):
    kmer_data = [tuple(x) for x in seqs_df["seq"]]
    model = Word2Vec(kmer_data, seed=1,size=embed, window=4, min_count=1, workers=1, iter=10)
    model.train(kmer_data, total_examples=len(kmer_data), epochs=10)
    word2vec1 = model.wv
    embeddings = np.random.rand(len(vocab), embed)
    for lin, word in enumerate(vocab):
        old_index = word2vec1.vocab[word].index
        embeddings[lin] = word2vec1.vectors[old_index]
    return embeddings

def to_ids(seqs):
    AA = 'AUCG'
    dd = zip(list(AA), range(len(AA)))
    AA_dict = dict(dd)
    seqs_new = []
    for seq in seqs:
        seq_new = [AA_dict[key] for key in seq]
        seqs_new.append(seq_new)
    return np.array(seqs_new)

def get_pos(seqs_num,seq_len):
    pos = np.array([list(range(seq_len))])
    pos = np.repeat(pos,seqs_num,axis=0)
    return pos


def pretrain_one_hot(vocab_dic):
    m = [1]*len(vocab_dic)
    n = np.diag(m)
    return n