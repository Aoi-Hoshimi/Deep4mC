# ANF
import numpy as np
import pandas as pd
import sys, os, platform

def CalculateMatrix(data, order):
    print("CalculateMatrix data:",data)
    # matrix = np.zeros((len(data[0]) - 2, 64))
    matrix = np.zeros((41 - 2, 64))
    # for i in range(len(data[0]) - 2): # position
    for i in range(41 - 2): # position
        for j in range(len(data)):
            if re.search('-', data[j][i:i+3]):
                pass
            else:
                matrix[i][order[data[j][i:i+3]]] += 1
    return matrix


def PSTNPss(fastas2, **kw):
    fastas = pd.DataFrame(fastas2,columns=["label","seq","is_train"])
    seq_id = ["Pos."+str(i) for i in range(len(fastas))]
    fastas["seq_id"] =  seq_id
    fastas = fastas.loc[:,["seq_id","seq","label","is_train"]]
    fastas["label"] = fastas["label"].astype("str")
    fastas = fastas.values.tolist()
    print(fastas[0])

    for i in fastas:
        if re.search('[^ACGT-]', i[1]):
            print('Error: illegal character included in the fasta sequences, only the "ACGT[U]" are allowed by this encoding scheme.')
            return 0

    encodings = []
    header = ['#', 'label']
    for pos in range(len(fastas[0][1])-2):
        header.append('Pos.%d' %(pos+1))
    encodings.append(header)

    # print(fastas[0])

    positive = []
    negative = []
    positive_key = []
    negative_key = []
    for i in fastas:
        if i[3] == 'training':
            if i[2] == '1':
                positive.append(i[1])
                positive_key.append(i[0])
            else:
                negative.append(i[1])
                negative_key.append(i[0])


    nucleotides = ['A', 'C', 'G', 'T']
    trinucleotides = [n1 + n2 + n3 for n1 in nucleotides for n2 in nucleotides for n3 in nucleotides]
    order = {}
    for i in range(len(trinucleotides)):
        order[trinucleotides[i]] = i

    print("encode1.py positive",positive,negative)
    matrix_po = CalculateMatrix(positive, order)
    matrix_ne = CalculateMatrix(negative, order)

    positive_number = len(positive)
    negative_number = len(negative)

    for i in fastas:
        if i[3] == 'testing':
            name, sequence, label = i[0], i[1], i[2]
            code = [name, label]
            for j in range(len(sequence) - 2):
                if re.search('-', sequence[j: j+3]):
                    code.append(0)
                else:
                    p_num, n_num = positive_number, negative_number
                    po_number = matrix_po[j][order[sequence[j: j+3]]]
                    if i[0] in positive_key and po_number > 0:
                        po_number -= 1
                        p_num -= 1
                    ne_number = matrix_ne[j][order[sequence[j: j+3]]]
                    if i[0] in negative_key and ne_number > 0:
                        ne_number -= 1
                        n_num -= 1
                    code.append(po_number/p_num - ne_number/n_num)
                    # print(sequence[j: j+3], order[sequence[j: j+3]], po_number, p_num, ne_number, n_num)
            encodings.append(code)

    return encodings



# def CalculateMatrix2(data, order):
#     '''
#     :param data: [seq,seq,...]
#     :param order: kmer_dict {kmer:id}
#     :return:
#     '''
#     # matrix = np.zeros((len(data[0]) - 2, 8))
#     matrix = np.zeros((41 - 2, 64))
#     # for i in range(len(data[0]) - 2): # position
#     for i in range(41 - 2):  # position
#         for j in range(len(data)):
#             if re.search('-', data[j][i:i+3]):
#                 pass
#             else:
#                 matrix[i][order[data[j][i:i+3]]] += 1
#     return matrix
#
#
# def PSTNPds(fastas2, **kw):
#     fastas = pd.DataFrame(fastas2, columns=["label", "seq", "is_train"])
#     seq_id = ["Pos." + str(i) for i in range(len(fastas))]
#     fastas["seq_id"] = seq_id
#     fastas = fastas.loc[:, ["seq_id", "seq", "label", "is_train"]]
#     fastas["label"] = fastas["label"].astype("str")
#     fastas = fastas.values.tolist()
#
#
#     for i in fastas:
#         if re.search('[^ACGT-]', i[1]):
#             print('Error: illegal character included in the fasta sequences, only the "ACGT[U]" are allowed by PSTNPds encoding scheme.')
#             return 0
#
#     for i in fastas:
#         i[1] = re.sub('T', 'A', i[1])
#         i[1] = re.sub('G', 'C', i[1])
#
#     encodings = []
#     header = ['#', 'label']
#     for pos in range(len(fastas[0][1])-2): # pos [0,39]
#         header.append('Pos.%d' %(pos+1))
#     encodings.append(header)
#
#     positive = []
#     negative = []
#     positive_key = []
#     negative_key = []
#     for i in fastas:
#         if i[3] == 'training':
#             if i[2] == '1':
#                 positive.append(i[1]) # seq_list
#                 positive_key.append(i[0]) # seq_id list
#             else:
#                 negative.append(i[1])
#                 negative_key.append(i[0])
#
#     nucleotides = ['A', 'C']
#     trinucleotides = [n1 + n2 + n3 for n1 in nucleotides for n2 in nucleotides for n3 in nucleotides] #kmer_list
#     order = {}
#     for i in range(len(trinucleotides)): # 构造kmer_dict
#         order[trinucleotides[i]] = i
#
#     matrix_po = CalculateMatrix2(positive, order)
#     matrix_ne = CalculateMatrix2(negative, order)
#
#     positive_number = len(positive) # 正负样本数目
#     negative_number = len(negative)
#
#     for i in fastas:
#         if i[3] == 'testing':
#             name, sequence, label = i[0], i[1], i[2]
#             code = [name, label] # code 最终形式 [seq_id,label,kmer_id,kmer_id,....]
#             for j in range(len(sequence) - 2):
#                 if re.search('-', sequence[j: j + 3]): #只要序列中的kmer含有-，就换0
#                     code.append(0)
#                 else:
#                     p_num, n_num = positive_number, negative_number
#                     po_number = matrix_po[j][order[sequence[j: j+3]]] # ?
#                     if i[0] in positive_key and po_number > 0:
#                         po_number -= 1
#                         p_num -= 1
#                     ne_number = matrix_ne[j][order[sequence[j: j+3]]]
#                     if i[0] in negative_key and ne_number > 0:
#                         ne_number -= 1
#                         n_num -= 1
#                     code.append(po_number/p_num - ne_number/n_num)
#                     # print(sequence[j: j+3], order[sequence[j: j+3]], po_number, p_num, ne_number, n_num)
#             encodings.append(code)
#     return encodings

def CalculateMatrix2(data, order):
    '''统计每个位置的kmer频次，所以需要先计算正样本的'''
    matrix = np.zeros((41 - 2, 8))
    # matrix = np.zeros((len(data[0]) - 2, 8))
    # for i in range(len(data[0]) - 2): # position
    for i in range(41 - 2): # position
        for j in range(len(data)):
            if re.search('-', data[j][i:i+3]):
                pass
            else:
                matrix[i][order[data[j][i:i+3]]] += 1
    return matrix


def PSTNPds(fastas, **kw):
    print("PSTNPds:", fastas[0])
    for i in fastas:
        if re.search('[^ACGT-]', i[1]):
            print('Error: illegal character included in the fasta sequences, only the "ACGT[U]" are allowed by PSTNPds encoding scheme.')
            return 0

    for i in fastas:
        i[1] = re.sub('T', 'A', i[1])
        i[1] = re.sub('G', 'C', i[1])

    encodings = []
    header = ['#', 'label']
    for pos in range(len(fastas[0][1])-2):
        header.append('Pos.%d' %(pos+1))
    encodings.append(header)

    positive = []
    negative = []
    positive_key = []
    negative_key = []
    for i in fastas:
        # if i[3] == 'training':
        if i[2] == 'training':
            # if i[2] == '1':
            if i[0] == '1':
                positive.append(i[1])
                positive_key.append(i[0])
            else:
                negative.append(i[1])
                negative_key.append(i[0])

    nucleotides = ['A', 'C']
    trinucleotides = [n1 + n2 + n3 for n1 in nucleotides for n2 in nucleotides for n3 in nucleotides]
    order = {}
    for i in range(len(trinucleotides)):
        order[trinucleotides[i]] = i

    matrix_po = CalculateMatrix2(positive, order)
    matrix_ne = CalculateMatrix2(negative, order)

    positive_number = len(positive)
    negative_number = len(negative)

    for i in fastas:
        # if i[3] == 'testing':
        if i[2] == 'testing':
            # name, sequence, label = i[0], i[1], i[2]
            name, sequence, label = i[0], i[1], i[2]
            code = [name, label]
            for j in range(len(sequence) - 2):
                if re.search('-', sequence[j: j + 3]):
                    code.append(0)
                else:
                    p_num, n_num = positive_number, negative_number
                    po_number = matrix_po[j][order[sequence[j: j+3]]]
                    if i[0] in positive_key and po_number > 0:
                        po_number -= 1
                        p_num -= 1
                    ne_number = matrix_ne[j][order[sequence[j: j+3]]]
                    if i[0] in negative_key and ne_number > 0:
                        ne_number -= 1
                        n_num -= 1
                    code.append(po_number/p_num - ne_number/n_num)
                    # print(sequence[j: j+3], order[sequence[j: j+3]], po_number, p_num, ne_number, n_num)
            encodings.append(code)
    return encodings

def kmerArray(sequence, k):
    kmer = []
    for i in range(len(sequence) - k + 1):
        kmer.append(sequence[i:i + k])
    return kmer


def RC(kmer):
    myDict = {
        'A': 'T',
        'C': 'G',
        'G': 'C',
        'T': 'A'
    }
    return ''.join([myDict[nc] for nc in kmer[::-1]])


def generateRCKmer(kmerList):
    rckmerList = set()
    myDict = {
        'A': 'T',
        'C': 'G',
        'G': 'C',
        'T': 'A'
    }
    for kmer in kmerList:
        rckmerList.add(sorted([kmer, ''.join([myDict[nc] for nc in kmer[::-1]])])[0])
    return sorted(rckmerList)


def RCKmer(fastas, k=2, upto=False, normalize=True, **kw):
    # print("RCM:",fastas[0])
    encoding = []
    header = ['#', 'label']
    NA = 'ACGT'

    if k < 1:
        print('Error: the k-mer value should larger than 0.')
        return 0

    if upto == True:
        for tmpK in range(1, k + 1):
            tmpHeader = []
            for kmer in itertools.product(NA, repeat=tmpK):
                tmpHeader.append(''.join(kmer))
            header = header + generateRCKmer(tmpHeader)
        myDict = {}
        for kmer in header[2:]:
            rckmer = RC(kmer)
            if kmer != rckmer:
                myDict[rckmer] = kmer
        encoding.append(header)
        for i in fastas:
            name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
            count = Counter()
            for tmpK in range(1, k + 1):
                kmers = kmerArray(sequence, tmpK)
                for j in range(len(kmers)):
                    if kmers[j] in myDict:
                        kmers[j] = myDict[kmers[j]]
                count.update(kmers)
                if normalize == True:
                    for key in count:
                        if len(key) == tmpK:
                            count[key] = count[key] / len(kmers)
            code = [name, label]
            for j in range(2, len(header)):
                if header[j] in count:
                    code.append(count[header[j]])
                else:
                    code.append(0)
            encoding.append(code)
    else:
        tmpHeader = []
        for kmer in itertools.product(NA, repeat=k):
            tmpHeader.append(''.join(kmer))
        header = header + generateRCKmer(tmpHeader)
        myDict = {}
        for kmer in header[2:]:
            rckmer = RC(kmer)
            if kmer != rckmer:
                myDict[rckmer] = kmer

        encoding.append(header)
        for i in fastas:
            name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]

            kmers = kmerArray(sequence, k)
            for j in range(len(kmers)):
                if kmers[j] in myDict:
                    kmers[j] = myDict[kmers[j]]
            count = Counter()
            count.update(kmers)
            if normalize == True:
                for key in count:
                    count[key] = count[key] / len(kmers)
            code = [name, label]
            for j in range(2, len(header)):
                if header[j] in count:
                    code.append(count[header[j]])
                else:
                    code.append(0)
            encoding.append(code)
    return encoding

def RCKmer1(fastas):
    return RCKmer(fastas, k=1)
def RCKmer2(fastas):
    return RCKmer(fastas, k=2)
def RCKmer3(fastas):
    return RCKmer(fastas, k=3)
def RCKmer4(fastas):
    return RCKmer(fastas, k=4)
def RCKmer5(fastas):
    return RCKmer(fastas, k=5)

def DNC(fastas, **kw):
    base = 'ACGT'

    encodings = []
    dinucleotides = [n1 + n2 for n1 in base for n2 in base]
    header = ['#', 'label'] + dinucleotides
    encodings.append(header)

    AADict = {}
    for i in range(len(base)):
        AADict[base[i]] = i

    for i in fastas:
        name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
        code = [name, label]
        tmpCode = [0] * 16
        for j in range(len(sequence) - 2 + 1):
            tmpCode[AADict[sequence[j]] * 4 + AADict[sequence[j+1]]] = tmpCode[AADict[sequence[j]] * 4 + AADict[sequence[j+1]]] +1
        if sum(tmpCode) != 0:
            tmpCode = [i/sum(tmpCode) for i in tmpCode]
        code = code + tmpCode
        encodings.append(code)
    return encodings

# 要求等长序列
def TNC(fastas, **kw):
    AA = 'ACGT'
    encodings = []
    triPeptides = [aa1 + aa2 + aa3 for aa1 in AA for aa2 in AA for aa3 in AA]
    header = ['#', 'label'] + triPeptides
    encodings.append(header)

    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i

    for i in fastas:
        name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
        code = [name, label]
        tmpCode = [0] * 64
        for j in range(len(sequence) - 3 + 1):
            tmpCode[AADict[sequence[j]] * 16 + AADict[sequence[j+1]]*4 + AADict[sequence[j+2]]] = tmpCode[AADict[sequence[j]] * 16 + AADict[sequence[j+1]]*4 + AADict[sequence[j+2]]] +1
        if sum(tmpCode) != 0:
            tmpCode = [i/sum(tmpCode) for i in tmpCode]
        code = code + tmpCode
        encodings.append(code)
    return encodings

def ANF(fastas, **kw):
    AA = 'ACGT'
    encodings = []
    header = ['#', 'label']
    for i in range(1, len(fastas[0][1]) + 1):
        header.append('ANF.' + str(i))
    encodings.append(header)

    for i in fastas:
        name, sequence, label = i[0], i[1], i[2]
        code = [name, label]
        for j in range(len(sequence)):
            code.append(sequence[0: j + 1].count(sequence[j]) / (j + 1))
        encodings.append(code)
    return encodings


# BINARY

import sys, os, platform


# 需要等长序列
def binary(fastas, **kw):
    AA = 'ACGT'
    encodings = []
    header = ['#', 'label']
    for i in range(1, len(fastas[0][1]) * 4 + 1):
        header.append('BINARY.F' + str(i))
    encodings.append(header)

    for i in fastas:
        name, sequence, label = i[0], i[1], i[2]
        code = [name, label]
        for aa in sequence:
            if aa == '-':
                code = code + [0, 0, 0, 0]
                continue
            for aa1 in AA:
                tag = 1 if aa == aa1 else 0
                code.append(tag)
        encodings.append(code)
    return encodings


import sys, os, platform


def CKSNAP(fastas, gap=5, **kw):
    if gap < 0:
        print('Error: the gap should be equal or greater than zero' + '\n\n')
        return 0

    AA = 'ACGT'
    encodings = []
    aaPairs = []
    for aa1 in AA:
        for aa2 in AA:
            aaPairs.append(aa1 + aa2)

    header = ['#', 'label']
    for g in range(gap + 1):
        for aa in aaPairs:
            header.append(aa + '.gap' + str(g))
    encodings.append(header)

    for i in fastas:
        name, sequence, label = i[0], i[1], i[2]
        code = [name, label]
        for g in range(gap + 1):
            myDict = {}
            for pair in aaPairs:
                myDict[pair] = 0
            sum_value = 0
            for index1 in range(len(sequence)):
                index2 = index1 + g + 1
                if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in AA and sequence[
                    index2] in AA:
                    myDict[sequence[index1] + sequence[index2]] = myDict[sequence[index1] + sequence[index2]] + 1
                    sum_value = sum_value + 1
            for pair in aaPairs:
                # 添加对分母为零的判断，若 sum_value 为零，则赋值为零，避免除以零错误
                if sum_value == 0:
                    code.append(0)
                else:
                    code.append(myDict[pair] / sum_value)
        encodings.append(code)
    return encodings


def CKSNAP1(fastas, **kw):
    return CKSNAP(fastas, gap=1)
def CKSNAP2(fastas, **kw):
    return CKSNAP(fastas, gap=2)
def CKSNAP3(fastas, **kw):
    return CKSNAP(fastas, gap=3)
def CKSNAP4(fastas, **kw):
    return CKSNAP(fastas, gap=4)
def CKSNAP5(fastas, **kw):
    return CKSNAP(fastas, gap=5)
def CKSNAP6(fastas, **kw):
    return CKSNAP(fastas, gap=6)
def CKSNAP7(fastas, **kw):
    return CKSNAP(fastas, gap=7)
def CKSNAP8(fastas, **kw):
    return CKSNAP(fastas, gap=8)
def CKSNAP9(fastas, **kw):
    return CKSNAP(fastas, gap=9)

# DNC
# !/usr/bin/env python
# _*_coding:utf-8_*_

import re


def DNC(fastas, **kw):
    base = 'ACGT'

    encodings = []
    dinucleotides = [n1 + n2 for n1 in base for n2 in base]
    header = ['#', 'label'] + dinucleotides
    encodings.append(header)

    AADict = {}
    for i in range(len(base)):
        AADict[base[i]] = i

    for i in fastas:
        name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
        code = [name, label]
        tmpCode = [0] * 16
        for j in range(len(sequence) - 2 + 1):
            tmpCode[AADict[sequence[j]] * 4 + AADict[sequence[j + 1]]] = tmpCode[AADict[sequence[j]] * 4 + AADict[
                sequence[j + 1]]] + 1
        if sum(tmpCode) != 0:
            tmpCode = [i / sum(tmpCode) for i in tmpCode]
        code = code + tmpCode
        encodings.append(code)
    return encodings


# dnc
import sys, os, platform


# 该种编码方式，必须是等长序列 nei
def EIIP(fastas, **kw):
    AA = 'ACGT'

    EIIP_dict = {
        'A': 0.1260,
        'C': 0.1340,
        'G': 0.0806,
        'U': 0.1335,
        'T': 0.1335,
        '-': 0,
    }

    encodings = []
    header = ['#', 'label']
    for i in range(1, len(fastas[0][1]) + 1):
        header.append('F' + str(i))
    encodings.append(header)

    for i in fastas:
        name, sequence, label = i[0], i[1], i[2]
        code = [name, label]
        for aa in sequence:
            code.append(EIIP_dict.get(aa, 0))
        encodings.append(code)
    return encodings


# ENAC
import re, sys, os, platform
from collections import Counter
import argparse


# 序列长度需要等长，且长度要大于窗口大小
#nei
def ENAC(fastas, window=5, **kw):
    if window < 1:
        print('Error: the sliding window should be greater than zero' + '\n\n')
        return 0

    #     AA = kw['order'] if kw['order'] != None else 'ACGT'
    AA = 'ACGT'
    encodings = []
    header = ['#', 'label']
    for w in range(1, len(fastas[0][1]) - window + 2):
        for aa in AA:
            header.append('SW.' + str(w) + '.' + aa)
    encodings.append(header)

    for i in fastas:
        name, sequence, label = i[0], i[1], i[2]
        code = [name, label]
        for j in range(len(sequence)):
            if j < len(sequence) and j + window <= len(sequence):
                count = Counter(sequence[j:j + window])
                for key in count:
                    count[key] = count[key] / len(sequence[j:j + window])
                for aa in AA:
                    code.append(count[aa])
        encodings.append(code)
    return encodings

def ENAC1(fastas):
    return ENAC(fastas, window=1)
def ENAC2(fastas):
    return ENAC(fastas, window=2)
def ENAC3(fastas):
    return ENAC(fastas, window=3)
def ENAC4(fastas):
    return ENAC(fastas, window=4)
def ENAC5(fastas):
    return ENAC(fastas, window=5)
def ENAC6(fastas):
    return ENAC(fastas, window=6)
def ENAC7(fastas):
    return ENAC(fastas, window=7)
def ENAC8(fastas):
    return ENAC(fastas, window=8)
def ENAC9(fastas):
    return ENAC(fastas, window=9)

# Kmer
import argparse
import re, sys, os, platform
import itertools
from collections import Counter


def kmerArray(sequence, k):
    kmer = []
    for i in range(len(sequence) - k + 1):
        kmer.append(sequence[i:i + k])
    return kmer

#seq nei
def Kmer(fastas, k=3, type="DNA", upto=False, normalize=True, **kw):
    encoding = []
    header = ['#', 'label']
    NA = 'ACGT'
    if type in ("DNA", 'RNA'):
        NA = 'ACGT'
    else:
        NA = 'ACDEFGHIKLMNPQRSTVWY'

    if k < 1:
        print('Error: the k-mer value should larger than 0.')
        return 0

    if upto == True:
        for tmpK in range(1, k + 1):
            for kmer in itertools.product(NA, repeat=tmpK):
                header.append(''.join(kmer))
        encoding.append(header)
        for i in fastas:
            name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
            count = Counter()
            for tmpK in range(1, k + 1):
                kmers = kmerArray(sequence, tmpK)
                count.update(kmers)
                if normalize == True:
                    for key in count:
                        if len(key) == tmpK:
                            count[key] = count[key] / len(kmers)
            code = [name, label]
            for j in range(2, len(header)):
                if header[j] in count:
                    code.append(count[header[j]])
                else:
                    code.append(0)
            encoding.append(code)
    else:
        for kmer in itertools.product(NA, repeat=k):
            header.append(''.join(kmer))
        encoding.append(header)
        for i in fastas:
            name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
            kmers = kmerArray(sequence, k)
            count = Counter()
            count.update(kmers)
            if normalize == True:
                for key in count:
                    count[key] = count[key] / len(kmers)
            code = [name, label]
            for j in range(2, len(header)):
                if header[j] in count:
                    code.append(count[header[j]])
                else:
                    code.append(0)
            encoding.append(code)
    return encoding

def Kmer1(fastas):
    return Kmer(fastas, k=1)
def Kmer2(fastas):
    return Kmer(fastas, k=2)
def Kmer3(fastas):
    return Kmer(fastas, k=3)
def Kmer4(fastas):
    return Kmer(fastas, k=4)
def Kmer5(fastas):
    return Kmer(fastas, k=5)



# NAC
import re
from collections import Counter

#seq 内
def NAC(fastas, **kw):
    #     NA = kw['order'] if kw['order'] != None else 'ACGT'
    NA = 'ACGT'
    encodings = []
    header = ['#', 'label']
    for i in NA:
        header.append(i) # header = ['#', 'label','A','C','G','U']
    encodings.append(header)

    for i in fastas:
        name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
        count = Counter(sequence)
        for key in count:
            count[key] = count[key] / len(sequence)
        code = [name, label]
        for na in NA:
            code.append(count[na])
        encodings.append(code)
    return encodings


# PseEIIP
import sys, os, re


def TriNcleotideComposition(sequence, base):
    trincleotides = [nn1 + nn2 + nn3 for nn1 in base for nn2 in base for nn3 in base]
    tnc_dict = {}
    for triN in trincleotides:
        tnc_dict[triN] = 0
    for i in range(len(sequence) - 2):
        tnc_dict[sequence[i:i + 3]] += 1
    for key in tnc_dict:
        tnc_dict[key] /= (len(sequence) - 2)
    return tnc_dict

#seq nei
def PseEIIP(fastas, **kw):
    for i in fastas:
        if re.search('[^ACGT-]', i[1]):
            print(
                'Error: illegal character included in the fasta sequences, only the "ACGT-" are allowed by this PseEIIP scheme.')
            return 0

    base = 'ACGT'

    EIIP_dict = {
        'A': 0.1260,
        'C': 0.1340,
        'G': 0.0806,
        'U': 0.1335,
        'T': 0.1335,
    }

    trincleotides = [nn1 + nn2 + nn3 for nn1 in base for nn2 in base for nn3 in base]
    EIIPxyz = {}
    for triN in trincleotides:
        EIIPxyz[triN] = EIIP_dict[triN[0]] + EIIP_dict[triN[1]] + EIIP_dict[triN[2]]

    encodings = []
    header = ['#', 'label'] + trincleotides
    encodings.append(header)

    for i in fastas:
        name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
        code = [name, label]
        trincleotide_frequency = TriNcleotideComposition(sequence, base)
        code = code + [EIIPxyz[triN] * trincleotide_frequency[triN] for triN in trincleotides]
        encodings.append(code)
    return encodings

import sys, os, platform


chemical_property = {
    'A': [1, 1, 1],
    'C': [0, 1, 0],
    'G': [1, 0, 0],
    'T': [0, 0, 1],
    'U': [0, 0, 1],
    '-': [0, 0, 0],
}

def NCP(fastas, **kw):

    AA = 'ACGT'
    encodings = []
    header = ['#', 'label']
    for i in range(1, len(fastas[0][1]) * 3 + 1):
        header.append('NCP.F'+str(i))
    encodings.append(header)

    for i in fastas:
        name, sequence, label = i[0], i[1], i[2]
        code = [name, label]
        for aa in sequence:
            code = code + chemical_property.get(aa, [0, 0, 0])
        encodings.append(code)
    return encodings