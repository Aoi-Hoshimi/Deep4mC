
import os
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
import numpy as np
import pandas as pd

from data.data import get_single_data
from prepare.prepare_ml import ml_code
from tools.tools import get_label


def Ttest(X, Y, ttest_num=0):
    Y1 = np.where(Y == 1)[0]
    Y0 = np.where(Y == 0)[0]
    p_list = []
    for i in range(len(X[0])):
        p_l = stats.levene(X[Y1, i], X[Y0, i])[1]
        equal_var = [True if p_l > 0.05 else False]
        p_list.append(stats.ttest_ind(X[Y1, i], X[Y0, i], equal_var=equal_var)[1])

    return p_list, None

def Wtest(X, Y, wtest_num=0):
    Y1 = np.where(Y == 1)[0]
    Y0 = np.where(Y == 0)[0]
    p_list = []
    for i in range(len(X[0])):
        p_list.append(stats.ranksums(X[Y1, i], X[Y0, i])[1])
    return p_list, None

def RF(X, Y, rtest_num=0):
    forest = RandomForestClassifier(random_state=0, n_jobs=1)
    forest.fit(X, Y)
    importance = forest.feature_importances_
    return 1 / (importance + 1e-10), None

def LR_RFE(X, Y, lrtest_num=0):
    clf = LinearRegression()
    rfe = RFE(clf, n_features_to_select=1)
    rfe.fit(X, Y)
    rank = rfe.ranking_
    return rank, None

def SVM_RFE(X, Y, srtest_num=0):
    clf = SVC(kernel='linear', random_state=0)
    rfe = RFE(clf, n_features_to_select=1)
    rfe.fit(X, Y)
    rank = rfe.ranking_
    return rank, None

