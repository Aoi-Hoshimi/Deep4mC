
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from dl.net import TA_keras
from ensemble1.model_factory import get_new_model
from prepare.prepare_ml import ml_code, xgb_feature_selection
import os
import pickle as pkl

class StackingModel():

    def __init__(self, config):
        self.config = config
        if config.base_models_name is None:
            self.base_models_name = ["RF", "AB", "LD", "ET", "GB", "XGB", "LGBM", "LR"]
        else:
            self.base_models_name = config.base_models_name
        self.base_models = []
        self.meta_model = LogisticRegression()
        self.n_folds = config.n_folds
        self.best_features = None
        self.best_features_type = None
        self.config.model_name = config.model_name
        self.validate_prediction = None
        self.validate_real_label = None
        self.test_prediction = None
        self.test_real_label = None

    def fit(self, X, y):
        data_train1, label_train, record_feature_type = ml_code(X, "training",self.config.encoding)
        ml_X = pd.DataFrame(data_train1)
        if self.config.is_feature_selection:
            filename1 = self.config.features_save_path + self.config.model_name + "_feature_selection.pkl"
            if not os.path.exists(filename1):
                print("train fs record don't exist,creating and saving..")
                best_features = xgb_feature_selection(data_train1, label_train)
                best_features_name = [record_feature_type[i] for i in best_features]
                best_features_type = dict(Counter(best_features_name))
                self.best_features = best_features
                self.best_features_type = best_features_type
            else:
                print("train fs record exist,loading..")
                with open(filename1, 'rb') as f:
                    self.best_features, self.best_features_type = pkl.load(f)
                print("number of features：", len(self.best_features))
        else:
            best_features = list(range(ml_X.shape[1]))
            best_features_name = [record_feature_type[i] for i in best_features]
            best_features_type = dict(Counter(best_features_name))
            self.best_features = best_features
            self.best_features_type = best_features_type

        base_models_5fold_prediction = np.zeros((X.shape[0], len(self.base_models_name)+1))
        skf = list(StratifiedKFold(n_splits=self.n_folds).split(X, y))
        for j, model_name in enumerate(self.base_models_name):
            single_type_model = []
            for i, (train_index, val_index) in enumerate(skf):
                X_train, y_train, X_val, y_val = ml_X.iloc[train_index,self.best_features], y[train_index], ml_X.iloc[val_index,self.best_features], y[val_index]
                X_train = X_train.values
                X_val = X_val.values
                model = get_new_model(model_name)
                model.fit(X_train, y_train)
                y_pred = model.predict_proba(X_val)[:, 1]
                base_models_5fold_prediction[val_index, j] = y_pred
                single_type_model.append(model)
            self.base_models.append(single_type_model)
        # ---------------------dl----------------------------------------------------
        single_type_model = []
        for i, (train_index, val_index) in enumerate(skf):
            X_train, y_train, X_val, y_val = X.iloc[train_index, :], y[train_index], X.iloc[val_index, :], y[val_index]
            ta = TA_keras(self.config)
            ta.fit(X_train, y_train)
            y_pred = ta.predict_proba(X_val)
            single_type_model.append(ta) # save model(all fold) to deal with test set
            base_models_5fold_prediction[val_index, len(self.base_models_name)] = y_pred
        self.base_models.append(single_type_model)
        # ---------------------------------------------------------------------------
        # the 2th layer of ensemble learning model
        self.meta_model.fit(base_models_5fold_prediction, y)
        self.validate_prediction = base_models_5fold_prediction
        self.validate_real_label = y
        return self

    def predict(self, X):
        X1, label_train, record_feature_type = ml_code(X, "testing")
        X1 = X1[:,self.best_features]
        base_models_ind_prediction = np.zeros((X.shape[0], len(self.base_models)))
        for i,single_type_models_list in enumerate(self.base_models):
            single_type_result = np.zeros((X.shape[0], len(self.base_models[0])))
            for j,model in enumerate(single_type_models_list):
                if i < len(self.base_models) - 1:
                    single_type_result[:, j] = model.predict_proba(X1)[:, 1]
                else:
                    single_type_result[:, j] = model.predict_proba(X)
            base_models_ind_prediction[:, i] = single_type_result.mean(1)
        y_pred = self.meta_model.predict_proba(base_models_ind_prediction) # for plot
        y_pred = y_pred[:, 1]
        return y_pred














