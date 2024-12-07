from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from colorama import Fore

def xgb_feature_selection(data_train, label_train):
    data_train2, data_dev, label_train2, label_dev = train_test_split(data_train, label_train, test_size=0.1)
    xgb_model = XGBClassifier(eval_metric="logloss")
    xgb_model.fit(data_train2, label_train2)
    cc = pd.Series(xgb_model.feature_importances_)
    cc = cc.sort_values(ascending=False)
    fs1 = [i for i in cc.index]
    result1 = []
    
    for i in tqdm(range(1, len(fs1), 10), bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.RED, Fore.RESET)):
        model = XGBClassifier(eval_metric='logloss')
        model.fit(data_train2[:, fs1[:i]], label_train2)
        y_submission_xgb = model.predict_proba(data_dev[:, fs1[:i]])[:, 1]
        y_submission_acc = [(1 if x > 0.5 else 0) for x in y_submission_xgb]
        acc1 = accuracy_score(label_dev, y_submission_acc)
        result1.append(acc1)
    
    index_best = result1.index(max(result1)) * 10 + 1
    print("Chosen", str(index_best), "features")
    return fs1[:index_best]