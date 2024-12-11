import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from prepare.prepare_ml import load_data, ml_code

from fs.encode1 import ENAC2, binary, NCP, EIIP, CKSNAP, ENAC, Kmer, NAC, PseEIIP, ANF, CKSNAP8, Kmer4, TNC, RCKmer5, \
    DNC
from fs.load_acc import TAC
from fs.load_pse import SCPseTNC, PCPseTNC


# 使用VotingClassifier评估不同编码方法在给定数据上的表现
def evaluate_encoding_methods_with_voting_classifier(data_train, data_test, feature_extract_method):
    results = {}
    for method_name in feature_extract_method.keys():
        X_train, y_train, _ = ml_code(data_train,
                                      feature_extract_method={method_name: feature_extract_method[method_name]})
        X_test, y_test, _ = ml_code(data_test,
                                    feature_extract_method={method_name: feature_extract_method[method_name]})

        result = evaluate_single_with_voting_classifier(X_train, y_train, X_test, y_test)
        results[method_name] = result
    return results


def evaluate_single_with_voting_classifier(X_train, y_train, X_test, y_test):
    # 定义几个基础分类器
    clf1 = DecisionTreeClassifier()
    clf2 = LogisticRegression()
    clf3 = KNeighborsClassifier()
    # 创建VotingClassifier实例，这里用硬投票
    eclf = VotingClassifier(estimators=[('dt', clf1), ('lr', clf2), ('knn', clf3)], voting='hard')
    eclf.fit(X_train, y_train)
    y_pred = eclf.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    return {
        "Accuracy": accuracy,
        "Balanced Accuracy": None, #其他按需添加
        "ROC AUC": None,
        "F1 Score": None,
        "Time Taken": None
    }


# 汇总各编码方法在所有数据集上的平均性能
def summarize_encoding_methods_performance(results_list, dataset_names):
    """
    results_list是包含不同数据集对应的模型评估结果的列表
    dataset_names是数据集名称列表
    本函数整合这些结果，计算所有编码方法的综合性能，并输出全部编码方法及其性能的表格
    """
    all_average_scores = {}
    all_scores = {}  # 用于记录每个编码方法在每个数据集下的具体scores
    for index, results in enumerate(results_list):
        for method_name, metrics in results.items():
            if method_name not in all_average_scores:
                all_average_scores[method_name] = []
            accuracy = metrics.get('Accuracy')
            all_average_scores[method_name].append(accuracy)
            if method_name not in all_scores:
                all_scores[method_name] = {}
            all_scores[method_name][dataset_names[index]] = accuracy

    # 计算所有数据集下的平均性能，处理None值情况
    avg_performance = {}
    for method, scores in all_average_scores.items():
        valid_scores = [s for s in scores if s is not None]
        if valid_scores:
            avg_performance[method] = np.mean(valid_scores)
        else:
            avg_performance[method] = None

    # 按照性能排序，先处理None值情况排在最后
    sorted_methods = sorted(avg_performance.items(), key=lambda x: (x[1] is None, x[1]), reverse=True)
    sorted_methods = [m[0] for m in sorted_methods]

    # 输出每种特征编码在所有数据集上的平均性能
    print("特征编码方法\t平均性能")
    print("----------------------")
    for method in sorted_methods:
        avg_value = avg_performance[method]
        print(f"{method}\t{avg_value if avg_value is not None else '无有效数据'}")

    # 表格输出每个数据集下各特征编码的性能
    print("\n各数据集下特征编码方法的性能：")
    print("\t".join(["特征编码方法"] + dataset_names))
    print("".join(["------" for _ in range(len(dataset_names) + 1)]))
    for method in sorted_methods:
        scores_per_dataset = [all_scores[method].get(ds_name, None) for ds_name in dataset_names]
        scores_str = "\t".join([str(s) if s is not None else '无有效数据' for s in scores_per_dataset])
        print(f"{method}\t{scores_str}")

    return sorted_methods


def main():
    base_dir = r'E:\4mC\DeepSF-4mC-master\data\4mC'
    dataset_names = ['4mC_C.equisetifolia', '4mC_F.vesca', '4mC_S.cerevisiae']
    all_evaluated_results = []
    for dataset_name in dataset_names:
        data_train, data_test = load_data(dataset_name)
        feature_extract_method = {
            "ENAC": ENAC2, "binary": binary, "NCP": NCP, "EIIP": EIIP, "Kmer4": Kmer4, "CKSNAP": CKSNAP8,
            "PseEIIP": PseEIIP, "TNC": TNC, "RCKmer5": RCKmer5, "SCPseTNC": SCPseTNC, "PCPseTNC": PCPseTNC,
            "ANF": ANF, "NAC": NAC, "TAC": TAC
        }
        evaluated_results = evaluate_encoding_methods_with_voting_classifier(data_train, data_test,
                                                                           feature_extract_method)
        all_evaluated_results.append(evaluated_results)

    top_encoding_methods = summarize_encoding_methods_performance(all_evaluated_results, dataset_names)
    return top_encoding_methods


if __name__ == "__main__":
    result = main()
    print("所有特征编码方法按性能降序排列（已输出详细信息）")