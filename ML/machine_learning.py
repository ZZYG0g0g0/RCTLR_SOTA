"""
Author: 邹致远
Email: www.pisyongheng@foxmail.com
Date Created: 2023/9/12
Last Updated: 2023/11/20
Version: 2.0.0
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from data_reblancing import smote
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt

#随机森林
def random_forest(n_splits, is_databalance, dataset_name,  feature_file_name):
    df = pd.read_excel('./dataset/'+dataset_name+'/'+feature_file_name+'.xlsx')
    labels = pd.read_excel('./dataset/'+dataset_name+'/labels.xlsx')
    # 1. 数据归一化
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(df)
    precisions = []
    recalls = []
    f1s = []

    n_repeats = 1
    for _ in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=None)

        for train_index, test_index in kf.split(X_normalized):
            X_train, X_test = X_normalized[train_index], X_normalized[test_index]
            y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

            if is_databalance == 1:
                # 使用SMOTE进行过采样
                X, y = smote(X_train, y_train)

                # 使用随机森林进行训练
                clf = RandomForestClassifier()
                clf.fit(X, y)
            else:
                # 使用随机森林进行训练
                clf = RandomForestClassifier()
                clf.fit(X_train, y_train)

            # 预测
            y_pred = clf.predict(X_test)

            # 计算精确率和召回率
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        # 计算并打印平均精确率、召回率和F1值
    print(f"Average Precision: {np.mean(precisions)}")
    print(f"Average Recall: {np.mean(recalls)}")
    print(f"Average F1: {np.mean(f1s)}")
    return round(np.mean(precisions),4), round(np.mean(recalls),4), round(np.mean(f1s),4)

#随机森林
def knn(n_splits, is_databalance, dataset_name,  feature_file_name):
    df = pd.read_excel('./dataset/'+dataset_name+'/'+feature_file_name+'.xlsx')
    labels = pd.read_excel('./dataset/'+dataset_name+'/labels.xlsx')
    # 1. 数据归一化
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(df)
    precisions = []
    recalls = []
    f1s = []

    n_repeats = 1
    for _ in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=None)

        for train_index, test_index in kf.split(X_normalized):
            X_train, X_test = X_normalized[train_index], X_normalized[test_index]
            y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

            if is_databalance == 1:
                # 使用SMOTE进行过采样
                X, y = smote(X_train, y_train)

                # 使用随机森林进行训练
                clf = KNeighborsClassifier(n_neighbors=5)
                clf.fit(X, y)
            else:
                # 使用随机森林进行训练
                clf = KNeighborsClassifier(n_neighbors=5)
                clf.fit(X_train, y_train)
            # 预测
            y_pred = clf.predict(X_test)

            # 计算精确率和召回率
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        # 计算并打印平均精确率、召回率和F1值
    print(f"Average Precision: {np.mean(precisions)}")
    print(f"Average Recall: {np.mean(recalls)}")
    print(f"Average F1: {np.mean(f1s)}")
    return round(np.mean(precisions),4), round(np.mean(recalls),4), round(np.mean(f1s),4)

#逻辑回归
def logister_regression(n_splits, is_databalance, dataset_name,  feature_file_name):
    df = pd.read_excel('./dataset/'+dataset_name+'/'+feature_file_name+'.xlsx')
    labels = pd.read_excel('./dataset/'+dataset_name+'/labels.xlsx')
    # 1. 数据归一化
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(df)
    precisions = []
    recalls = []
    f1s = []

    n_repeats = 1
    for _ in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=None)

        for train_index, test_index in kf.split(X_normalized):
            X_train, X_test = X_normalized[train_index], X_normalized[test_index]
            y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

            if is_databalance == 1:
                # 使用SMOTE进行过采样
                X, y = smote(X_train, y_train)

                # 使用随机森林进行训练
                clf = LogisticRegression()
                clf.fit(X, y)
            else:
                # 使用随机森林进行训练
                clf = LogisticRegression()
                clf.fit(X_train, y_train)
            # 预测
            y_pred = clf.predict(X_test)

            # 计算精确率和召回率
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        # 计算并打印平均精确率、召回率和F1值
    print(f"Average Precision: {np.mean(precisions)}")
    print(f"Average Recall: {np.mean(recalls)}")
    print(f"Average F1: {np.mean(f1s)}")
    return round(np.mean(precisions),4), round(np.mean(recalls),4), round(np.mean(f1s),4)

#朴素贝叶斯
def naive_bayes(n_splits, is_databalance, dataset_name,  feature_file_name):
    df = pd.read_excel('./dataset/'+dataset_name+'/'+feature_file_name+'.xlsx')
    labels = pd.read_excel('./dataset/'+dataset_name+'/labels.xlsx')
    # 1. 数据归一化
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(df)
    precisions = []
    recalls = []
    f1s = []

    n_repeats = 1
    for _ in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=None)

        for train_index, test_index in kf.split(X_normalized):
            X_train, X_test = X_normalized[train_index], X_normalized[test_index]
            y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

            if is_databalance == 1:
                # 使用SMOTE进行过采样
                X, y = smote(X_train, y_train)

                # 使用随机森林进行训练
                clf = GaussianNB()
                clf.fit(X, y)
            else:
                # 使用随机森林进行训练
                clf = GaussianNB()
                clf.fit(X_train, y_train)

            # 预测
            y_pred = clf.predict(X_test)

            # 计算精确率和召回率
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        # 计算并打印平均精确率、召回率和F1值
    print(f"Average Precision: {np.mean(precisions)}")
    print(f"Average Recall: {np.mean(recalls)}")
    print(f"Average F1: {np.mean(f1s)}")
    return round(np.mean(precisions),4), round(np.mean(recalls),4), round(np.mean(f1s),4)

#SVM
def svm(n_splits, is_databalance, dataset_name,  feature_file_name):
    df = pd.read_excel('./dataset/'+dataset_name+'/'+feature_file_name+'.xlsx')
    labels = pd.read_excel('./dataset/'+dataset_name+'/labels.xlsx')
    # 1. 数据归一化
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(df)
    precisions = []
    recalls = []
    f1s = []

    n_repeats = 1
    for _ in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=None)

        for train_index, test_index in kf.split(X_normalized):
            X_train, X_test = X_normalized[train_index], X_normalized[test_index]
            y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
            if is_databalance == 1:
                # 使用SMOTE进行过采样
                X, y = smote(X_train, y_train)

                # 使用随机森林进行训练
                clf = SVC()
                clf.fit(X, y)
            else:
                # 使用随机森林进行训练
                clf = SVC()
                clf.fit(X_train, y_train)

            # 预测
            y_pred = clf.predict(X_test)

            # 计算精确率和召回率
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        # 计算并打印平均精确率、召回率和F1值
    print(f"Average Precision: {np.mean(precisions)}")
    print(f"Average Recall: {np.mean(recalls)}")
    print(f"Average F1: {np.mean(f1s)}")
    return round(np.mean(precisions),4), round(np.mean(recalls),4), round(np.mean(f1s),4)

#投票
def Voted(n_splits, is_databalance, dataset_name,  feature_file_name):
    df = pd.read_excel('./dataset/'+dataset_name+'/'+feature_file_name+'.xlsx')
    labels = pd.read_excel('./dataset/'+dataset_name+'/labels.xlsx')
    # 1. 数据归一化
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(df)
    precisions = []
    recalls = []
    f1s = []

    n_repeats = 1
    for _ in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=None)

        for train_index, test_index in kf.split(X_normalized):
            X_train, X_test = X_normalized[train_index], X_normalized[test_index]
            y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

            if is_databalance == 1:
                # 使用SMOTE进行过采样
                X, y = smote(X_train, y_train)

                # 使用随机森林进行训练
                clf = VotingClassifier(estimators=[
                    ('knn', KNeighborsClassifier(n_neighbors=5)),
                    ('nb', GaussianNB()),
                    ('svm', SVC(probability=True)),
                    ('logreg', LogisticRegression()),
                    ('rf', RandomForestClassifier())], voting='soft')
                clf.fit(X, y)
            else:
                # 使用随机森林进行训练
                clf = VotingClassifier(estimators=[
                    ('knn', KNeighborsClassifier(n_neighbors=5)),
                    ('nb', GaussianNB()),
                    ('svm', SVC(probability=True)),
                    ('logreg', LogisticRegression()),
                    ('rf', RandomForestClassifier())], voting='soft')
                clf.fit(X_train, y_train)

            # 预测
            y_pred = clf.predict(X_test)

            # 计算精确率和召回率
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        # 计算并打印平均精确率、召回率和F1值
    print(f"Average Precision: {np.mean(precisions)}")
    print(f"Average Recall: {np.mean(recalls)}")
    print(f"Average F1: {np.mean(f1s)}")
    return round(np.mean(precisions),4), round(np.mean(recalls),4), round(np.mean(f1s),4)



if __name__ == '__main__':
    # classifiers = ['Random Forest', 'KNN', 'Logistic Regression', 'Naive Bayes', 'SVM', 'Voting']
    precision_scores = []
    recall_scores = []
    f1_scores = []

    # classifiers = ['Random Forest', 'KNN', 'Logistic Regression', 'Naive Bayes', 'SVM', 'Voting']
    # for classifier in [random_forest, knn, logister_regression, naive_bayes, svm, Voted]:
    #     p, r, f1 = classifier()
    #     precision_scores.append(p)
    #     recall_scores.append(r)
    #     f1_scores.append(f1)
    #     print(f"{classifier.__name__} Precision:{p} Recall:{r} F1:{f1}")
    #
    # # # 设置柱状图的位置和宽度
    # # width = 0.2
    # x = np.arange(len(classifiers))
    #
    # # 绘制柱状图
    # plt.bar(x - width, precision_scores, width=width, label='Precision', align='center')
    # plt.bar(x, recall_scores, width=width, label='Recall', align='center')
    # plt.bar(x + width, f1_scores, width=width, label='F1 Score', align='center')
    #
    # # 在柱状图上方添加数字标签
    # for i in range(len(classifiers)):
    #     plt.text(x[i] - width, precision_scores[i] + 0.01, f'{precision_scores[i]:.2f}', ha='center', fontsize=8)
    #     plt.text(x[i], recall_scores[i] + 0.01, f'{recall_scores[i]:.2f}', ha='center', fontsize=8)
    #     plt.text(x[i] + width, f1_scores[i] + 0.01, f'{f1_scores[i]:.2f}', ha='center', fontsize=8)
    #
    # # 添加标签和标题
    # plt.xlabel('Classifiers')
    # plt.ylabel('Scores')
    # plt.title('Comparison of Different Classifiers')
    # plt.xticks(x, classifiers, fontsize=8)
    # plt.legend()
    #
    # # 调整坐标轴标签和标题的字体大小
    # plt.xlabel('Classifiers', fontsize=10)
    # plt.ylabel('Scores', fontsize=10)
    # plt.title('Comparison of Different Classifiers', fontsize=12)
    #
    # # 显示图形
    # plt.tight_layout()
    # plt.show()

