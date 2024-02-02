import feature_select
from machine_learning import random_forest, knn, logister_regression, naive_bayes, svm, Voted

if __name__ == '__main__':
    datasets = ['Albergate','Eanci','iTrust','SMOS','maven']
    for dataset in datasets:
        file = ["./dataset/"+dataset+"/IR.xlsx", "./dataset/"+dataset+"/QQ.xlsx", "./fine-grain/"+dataset+"/fine_grain.xlsx","./dataset/"+dataset+"/labels.xlsx"]
        feature_select.label_generate(dataset)
        test = feature_select.feature_generate(file)
        feature_select.feature_select(test, "selected_features", dataset)
        n_splits_sets = [2, 5, 10]
        is_data_balance = [0, 1]
        for classifier in [random_forest, knn, logister_regression, naive_bayes, svm, Voted]:
            for n_splits in n_splits_sets:
                for data_balance in is_data_balance:
                    p, r, f1 = classifier(n_splits, data_balance, dataset, "selected_features")
                    with open('./res1.txt', 'a', encoding='ISO8859-1')as rf:
                        rf.write(f"dataset:{dataset};  classifier:{classifier.__name__} n_splits:{n_splits}; is_data_balance:{data_balance}; Precision:{p}; Recall:{r}; F1:{f1}\n")


