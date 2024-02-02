"""
Author: 邹致远
Email: www.pisyongheng@foxmail.com
Date Created: 2023/9/15
Last Updated: 2023/9/15
Version: 1.0.0
"""
import pandas as pd
import numpy as np
import os
from itertools import islice

import matplotlib.pyplot as plt

def plot_precision_recall(precision1, recall1, precision2, recall2):
    plt.figure(figsize=(10, 7))

    # 绘制第一条曲线
    plt.plot(recall1, precision1, label='VSM')

    # 绘制第二条曲线
    plt.plot(recall2, precision2, label='ConPos', color='red')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.legend()  # 显示图例
    plt.show()

def calculate_f1(precision, recall):
    return 2 * precision * recall / (precision + recall)

def verification_by_topn(fname, tname, method, dataset_name, ir_model):
    if method == 'ConPos':
        df = pd.read_excel('./docs/' + dataset_name + '/sim_matrix_' + ir_model.__name__ + '.xlsx', index_col=0)
    else:
        df = ir_model(fname, tname)

    true_set = []
    with open('./dataset/' + dataset_name + '/true_set.txt', 'r', encoding='ISO8859-1') as tf:
        lines = tf.readlines()
        for line in lines:
            true_set.append(line.strip())

    cc_filenames = [f.strip() for f in os.listdir('./dataset/' + dataset_name + '/cc') if
                    os.path.isfile(os.path.join('./dataset/' + dataset_name + '/cc', f))]
    uc_filenames = [f.strip() for f in os.listdir('./dataset/' + dataset_name + '/uc') if
                    os.path.isfile(os.path.join('./dataset/' + dataset_name + '/uc', f))]

    for i in range(len(uc_filenames)):
        uc_filenames[i] = uc_filenames[i].split('.')[0]

    sim_dict = {}
    for row in range(df.shape[0]):
        for col in range(df.shape[1]):
            value = df.iloc[row, col]
            key = (row, col)
            sim_dict[key] = value

    sorted_sim_dict = dict(sorted(sim_dict.items(), key=lambda item: item[1], reverse=True))

    # Select top 200 links
    top_links = dict(islice(sorted_sim_dict.items(), 200))

    correct_link = 0
    for key, _ in top_links.items():
        if (uc_filenames[key[0]] + ' ' + cc_filenames[key[1]] in true_set):
            correct_link += 1
    print(correct_link)
    p = correct_link / 200  # Precision
    r = correct_link / len(true_set)  # Recall
    f1 = calculate_f1(p, r)  # F1 score

    print('verification_topn over')
    return p, r, f1

def verification(fname, tname, method, dataset_name, ir_model):
    if method == 'ConPos':
        df = pd.read_excel('./docs/'+dataset_name+'/sim_matrix_'+ir_model.__name__+'.xlsx', index_col=0)
    else:
        df = ir_model(fname, tname)
    true_set = []
    with open('./dataset/' + dataset_name + '/true_set.txt', 'r', encoding='utf-8')as tf:
        lines = tf.readlines()
        for line in lines:
            true_set.append(line.strip())
    cc_filenames = [f.strip() for f in os.listdir('./dataset/'+dataset_name + '/cc') if os.path.isfile(os.path.join('./dataset/'+dataset_name + '/cc', f))]
    uc_filenames = [f.strip() for f in os.listdir('./dataset/'+dataset_name + '/uc') if os.path.isfile(os.path.join('./dataset/'+dataset_name + '/uc', f))]
    for i in range(len(uc_filenames)):
        uc_filenames[i] = uc_filenames[i].split('.')[0]
    sim_dict = {}
    for row in range(df.shape[0]):
        for col in range(df.shape[1]):
            value = df.iloc[row, col]
            # if value != 0:
            key = (row, col)
            sim_dict[key] = value
    sorted_sim_dict = dict(sorted(sim_dict.items(), key=lambda item: item[1], reverse=True))
    thresholds = np.linspace(0.01, 1, 100)#阈值选取
    precision = []
    recall = []
    res_f = res_p = res_r = -1
    threshold = -1
    for t in thresholds:
        correct_link = 0
        candidate_link_len = int(len(sorted_sim_dict)*t)
        top_links = dict(islice(sorted_sim_dict.items(), candidate_link_len))
        for key, _ in top_links.items():
            if(uc_filenames[key[0]]+' '+cc_filenames[key[1]] in true_set):
                correct_link += 1
        p = correct_link/candidate_link_len
        r = correct_link/len(true_set)
        f1 = calculate_f1(p, r)
        if f1 > res_f:
            res_f = f1
            res_p = p
            res_r = r
            threshold = t
        precision.append(p)
        recall.append(r)
    print('verification over')
    return precision, recall, res_p, res_r, res_f, threshold

def verification_threshold_between_low_to_high(fname, tname, method, dataset_name, ir_model):
    if method == 'ConPos':
        df = pd.read_excel('./docs/'+dataset_name+'/sim_matrix_'+ir_model.__name__+'.xlsx', index_col=0)
    else:
        df = ir_model(fname, tname)
    true_set = []
    with open('./dataset/' + dataset_name + '/true_set.txt', 'r', encoding='utf-8')as tf:
        lines = tf.readlines()
        for line in lines:
            true_set.append(line.strip())
    cc_filenames = [f.strip() for f in os.listdir('./dataset/'+dataset_name + '/cc') if os.path.isfile(os.path.join('./dataset/'+dataset_name + '/cc', f))]
    uc_filenames = [f.strip() for f in os.listdir('./dataset/'+dataset_name + '/uc') if os.path.isfile(os.path.join('./dataset/'+dataset_name + '/uc', f))]
    for i in range(len(uc_filenames)):
        uc_filenames[i] = uc_filenames[i].split('.')[0]
    sim_dict = {}
    for row in range(df.shape[0]):
        for col in range(df.shape[1]):
            value = df.iloc[row, col]
            # if value != 0:
            key = (row, col)
            sim_dict[key] = value
    sorted_sim_dict = dict(sorted(sim_dict.items(), key=lambda item: item[1], reverse=True))
    thresholds = np.linspace(0.05, 0.31, 27)#阈值选取
    precision = []
    recall = []
    threshold_res = []
    fs= []
    res_f = res_p = res_r = -1
    threshold = -1
    for t in thresholds:
        correct_link = 0
        candidate_link_len = int(len(sorted_sim_dict)*t)
        top_links = dict(islice(sorted_sim_dict.items(), candidate_link_len))
        for key, _ in top_links.items():
            if(uc_filenames[key[0]]+' '+cc_filenames[key[1]] in true_set):
                correct_link += 1
        p = correct_link/candidate_link_len
        r = correct_link/len(true_set)
        f1 = calculate_f1(p, r)
        f1_rounded = round(f1, 4) * 100
        precision.append(p)
        recall.append(r)
        threshold_res.append(t)
        fs.append(f1_rounded)
    print('verification over')
    return precision, recall, res_p, res_r, res_f, threshold_res,fs



# if __name__ == '__main__':
#     precision_vsm, recall_vsm = verification('./docs/iTrust/cc/cc_doc.txt', './docs/iTrust/uc/uc_doc.txt', 'vsm')
#     precision_ConPos, recall_ConPos = verification('./docs/iTrust/cc/cc_doc.txt', './docs/iTrust/uc/uc_doc.txt', 'ConPos')
#     plot_precision_recall(precision_vsm, recall_vsm, precision_ConPos, recall_ConPos)