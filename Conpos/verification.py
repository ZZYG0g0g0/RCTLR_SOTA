from resort import resort
import numpy as np
from matplotlib import pyplot as plt
from vsm import vsm
import os
from itertools import islice
import pandas as pd

def verification(dataset_name, ir_model):
    res = resort(dataset_name, ir_model)
    vsm_res = ir_model('./data/'+dataset_name+'/uc_words_removed.txt','./data/'+dataset_name+'/cc_words_removed.txt')
    v_res = vsm_res[2]

    true_sets = []
    """add"""
    with open('./dataset/' + dataset_name + '/true_set.txt', 'r', encoding='utf-8') as tf:
        lines = tf.readlines()
        for line in lines:
            true_sets.append(line.strip())
    cc_filenames = [f.strip() for f in os.listdir('./dataset/' + dataset_name + '/cc') if
                    os.path.isfile(os.path.join('./dataset/' + dataset_name + '/cc', f))]
    uc_filenames = [f.strip() for f in os.listdir('./dataset/' + dataset_name + '/uc') if
                    os.path.isfile(os.path.join('./dataset/' + dataset_name + '/uc', f))]
    for i in range(len(uc_filenames)):
        uc_filenames[i] = uc_filenames[i].split('.')[0]
    sim_dict1 = {}
    sim_dict2 = {}
    for item in res:
        key = (item[0],item[1])
        sim_dict1[key] = item[2]
    for item in v_res:
        key = (item[0],item[1])
        sim_dict2[key] = item[2]
    sorted_sim_dict1 = dict(sorted(sim_dict1.items(), key=lambda item: item[1], reverse=True))
    sorted_sim_dict2 = dict(sorted(sim_dict2.items(), key=lambda item: item[1], reverse=True))
    """add end"""

    thresholds = np.linspace(0.01,1,100)#set thresholds vector

    # uc_labels = []
    # with open('./data/'+dataset_name+'/uc_labels.txt','r',encoding='ISO8859-1')as ulf:#read uc_label
    #     lines = ulf.readlines()
    # for line in lines:
    #     uc_labels.append(line.strip().split('.')[0])
    #
    # cc_labels = []
    # with open('./data/'+dataset_name+'/cc_labels.txt','r',encoding='ISO8859-1')as clf:#read cc_label
    #     lines = clf.readlines()
    # for line in lines:
    #     cc_labels.append(line.strip())

    r_len = len(res)
    best_f1 = -1
    best_p = 0
    best_r = 0
    t = 0
    precisions_new = []  # P-value for IR_CRT
    recalls_new = []  # R-value for IR_CRT
    precisions_vsm = []  # P-value for IR(vsm)
    recalls_vsm = []  # R-value for IR(vsm)
    # for threshold in thresholds:
    #     print(f"current:p:{best_p},r:{best_r},f1{best_f1}")
    #     num = int(threshold*r_len)
    #     threshold_res_new = res[0:num]
    #     threshold_res_vsm = v_res[0:num]
    #     true_new = 0
    #     true_vsm = 0
    #     """2023 12 1"""
    #     # top_links = dict(islice(sorted_sim_dict.items(), candidate_link_len))
    #     # for key, _ in top_links.items():
    #     #     if(uc_filenames[key[0]]+' '+cc_filenames[key[1]] in true_set):
    #     #         correct_link += 1
    #     for item in threshold_res_new:
    #         for i in true_sets:
    #             if i[0]==uc_labels[item[0]-1] and i[1]==cc_labels[item[1]-1]:#check candidate links of IR_CRT
    #                 true_new += 1
    #                 break
    #     for item in threshold_res_vsm:
    #         for i in true_sets:
    #             if i[0] == uc_labels[item[0] - 1] and i[1] == cc_labels[item[1] - 1]:#check candidate links of IR
    #                 true_vsm += 1
    #                 break
    #     #calculate R-value and P-value
    #     p = true_new/num
    #     r = true_new/len(true_sets)
    #     recalls_new.append(r)
    #     precisions_new.append(p)
    #     f1 = 2*p*r/(p+r)
    #     if f1 > best_f1:
    #         best_f1 = f1
    #         t = threshold
    #         best_p = p
    #         best_r = r
    #     recalls_vsm.append(true_vsm / len(true_sets))
    #     precisions_vsm.append(true_vsm / num)
    for threshold in thresholds:
        correct_link1 = 0
        correct_link2 = 0
        candidate_link_len = int(len(sorted_sim_dict1)*threshold)
        top_links1 = dict(islice(sorted_sim_dict1.items(), candidate_link_len))
        top_links2 = dict(islice(sorted_sim_dict2.items(), candidate_link_len))
        for key, _ in top_links1.items():
            if(uc_filenames[key[0]-1]+' '+cc_filenames[key[1]-1] in true_sets):
                correct_link1 += 1
        for key, _ in top_links2.items():
            if (uc_filenames[key[0]-1] + ' ' + cc_filenames[key[1]-1] in true_sets):
                correct_link2 += 1
        p = correct_link1/candidate_link_len
        r = correct_link1/len(true_sets)
        if p+r == 0:
            f1=0
        else:
            f1 = 2*p*r/(p+r)
        if f1 > best_f1:
            best_f1 = f1
            best_p = p
            best_r = r
            t = threshold
        p_vsm = correct_link2/candidate_link_len
        r_vsm = correct_link2/len(true_sets)
        precisions_vsm.append(p_vsm)
        recalls_vsm.append(r_vsm)
        precisions_new.append(p)
        recalls_new.append(r)
    # Set horizontal axis scale and axis annotation
    # plt.xlim(0,1)
    # plt.xlabel('Recall')
    # plt.xticks(np.linspace(0,1,11))
    # # Set vertical axis scale and axis annotation
    # plt.ylim(0,1)
    # plt.ylabel('Precision')
    # plt.yticks(np.linspace(0,1,11))
    # plt.plot(recalls_new,precisions_new,linewidth=2,alpha=0.8,color='blue',label='IR_CRT(5% & 10%)')
    # plt.plot(recalls_vsm,precisions_vsm,linewidth=2,alpha=0.8,color='green',label='IR')
    # plt.legend()
    # plt.show()
    print('verification over')
    return best_p, best_r, best_f1, t

def verification_topn(dataset_name, ir_model):
    res = resort(dataset_name, ir_model)

    true_sets = []
    with open('./dataset/' + dataset_name + '/true_set.txt', 'r', encoding='utf-8') as tf:
        lines = tf.readlines()
        for line in lines:
            true_sets.append(line.strip())

    cc_filenames = [f.strip() for f in os.listdir('./dataset/' + dataset_name + '/cc') if os.path.isfile(os.path.join('./dataset/' + dataset_name + '/cc', f))]
    uc_filenames = [f.strip() for f in os.listdir('./dataset/' + dataset_name + '/uc') if os.path.isfile(os.path.join('./dataset/' + dataset_name + '/uc', f))]
    for i in range(len(uc_filenames)):
        uc_filenames[i] = uc_filenames[i].split('.')[0]

    sim_dict = {}
    for item in res:
        key = (item[0], item[1])
        sim_dict[key] = item[2]

    sorted_sim_dict = dict(sorted(sim_dict.items(), key=lambda item: item[1], reverse=True))

    # Select top 200 links
    top_links = dict(islice(sorted_sim_dict.items(), 200))

    correct_link = 0
    for key, _ in top_links.items():
        if uc_filenames[key[0] - 1] + ' ' + cc_filenames[key[1] - 1] in true_sets:
            correct_link += 1

    precision = correct_link / 200  # Precision for the model
    recall = correct_link / len(true_sets)  # Recall for the model

    print('verification over')
    return precision, recall, 2*precision*recall/(precision + recall)

def verification_threshold(dataset_name, ir_model):
    res = resort(dataset_name, ir_model)

    true_sets = []
    with open('./dataset/' + dataset_name + '/true_set.txt', 'r', encoding='ISO8859-1') as tf:
        lines = tf.readlines()
        for line in lines:
            true_sets.append(line.strip())

    cc_filenames = [f.strip() for f in os.listdir('./dataset/' + dataset_name + '/cc') if os.path.isfile(os.path.join('./dataset/' + dataset_name + '/cc', f))]
    uc_filenames = [f.strip() for f in os.listdir('./dataset/' + dataset_name + '/uc') if os.path.isfile(os.path.join('./dataset/' + dataset_name + '/uc', f))]
    for i in range(len(uc_filenames)):
        uc_filenames[i] = uc_filenames[i].split('.')[0]

    sim_dict = {}
    for item in res:
        key = (item[0], item[1])
        sim_dict[key] = item[2]

    sorted_sim_dict = dict(sorted(sim_dict.items(), key=lambda item: item[1], reverse=True))

    # Select links with a score greater than 0.7
    selected_links = {k: v for k, v in sorted_sim_dict.items() if v > 0.7}

    correct_link = 0
    for key, _ in selected_links.items():
        if uc_filenames[key[0] - 1] + ' ' + cc_filenames[key[1] - 1] in true_sets:
            correct_link += 1

    selected_links_count = len(selected_links)  # Number of selected links

    precision = correct_link / selected_links_count if selected_links_count > 0 else 0  # Precision for the model
    recall = correct_link / len(true_sets)  # Recall for the model
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print('verification over')
    return precision, recall, f1_score

def verification_between_low_and_high(dataset_name, ir_model):
    res = resort(dataset_name, ir_model)
    vsm_res = ir_model('./data/'+dataset_name+'/uc_words_removed.txt','./data/'+dataset_name+'/cc_words_removed.txt')
    v_res = vsm_res[2]

    true_sets = []
    """add"""
    with open('./dataset/' + dataset_name + '/true_set.txt', 'r', encoding='utf-8') as tf:
        lines = tf.readlines()
        for line in lines:
            true_sets.append(line.strip())
    cc_filenames = [f.strip() for f in os.listdir('./dataset/' + dataset_name + '/cc') if
                    os.path.isfile(os.path.join('./dataset/' + dataset_name + '/cc', f))]
    uc_filenames = [f.strip() for f in os.listdir('./dataset/' + dataset_name + '/uc') if
                    os.path.isfile(os.path.join('./dataset/' + dataset_name + '/uc', f))]
    for i in range(len(uc_filenames)):
        uc_filenames[i] = uc_filenames[i].split('.')[0]
    sim_dict1 = {}
    sim_dict2 = {}
    for item in res:
        key = (item[0],item[1])
        sim_dict1[key] = item[2]
    for item in v_res:
        key = (item[0],item[1])
        sim_dict2[key] = item[2]
    sorted_sim_dict1 = dict(sorted(sim_dict1.items(), key=lambda item: item[1], reverse=True))
    sorted_sim_dict2 = dict(sorted(sim_dict2.items(), key=lambda item: item[1], reverse=True))
    """add end"""
    thresholds = np.linspace(0.05,0.31,27)#set thresholds vector
    best_f1 = -1
    best_p = 0
    best_r = 0
    t = 0
    precisions_new = []  # P-value for IR_CRT
    recalls_new = []  # R-value for IR_CRT
    precisions_vsm = []  # P-value for IR(vsm)
    recalls_vsm = []  # R-value for IR(vsm)
    threshold_res = []
    fs = []
    for threshold in thresholds:
        correct_link1 = 0
        correct_link2 = 0
        candidate_link_len = int(len(sorted_sim_dict1)*threshold)
        top_links1 = dict(islice(sorted_sim_dict1.items(), candidate_link_len))
        top_links2 = dict(islice(sorted_sim_dict2.items(), candidate_link_len))
        for key, _ in top_links1.items():
            if(uc_filenames[key[0]-1]+' '+cc_filenames[key[1]-1] in true_sets):
                correct_link1 += 1
        for key, _ in top_links2.items():
            if (uc_filenames[key[0]-1] + ' ' + cc_filenames[key[1]-1] in true_sets):
                correct_link2 += 1
        p = correct_link1/candidate_link_len
        r = correct_link1/len(true_sets)
        if p+r == 0:
            f1=0
        else:
            f1 = 2*p*r/(p+r)
        t = threshold
        fs.append(f1)
        threshold_res.append(t)
        p_vsm = correct_link2/candidate_link_len
        r_vsm = correct_link2/len(true_sets)
        precisions_vsm.append(p_vsm)
        recalls_vsm.append(r_vsm)
        precisions_new.append(p)
        recalls_new.append(r)
    # Set horizontal axis scale and axis annotation
    plt.xlim(0,1)
    plt.xlabel('Recall')
    plt.xticks(np.linspace(0,1,11))
    # Set vertical axis scale and axis annotation
    plt.ylim(0,1)
    plt.ylabel('Precision')
    plt.yticks(np.linspace(0,1,11))
    plt.plot(recalls_new,precisions_new,linewidth=2,alpha=0.8,color='blue',label='IR_CRT(5% & 10%)')
    plt.plot(recalls_vsm,precisions_vsm,linewidth=2,alpha=0.8,color='green',label='IR')
    plt.legend()
    plt.show()
    print('verification over')
    return best_p, best_r, best_f1, t,threshold_res,fs


if __name__ == '__main__':
    # verification('iTrust')
    verification('EasyClinic')