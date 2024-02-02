"""
Author: 邹致远
Email: www.pisyongheng@foxmail.com
Date Created: 2023/9/15
Last Updated: 2023/9/15
Version: 1.0.0
"""
from processing import pos_tag
from gensim import corpora, similarities
from gensim import models
import re
import pandas as pd
import scipy
import numpy as np


def set_generation(file):
    with open(file, 'r', encoding='ISO-8859-1')as f:
        lines_T = f.readlines()
    set_lines = []
    for line in lines_T:
        word = line.split(' ')
        word = [re.sub('\s', '', i) for i in word]
        word = [i for i in word if len(i) > 0]
        set_lines.append(word)
    return set_lines

# VSM相似度计算
def vsm_similarity(queried_file, query_file):
    # 生成被查询集
    queried_line = set_generation(queried_file)
    # 生成查询集
    query_line = set_generation(query_file)

    # 被查询集生成词典和corpus
    dictionary = corpora.Dictionary(queried_line)
    corpus = [dictionary.doc2bow(text) for text in queried_line]

    # 计算tfidf值
    tfidf_model = models.TfidfModel(corpus)
    corpus_tfidf = tfidf_model[corpus]

    # 待检索的文档向量初始化一个相似度计算的对象
    corpus_sim = similarities.MatrixSimilarity(corpus_tfidf)

    # 查询集生成corpus和tfidf值
    query_corpus = [dictionary.doc2bow(text) for text in query_line]  # 在每句话中每个词语出现的频率
    query_tfidf = tfidf_model[query_corpus]
    sim = pd.DataFrame(corpus_sim[query_tfidf])

    return sim

# LSI相似度计算
def lsi_similarity(queried_file, query_file):
    queried_line = set_generation(queried_file)
    query_line = set_generation(query_file)

    # 被查询集生成词典和corpus
    dictionary = corpora.Dictionary(queried_line)
    corpus = [dictionary.doc2bow(text) for text in queried_line]

    # 计算tfidf值
    tfidf_model = models.TfidfModel(corpus)
    corpus_tfidf = tfidf_model[corpus]

    # 生成lsi主题
    lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary)
    corpus_lsi = lsi_model[corpus_tfidf]

    # 待检索的文档向量初始化一个相似度计算的对象
    corpus_sim = similarities.MatrixSimilarity(corpus_lsi)

    # 查询集生成corpus和tfidf值
    query_corpus = [dictionary.doc2bow(text) for text in query_line]  # 在每句话中每个词语出现的频率
    query_tfidf = tfidf_model[query_corpus]
    query_lsi = lsi_model[query_tfidf]
    sim = pd.DataFrame(corpus_sim[query_lsi])

    return sim

def HellingerDistance(p, q):
    """
    计算HellingerDistance距离
    :param p:
    :param q:
    :return: float距离
    """
    return 1 - (1 / np.sqrt(2) * np.linalg.norm(np.sqrt(p) - np.sqrt(q)))

def hellingerSim(A_set, B_set, topic_number):
    """
    计算两个集合中每条数据之间的Hellinger距离
    :param A_set: 被查询集
    :param B_set: 查询集
    :return: 一个 len(B_set) * len(A_set) 的 pandas.DataFrame
    """
    df = pd.DataFrame(index=range(len(B_set)), columns=range(len(A_set)))
    A_matrix = np.zeros((len(A_set), topic_number))
    B_matrix = np.zeros((len(B_set), topic_number))

    # 将A_set和B_set分别转化为List[List[float]](e.i. 二维矩阵)
    row = 0
    for tu in A_set:
        for i in tu:
            A_matrix[row][i[0]] = i[1]
        row = row + 1
    row = 0
    for tu in B_set:
        for i in tu:
            B_matrix[row][i[0]] = i[1]
        row = row + 1

    # 开始计算Hellinger距离
    for row in range(len(B_set)):
        for column in range(len(A_set)):
            df.iloc[[row], [column]] = HellingerDistance(B_matrix[row], A_matrix[column])  # B_matrix为查询集，所以放前面
    return df

# LDA相似度计算
def lda_similarity(queried_file, query_file):
    queried_line = set_generation(queried_file)
    query_line = set_generation(query_file)

    # 被查询集生成词典和corpus
    dictionary = corpora.Dictionary(queried_line)
    corpus = [dictionary.doc2bow(text) for text in queried_line]

    # 生成lda主题
    topic_number = 100
    lda_model = models.LdaModel(corpus, id2word=dictionary, num_topics=topic_number, random_state=0)
    document_topic = lda_model.get_document_topics(corpus)

    # 查询集生成corpus和tfidf值
    query_corpus = [dictionary.doc2bow(text) for text in query_line]  # 在每句话中每个词语出现的频率
    query_lda = lda_model.get_document_topics(query_corpus)

    sim = hellingerSim(document_topic, query_lda, topic_number)

    return sim

# JS散度相似度计算
def JS_similarity(queried_file, query_file):
    queried_line = set_generation(queried_file)
    query_line = set_generation(query_file)

    # 被查询集生成词典和corpus
    dictionary = corpora.Dictionary(queried_line + query_line)
    corpus = [dictionary.doc2bow(text) for text in queried_line]
    corpus2 = [dictionary.doc2bow(text) for text in query_line]
    A_matrix = np.zeros((len(queried_line), len(dictionary)))
    B_matrix = np.zeros((len(query_line), len(dictionary)))

    row = 0
    for document in corpus:
        for word_id, freq in document:
            A_matrix[row][word_id] = freq
        row = row + 1

    row = 0
    for document in corpus2:
        for word_id, freq in document:
            B_matrix[row][word_id] = freq
        row = row + 1

    sum_matrix = np.sum(np.vstack((A_matrix, B_matrix)), axis=0)
    probability_A = A_matrix / sum_matrix
    probability_B = B_matrix / sum_matrix

    sim = JS_Sim(probability_A, probability_B)

    return sim

def JS_Sim(A_set, B_set) -> pd.DataFrame:
    df = pd.DataFrame(index=range(len(B_set)), columns=range(len(A_set)))
    # 开始计算JS相似度
    for row in range(len(B_set)):
        for column in range(len(A_set)):
            df.iloc[[row], [column]] = JS_divergence(B_set[row], A_set[column])  # B_set为查询集，所以放前面
    return df

def JS_divergence(p, q):
    M = (p + q) / 2
    pk = np.asarray(p)
    pk2 = np.asarray(q)
    a = 0
    b = 0
    if (np.sum(pk, axis=0, keepdims=True) != 0):
        a = 0.5 * scipy.stats.entropy(p, M)
    if (np.sum(pk2, axis=0, keepdims=True) != 0):
        b = 0.5 * scipy.stats.entropy(q, M)
    return a + b  # 选用自然对数

def rerank(fname, tname,dataset_name, ir_model):
    sim_matrix = ir_model(fname, tname)
    res = []
    pos_tag_dict = {}
    with open(fname, 'r', encoding='ISO8859-1')as cf:
        lines = cf.readlines()
    for index, line in enumerate(lines):
        pos_line = pos_tag(line)
        pos_tag_dict[index] = pos_line
    with open(tname, 'r', encoding='ISO8859-1')as uf:
        lines = uf.readlines()
        for key, value in pos_tag_dict.items():#遍历每个cc
            for index, line in enumerate(lines):#遍历每个uc行
                param_lambda = 0
                line_list = line.split(' ')
                for item in value:
                    if item[1] == 'VB' and item[0] in line_list:#如果词为动词，且在uc中
                        param_lambda += 1
                if param_lambda != 0:
                    sim_value = sim_matrix.iloc[index, key]
                    sim_matrix.iloc[index, key] += param_lambda*sim_value/10
                else:
                    sim_matrix.iloc[index, key] = 0
    sim_matrix.to_excel('./docs/' + dataset_name + '/sim_matrix_'+ir_model.__name__+'.xlsx', engine='openpyxl')
    print('rerank over')
    return sim_matrix

# if __name__ == '__main__':
#     rerank('./docs/iTrust/cc/cc_doc.txt', './docs/iTrust/uc/uc_doc.txt','iTrust')