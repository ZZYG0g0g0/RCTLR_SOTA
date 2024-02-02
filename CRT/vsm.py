import operator

from gensim import corpora
from gensim import models
from gensim import similarities
import re
import pandas as pd
import scipy
import scipy.stats
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
    # sort
    result = []
    sim = sim.T
    sim_list = sim.values.tolist()
    for i in range(len(sim_list)):
        # print(len(sim))
        for j in range(len(sim_list[i])):
            # print(len(sim.iloc[i]))
            result.append((i+1, j+1, sim_list[i][j]))
    result = sorted(result, key=operator.itemgetter(2), reverse=True)
    k1 = 0.05
    k1_res_cnt = int(k1 * len(result))

    print('lsi over')
    return result[:k1_res_cnt + 1], sim_list, result


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

    # sort
    result = []
    sim = sim.T
    sim_list = sim.values.tolist()
    for i in range(len(sim_list)):
        # print(len(sim))
        for j in range(len(sim_list[i])):
            # print(len(sim.iloc[i]))
            result.append((i + 1, j + 1, sim_list[i][j]))
    result = sorted(result, key=operator.itemgetter(2), reverse=True)
    k1 = 0.05
    k1_res_cnt = int(k1 * len(result))

    print('lda over')
    return result[:k1_res_cnt + 1], sim_list, result
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

    # sort
    result = []
    sim = sim.T
    sim_list = sim.values.tolist()
    for i in range(len(sim_list)):
        # print(len(sim))
        for j in range(len(sim_list[i])):
            # print(len(sim.iloc[i]))
            result.append((i + 1, j + 1, sim_list[i][j]))
    result = sorted(result, key=operator.itemgetter(2), reverse=True)
    k1 = 0.05
    k1_res_cnt = int(k1 * len(result))

    print('js over')
    return result[:k1_res_cnt + 1], sim_list, result

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


def vsm(data_file,test_file):
    uc_texts = []
    cc_texts = []
    with open(data_file,'r',encoding='ISO8859-1')as df:
        lines = df.readlines()
        for line in lines:
            uc_texts.append(line.strip().split(' '))
    with open(test_file,'r',encoding='ISO8859-1')as tf:
        lines = tf.readlines()
        for line in lines:
            cc_texts.append(line.strip().split(' '))

    #generate bag-of-words model
    dictionary = corpora.Dictionary(uc_texts)
    cc_corpus = [dictionary.doc2bow(text) for text in cc_texts]

    #calculate tf-idf
    tfidf_model = models.TfidfModel(cc_corpus)
    cc_corpus_tfidf = tfidf_model[cc_corpus]

    # Form a similarity matrix between cc documents
    cc_corpus_sim = similarities.MatrixSimilarity(cc_corpus_tfidf)

    # Calculate the similarity between uc and cc
    sim = []
    for i in range(len(uc_texts)):
        print(i)
        test_bow = dictionary.doc2bow(uc_texts[i])
        test_tfidf = tfidf_model[test_bow]
        test_sim = cc_corpus_sim[test_tfidf]
        sim.append(test_sim)

    #sort
    result = []
    for i in range(len(sim)):
        for j in range(len(sim[i])):
            result.append((i+1,j+1,sim[i][j]))
    result = sorted(result,key=operator.itemgetter(2),reverse=True)
    k1 = 0.05
    k1_res_cnt = int(k1 * len(result))
    print('vsm over')
    return result[:k1_res_cnt+1],sim,result
