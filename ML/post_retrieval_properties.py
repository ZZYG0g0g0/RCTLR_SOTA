"""
Author: 邹致远
Email: www.pisyongheng@foxmail.com
Date Created: 2023/9/7
Last Updated: 2023/9/7
Version: 1.0.0
"""
#后检索特征
# import math
# import os
# import shutil
#
# import numpy as np
# import IR_based_feature as ir
# from scipy.stats import spearmanr
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import pandas as pd
#
# options = [ir.vsm_similarity, ir.bm25_score, ir.JS_similarity]
#
# #检索后特征1
# def term_query_set(tname,dataset_name):
#     # 生成每个查询术语的查询集
#     with open(tname,'r',encoding='utf-8')as ff:
#         lines = ff.readlines()
#         idx = 0
#         if not os.path.exists('./docs/' + dataset_name +'/uc/word'):
#             os.makedirs('./docs/' + dataset_name +'/uc/word')
#         if not os.path.exists('./docs/' + dataset_name +'/cc/word'):
#             os.makedirs('./docs/' + dataset_name +'/cc/word')
#         for line in lines:
#             i = 0
#             words = line.split(' ')
#             with open('./docs/' + dataset_name + '/' + tname.split('/')[3] + '/word/' + str(idx) + '.txt','w', encoding='utf-8') as wf:
#                 for word in words:
#                     wf.write(word.strip()+'\n')
#             idx += 1
#
# #计算重叠度
# def calculate_overlap(list1, list2, k=10):
#     return len(set(list1[:k]).intersection(set(list2[:k])))
#
#
# #计算subquery_overlap特征(一般般慢)
# def subquery_overlap(fname, tname, dataset_name, output_fname=None):
#     term_query_set(tname, dataset_name)
#     results = {}
#     for option in options:
#         # 生成候选链接
#         sim = option(fname, tname)
#         sim = sim.astype(float)
#         top_10_cols_original = sim.apply(lambda row: row.nlargest(10).index.tolist(), axis=1)#对于每一行的listR里的top10
#         stddev_list = []
#         word_file_names = os.listdir('./docs/' + dataset_name + '/' + tname.split('/')[3] + '/word')
#         for idx, name in enumerate(word_file_names):
#             sim_term = option('./docs/' + dataset_name + '/' + tname.split('/')[3] + '/word/' + name, tname)
#             sim_term = sim_term.astype(float)
#             top_10_cols_term = sim_term.apply(lambda row: row.nlargest(10).index.tolist(), axis=1)#对每个文档里的术语也选top10
#
#             overlaps_for_term = []
#             for original_row, term_row in zip(top_10_cols_original, top_10_cols_term):
#                 overlap = calculate_overlap(original_row, term_row)
#                 overlaps_for_term.append(overlap)
#
#             # 计算标准差
#             stddev = np.std(overlaps_for_term)
#             stddev_list.append(stddev)
#
#         results[option.__name__+'_subquery_overlap'] = stddev_list
#     return results
#
# #检索后特征2(超级慢),由于跑的太慢，这个特征单独跑(我把列名还改了，记得改回来)
# def robustness_score(fname,tname,dataset_name,output_fname=None):
#     results = {}#存结果，为字典形式key = model name，value = feature matrix
#     # 获取文件的目录路径
#     dir_path = os.path.dirname(tname)
#     # 创建 'robust/query_set' 子目录（如果尚不存在）
#     query_set_path = os.path.join(dir_path, 'robust', 'query_set')
#     if not os.path.exists(query_set_path):
#         os.makedirs(query_set_path)
#     with open(tname, 'r', encoding='utf-8') as tf:
#         # lines = len(tf.readlines())  # 获取查询集的长度
#         tname_lines = tf.readlines()
#         lines = len(tname_lines)
#         for i, line in enumerate(tname_lines):
#             with open(os.path.join(query_set_path, f'tname_{i}.txt'), 'w', encoding='utf-8') as qf:
#                 qf.write(line)
#
#     for option in options:
#         # 创建'robust'子目录（如果尚不存在）
#         robust_path = os.path.join(dir_path, 'robust' ,option.__name__)
#         if not os.path.exists(robust_path):
#             os.makedirs(robust_path)
#         # 复制文件到'robust'目录并重命名为'copyi.txt'
#         for i in range(lines):
#             dest_path = os.path.join(robust_path, 'copy'+str(i)+'.txt')
#             shutil.copy(fname, dest_path)#将被查询集的文档复制到tname对应的路径下
#         avg_correlations = {j: [] for j in range(lines)}
#         for i in range(100):
#             # 生成每个查询集的候选链接
#             for j in range(lines):
#                 path = os.path.join(query_set_path,f'tname_{j}.txt')
#                 sim1 = option(os.path.join(robust_path, 'copy'+str(j)+'.txt'), path)
#                 sim1 = sim1.astype(float)
#                 num_cols = sim1.shape[1]#看看被查询集有多少
#                 n = 50 if num_cols >= 50 else num_cols // 2#如果被查询集大于50则取top_n=50，否则取前50%，这是因为iTrust中如果uc作为被查询集，则不满足50条，所以选取前50%，这与原文不符，但是是无奈之举
#                 # 获取前top_n的列
#                 row_series = sim1.iloc[0]
#                 top_n_cols_indices = row_series.nlargest(n).index.tolist()
#                 #分割文档和查询到单词列表
#                 with open(path,'r',encoding='utf-8')as qf:
#                     query = qf.readline()
#                 queried_set = []
#                 with open(os.path.join(robust_path, 'copy'+str(j)+'.txt'))as qf:#打开查询集对应的被查询集表
#                     queried = qf.readlines()
#                 for idx in top_n_cols_indices:#将需要修改的行放入queried_set中
#                     queried_set.append(queried[idx])
#                 change_doc = []#修改后的文档d'
#                 for item in queried_set:
#                     doc_terms = item.split()
#                     query_terms = query.split()
#                     doc_freq = {term: doc_terms.count(term) for term in set(doc_terms)}
#                     #初始化扰动文档
#                     perturbed_doc_terms = []
#                     #对文档中的每个术语
#                     for term, freq in doc_freq.items():
#                         if term not in query_terms:#条件b
#                             perturbed_doc_terms.extend([term] * freq)
#                         else:#条件c
#                             perturbed_freq = np.random.poisson(freq)#泊松分布
#                             perturbed_doc_terms.extend([term] * perturbed_freq)
#                     change_doc.append(' '.join(perturbed_doc_terms))
#                 for idx, line_idx in enumerate(top_n_cols_indices):
#                     queried[line_idx] = change_doc[idx] + '\n'
#                 with open(os.path.join(robust_path, 'copy'+str(j)+'.txt'),'w',encoding='utf-8')as file:#用扰动文档更新原文档
#                     file.writelines(queried)
#                 sim2 = option(os.path.join(robust_path, 'copy'+str(j)+'.txt'), path)
#                 sim2 = sim2.astype(float)
#                 new_row_series = sim2.iloc[0]
#                 new_rank_values = [new_row_series[idx] for idx in top_n_cols_indices]
#                 sorted_indices = sorted(range(len(new_rank_values)),key = lambda k: new_rank_values[k],reverse=True)
#                 new_ranks = [sorted_indices.index(x)+1 for x in range(len(new_rank_values))]
#                 rank_old = list(range(1,n+1))
#                 correlation,_ = spearmanr(rank_old, new_ranks)
#                 avg_correlations[j].append(correlation)
#         res_list = []  # 新建一个临时的结果列表
#         for j,corrs in avg_correlations.items():
#             avg_corr = sum(corrs)/len(corrs)
#             avg_correlations[j] = avg_corr
#             res_list.append(avg_corr)
#             # res_list.extend([avg_corr] * num_cols)  # 对于每个查询，创建一个列表，其中所有元素都是avg_corr
#         results[option.__name__+'_robustness_score'] = res_list
#     return results
#
# #检索后特征3(超级慢)，这个也单独跑
# def first_rank_change(fname,tname,dataset_name,output_fname=None):
#     results = {}#存结果，为字典形式key = model name，value = feature matrix
#     # 获取文件的目录路径
#     dir_path = os.path.dirname(tname)
#     # 创建 'robust/query_set' 子目录（如果尚不存在）
#     query_set_path = os.path.join(dir_path, 'robust', 'query_set')
#     if not os.path.exists(query_set_path):
#         os.makedirs(query_set_path)
#     with open(tname, 'r', encoding='utf-8') as tf:
#         # lines = len(tf.readlines())  # 获取查询集的长度
#         tname_lines = tf.readlines()
#         lines = len(tname_lines)
#         for i, line in enumerate(tname_lines):
#             with open(os.path.join(query_set_path, f'tname_{i}.txt'), 'w', encoding='utf-8') as qf:
#                 qf.write(line)
#
#     for option in options:
#         # 创建'robust'子目录（如果尚不存在）
#         robust_path = os.path.join(dir_path, 'robust' ,option.__name__)
#         if not os.path.exists(robust_path):
#             os.makedirs(robust_path)
#         # 复制文件到'robust'目录并重命名为'copyi.txt'
#         for i in range(lines):
#             dest_path = os.path.join(robust_path, 'copy'+str(i)+'.txt')
#             shutil.copy(fname, dest_path)#将被查询集的文档复制到tname对应的路径下
#         #在这里初始化让所有的查询集对应的score=0
#         scores = {j: 0 for j in range(lines)}
#         for i in range(100):
#             # 生成每个查询集的候选链接
#             for j in range(lines):
#                 path = os.path.join(query_set_path,f'tname_{j}.txt')
#                 sim1 = option(os.path.join(robust_path, 'copy'+str(j)+'.txt'), path)
#                 sim1 = sim1.astype(float)
#                 num_cols = sim1.shape[1]#看看被查询集有多少
#                 n = 50 if num_cols >= 50 else num_cols // 2#如果被查询集大于50则取top_n=50，否则取前50%，这是因为iTrust中如果uc作为被查询集，则不满足50条，所以选取前50%，这与原文不符，但是是无奈之举
#                 # 获取前top_n的列
#                 row_series = sim1.iloc[0]
#                 top_n_cols_indices = row_series.nlargest(n).index.tolist()
#                 #分割文档和查询到单词列表
#                 with open(path,'r',encoding='utf-8')as qf:
#                     query = qf.readline()
#                 queried_set = []
#                 with open(os.path.join(robust_path, 'copy'+str(j)+'.txt'))as qf:#打开查询集对应的被查询集表
#                     queried = qf.readlines()
#                 for idx in top_n_cols_indices:#将需要修改的行放入queried_set中
#                     queried_set.append(queried[idx])
#                 change_doc = []#修改后的文档d'
#                 for item in queried_set:
#                     doc_terms = item.split()
#                     query_terms = query.split()
#                     doc_freq = {term: doc_terms.count(term) for term in set(doc_terms)}
#                     #初始化扰动文档
#                     perturbed_doc_terms = []
#                     #对文档中的每个术语
#                     for term, freq in doc_freq.items():
#                         if term not in query_terms:#条件b
#                             perturbed_doc_terms.extend([term] * freq)
#                         else:#条件c
#                             perturbed_freq = np.random.poisson(freq)#泊松分布
#                             perturbed_doc_terms.extend([term] * perturbed_freq)
#                     change_doc.append(' '.join(perturbed_doc_terms))
#                 for idx, line_idx in enumerate(top_n_cols_indices):
#                     queried[line_idx] = change_doc[idx] + '\n'
#                 with open(os.path.join(robust_path, 'copy'+str(j)+'.txt'),'w',encoding='utf-8')as file:#用扰动文档更新原文档
#                     file.writelines(queried)
#                 sim2 = option(os.path.join(robust_path, 'copy'+str(j)+'.txt'), path)
#                 sim2 = sim2.astype(float)
#                 new_row_series = sim2.iloc[0]
#                 new_top_n_cols_indices = new_row_series.nlargest(n).index.tolist()
#                 # 检查扰动前后的第一个文档是否相同，如果是则score +1
#                 if top_n_cols_indices[0] == new_top_n_cols_indices[0]:
#                     scores[j] += 1
#         results[option.__name__ + '_first_rank_change'] = [scores[k] for k in scores]
#
#     return results
#
# #检索后特征4
# def clustering_tendency(fname, tname, dataset_name, output_fname=None):
#     return None
#
# #检索后特征5(很快)
# def spatial_auto_correlation(fname, tname, dataset_name, output_fname=None):
#     results = {}
#     # 读取文档
#     with open(fname, 'r', encoding='utf-8') as f:
#         documents = f.readlines()
#
#     for option in options:
#         sim = option(fname, tname)
#         sim = sim.astype(float)
#         num_cols = sim.shape[1]
#         n = 50 if num_cols >= 50 else num_cols // 2
#         top_n_cols_original = sim.apply(lambda row: row.nlargest(n).index.tolist(), axis=1)
#         correlations = []  # 存放相似度
#         for index, top_n_indices in top_n_cols_original.items():  # 修改此处
#             top_n_docs = [documents[i] for i in top_n_indices]
#             # 使用tf-idf进行向量化
#             vectorizer = TfidfVectorizer()
#             tfidf_matrix = vectorizer.fit_transform(top_n_docs)
#
#             # 计算所有文档对的余弦相似度
#             cosine_matrix = cosine_similarity(tfidf_matrix)
#             # 对于每个文档，找到与其最相似的5个文档并计算它们的平均相似度得分
#             derived_scores = []
#             for i in range(n):
#                 # 获取与文档d最相似的5个文档的索引
#                 similar_docs_indices = cosine_matrix[i].argsort()[-6:-1]  # -6是因为包含文档d本身
#                 similar_docs_scores = [sim.iloc[index, top_n_indices[idx]] for idx in similar_docs_indices]
#                 derived_scores.append(sum(similar_docs_scores) / 5)
#             origin_scores = [sim.iloc[index, i] for i in top_n_indices]
#             correlation_coefficient = np.corrcoef(origin_scores, derived_scores)[0][1]
#             correlations.append(correlation_coefficient)
#         results[option.__name__+'spatial_auto_correlation'] = correlations
#     return results
#
# #检索后特征6(没有告诉k的取值,且Pr(t|D)可能为0的情况（没有告诉如何进行平滑操作）)
# def WIG(fname, tname, dataset_name, output_fname=None):
#     return None
#
# #检索后特征7(快)
# def NQC(fname, tname, dataset_name, output_fname=None):
#     results = {}#存放结果
#     for option in options:#选择方法
#         sim = option(fname,tname)
#         sim = sim.astype(float)
#         num_cols = sim.shape[1]
#         n = 100 if num_cols >= 100 else num_cols // 2#选top100，但由于特殊情况（被查询集没有100项），所以如果不满100项则取前50%
#         top_n_cols_original = sim.apply(lambda row: row.nlargest(n).index.tolist(), axis=1)
#         mu_set = []#求mu
#         for index, top_n_indices in top_n_cols_original.items():
#             mu_set.append(sum([sim.iloc[index, idx] for idx in top_n_indices])/n)
#         NQC = []#存每个查询的NQC
#         for index, top_n_indices in top_n_cols_original.items():
#             score_d_mu = 0
#             for idx in top_n_indices:
#                 score_d_mu += pow(sim.iloc[index, idx] - mu_set[index], 2)#(score_d-mu)^2
#             NQC.append(math.sqrt(score_d_mu/n)/sim.iloc[index].sum())#NQC
#         results[option.__name__+'NQC'] = NQC
#     return results
#
# def post_retrieval_feature_generate(fname, tname, dataset_name, output_fname=None):
#     ops = [subquery_overlap, robustness_score, first_rank_change, spatial_auto_correlation, NQC]
#     with open(fname,'r',encoding='utf-8')as ff:
#         f_lines_len = len(ff.readlines())
#     with open(tname, 'r',encoding='utf-8')as tf:
#         t_lines_len = len(tf.readlines())
#     all_features = pd.DataFrame()
#     for op in ops:
#         pos = op(fname, tname, dataset_name)
#         for key in pos:
#             expanded_df = pd.concat([pd.DataFrame([value] * f_lines_len) for value in pos[key]], ignore_index=True)
#             all_features[key+'_pos'] = expanded_df.stack().reset_index(drop=True)
#         neg = op(tname, fname, dataset_name)
#         for key in neg:
#             # 对于每个值创建一个单独的列
#             df_list = [pd.DataFrame([value] * t_lines_len) for value in neg[key]]
#             combined_df = pd.concat(df_list, axis=1)
#             # 最后，堆叠并保存到all_features的新列中
#             all_features[key+'_neg'] = combined_df.stack().reset_index(drop=True)
#     output_file = output_fname if output_fname is not None else 'all_feature.xlsx'
#     all_features.to_excel(output_file)
#
# if __name__ == '__main__':
#     # subquery_overlap('./docs/iTrust/cc/cc_doc.txt','./docs/iTrust/uc/uc_doc.txt','iTrust')
#     # robustness_score('./docs/iTrust/cc/cc_doc.txt','./docs/iTrust/uc/uc_doc.txt','iTrust')
#     # first_rank_change('./docs/iTrust/cc/cc_doc.txt', './docs/iTrust/uc/uc_doc.txt', 'iTrust')
#     # spatial_auto_correlation('./docs/iTrust/cc/cc_doc.txt','./docs/iTrust/uc/uc_doc.txt','iTrust')
#     # NQC('./docs/iTrust/cc/cc_doc.txt','./docs/iTrust/uc/uc_doc.txt','iTrust')
#     post_retrieval_feature_generate('./docs/eTour/cc/cc_doc.txt','./docs/eTour/uc/uc_doc.txt','eTour','./dataset/eTour/post_retrieval.xlsx')
