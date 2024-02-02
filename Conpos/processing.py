# @Time : 2023/09/06 15:21
# @Author : zzy
import os
import re

import stanfordcorenlp
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
import json
from nltk.stem import PorterStemmer

nlp = stanfordcorenlp.StanfordCoreNLP(r'./CoreNLP/stanford-corenlp-4.5.4')

#1.去除驼峰命名
def split_camel_case(s):
    # 将字符串s切分为单词列表
    words = s.split()
    # 对于列表中的每一个单词，如果它是驼峰命名的，则拆分它
    for index, word in enumerate(words):
        splitted = ' '.join(re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', word))
        if splitted:  # 如果成功拆分，则替换原单词
            words[index] = splitted

    # 将处理后的单词列表连接起来，并返回
    return ' '.join(words)

#2.去除下滑线命名
def split_under_line(s):
    words = s.split()
    for index, word in enumerate(words):
        splitted = ' '.join(re.split(r'_', word))
        if splitted:
            words[index] = splitted
    return ' '.join(words)

#8.去除所有非字母字符
def remove_non_alpha(s):
    str1 = ' '.join(s)
    # 使用正则表达式移除非字母字符
    return re.sub(r'[^a-zA-Z\s]', '', str1)


#3.分词
def tokenize_text(s):
    return ' '.join(word_tokenize(s))

#4.将词转为小写
def words_to_lowercase(s):
    words = s.split()
    return ' '.join([word.lower() for word in words])

#5.去除停用词，标点和数字
def filter_words(s):
    # nltk.download('stopword')
    stop_words = set(stopwords.words('english'))  # 停用词列表
    punctuation_symbols = set(string.punctuation)  # 标点符号列表
    words = []
    word_list = s.split()
    for word in word_list:
        if '.' in word:
            for each in word.split('.'):
                words.append(each)
        words.append(word)
    # 过滤停用词、标点符号和数字
    filtered_words = [word for word in words if word.lower() not in stop_words and
                      not any(char in punctuation_symbols for char in word) and
                      word.isalpha()]
    return ' '.join(filtered_words)

#6.词形还原
def extract_restore(text):
    # Initialize NLTK's PorterStemmer
    stemmer = PorterStemmer()

    def split_text(text, max_length):
        return [text[i:i + max_length] for i in range(0, len(text), max_length)]

    chunks = split_text(text, 100000)
    all_stems = []
    for chunk in chunks:
        doc = nlp.annotate(chunk, properties={
            'annotators': 'lemma',
            'pipelineLanguage': 'en',
            'outputFormat': 'json'
        })
        doc = json.loads(doc)
        lemmas = [word['lemma'] for sentence in doc['sentences'] for word in sentence['tokens']]
        stems = [stemmer.stem(token) for token in lemmas]
        all_stems.extend(stems)

    return ' '.join(all_stems)

#7.词性标注
def pos_tag(text):
    def split_text(text, max_length):
        return [text[i:1+max_length] for i in range(0, len(text), max_length)]

    chunks = split_text(text, 50000)
    tagged_words = []
    # 使用StanfordCoreNLP进行词性标注
    for chunk in chunks:
        annotated_text = nlp.annotate(chunk, properties={
            'annotators': 'pos',
            'pipelineLanguage': 'en',
            'outputFormat': 'json'
        })
        annotated_text = json.loads(annotated_text)
        # 将结果从JSON格式转换为Python的字典

        for sentence in annotated_text["sentences"]:
            for token in sentence["tokens"]:
                # 提取标注的单词和其词性标签
                tagged_words.append((token["word"], token["pos"]))
    return tagged_words

#预处理
def preprocessing(dataset_name):
    if not os.path.exists('./docs/'+dataset_name+'/cc'):
        os.makedirs('./docs/'+dataset_name+'/cc')
    if not os.path.exists('./docs/'+dataset_name+'/uc'):
        os.makedirs('./docs/'+dataset_name+'/uc')
    options = [split_camel_case, split_under_line, tokenize_text, words_to_lowercase, filter_words, extract_restore]
    file_names_cc = os.listdir('./dataset/'+ dataset_name +'/cc')#读文件夹cc的文件
    file_names_uc = os.listdir('./dataset/' + dataset_name + '/uc')  # 读文件夹uc的文件
    open('./docs/'+dataset_name+'/cc/cc_doc.txt', 'w').close()
    open('./docs/'+dataset_name+'/uc/uc_doc.txt', 'w').close()
    for file_name in file_names_cc:
        print(file_name)
        with open('./dataset/'+dataset_name+'/cc/'+file_name,'r',encoding='ISO8859-1')as cf:
            text = ""
            lines = cf.readlines()
            for line in lines:
                text += line.strip()
            for option in options:
                text = option(text)
        with open('./docs/'+dataset_name+'/cc/cc_doc.txt','a',encoding='ISO8859-1')as cwf:
            cwf.write(text)
            cwf.write('\n')
    for file_name in file_names_uc:
        with open('./dataset/'+dataset_name+'/uc/'+file_name,'r',encoding='ISO8859-1')as uf:
            text = ""
            lines = uf.readlines()
            for line in lines:
                text += line.strip()
            for option in options:
                text = option(text)
        with open('./docs/'+dataset_name+'/uc/uc_doc.txt','a',encoding='ISO8859-1')as uwf:
            uwf.write(text)
            uwf.write('\n')
    print('preprocessing over')

if __name__ == '__main__':
    preprocessing('iTrust')
