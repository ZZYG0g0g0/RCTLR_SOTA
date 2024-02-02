import os
import re
import string
import json
import nltk
from nltk.corpus import stopwords
import stanfordcorenlp
from nltk.stem import PorterStemmer

nlp = stanfordcorenlp.StanfordCoreNLP(r'./CoreNLP/stanford-corenlp-4.5.4')

def wordToken(sentence):#split words
    words = nltk.word_tokenize(sentence)
    return words

def split_camel_case_word(word):#Tokenize words in the code according to camelCase naming convention
    words = re.findall(r'[A-Z]?[a-z]+',word)
    return words

#read uc files
def read_files_uc(dataset_name):
    if not os.path.exists('./data/' + dataset_name +'/uc'):
        os.makedirs('./data/' + dataset_name +'/uc')
    file_list = os.listdir('./dataset/' + dataset_name + '/uc/')
    file_labels = []
    with open('./data/' + dataset_name + '/uc_words.txt', 'w', encoding='ISO-8859-1') as wf:#Delete the original file content
        wf.truncate()
    for file_name in file_list:
        print(file_name)
        file_labels.append(file_name)#add file name
        file_path = os.path.join('./dataset/' + dataset_name + '/uc/', file_name)#generate file path
        with open(file_path, 'r', encoding='ISO-8859-1') as f:#read file
            lines = f.readlines()
            file_words = []
            for line in lines:
                file_words.append(wordToken(line))
                if not os.path.exists('./data/'+dataset_name+'/'):
                    os.makedirs('./data/'+dataset_name+'/')#create file path
            with open('./data/'+dataset_name+'/uc_words.txt','a',encoding='ISO8859-1') as wf:
                for words in file_words:
                    for word in words:
                        wf.write(word+' ')
                wf.write('\n')
    with open('./data/' + dataset_name + '/uc_labels.txt','w',encoding='ISO8859-1')as lf:
        for label in file_labels:
            lf.write(label+'\n')

#read cc files
def read_files_cc(dataset_name):
    file_list = os.listdir('./dataset/' + dataset_name + '/cc/')
    file_labels = []
    with open('./data/' + dataset_name + '/cc_words.txt', 'w', encoding='ISO8859-1') as wf:#Delete the original file content
        wf.truncate()
    for file_name in file_list:
        print(file_name)
        file_labels.append(file_name)#add file name
        file_path = os.path.join('./dataset/' + dataset_name + '/cc/', file_name)#generate file path
        with open(file_path, 'r', encoding='ISO8859-1') as f:#read file
            lines = f.readlines()
            file_words = []
            for line in lines:
                file_words.append(wordToken(line))
                if not os.path.exists('./data/'+dataset_name+'/'):
                    os.makedirs('./data/'+dataset_name+'/')#create file path
            with open('./data/'+dataset_name+'/cc_words.txt','a',encoding='ISO8859-1') as wf:
                for words in file_words:
                    for word in words:
                        word_without_punctuation = re.split(r'[./]',word)#split by '.' and '/' to split word like 'arrayList.add()' or '/a/b.jsp' etc
                        for word_w_p in word_without_punctuation:
                            split_words = split_camel_case_word(word_w_p)#Tokenize words in the code according to camelCase naming convention
                            for each in split_words:
                                wf.write(each+' ')
                wf.write('\n')
    with open('./data/' + dataset_name + '/cc_labels.txt','w',encoding='ISO8859-1')as lf:
        for label in file_labels:
            lf.write(label+'\n')

#Remove stop words, punctuation marks, and specific words.
def remove_words(dataset_name):
    file_path_cc = './data/'+dataset_name+'/cc_words.txt'
    file_path_uc = './data/'+dataset_name+'/uc_words.txt'
    process_file(file_path_uc,dataset_name)    #process uc file
    process_file(file_path_cc,dataset_name)    #process cc file

def process_file(file_path,dataset_name):
    # nltk.download('stopwords')#download stopwords data if you haven't download it yet
    stop_words = set(stopwords.words('english'))#stop words list
    specific_words = set([dataset_name,dataset_name.lower()])#specific words list
    punctuation_symbols = string.punctuation#punctuation list
    with open(file_path, 'r', encoding='ISO8859-1') as wf:
        if file_path.split('/')[-1][0:2]=='cc':#Determine if it is a cc file or a uc file
            store_name = '/cc_words_removed.txt'
        else:
            store_name = '/uc_words_removed.txt'
        with open('./data/' + dataset_name + store_name, 'w', encoding='ISO8859-1') as rf:#create file if it not exist and empty file content if it exist
            rf.truncate()
        lines = wf.readlines()
        for line in lines:
            words = line.strip().split(' ')
            words = [word for word in words if not any(
                char in punctuation_symbols for char in word)]  # remove punctuation and any words with punctuation
            filtered_words = [word for word in words if
                              word.lower() not in stop_words and word not in specific_words]  # remove stopwords and specific words
            with open('./data/' + dataset_name + store_name, 'a', encoding='ISO8859-1') as crf:
                crf.write(' '.join(filtered_words) + '\n')

#Extract stems and restore word forms
# def extract_restore(dataset_name,artifact):
#     # Initialize NLTK's PorterStemmer
#     stemmer = PorterStemmer()
#     with open('./data/' + dataset_name + '/'+artifact+'_words_final.txt', 'w', encoding='ISO8859-1') as uff:
#         uff.truncate()
#     with open('./data/'+dataset_name+'/'+artifact+'_words_removed.txt','r',encoding='ISO8859-1') as rf:
#         lines = rf.readlines()
#         for line in lines:
#             doc = nlp.annotate(line,properties={
#                 'annotators':'lemma',
#                 'pipelineLanguage':'en',
#                 'outputFormat':'json'
#             })
#             doc = json.loads(doc)
#             lemmas = [word['lemma'] for sentence in doc['sentences'] for word in sentence['tokens']]
#             stems = [stemmer.stem(token) for token in lemmas]
#             with open('./data/' + dataset_name + '/'+artifact+'_words_final.txt', 'a', encoding='utf-8') as uff:
#                 print(stems)
#                 for word in stems:
#                     uff.write(word+' ')
#                 uff.write('\n')

def extract_restore(dataset_name, artifact):
    # Initialize NLTK's PorterStemmer
    stemmer = PorterStemmer()

    def split_text(text, max_length):
        return [text[i:i + max_length] for i in range(0, len(text), max_length)]

    with open('./data/' + dataset_name + '/'+artifact+'_words_final.txt', 'w', encoding='ISO8859-1') as uff:
        uff.truncate()

    with open('./data/'+dataset_name+'/'+artifact+'_words_removed.txt','r',encoding='ISO8859-1') as rf:
        lines = rf.readlines()
        for line in lines:
            chunks = split_text(line, 50000)  # Assuming 100000 is a safe chunk size
            all_stems = []
            for chunk in chunks:
                doc = nlp.annotate(chunk, properties={
                    'annotators':'lemma',
                    'pipelineLanguage':'en',
                    'outputFormat':'json'
                })
                doc = json.loads(doc)
                lemmas = [word['lemma'] for sentence in doc['sentences'] for word in sentence['tokens']]
                stems = [stemmer.stem(token) for token in lemmas]
                all_stems.extend(stems)

            with open('./data/' + dataset_name + '/'+artifact+'_words_final.txt', 'a', encoding='utf-8') as uff:
                for word in all_stems:
                    uff.write(word + ' ')
                uff.write('\n')


def preprocessed(dataset_name):
    #1.read file and tokenize words
    #read uc file
    read_files_uc(dataset_name)
    #read cc file
    read_files_cc(dataset_name)

    #2Remove stop words, punctuation marks, and specific words.
    remove_words(dataset_name)

    #3Extract stems and restore word forms
    extract_restore(dataset_name,'cc')
    extract_restore(dataset_name,'uc')
    print('preprocessing over')



if __name__ == '__main__':
    # preprocessed('iTrust')
    preprocessed('EasyClinic')
