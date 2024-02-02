import gensim
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def generate_finegrain_feature(dataset_name):
    # Load the pretrained Word2Vec model
    word_vec = gensim.models.keyedvectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
    file_names = ['attributes.txt', 'classComments.txt',  'classNames.txt', 'methodComments.txt', 'methodNames.txt', 'methodParameters.txt', 'methodReturns.txt']
    with open('./docs/' + dataset_name + '/uc/uc_doc.txt','r', encoding='ISO8859-1') as urf:
        u_lines = urf.readlines()
    uc_document_vectors = []
    for line in u_lines:
        words = line.strip().split()
        # Calculate the word vector average of the document
        uc_document_vector = [word_vec[word] for word in words if word in word_vec]
        if uc_document_vector:
            uc_document_vector = sum(uc_document_vector) / len(uc_document_vector)
            uc_document_vectors.append(uc_document_vector)
        else:
            uc_document_vectors.append([0 for i in range(300)])
    res = []
    score1 = []
    for filename in file_names:
        with open('./fine-grain/'+dataset_name+'/cc/'+filename, 'r', encoding='ISO8859-1') as crf:
            c_lines = crf.readlines()
        # Process word vectors for each document
        cc_document_vectors = []
        for line in c_lines:
            words = line.strip().split()
            # Calculate the word vector average of the document
            cc_document_vector = [word_vec[word] for word in words if word in word_vec]
            if cc_document_vector:
                cc_document_vector = sum(cc_document_vector) / len(cc_document_vector)
                cc_document_vectors.append(cc_document_vector)
            else:
                cc_document_vectors.append([0 for i in range(300)])
        res_similarity = []
        scores = []
        # Calculate the similarity between documents
        for i in range(len(uc_document_vectors)):
            for j in range(len(cc_document_vectors)):
                similarity_score = cosine_similarity([uc_document_vectors[i]], [cc_document_vectors[j]])
                score = similarity_score[0][0]
                res_similarity.append((i + 1, j + 1, score))
                scores.append(score)
        score1.append(scores)
        res.append(res_similarity)
        print(filename)
    s_df = pd.DataFrame(score1)
    s_df = s_df.T
    s_df.columns = ['attr','classComm','className','methodComm','methodName','methodParam','methodReturn']
    s_df.to_excel('./fine-grain/'+dataset_name+'/fine_grain.xlsx', index=True)
    #TODO
    return res

if __name__ == '__main__':
    # datasets = ['Albergate', 'Derby', 'Drools', 'Eanci', 'EasyClinic', 'eTour', 'Groovy', 'Infinispan', 'iTrust',
                # 'maven', 'Pig', 'Seam', 'SMOS']
    # datasets = ['Infinispan','iTrust','maven','Pig','Seam','SMOS']
    datasets = ['Seam']
    for dataset in datasets:
        print(dataset)
        generate_finegrain_feature(dataset)