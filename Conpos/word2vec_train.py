import gensim.models.keyedvectors
from sklearn.metrics.pairwise import cosine_similarity

def word2vec_train(dataset_name):
    # Load the pretrained Word2Vec model
    word_vec = gensim.models.keyedvectors.load_word2vec_format("GoogleNews-vectors-negative300.bin",binary=True)

    with open('./data/'+dataset_name+'/cc_words_removed.txt') as crf:
        lines = crf.readlines()

    # Process word vectors for each document
    document_vectors = []
    for line in lines:
        words = line.strip().split()
        # Calculate the word vector average of the document
        document_vector = [word_vec[word] for word in words if word in word_vec]
        if document_vector:
            document_vector = sum(document_vector)/len(document_vector)
            document_vectors.append(document_vector)
    res_similarity = []
    #Calculate the similarity between documents
    for i in range(len(document_vectors)):
        for j in range(i+1,len(document_vectors)):
            similarity_score = cosine_similarity([document_vectors[i]],[document_vectors[j]])
            score = similarity_score[0][0]
            res_similarity.append((i+1,j+1,score))

    sorted_res_similarity = sorted(res_similarity,key=lambda x:x[2],reverse=True)
    k2 = 0.10
    k2_res_cnt = int(k2 * len(sorted_res_similarity))
    return sorted_res_similarity[:k2_res_cnt+1]

if __name__ == '__main__':
    word2vec_train('iTrust')