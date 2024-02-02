import numpy as np
import operator
from vsm import vsm
from word2vec_train import word2vec_train

def resort(dataset_name, ir_model):
    vsm_res = ir_model('./data/'+dataset_name+'/uc_words_removed.txt','./data/'+dataset_name+'/cc_words_removed.txt')
    res1, res_total_vsm = vsm_res[0],vsm_res[1]
    max_col = len(res_total_vsm)
    max_row = len(res_total_vsm[0])
    matrix = [[0 for _ in range(max_row + 1)] for _ in range(max_col + 1)]
    for item in res1:
        matrix[item[0]][item[1]] = item[2]
    #calculate bonus delta
    vi_vector = []
    for i in res_total_vsm:
        vi_vector.append((max(i)-min(i))/2)
    median = np.median(vi_vector)#delta is vi_vector's median num
    print('before_train')
    res2 = word2vec_train(dataset_name)
    print('after_train')
    print("start")
    test1 = 0
    test2 = 0
    for i in res2:
        print(f"{i}")
        cc1, cc2 = i[0], i[1]
        for row in range(len(matrix)):
            if matrix[row][cc1] != 0 and matrix[row][cc2] == 0:
                res_total_vsm[row-1][cc2-1] += median*res_total_vsm[row-1][cc2-1]
                test1+=1
            elif matrix[row][cc1] == 0 and matrix[row][cc2] != 0:
                res_total_vsm[row-1][cc1-1] += median*res_total_vsm[row-1][cc1-1]
                test2+=1
    print(f"{test1} and {test2}")
    print('end')
    # sort
    final = []
    for i in range(len(res_total_vsm)):
        for j in range(len(res_total_vsm[i])):
            # print('keep going')
            final.append((i + 1, j + 1, res_total_vsm[i][j]))
    res = sorted(final, key=operator.itemgetter(2), reverse=True)
    print(len(res))
    return res

if __name__ == '__main__':
    resort('iTrust')