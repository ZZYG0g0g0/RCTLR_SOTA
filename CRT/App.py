from verification import verification_between_low_and_high
from verification import verification
from verification import verification_topn
from verification import verification_threshold
from preprocessed import preprocessed
import vsm
import pandas as pd


if __name__ == '__main__':
    # datasets = ['iTrust','Albergate','EasyClinic','eTour', 'SMOS', 'Derby','Drools', 'Groovy', 'Infinispan', 'maven','Pig' ,'Seam']
    datasets = ['Albergate']
    with open('./res.txt','w',encoding='ISO8859-1')as rf:
        rf.close()
    """截点"""
    ir_models = [vsm.vsm, vsm.lsi_similarity, vsm.lda_similarity,vsm.JS_similarity]
    for dataset in datasets:
        # preprocessed(dataset)
        for ir_model in ir_models:
            p,r,f,t = verification(dataset, ir_model)
            with open('./res.txt','a',encoding='ISO8859-1')as rf:
                rf.write(f'model:{ir_model.__str__()};dataset:{dataset};p:{p};r{r};f1{f};t{t}\n')
    """top_n"""
    # ir_models = [vsm.vsm,vsm.lda_similarity, vsm.JS_similarity,vsm.lsi_similarity]
    # for dataset in datasets:
    #     for ir_model in ir_models:
    #         p,r,f = verification_topn(dataset,ir_model)
    #         with open('./res_topn.txt', 'a', encoding='ISO8859-1')as rf:
    #             rf.write(f'model:{ir_model.__str__()};dataset:{dataset};p:{p};r{r};f1{f}\n')
    """thresholds"""
    # ir_models = [vsm.vsm,vsm.lsi_similarity, vsm.lda_similarity,vsm.JS_similarity]
    # results = []
    # for ir_model in ir_models:
    #     # preprocessed(dataset)
    #     for dataset in datasets:
    #         _,_,_,_,ts,fs = verification_between_low_and_high(dataset, ir_model)
    #         for i in range(len(ts)):
    #                 results.append({
    #                     'Dataset': dataset,
    #                     'IR Model': str(ir_model),
    #                     # 'Precision': precisions[i],
    #                     # 'Recall': recalls[i],
    #                     'F1 Score': fs[i],
    #                     'Threshold': ts[i]
    #                 })
    # df = pd.DataFrame(results)
    # df.to_excel('./results.xlsx', index=False)
            # with open('./res.txt','a',encoding='ISO8859-1')as rf:
                # rf.write(f'model:{ir_model.__str__()};dataset:{dataset};p:{p};r{r};f1{f};t{t}\n')
    # df = pd.read_excel('./results.xlsx')
    # df['F1 Score'] *= 100
    # # Group the data by both 'Dataset' and 'Threshold'
    # for (dataset, threshold), group in df.groupby(['Dataset', 'Threshold']):
    #     f1_scores = group['F1 Score'].round(2)  # Round F1 scores to 2 decimal places
    #     f1_scores_str = '&'.join(f1_scores.astype(str))  # Join F1 scores with '&'
    #     print(f"{threshold} {dataset} {f1_scores_str}")