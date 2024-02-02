"""
Author: 邹致远
Email: www.pisyongheng@foxmail.com
Date Created: 2023/11/27
Last Updated: 2023/11/27
Version: 1.0.0
"""
import verification
import vsm
import pandas as pd


if __name__ == '__main__':
    # datasets = ['iTrust', 'eTour', 'Albergate', 'EasyClinic', 'SMOS', 'Derby', 'Drools', 'Groovy', 'Infinispan', 'maven', 'pig', 'Seam']
    # datasets = ['Derby', 'Drools', 'Groovy', 'Infinispan', 'maven', 'pig', 'Seam']
    # with open('./res.txt', 'w', encoding='ISO-8859-1')as rf:
    #     rf.close()
    datasets = ['iTrust', 'eTour', 'Albergate', 'EasyClinic', 'SMOS']
    # ir_models = [vsm.lsi_similarity, vsm.lda_similarity, vsm.JS_similarity]
    ir_models = [vsm.vsm_similarity,vsm.lsi_similarity,vsm.lda_similarity,vsm.JS_similarity]
    results = []
    for ir_model in ir_models:
        for dataset in datasets:
            # processing.preprocessing(dataset)
            vsm.rerank('./docs/'+dataset+'/cc/cc_doc.txt', './docs/'+dataset+'/uc/uc_doc.txt',dataset, ir_model)
            # precision_vsm, recall_vsm = verification.verification('./docs/'+dataset+'/cc/cc_doc.txt', './docs/'+dataset+'/uc/uc_doc.txt', 'vsm', dataset)
            # precision_ConPos, recall_ConPos = verification.verification('./docs/'+dataset+'/cc/cc_doc.txt', './docs/'+dataset+'/uc/uc_doc.txt', 'ConPos', dataset)
            """截点"""
            precisions,recalls,_,_,_,threshold,f1s = verification.verification_threshold_between_low_to_high('./docs/'+dataset+'/cc/cc_doc.txt', './docs/'+dataset+'/uc/uc_doc.txt', 'ConPos', dataset,ir_model)
            # verification.plot_precision_recall(precision_vsm, recall_vsm, precision_ConPos ,recall_ConPos)
            # print(f"dataset:{dataset},p:{res_p},r:{res_r},f:{res_f},t:{threshold}")
            for i in range(len(threshold)):
                results.append({
                    'Dataset': dataset,
                    'IR Model': str(ir_model),
                    'Precision': precisions[i],
                    'Recall': recalls[i],
                    'F1 Score': f1s[i],
                    'Threshold': threshold[i]
                })

                # with open('./res.txt', 'a' , encoding='ISO8859-1')as rf:
                #     rf.write(f"ir_model:{ir_model.__name__};dataset:{dataset},p:{res_p},r:{res_r},f:{res_f},t:{threshold}\n")
    # Create a DataFrame from the results list
    df = pd.DataFrame(results)

    # Write the DataFrame to an Excel file
    df.to_excel('./results2.xlsx', index=False)
            #"""topn"""
            # res_p, res_r, res_f = verification.verification_by_topn('./docs/'+dataset+'/cc/cc_doc.txt', './docs/'+dataset+'/uc/uc_doc.txt', 'ConPos', dataset,ir_model)
            # print(f"dataset:{dataset},p:{res_p},r:{res_r},f:{res_f}")
            # with open('./res_top200.txt', 'a' ,encoding='ISO8859-1')as rf:
            #     rf.write(f"ir_model:{ir_model.__name__};dataset:{dataset},p:{res_p},r:{res_r},f:{res_f}\n")
    # Load the data from the Excel file
    # df = pd.read_excel('./results.xlsx')
    #
    # # Group the data by both 'Dataset' and 'Threshold'
    # for (dataset, threshold), group in df.groupby(['Dataset', 'Threshold']):
    #     f1_scores = group['F1 Score'].round(2)  # Round F1 scores to 2 decimal places
    #     f1_scores_str = '&'.join(f1_scores.astype(str))  # Join F1 scores with '&'
    #     print(f"{threshold} {dataset} {f1_scores_str}")