#!/usr/bin/env python
import glob
import pandas as pd

output_folders = glob.glob(f'../saved_output/classification/*')

#!/usr/bin/env python
import glob
import pandas as pd

output_folders = glob.glob(f'saved_output/classification/*')

for folder in output_folders:
    file_names = glob.glob(f'{folder}/*/eval_results_None.txt')
    print(f'No of files in {folder} : {len(file_names)}')
    acc_results = {}
    f1_results = {}
    for file in file_names:
        cat = file.split('/')[-2].split('_')[0]
        method = "_".join(file.split('/')[-2].split('_')[1:])
        with open(file, 'r') as f:
            for i, line in enumerate(f):
                if i == 1:
                    a_l = line.split()
                    metric = a_l[0]
                    value = a_l[-1]                   
                    if cat not in acc_results.keys():
                        acc_results[cat] = [{method: value}]
                    else:
                        acc_results[cat].append({method: value})
                if i == 2:
                    f_l = line.split()
                    metric = f_l[0]
                    value = f_l[-1]
                    if cat not in f1_results.keys():
                        f1_results[cat] = [{method: value}]
                    else:
                        f1_results[cat].append({method: value})
        f.close()
    
    acc_df = pd.DataFrame.from_dict(acc_results, orient='index')
    f1_df = pd.DataFrame.from_dict(f1_results, orient='index')

    acc_df2 = acc_df.transpose()
    f1_df2 = f1_df.transpose()

    methods = ['bert_mlm','mlm_emolex', 'bert', 'emo_mlm', \
             'mlm_health', 'emo_only_mlm', 'emo_wp_mlm', \
             'emo_wp_mlm_only'
            ]
    acc_df2.index = methods
    f1_df2.index = methods

    acc_df3 = acc_df2
    f1_df3 = acc_df2

    acc_df3[:] = None
    f1_df3[:] = None

    for i, j in acc_results.items():
        for k in j:
            index = list(k.keys())[0]
            acc_df3.loc[index, i] = k[index]
            f1_df3.loc[index, i] = k[index]

    acc_df3 = acc_df3.reindex(sorted(acc_df3.columns), axis=1)
    f1_df3 = f1_df3.reindex(sorted(f1_df3.columns), axis=1)

    acc_df3.to_csv(f'../data/senwave/{folder.split("/")[-1]}_acc_results.csv')
    f1_df3.to_csv(f'../data/senwave/{folder.split("/")[-1]}_f1_results.csv')