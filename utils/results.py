#!/usr/bin/env python
import glob
import pandas as pd

output_folders = glob.glob(f'saved_output/classification/*')

num_results_per_split = 80

li = {}
for folder in output_folders:
    file_names = glob.glob(f'{folder}/*/test_results.txt')
    if len(file_names) != num_results_per_split:
        print(f'NOT COMPLETE!!! No of files in {folder}: {len(file_names)} ')
        continue
    results = {}
    models = []
    for file in file_names:
        cat = file.split('/')[-2].split('_')[0]
        model = "_".join(file.split('/')[-2].split('_')[1:])
        models.append(model)
        with open(file, 'r') as f:
            for i, line in enumerate(f):
                l = line.split()
                metric = l[0]
                value = l[-1]
#                 import pdb; pdb.set_trace()
                if metric not in results.keys():
                    temp = {}
                    temp[cat] = [{model: value}]
                    results[metric] = temp
                else:
                    if cat not in results[metric].keys():
                        results[metric][cat] = [{model: value}]
                    else:
                        results[metric][cat].append({model: value})
    for key in results:
        df = pd.DataFrame.from_dict(results[key], orient='index', dtype=float)
        df2 = df.transpose()
        
        models = list(set(models))
        df2.index = models
        
        df3 = df2

        df3[:] = None
        
        for i, j in results[key].items():
            for k in j:
                index = list(k.keys())[0]
                df3.loc[index, i] = k[index]
                
        df3 = df3.reindex(sorted(df3.columns), axis=1)
        
        if key not in li.keys():
            li[key] = [df3]
        else:
            li[key].append(df3)

for key in li.keys():
    metric_df = pd.concat(li[key], axis=0)
    metric_df.to_csv(f'../data/senwave/{key}_results.csv')