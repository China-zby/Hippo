import os
import csv
import math

import yaml
from prettytable import PrettyTable

time_weight = 0.45
save_dir = "./experiments/experiment1"
result_dir = "./result"
object_classes = ['car', 'truck']

datasets = ['amsterdam']
methods = ['otif', 'dynamic_safe', 'cdbtune',
           'streamline', 'streamline_scl', 
           'streamline_lds', 'streamline_lds_scl',
           'streamline_rgo', 'streamline_sgo', 
           'streamline_sgo2', 'chameleon', 'streamline_lds_scl_sgo2', 'streamline_lds_scl_sgo3'] # os.listdir(result_dir)

output_result = {}
for dataset in datasets:
    if dataset not in output_result:
        output_result[dataset] = PrettyTable()
        field_names = ["Method"]
        for object_class in object_classes:
            field_names.append(object_class.capitalize() + " F1")
            field_names.append(object_class.capitalize() + " Accuracy")
            field_names.append(object_class.capitalize() + " Count Error")
            field_names.append(object_class.capitalize() + " Count Accuracy")
            field_names.append(object_class.capitalize() + " Topk Accuracy")
            field_names.append(object_class.capitalize() + " HOTA")
        field_names.append("Time")
        field_names.append("Score")
        output_result[dataset].field_names = field_names
    data_rows = []
    for method in methods:
        if not os.path.exists(os.path.join(result_dir, method, dataset, "results.yaml")):
            continue
        with open(os.path.join(result_dir, method, dataset, "results.yaml"), "r") as f:
            results = yaml.safe_load(f)
        with open(os.path.join("./configs", dataset, f"{method}.yaml"), "r") as f:
            config = yaml.safe_load(f)
        # print(f"./TrackEval/data/trackers/videodb/{config['database']['dataname']}S{config['videobase']['skipnumber']}-{config['database']['datatype']}/{config['methodname']}/time.txt") 
        time_path = f"./TrackEval/data/trackers/videodb/{config['database']['dataname']}S{config['videobase']['skipnumber']}-{config['database']['datatype']}/{config['methodname']}/time.txt"
        time = round(float(open(time_path, "r").readline().strip()), 1)
        
        data_row = [method]
        SCORES = []
        
        for object_class in object_classes:
            data_row.append(results[object_class]['f1'])
            SCORES.append(results[object_class]['f1'])
            
            data_row.append(results[object_class]['acc'])
            SCORES.append(results[object_class]['acc'])
            
            data_row.append(abs(results[object_class]['pred_count'] -
                                results[object_class]['gt_count']))
            data_row.append(round(100.0 - abs(results[object_class]['pred_count'] -
                                  results[object_class]['gt_count']) / results[object_class]['gt_count'] * 100.0, 1))
            SCORES.append(100.0 - abs(results[object_class]['pred_count'] -
                                      results[object_class]['gt_count']) / results[object_class]['gt_count'] * 100.0)
            
            data_row.append(results[object_class]['acc_topk'])
            SCORES.append(results[object_class]['acc_topk'])
            
            data_row.append(results[object_class]['HOTA'])
            SCORES.append(results[object_class]['HOTA'])

        data_row.append(time)
        SCORES.append(math.exp(-time / 500) * 100)
        SCORE = sum(SCORES[:-1]) / len(SCORES[:-1]) * (1 - time_weight) + SCORES[-1] * time_weight
        
        data_row.append(round(SCORE, 1))
        data_rows.append(data_row)
    data_rows = sorted(data_rows, key=lambda x: x[-1], reverse=False)
    for data_row in data_rows: output_result[dataset].add_row(data_row)
    
    print(output_result[dataset])
        
    header = output_result[dataset].field_names

    # 保存为CSV文件
    with open(f'./paper_experiments/experiment1_{dataset}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data_rows)
