import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

# 方法定义
methods = ['hippo', 'hippo_scluster']
# methods = ['otif', 'hippo']

# 初始化要打印或画图的指标
# metrics = {"Q-1": "CarSel", 
#         #    "Q3": "BusSel",
#            "Q-4": "CarAgg", 
#         #    "Q6": "BusAgg",
#            "Q-7": "CarTopk",
#         #    "Q9": "BusTopk",
           
#            "Q-10": "CarCSel", 
#         #    "Q12": "BusCSel",
#             "Q-13": "CarCAgg", 
#             "Q-15": "BusCAgg",
#             "Q-16": "CarCTopk", 
#             # "Q18": "BusCTopk",
            
#             "Q-19": "CarCQ1", 
#             # "Q21": "BusCQ1",
#             "Q-22": "CarCQ2",
#             # "Q25": "CarCQ3",
#             "Q-28": "CarCCQ1", 
#             # "Q30": "BusCCQ1",
            
#         #    "Car HOTA": "CarHota", "BUS HOTA": "BusHota"
#            } # "Truck HOTA": "TruckHota",

metrics = {
           "Q1": "CarSel",
           "Q2": "BusSel",
           "Q3": "TruckSel", 
           
           "Q4": "CarAgg",
           "Q5": "BusAgg",
           "Q6": "TruckAgg",
           
           "Q7": "CarCQ3",
           "Q8": "BusCQ3",
           "Q9": "TruckCQ3",
           
           "MOT Quality": "CarHota",
        }

# 文件夹路径
folder_path = 'ingestion_results'

# 初始化一个空的字典
results = {}
all_metric_best_methods = {}

# 遍历文件夹中的每个子文件夹
for subfolder in ['200']: # os.listdir(folder_path): , '150', '200', '250', '300'
    metric_best_methods = {metric: [None, -float('inf')] for metric in metrics}
    video_number = int(subfolder.split('_')[0])
    if video_number not in results:
        results[video_number] = {}
    subfolder_path = os.path.join(folder_path, subfolder)
    if os.path.isdir(subfolder_path):
        for method in methods:
            # 读取json文件
            json_file = os.path.join(subfolder_path, method + '.json')
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    method_result = json.load(f)
                    # print(method_result)
                    # 所有数据保留两位小数
                    method_result = {k: round(v, 4)
                                     for k, v in method_result.items()}
                    if method not in results[video_number]:
                        results[video_number][method] = method_result
                    for metric in metrics:
                        if method_result[metrics[metric]] >= metric_best_methods[metric][1]:
                            metric_best_methods[metric] = [
                                method, method_result[metrics[metric]]]
    all_metric_best_methods[video_number] = metric_best_methods

# 打印表格
for video_number in results:
    table = PrettyTable()
    table.field_names = ["Method"] + list(metrics.keys()) + ["Mean"]
    for method in methods:
        row = [method]
        metric_values = []
        for metric in metrics:
            if method not in results[video_number]:
                row.append("N/A")
                continue
            if method == all_metric_best_methods[video_number][metric][0]:
                row.append(
                    f"\033[31m{results[video_number][method][metrics[metric]]}\033[0m")
            else:
                row.append(f"{results[video_number][method][metrics[metric]]}")
            metric_values.append(results[video_number][method][metrics[metric]])
        if len(metric_values) > 0:
            row.append(f"{sum(metric_values)/len(metric_values):.4f}")
        else:
            row.append("N/A")
        table.add_row(row)
    print(f"Video {video_number}")
    print(table) #file=open("./sample_seeds.txt", "a")