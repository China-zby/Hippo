import os
import csv
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

# 方法定义
methods = ['otif', 'skyscraper', 'unitune', 'hippo']
# methods = ['otif', 'hippo']

# 初始化要打印或画图的指标
metrics = {
           "Q-1": "CarSel",
           "Q-2": "BusSel",
           "Q-3": "TruckSel", 
           
           "Q-4": "CarAgg",
           "Q-5": "BusAgg",
           "Q-6": "TruckAgg",
           
           "Q-7": "CarCQ3",
           "Q-8": "BusCQ3",
           "Q-9": "TruckCQ3"
        }

# "Car HOTA": "CarHota", "Truck HOTA": "TruckHota", "BUS HOTA": "BusHota",
# "Car MOTA": "CarMota", "Car MOTP":  "CarMotp", "Truck MOTP": "TruckMotp", "Bus MOTP": "BusMotp",
# "Car IDF1": "CarIdf1", "Truck IDF1": "TruckIdf1", "Bus IDF1": "BusIdf1",

# 文件夹路径
folder_path = 'ingestion_results'

# 初始化一个空的字典
results = {}
video_gap = 50
video_gap_number = 7
all_metric_best_methods = {}

# 遍历文件夹中的每个子文件夹
save_video_writer = csv.writer(open('experiment2.csv', 'w'))
save_video_writer.writerow(['method', '100', '150', '200', '250', '300'])
result_video_number_list = list(map(int, os.listdir(folder_path)))
for method in methods:
    safe_rates = []
    for subfolder in [100, 150, 200, 250, 300]:  # os.listdir(folder_path):
        metric_best_methods = {metric: [None, -float('inf')] for metric in metrics}
        sub_result_video_number_list = list(filter(lambda x: subfolder <= x < subfolder + video_gap, result_video_number_list))
        
        hit_number = 0
        for sub_result_video_number in sub_result_video_number_list:
            json_file = os.path.join(folder_path, f"{sub_result_video_number}", method + '.json')
            if os.path.exists(json_file):
                hit_number += 1
        safe_rates.append(hit_number / video_gap_number)
    print(method, safe_rates)
    save_video_writer.writerow([method] + safe_rates)