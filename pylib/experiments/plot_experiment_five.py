import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

# 方法定义
# methods = ['otif', 'skyscraper', 'unitune', 'hippo']
# methods = ['otif', 'hippo']
methods = ['hippo_wo_age', 'hippo_wo_prl', 'hippo_wo_il', 'hippo']

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

if __name__ == "__main__":
    # 文件夹路径
    folder_path = 'ingestion_results'

    # 初始化一个空的字典
    results = {}
    all_metric_best_methods = {}

    # 遍历文件夹中的每个子文件夹
    for subfolder in ['100', '150']: # ['100', '150', '200', '250', '300']:  # os.listdir(folder_path):
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
                        # 所有数据保留两位小数
                        method_result = {k: round(v, 2)
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
                row.append(f"{sum(metric_values)/len(metric_values):.2f}")
            else:
                row.append("N/A")
            table.add_row(row)
        print(f"Video {video_number}")
        print(table)

    # 画图, 为每个指标画一个随视频数量变多的多柱状图，并存储到 ./paretos/results 下面, sns修改风格
    fontsize = 36
    bar_width = 0.18
    shift_x = bar_width / 2.0
    gap_x = 2
    markersize = 15
    linewidth = 5
    markers = ['o-', 's-', '^-', '*-']
    for metric in metrics:
        # sns.set_style("whitegrid")
        sns.set_context("paper")
        sns.set_palette("Set1")
        sns.set_color_codes()
        plt.figure()
        # method_x_values = [xi for xi in range(1, 1+len(results))]
        min_value, max_value = float('inf'), -float('inf')
        for j, method in enumerate(methods):
            method_values = []
            for video_number in results:
                if method in results[video_number]:
                    method_values.append(
                        results[video_number][method][metrics[metric]])
                    if results[video_number][method][metrics[metric]] < min_value:
                        min_value = results[video_number][method][metrics[metric]]
                    if results[video_number][method][metrics[metric]] > max_value:
                        max_value = results[video_number][method][metrics[metric]]
                # else:
                #     method_values.append(0)
                    
            # plt.bar([(x + j*bar_width - shift_x) * gap_x for x in method_x_values],
            #         method_values, bar_width * gap_x, label=method)
            method_x_values = [xi for xi in range(1, 1+len(method_values))]
            if method == "hippo":
                plt.plot([(x + bar_width) * gap_x for x in method_x_values],
                            method_values, markers[j], markersize=markersize + 3, linewidth=linewidth)
            else:
                plt.plot([(x + bar_width) * gap_x for x in method_x_values],
                            method_values, markers[j], markersize=markersize, linewidth=linewidth)
        # Adjust the axis font and title font size.
        plt.xticks(fontsize=fontsize-5)
        plt.yticks(fontsize=fontsize-5)
        plt.tight_layout()
        
        # 设置边框粗度
        plt.gca().spines['top'].set_linewidth(2)
        plt.gca().spines['right'].set_linewidth(2)
        plt.gca().spines['bottom'].set_linewidth(2)
        plt.gca().spines['left'].set_linewidth(2)
        
        # 手动设置y轴的刻度位置
        ticks = np.linspace(min_value, max_value, 8)  # 生成5个刻度
        plt.yticks(ticks, [f"{tick:.2f}" for tick in ticks])  # 设置刻度标签，并保留两位小数
        plt.ylim(min_value-0.01, max_value+0.01)
        
        plt.xlabel('Video Number', fontsize=fontsize)
        # plt.ylabel(metric, fontsize=fontsize)
        plt.title(metric, fontsize=fontsize)
        # plt.ylim(min_value-0.05*max_value, max_value+0.05*max_value)
        plt.xticks([(x + bar_width) * gap_x for x in method_x_values], list(results.keys()))
        # plt.legend()
        metric = metric.replace('-', '')
        print("Saving the figures to path ./paretos/results/{}.png".format(metric))
        plt.savefig(f'./paretos/results/{metric}.png', bbox_inches='tight', dpi=300)
        plt.close()
