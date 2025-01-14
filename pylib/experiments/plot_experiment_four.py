import os
import math
import random
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
def is_dominated(x, y):
    return all(x[i] <= y[i] for i in range(2)) and any(x[i] < y[i] for i in range(2))

def identify_pareto(solutions):
    pareto_front = []
    for solution in solutions:
        if not any(is_dominated(solution, other) for other in solutions):
            if ",".join([str(int(res)) for res in solution[-1]]) not in [",".join([str(int(res)) for res in pareto[-1]]) for pareto in pareto_front]:
                pareto_front.append(solution)
    return pareto_front

def collocate_pareto_set(video_id, camera_type="train", cache_dir="./cache/global/one_camera_with_config"):
    ingestion_result_path_name_list = os.listdir(cache_dir)
    filtered_ingestion_result_path_name_list = [
        path_name for path_name in ingestion_result_path_name_list if f"{camera_type}_{video_id}" in path_name]
    result_sets = []
    for ingestion_result_path_name in filtered_ingestion_result_path_name_list:
        ingestion_result_path = os.path.join(
            cache_dir, ingestion_result_path_name)
        with open(ingestion_result_path, "rb") as f:
            ingestion_result = pickle.load(f)
            #print(ingestion_result)
            cache_info = ingestion_result_path_name.rstrip(".pkl").split("_")[
                1:]
            video_id, config_vector = int(cache_info[0]), cache_info[1:]
            config_vector = [int(res) for res in config_vector]
            _, ingestionvalue, _, acc, record, cmetric, _ = ingestion_result
            #print(cmetric)
            # result_sets.append(
            #     [ingestionvalue, cmetric[-1], config_vector])
            counts= []
            for objectname in ['Car','Bus','Truck']:
               counts.append(record[f'{objectname.capitalize()}GtCount'])
            result_sets.append(
                [ingestionvalue, cmetric[-1], (acc[0]*counts[0]+acc[1]*counts[1]+acc[2]*counts[2])/sum(counts),config_vector])
            
    pareto_set = identify_pareto(result_sets)
    pareto_set = sorted(pareto_set, key=lambda x: x[0])
    return pareto_set

def evenly_sample_indices(data, num_samples=10):
    # 确定每个采样点所在的分位数区间
    quantiles = np.linspace(0, 1, num_samples)  # 避免0和1，这样不会取到最小和最大值之外的数
    # 计算每个分位数对应的数据值
    quantile_values = np.quantile(data, quantiles)

    # 对原数据进行排序，获取排序后的索引
    sorted_indices = np.argsort(data)
    sorted_data = data[sorted_indices]

    # 为每个分位数找到最接近的数据点索引
    sample_indices = []
    for value in quantile_values:
        # 找到最接近分位数值的数据点索引
        idx = np.abs(sorted_data - value).argmin()
        sample_indices.append(sorted_indices[idx])

    return np.unique(sample_indices)

if __name__ == "__main__":
    temp = 10.0
    linewidth = 3
    camera_ids = [0, 100, 150, 300]
    font_size = 19
    sns.set_context("paper")
    sns.set_palette("Set1")
    sns.set_color_codes()
    marker_size = 80
    pareto_size = 10
    camera_type = "train"
    names = ["SKY","OTIF","MED","FULL"]
    name_dicts = {"SKY": "Skyscraper", 
                  "OTIF": "OTIF",
                  "MED": "Median", 
                  "FULL": "Hippo"}
    # 指定字体
    font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
    font_prop = FontProperties(fname=font_path,size = 15)

    for camera_id in camera_ids:
        if camera_id != 0:
            break
        plt.figure()
        #colors = sns.color_palette("Set2")
        colors = ['#b6ccd8', '#d4eaf7', '#71c4ef', '#00668c']
        markers = ["o", "^", "s", "*"]
        for ni, name in enumerate(names):
            if name == "MED":
                continue
            cache_store_path = f"./pylib/experiments/global/one_camera_with_config_with_searchspace_{name}"
            if not os.path.exists(cache_store_path):
                pareto_set = [] # [[random.random(), random.random() * 0.5, [random.randint(0, 1)]] for _ in range(pareto_size)]
            else:
                pareto_set = collocate_pareto_set(camera_id,
                                                camera_type=camera_type,
                                                cache_dir=cache_store_path)
                
                if len(pareto_set) > pareto_size:
                    pareto_set_gt_idxs = evenly_sample_indices(
                                    np.array([res[0] for res in pareto_set]), num_samples=10)
                    pareto_set_gt = [pareto_set[idx] for idx in pareto_set_gt_idxs]
                    pareto_set = sorted(pareto_set_gt, key=lambda x: x[0], reverse=True)
            
            pareto_set = sorted(pareto_set, key=lambda x: x[2])
            
            if name == "FULL" and camera_id == 0:
                print(pareto_set)

            accs, fpss = [], []
            for res in pareto_set:
                _, lat, acc ,_= res
                # if acc>0.47 and acc<0.48 and lat>0.91 and lat<0.92:
                #     continue
                # if acc>0.70 and acc<0.71 and lat>0.67 and lat<0.68:
                #     continue
                fps = 30*60/(- math.log(lat) * temp)*(1.07/1.19)
                time = 30*60*5/fps
                accs.append(acc)
                fpss.append(time)
            method_name = name_dicts[name]
            if markers[ni] == "*":        
                plt.scatter(accs, fpss, label=method_name, marker=markers[ni], color=colors[ni], s=marker_size + 20)
            else:
                plt.scatter(accs, fpss, label=method_name, marker=markers[ni], color=colors[ni], s=marker_size)
            plt.plot(accs, fpss, color=colors[ni], linewidth=linewidth, linestyle="dashed")
            if name == "FULL" and camera_id == 0:
                for i, (acc, fps, res) in enumerate(zip(accs, fpss, pareto_set)):
                    if i == 2:
                        label = "P1"
                        plt.scatter(acc, fps, color='#3b3c3d',marker=markers[ni], s=marker_size,zorder=3)  
                        plt.annotate(label, (acc, fps), textcoords="offset points", xytext=(0, -15), ha='center')
                    if i == 3:
                        label = "P2"
                        plt.scatter(acc, fps, color='#3b3c3d',marker=markers[ni], s=marker_size,zorder=3)  
                        plt.annotate(label, (acc, fps), textcoords="offset points", xytext=(0, -15), ha='center', va='bottom')
                    if i == 6:
                        label = "P3"
                        plt.scatter(acc, fps, color='#3b3c3d',marker=markers[ni], s=marker_size,zorder=3)  
                        plt.annotate(label, (acc, fps), textcoords="offset points", xytext=(0, -15), ha='center', va='bottom')
                    if i == 7:
                        label = "P4"
                        plt.scatter(acc, fps, color='#3b3c3d',marker=markers[ni], s=marker_size,zorder=3)  
                        plt.annotate(label, (acc, fps), textcoords="offset points", xytext=(0, -15), ha='left', va='bottom')

            if name == "OTIF" and camera_id == 0:
                for i, (acc, fps, res) in enumerate(zip(accs, fpss, pareto_set)):
                    if i == 0:
                        label = "P5"
                        plt.scatter(acc, fps, color='#3b3c3d',marker=markers[ni], s=marker_size,zorder=3)  
                        plt.annotate(label, (acc, fps), textcoords="offset points", xytext=(0, -10), ha='center', va='top')
                    if i == 1:
                        label = "P6"
                        plt.scatter(acc, fps, color='#3b3c3d',marker=markers[ni], s=marker_size,zorder=3)  
                        plt.annotate(label, (acc, fps), textcoords="offset points", xytext=(0, -10), ha='center', va='top')

            if name == "SKY" and camera_id == 0:
                for i, (acc, fps, res) in enumerate(zip(accs, fpss, pareto_set)):
                    if i == 0:
                        label = "P7"
                        plt.scatter(acc, fps, color='#3b3c3d',marker=markers[ni], s=marker_size,zorder=3)  
                        plt.annotate(label, (acc, fps), textcoords="offset points", xytext=(0, 3), ha='center', va='bottom')
                    if i == 1:
                        label = "P8"
                        plt.scatter(acc, fps, color='#3b3c3d',marker=markers[ni], s=marker_size,zorder=3)  
                        plt.annotate(label, (acc, fps), textcoords="offset points", xytext=(0, 15), ha='center', va='bottom')


        plt.xlabel("HOTA", fontproperties=font_prop, fontsize=font_size)
        plt.xticks(fontsize=font_size - 4)
        plt.ylabel("Processing Time(s)", fontproperties=font_prop, fontsize=font_size)
        plt.yticks(fontsize=font_size - 4)
        plt.ylim(0, 25)
        plt.legend(prop=font_prop, fontsize=font_size - 6, loc="lower left", borderpad=0.2, labelspacing=0.2)
        plt.savefig(f"./pareto_sets_with_searchspace/{camera_id}_time.png", bbox_inches="tight", dpi=300)
        #plt.savefig(f"./pareto_sets_with_searchspace/{camera_id}_time/{names[3]}.png", bbox_inches="tight", dpi=300)
        plt.close()
        plt.cla()
        plt.clf()