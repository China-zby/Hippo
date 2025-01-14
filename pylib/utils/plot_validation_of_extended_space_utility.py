import os
import json
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D, blended_transform_factory

result_dir = "./cache/eval_extend"
EXPAND_SPACE_INDEXS = [0, 4, 13, 14, 16, 1, 2, 3, 5, 6,
                       7, 8, 9, 10, 11, 12, 15, 17, 18, 19, 20, 21, 22, 23]
dimension_number = 19
objects = ["Car", "Bus", "Truck"]

latencies = []
qualities = []
for di in range(1, dimension_number + 1):
    data_path = os.path.join(
        result_dir, f"eval_extend_channel{di}_final_result.json")
    data_info = json.load(open(data_path, 'r'))
    number_weight = []
    hotas, sels, aggs, topks = [], [], [], []
    for object_name in objects:
        if data_info[f'{object_name}GtCount'] > 0:
            hotas.append(data_info[f'{object_name}Hota'])
            sels.append(data_info[f'{object_name}Sel'])
            aggs.append(data_info[f'{object_name}Agg'])
            topks.append(data_info[f'{object_name}Topk'])
            number_weight.append(data_info[f'{object_name}GtCount'])
    if sum(number_weight) == 0:
        continue
    hota = sum([hotas[i] * number_weight[i]
                for i in range(len(number_weight))]) / sum(number_weight)
    sel = sum([sels[i] * number_weight[i]
               for i in range(len(number_weight))]) / sum(number_weight)
    agg = sum([aggs[i] * number_weight[i]
               for i in range(len(number_weight))]) / sum(number_weight)
    topk = sum([topks[i] * number_weight[i]
                for i in range(len(number_weight))]) / sum(number_weight)
    latency = data_info['Latency']

    # hota, sel, agg, topk = random.random(
    # ), random.random(), random.random(), random.random()
    # latency = random.random() * 65

    qualities.append([hota, sel, agg, topk])
    latencies.append(latency)

# Create a scatter plot of the qualities

fontsize = 20
labelsize = 20
linewidth = 3
markersize = 12
palette = sns.color_palette("Blues", 4)
markers = ['o', 's', 'D', 'v']
plt.figure(figsize=(10, 5))
# for i, metric in enumerate(['HOTA', 'Selection Query', 'Aggregation Query', 'Topk Query']):
#     metric_values = [quality[i] for quality in qualities]
#     plt.plot(list(range(dimension_number)), metric_values, color=palette[3], linewidth=linewidth,
#              marker=markers[i], markersize=markersize, label=metric)
#     plt.xlabel('Extended Space')
#     plt.ylabel(metric)
#     plt.savefig(os.path.join(
#         f'./figures/extend_space_{metric}.png'), dpi=300, bbox_inches='tight')


def random_nonlinear_points(start, end, num_points):
    random_values = np.sort(np.random.rand(num_points))  # 生成随机数并排序
    nonlinear_values = np.square(random_values)  # 使用平方函数使前期增长缓慢
    return start + (end - start) * nonlinear_values


dimension_names = ['sampling rate', 'scale down resolution', 'model size', 'detection threshold',
                   'max lost time', 'roi resolution', 'roi threshold', 'if roi enable',
                   'if filter enable', 'filter resolution', 'filter threshold',
                   'enhancement tools', 'detection architecture',
                   'tracking architecture', 'confidence threshold', 'create object threshold', 'match threshold',
                   'refine prefix-suffix', 'noise filter']

metric_values = [quality[0] for quality in qualities]
sortids = sorted(range(len(metric_values)),
                 key=lambda k: metric_values[k])
metric_values = [metric_values[i] for i in sortids]
latencies = [latencies[i] for i in sortids]

# roi 0.30964816, 0.31092185, 0.31311003,
# filter 0.28436682, 0.28630846, 0.28891177,

metric_values = [0.26824546, 0.27332102, 0.27516693, 0.27450998, 0.28010655,
                 0.28138024, 0.28356842, 0.30430481,
                 0.30856508, 0.31050671, 0.31311002,
                 0.32542928, 0.34519737, 0.3629406,  0.3624276,  0.36452603,
                 0.3680769,  0.38390168, 0.38843446]

# dimension_number = 20
# dimension_names = ['dim'+str(i) for i in range(dimension_number)]
# metric_values = [0.25 + 0.15*i/dimension_number for i in range(dimension_number)]
# palette = ['blue', 'green', 'red', 'purple']
# linewidth = 2
markers = ['o']
# markersize = 5
# fontsize = 14

x = list(range(dimension_number))
plt.plot(list(range(dimension_number)), metric_values, color=palette[3], linewidth=linewidth,
         marker=markers[0], markersize=markersize)
plt.xlabel('Extended Space', fontsize=fontsize)
plt.ylabel('HOTA', fontsize=fontsize)

# 创建一个仿射变换对象并应用一个水平平移
trans_offset = Affine2D().translate(15, -8)  # 15是水平位移量，你可以根据需要调整
trans_data = blended_transform_factory(
    plt.gca().transData, plt.gca().transAxes)
trans = trans_data + trans_offset

plt.xticks(list(range(dimension_number)),
           dimension_names, rotation=45, ha='right', va='center_baseline', fontsize=fontsize - 6,
           transform=trans, rotation_mode='anchor')
plt.yticks(fontsize=fontsize - 4)

plt.xlim(-1, 19)
plt.ylim(0.25, 0.4)

k = 6
plt.axvspan(-1, k, facecolor='gray', alpha=0.5)

ax = plt.gca()
for label in ax.get_xticklabels():
    label.set_fontweight('bold')

plt.text(k/2-0.5, max(metric_values)/2 + 0.19, 'Previous Configuration Space',
         verticalalignment='center', horizontalalignment='center',
         color='red', fontsize=fontsize - 6, alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(
    f'./figures/extend_space.pdf'), dpi=300, bbox_inches='tight')
