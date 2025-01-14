import matplotlib.pyplot as plt
import plotly.graph_objects as go
import re
import os
import csv
import json
import math
import numpy as np

import plotly.express as px
import pandas as pd
import seaborn as sns

import plotly.io as pio

tempature = 100.0
objects = ["car", "truck"]
result_dir = "./experiments/ablation_searchdimension"
all_result_paths = os.listdir(result_dir)
result_paths = [os.path.join(result_dir, result_path)
                for result_path in all_result_paths if result_path.endswith(".json") and "nomask" not in result_path]

nomask_result_path = [os.path.join(result_dir, result_path)
                      for result_path in all_result_paths if result_path.endswith(".json") and "nomask" in result_path][0]

# * 1. transform these json data to csv data.
save_csv_path = "./experiments/ablation_searchdimension/data.csv"
with open(save_csv_path, "w") as save_writer:
    csv_writer = csv.writer(save_writer)
    csv_writer.writerow(["config", "Latency", "CarSel", "TruckSel",
                        "CarAgg", "TruckAgg", "CarHota", "TruckHota"])

    nomask_result_data = json.load(open(nomask_result_path, "r"))
    csv_writer.writerow(["NoMask", math.exp(-nomask_result_data["Latency"]/tempature),
                        nomask_result_data["CarSel"], nomask_result_data["TruckSel"],
                        nomask_result_data["CarAgg"], nomask_result_data["TruckAgg"],
                        nomask_result_data["CarHota"], nomask_result_data["TruckHota"]])

    for result_path in result_paths:
        result_data = json.load(open(result_path, "r"))
        config_name = os.path.basename(result_path).split(".")[0]
        csv_writer.writerow([config_name, math.exp(-result_data["Latency"]/tempature),
                            result_data["CarSel"], result_data["TruckSel"],
                            result_data["CarAgg"], result_data["TruckAgg"],
                            result_data["CarHota"], result_data["TruckHota"]])
    print(f"Save csv data to {save_csv_path}.")
    save_writer.close()


def simplify_config_name(config_name):
    if config_name == 'nomask':
        return config_name
    parts = config_name.split('_')
    if len(parts) > 2:
        return parts[1].replace("base", "_").lstrip('_').rstrip('_')
    else:
        return config_name


# # * 2. Load csv data
# data = pd.read_csv(save_csv_path)
# data['simplified_config_name'] = data['config'].astype('category').cat.codes

# unique_configs = data['simplified_config_name'].unique()
# colors = sns.color_palette("hls", len(unique_configs)).as_hex()
# custom_colors = dict(zip(unique_configs, colors))

# # * 3. Normalize data
# columns_to_normalize = ['Latency', 'CarSel', 'TruckSel',
#                         'CarAgg', 'TruckAgg', 'CarHota', 'TruckHota']
# # data[columns_to_normalize] = data[columns_to_normalize].apply(
# #     lambda x: (x - x.min()) / (x.max() - x.min()))

# scaled_colors = [[i/(len(colors)-1), color] for i, color in enumerate(colors)]

# # * 4. Plot
# fig = px.parallel_coordinates(data, color='simplified_config_name',
#                               dimensions=columns_to_normalize,
#                               color_continuous_scale=scaled_colors,
#                               labels={"config": "Configuration",
#                                       "Latency": "Latency",
#                                       "CarSel": "CarSel",
#                                       "TruckSel": "TruckSel",
#                                       "CarAgg": "CarAgg",
#                                       "TruckAgg": "TruckAgg",
#                                       "CarHota": "CarHota",
#                                       "TruckHota": "TruckHota"},
#                               title="Performance of Configurations with Different Search Dimensions",
#                               color_continuous_midpoint=(len(colors)-1)/2)

# reverse_config_mapping = dict(
#     enumerate(data['config'].astype('category').cat.categories))

# colorbar_tickvals = list(range(len(custom_colors)))
# colorbar_ticktext = [simplify_config_name(
#     reverse_config_mapping[key]) for key in custom_colors.keys()]

# fig.update_layout(coloraxis_colorbar=dict(
#     tickvals=colorbar_tickvals, ticktext=colorbar_ticktext))

# # * 5. Save
# pio.write_image(
#     fig, "./experiments/ablation_searchdimension/plot.png", scale=2.0)


# # * 6. Create separate subplots for each metric using matplotlib

# # Extract the 'NoMask' values for 'CarHota', 'TruckHota', and 'Latency'.
# no_mask_values = data[data['config'] == 'NoMask'][[
#     'CarHota', 'TruckHota', 'Latency']].values[0]

# # Prepare data for bar chart.
# labels = data['config'].values[1:]

minhota, maxhota = 9999.9, -9999.9
with open(save_csv_path, "r") as file_reader:
    csv_reader = csv.reader(file_reader)
    header = next(csv_reader)
    nomask_data = next(csv_reader)
    nomask_hota = float(nomask_data[-2])
    minhota = min(minhota, nomask_hota)
    maxhota = max(maxhota, nomask_hota)
    maskdimensions, hotas = [], []
    print(header)
    print(nomask_data)
    for dataline in csv_reader:
        maskdimensions.append(dataline[0])
        minhota = min(minhota, float(dataline[-2]))
        maxhota = max(maxhota, float(dataline[-2]))
        hotas.append(float(dataline[-2]))
    print(hotas)

print("minhota: ", minhota)
print("maxhota: ", maxhota)

fontsize = 20
plt.rc('font', size=fontsize)          # controls default text sizes

colors = sns.color_palette("Blues", len(maskdimensions) + 1).as_hex()

# xticks = ['detector threshold',
#           'filter resolution',
#           'sampling rate',
#           'tracking parameter-1',
#           'tracking architecture',
#           'frame filter',
#           'roi segmentation',
#           'filter threshold',
#           'tracking parameter-2',
#           'scale down resolution',
#           'roi resolution',
#           'tracking parameter-3',
#           'tracking parameter-4',
#           'image enhancement',
#           'noise filter',
#           'roi threshold',
#           'tracking parameter-5',
#           'refine start&end',
#           'tracking parameter-6',
#           'tracking parameter-7',
#           'detector architecture',]

# xticks_new = ['sampling rate',
#               'frame filter',
#               'filter threshold',
#               'filter resolution',
#               'scale down resolution',
#               'roi segmentation',
#               'roi threshold',
#               'roi resolution',
#               'image enhancement',
#               'detector architecture',
#               'detector threshold',
#               'tracking architecture',
#               'tracking parameter-1',
#           'tracking parameter-2',
#           'tracking parameter-3',
#           'tracking parameter-4',
#           'tracking parameter-5',
#           'tracking parameter-6',
#           'tracking parameter-7',
#           'refine start&end',
#           'noise filter',]

xticks = ['detector threshold',
          'filter resolution',
          'sampling rate',
          'tracking parameter-1',
          'tracking architecture',

          'frame filter',
          'roi segmentation',
          'filter threshold',
          'tracking parameter-2',
          'scale down resolution',

          'roi resolution',
          'tracking parameter-3',
          'tracking parameter-4',
          'image enhancement',
          'noise filter',

          'roi threshold',
          'tracking parameter-5',
          'refine start&end',
          'tracking parameter-6',
          'tracking parameter-7',

          'detector architecture',]

xticks_new = ['sampling rate',
              'frame filter',
              'filter threshold',
              'filter resolution',
              'scale down resolution',
              'roi segmentation',
              'roi threshold',
              'roi resolution',
              'image enhancement',
              'detector architecture',
              'detector threshold',
              'tracking architecture',
              'tracking parameter-1',
              'tracking parameter-2',
              'tracking parameter-3',
              'tracking parameter-4',
              'tracking parameter-5',
              'tracking parameter-6',
              'tracking parameter-7',
              'refine start&end',
              'noise filter',]

sorted_ids = [xticks.index(xtick) for xtick in xticks_new]

hotas = [hotas[sorted_id] for sorted_id in sorted_ids]
maskdimensions = [maskdimensions[sorted_id] for sorted_id in sorted_ids]

print("len(xticks): ", len(xticks), len(hotas), maskdimensions)

# print the filtered configs
for hi, hota in enumerate(hotas):
    if abs(hota - nomask_hota) < 0.001:
        print(xticks_new[hi], hota, maskdimensions[hi])

xticks = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
          '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']

width = 0.7
# *1 Just plot the car hota in figure
plt.figure(figsize=(12, 10))

legend_font_size = 24
title_font_size = 32
tick_font_size = 24

x = list(range(len(hotas)))
x = [xi * 1.5 for xi in x]

plt.bar(x, hotas, width, label='Hota',
        color=colors[12], edgecolor='black')
plt.ylabel('Hota', fontsize=title_font_size)
plt.xlabel('The Configurations of Masked Dimensions',
           fontsize=title_font_size)
plt.xticks([])
# plt.xticks(x, xticks, fontsize=tick_font_size, ha='center')
plt.yticks(fontsize=tick_font_size)
plt.ylim((minhota-0.001, maxhota+0.001))

# plot horizontal line
plt.axhline(y=nomask_hota, color=colors[21], linestyle='--')
plt.text(-1.4, nomask_hota+0.001, "Original",
         color=colors[21], fontsize=legend_font_size)

plt.savefig("./experiments/ablation_searchdimension/CarHota.png",
            dpi=100, bbox_inches='tight')
