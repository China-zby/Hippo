import os
import toml
import numpy as np
import seaborn as sns

from glob import glob
import matplotlib.pyplot as plt

tempature = 100.0
objects = ["car", "truck"]
result_dir = "./experiments/hota_influence"

data_dir = "./cache/ardent/records/epoch_*"

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

HOTADICT = {}

for data_path in glob(data_dir):
    result_list = os.listdir(data_path)
    for result_name in result_list:
        result_path = os.path.join(data_path, result_name)
        result_data = toml.load(open(result_path, "r"))
        for object_name in objects:
            if object_name not in HOTADICT.keys():
                HOTADICT[object_name] = {"sel": [],
                                         "agg": [],
                                         "count": []}
            if result_data[f"{object_name.capitalize()}GtCount"] > 0:
                countRate = 1.0 - abs(result_data[f"{object_name.capitalize()}GtCount"] -
                                      result_data[f"{object_name.capitalize()}PredCount"]) / result_data[f"{object_name.capitalize()}GtCount"]
            else:
                countRate = 0.0

            if result_data[f"{object_name.capitalize()}GtCount"] <= 0 or \
                result_data[f"{object_name.capitalize()}Sel"] <= 0 or \
                    result_data[f"{object_name.capitalize()}Agg"] <= 0 or \
                result_data[f"{object_name.capitalize()}Hota"] <= 0 or \
            countRate <= 0:
                continue
            HOTADICT[object_name]["sel"].append([result_data[f"{object_name.capitalize()}Hota"],
                                                 result_data[f"{object_name.capitalize()}Sel"]])
            HOTADICT[object_name]["agg"].append([result_data[f"{object_name.capitalize()}Hota"],
                                                 result_data[f"{object_name.capitalize()}Agg"]])
            HOTADICT[object_name]["count"].append([result_data[f"{object_name.capitalize()}Hota"],
                                                   countRate])

colors = sns.color_palette("hls", 8)

legend_font_size = 24
title_font_size = 32
tick_font_size = 24

for object_name in objects:
    sel_color = colors[0]
    agg_color = colors[1]
    plt.figure(figsize=(12, 10))
    plt.scatter([x[0] for x in HOTADICT[object_name]["sel"]],
                [x[1] for x in HOTADICT[object_name]["sel"]],
                label="Sel",
                marker="x",
                color=sel_color,
                alpha=0.8,
                s=75,
                linewidths=2)

    sel_x = [x[0] for x in HOTADICT[object_name]["sel"]]
    sel_y = [x[1] for x in HOTADICT[object_name]["sel"]]

    correlation_coefficient = np.corrcoef(sel_x, sel_y)[0, 1]

    sel_x_mean = np.mean(sel_x)
    sel_y_mean = np.mean(sel_y)
    slope = np.sum(np.array(sel_x) * np.array(sel_y)) / \
        np.sum(np.array(sel_x)**2)

    regression_line_x = np.linspace(0.0, max(sel_x), 100)
    regression_line_y = slope * regression_line_x
    plt.plot(regression_line_x, regression_line_y,
             color=sel_color, linestyle='--')

    print(f"{object_name}-sel:", correlation_coefficient)

    plt.scatter([x[0] for x in HOTADICT[object_name]["agg"]],
                [x[1] for x in HOTADICT[object_name]["agg"]],
                label="Agg",
                marker='+',
                color=agg_color,
                alpha=0.8,
                s=75,
                linewidths=2)

    agg_x = [x[0] for x in HOTADICT[object_name]["agg"]]
    agg_y = [x[1] for x in HOTADICT[object_name]["agg"]]

    correlation_coefficient = np.corrcoef([x[0] for x in HOTADICT[object_name]["agg"]],
                                          [x[1] for x in HOTADICT[object_name]["agg"]])[0, 1]

    agg_x_mean = np.mean(agg_x)
    agg_y_mean = np.mean(agg_y)
    slope = np.sum(np.array(agg_x) * np.array(agg_y)) / \
        np.sum(np.array(agg_x)**2)

    regression_line_x = np.linspace(0.0, max(agg_x), 100)
    regression_line_y = slope * regression_line_x
    plt.plot(regression_line_x, regression_line_y,
             color=agg_color, linestyle='--')

    print(f"{object_name}-agg:", correlation_coefficient)

    # plt.scatter([x[0] for x in HOTADICT[object_name]["count"]],
    #             [x[1] for x in HOTADICT[object_name]["count"]],
    #             label="count",
    #             marker="^",
    #             color="green",
    #             alpha=0.5,
    #             s=10,
    #             linewidths=1)
    plt.xlabel("HOTA", fontsize=title_font_size)
    plt.ylabel("Influence", fontsize=title_font_size)
    plt.ylim(0, 1.02)
    plt.xlim(0, 1.02)
    plt.xticks(fontsize=tick_font_size)
    plt.yticks(fontsize=tick_font_size)
    plt.legend(fontsize=legend_font_size)
    plt.savefig(os.path.join(
        result_dir, f"{object_name}_influence.png"), dpi=100, bbox_inches='tight')
    plt.close()
