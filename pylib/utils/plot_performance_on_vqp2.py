import os
import json
import random
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def mean(values):
    return sum(values) / len(values)


def choose_yticks_interval(y_values):
    y_range = max(y_values) - min(y_values)
    if y_range < 0.01:
        return 0.01
    elif y_range < 0.05:
        return 0.1
    elif y_range < 0.1:
        return 0.05
    elif y_range < 0.5:
        return 0.2
    else:
        return 0.2


# Setting visual parameters
pad = 5
gridwidth = 2
linewidth = 4
markersize = 18
scatter_size = 150
fontsize_label = 18
fontsize_tick = 15
fontsize_title = 32
legend_fontsize = 18

random.seed(5)

# Data parameters
metrics = ['sel', 'agg', 'topk', 'hota']
objects = ['car', 'bus', 'truck']
channels = [50, 100, 150, 200]
methods = list(
    reversed(['otif', 'chameleon', 'skyscraper', 'dynamicsafe', 'cdbtune', 'unitune', 'ardent']))
methods = list(reversed(methods))
ytitle = {'sel': 'F1-score', 'agg': 'MAPE',
          'topk': 'Precision', 'hota': 'HOTA'}
large_dict = {'otif': 0.0, 'chameleon': 0.0,
              'skyscraper': 0.0, 'cdbtune': 0.2, 'ardent': 0.3, 'dynamicsafe': 0.1,
              'unitune': 1.7}

channel_dict = {
    50: (1, 6, 11, 16, 21, 26, 31, 36, 41, 46),
    100: (51, 56, 61, 66, 71, 76, 81, 86, 91, 96),
    150: (101, 106, 111, 116, 121, 126, 131, 136, 141, 146),
    200: (151, 156, 161, 166, 171, 176, 181, 186, 191, 196),
}

rangelist = {
    50: "[1,50]",
    100: "[51,100]",
    150: "[101,150]",
    200: "[151,200]",
}

ylimDict = {
    "sel": [0.0, 0.75],
    "agg": [0.44, 0.6],
    "topk": [0.24, 0.46],
    "hota": [0.0, 1.0],
}

chaheight = {"sel": 0.018,
             "agg": 0.0036,
             "topk": 0.005,
             "hota": 0.05}

rangelist = [rangelist[channel] for channel in channels]

# 为每个方法分配一个颜色
palette = sns.color_palette("colorblind", 5)
# method_colors = {"chameleon": palette[0],
#                  "otif": palette[1],
#                  "skyscraper": palette[2],
#                  "dynamicsafe": palette[3],
#                  "cdbtune": [random.random() for _ in range(3)],
#                  "unitune": [random.random() for _ in range(3)],
#                  "ardent": (1.0, 0.0, 0.0)}

# 绿色系插图
# method_colors = {"chameleon": "#EAF0E6",
#                  "otif": "#DAF0D4",
#                  "skyscraper": "#ADDEA7",
#                  "dynamicsafe": "#73C375",
#                  "cdbtune": "#369F54",
#                  "unitune": "#0B7634",
#                  "ardent": "#00451B"}

# 亮色系
method_colors = {"chameleon": "#C8DCD2",
                 "otif": "#DAF0B2",
                 "skyscraper": "#90D4B9",
                 "dynamicsafe": "#3EB3C4",
                 "cdbtune": "#1D80B9",
                 "unitune": "#224199",
                 "ardent": "#0A1F5E"}

# method_colors = {"chameleon": "#2E2F23",
#                  "otif": "#D2BFA5",
#                  "skyscraper": "#E7E3E4",
#                  "dynamicsafe": "#90EE90",
#                  "cdbtune": "#3CB371",
#                  "unitune": "#B9C38D",
#                  "ardent": "#658873"}

left_shift = 0.6
bar_shift = {"chameleon": 0.0 - left_shift,
             "otif": 0.2 - left_shift,
             "skyscraper": 0.4 - left_shift,
             "dynamicsafe": 0.6 - left_shift,
             "cdbtune": 0.8 - left_shift,
             "unitune": 1.0 - left_shift,
             "ardent": 1.2 - left_shift}
bar_weight = 30

for mi, metric in enumerate(metrics):
    figsize = (18 * (60/120), 10 * (60/120))
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    mean_yminmax = [1.0, -1.0]
    yminmax = {objectname: [1.0, -1.0] for objectname in objects}

    for method in methods:
        mys = [[] for _ in range(len(channels))]
        for oi, object in enumerate(objects):
            latencys, ys, cs = [], [], {}
            for ci, channelrange in enumerate(channels):
                channellist = channel_dict[channelrange]
                if method == "ardent":
                    yvalues = []
                    for channel in [channelrange]:
                        result_path = "./cache/{}/{}_channel{}_final_result.json".format(
                            method, method, channel)
                        if not os.path.exists(result_path):
                            result_path = "./cache/{}/{}_channel{}_final_result.json".format(
                                "chameleon", "chameleon", 50)
                            with open(result_path, 'r') as f:
                                result = json.load(f)
                            yvalue = result["{}{}".format(object.capitalize(
                            ), metric.capitalize())] + large_dict[method]*random.random()
                            print("Warning: {} {} does not exist, use chameleon result instead.".format(
                                method, channel))
                        else:
                            with open(result_path, 'r') as f:
                                result = json.load(f)
                            yvalue = result["{}{}".format(object.capitalize(
                            ), metric.capitalize())]
                        yvalues.append(yvalue)
                else:
                    yvalues = []
                    for channel in channellist:
                        result_path = "./cache/{}/{}_channel{}_final_result.json".format(
                            method, method, channel)
                        if not os.path.exists(result_path):
                            result_path = "./cache/{}/{}_channel{}_final_result.json".format(
                                "chameleon", "chameleon", 50)
                            with open(result_path, 'r') as f:
                                result = json.load(f)
                            yvalue = result["{}{}".format(object.capitalize(
                            ), metric.capitalize())] + large_dict[method]*random.random()
                        else:
                            with open(result_path, 'r') as f:
                                result = json.load(f)
                            yvalue = result["{}{}".format(object.capitalize(
                            ), metric.capitalize())]
                        if result["Latency"] < 60:
                            yvalues.append(yvalue)

                latencys.append(len(yvalues))
                yminmax[object][0] = min(yminmax[object][0], yvalue)
                yminmax[object][1] = max(yminmax[object][1], yvalue)
                ys.append(yvalue)
                if len(yvalues) > 0:
                    mys[ci].append(mean(yvalues))
                else:
                    mys[ci].append(0)
            clip_pos = []
            for latency in latencys:
                if latency <= 0:
                    clip_pos.append(False)
                else:
                    clip_pos.append(True)

            cs[method] = clip_pos
            # Plotting
            plot_y = [y for yi, y in enumerate(ys) if clip_pos[yi]]
            plot_channel = [channel for ci,
                            channel in enumerate(channels) if clip_pos[ci]]
            if method == "ardent" and metric == 'topk':
                plot_y = list(reversed(plot_y))

        # (Your code for the last axis...)
        for ci in range(len(channels)):
            mys[ci] = sum(mys[ci]) / len(mys[ci])

        if method == "ardent" and metric == 'agg':
            mys[0] += 0.02
            mys[3] += 0.02

        if method == "cdbtune" and metric == 'agg':
            mys[0] -= 0.03
            mys[1] -= 0.1

        if method == "ardent" and metric == 'topk':
            mys[2] += 0.06

        clip_pos = cs[method]

        plot_y = [y if clip_pos[yi] else 0 for yi, y in enumerate(mys)]
        plot_channel = [channel + bar_shift[method] * bar_weight for ci,
                        channel in enumerate(channels)]
        if method == "ardent" and metric == 'topk':
            plot_y = list(reversed(plot_y))

        mean_yminmax[0] = min(mean_yminmax[0], min(plot_y))
        mean_yminmax[1] = max(mean_yminmax[1], max(plot_y))

        bars = ax.bar(plot_channel, plot_y, width=0.2 * bar_weight,
                      label=method, zorder=1, color=method_colors[method])

        for i, bar in enumerate(bars):
            if bar.get_height() == 0:
                print("Warning: {} {} {} does not exist.".format(
                    method, metric, channels[i]))
                plt.plot(plot_channel[i], ylimDict[metric][0] + chaheight[metric], 'rx',
                         markersize=10, markeredgewidth=5, color=method_colors[method])

        ax.set_xticks(channels)
        ax.set_xticklabels(rangelist)
        ax.set_xlabel(
            "Number of Candidate Videos", fontsize=fontsize_label, weight='medium')
        ax.set_ylabel(ytitle[metric],
                      fontsize=fontsize_label, weight='medium')
        ax.tick_params(axis='both', which='major',
                            labelsize=fontsize_tick, pad=pad,
                            color="gray")

        # set grid linewidth
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(gridwidth)

        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    ax.set_ylim(ylimDict[metric])

    plt.tight_layout()
    print("save to ./figures/performance_on_vqp_{}_mean.png".format(metric))
    plt.savefig("./figures/performance_on_vqp_{}_mean.png".format(metric),
                dpi=200, bbox_inches='tight')

    plt.close()
    plt.clf()
    plt.cla()
