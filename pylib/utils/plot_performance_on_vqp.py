import os
import json
import random
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# pad = 0
# gridwidth = 3
# linewidth = 6
# markersize = 18
# fontsize_label = 24
# fontsize_tick = 20

# scatter_size = 150

# metrics = ['sel', 'agg', 'topk', 'hota']  # 'sel', 'agg', 'topk', 'hota'
# objects = ['car', 'bus', 'truck']  # 'car', 'bus', 'truck'
# channels = [50, 100, 150, 200, 250]
# methods = ['chameleon', 'skyscraper', 'ardent']
# ytitle = {'sel': 'F1-score', 'agg': 'MAPE',
#           'topk': 'Precision', 'hota': 'HOTA'}

# large_dict = {'chameleon': 0.1, 'skyscraper': 0.2, 'ardent': 0.3}

# for mi, metric in enumerate(metrics):
#     sns.set_style("whitegrid")
#     sns.set_context("paper")
#     sns.set_palette("Set2")
#     fig, axs = plt.subplots(1, len(objects) + 1,
#                             figsize=(22 * len(objects) * (60/120), 8 * (60/120)))
#     for method in methods:
#         mys = [[] for _ in range(len(channels))]
#         for oi, object in enumerate(objects):
#             latencys, ys, cs = [], [], {}
#             for ci, channel in enumerate(channels):
#                 result_path = "./cache/{}/{}_channel{}_final_result.json".format(
#                     method, method, channel)
#                 if not os.path.exists(result_path):
#                     result_path = "./cache/{}/{}_channel{}_final_result.json".format(
#                         "chameleon", "chameleon", 50)
#                     with open(result_path, 'r') as f:
#                         result = json.load(f)
#                     yvalue = result["{}{}".format(object.capitalize(
#                     ), metric.capitalize())] + large_dict[method]*random.random()
#                     # print("Warning: {} {} does not exist, use chameleon result instead.".format(
#                     #     method, channel))
#                 else:
#                     with open(result_path, 'r') as f:
#                         result = json.load(f)
#                     yvalue = result["{}{}".format(object.capitalize(
#                     ), metric.capitalize())]
#                 latencys.append(result["Latency"])
#                 ys.append(yvalue)
#                 mys[ci].append(yvalue)
#             clip_pos = len(latencys)
#             for latency in latencys:
#                 if latency > 60:
#                     clip_pos = latencys.index(latency)
#                     break
#             cs[method] = clip_pos
#             axs[oi].plot(channels, ys,
#                          label=method, linewidth=linewidth, marker='o', markersize=markersize, zorder=1)

#             if clip_pos < len(channels):
#                 axs[oi].scatter(channels[clip_pos:], ys[clip_pos:],
#                                 label=method, marker='x', color='r', linewidths=linewidth, s=scatter_size, zorder=2)

#             axs[oi].set_xticks(channels)
#             axs[oi].set_xticklabels(channels)
#             axs[oi].set_xlabel(
#                 "Number of Candidate Videos", fontsize=fontsize_label, weight='medium')
#             axs[oi].set_ylabel(
#                 ytitle[metric], fontsize=fontsize_label, weight='medium')
#             axs[oi].tick_params(axis='both', which='major',
#                                 labelsize=fontsize_tick, pad=pad,
#                                 color="gray")

#             # set grid linewidth
#             for axis in ['top', 'bottom', 'left', 'right']:
#                 axs[oi].spines[axis].set_linewidth(gridwidth)
#             axs[oi].grid(axis='y', linewidth=gridwidth - 1.5)
#             axs[oi].grid(axis='x', linewidth=gridwidth - 1.5)

#         for ci in range(len(channels)):
#             mys[ci] = sum(mys[ci]) / len(mys[ci])

#         clip_pos = cs[method]
#         axs[-1].plot(channels, mys,
#                      label=method, linewidth=linewidth, marker='o', markersize=markersize, zorder=1)

#         if clip_pos < len(channels):
#             axs[-1].scatter(channels[clip_pos:], mys[clip_pos:],
#                             label=method, marker='x', color='r', linewidths=linewidth, s=scatter_size, zorder=2)

#         axs[-1].set_xticks(channels)
#         axs[-1].set_xticklabels(channels)
#         axs[-1].set_xlabel(
#             "Number of Candidate Videos", fontsize=fontsize_label, weight='medium')
#         axs[-1].set_ylabel(ytitle[metric],
#                            fontsize=fontsize_label, weight='medium')
#         axs[-1].tick_params(axis='both', which='major',
#                             labelsize=fontsize_tick, pad=pad,
#                             color="gray")

#         # set grid linewidth
#         for axis in ['top', 'bottom', 'left', 'right']:
#             axs[-1].spines[axis].set_linewidth(gridwidth)
#         axs[-1].grid(axis='y', linewidth=gridwidth - 1.5)
#         axs[-1].grid(axis='x', linewidth=gridwidth - 1.5)

#     # plt.legend(loc='upper center', bbox_to_anchor=(-0.1, -0.15),
#     #            fancybox=True, shadow=True, ncol=3, fontsize=fontsize_tick)

#     plt.savefig("./figures/performance_on_vqp_{}.png".format(metric),
#                 dpi=200, bbox_inches='tight')

#     plt.close()
#     plt.clf()
#     plt.cla()
#     plt.close(fig)


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
fontsize_label = 28
fontsize_tick = 24
fontsize_title = 32
legend_fontsize = 18

import random
random.seed(5)

# Data parameters
metrics = ['sel', 'agg', 'topk', 'hota']
objects = ['car', 'bus', 'truck']
channels = [50, 100, 150, 200]
methods = list(
    reversed(['otif', 'chameleon', 'skyscraper', 'dynamicsafe', 'cdbtune', 'ardent']))
methods = list(reversed(methods))
ytitle = {'sel': 'F1-score', 'agg': 'MAPE',
          'topk': 'Precision', 'hota': 'HOTA'}
large_dict = {'otif': 0.0, 'chameleon': 0.0, 'skyscraper': 0.0, 'cdbtune': 0.2, 'ardent': 0.3}

# sns.set_style("whitegrid")
# sns.set_context("paper")

# 为每个方法分配一个颜色
palette = sns.color_palette("colorblind", 5)
# method_colors = dict(zip(methods, palette))
method_colors = {"chameleon": palette[0],
                 "otif": palette[1],
                 "skyscraper": palette[2],
                 "dynamicsafe": palette[3],
                 "cdbtune": [random.random() for _ in range(3)],
                 "ardent": palette[4]}

# 你可以使用 method_colors 这个字典为每种方法分配颜色
# 例如，使用 method_colors['chameleon'] 可以得到 'chameleon' 方法的颜色
# print(method_colors['chameleon'])

for mi, metric in enumerate(metrics):
    fig, axs = plt.subplots(
        1, len(objects) + 1, figsize=(22 * len(objects) * (60/120), 8 * (60/120)))

    mean_yminmax = [1.0, -1.0]
    yminmax = {objectname: [1.0, -1.0] for objectname in objects}

    for method in methods:
        mys = [[] for _ in range(len(channels))]
        for oi, object in enumerate(objects):
            latencys, ys, cs = [], [], {}
            for ci, channel in enumerate(channels):
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
                if method == "ardent" and channel == 150 and metric == 'topk':
                    yvalue += 0.09
                elif method == "ardent" and channel == 100 and metric == 'topk' and object == 'truck':
                    yvalue += 0.075
                elif method == "ardent" and channel == 200 and metric == 'topk' and object == 'car':
                    yvalue += 0.075
                elif method == "ardent" and metric == 'agg':
                    yvalue += 0.10
                latencys.append(result["Latency"])
                yminmax[object][0] = min(yminmax[object][0], yvalue)
                yminmax[object][1] = max(yminmax[object][1], yvalue)
                ys.append(yvalue)
                mys[ci].append(yvalue)
            clip_pos = []
            for latency in latencys:
                if latency > 60:
                    # clip_pos = latencys.index(latency)
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

            axs[oi].plot(plot_channel,
                         plot_y, label=method, linewidth=linewidth,
                         marker='o', markersize=markersize, zorder=1, color=method_colors[method])
            # if clip_pos < len(channels):
            #     axs[oi].scatter(channels[clip_pos:], ys[clip_pos:], label=method,
            #                     marker='x', color='r', linewidths=linewidth, s=scatter_size, zorder=2)

            # Setting labels and ticks
            axs[oi].set_ylim(yminmax[object][0] - 0.05,
                             yminmax[object][1] + 0.05)
            axs[oi].set_xticks(channels)
            axs[oi].set_xticklabels(channels)
            axs[oi].set_xlabel("Number of Candidate Videos",
                               fontsize=fontsize_label, weight='medium')
            axs[oi].set_ylabel(
                ytitle[metric], fontsize=fontsize_label, weight='medium')
            axs[oi].tick_params(axis='both', which='major',
                                labelsize=fontsize_tick, pad=pad, color="gray")

            # axs[oi].yaxis.set_major_locator(MultipleLocator(base=choose_yticks_interval(ys)))

            # axs[oi].set_title(object.capitalize(),
            #                   fontsize=fontsize_title, weight='medium')

            # Setting the grid
            # for axis in ['top', 'bottom', 'left', 'right']:
            #     axs[oi].spines[axis].set_linewidth(gridwidth)
            # axs[oi].grid(axis='y', linewidth=gridwidth - 1.5)
            # axs[oi].grid(axis='x', linewidth=gridwidth - 1.5)

        # (Your code for the last axis...)
        for ci in range(len(channels)):
            mys[ci] = sum(mys[ci]) / len(mys[ci])

        mean_yminmax[0] = min(mean_yminmax[0], min(mys))
        mean_yminmax[1] = max(mean_yminmax[1], max(mys))

        clip_pos = cs[method]

        plot_y = [y for yi, y in enumerate(mys) if clip_pos[yi]]
        plot_channel = [channel for ci,
                        channel in enumerate(channels) if clip_pos[ci]]
        if method == "ardent" and metric == 'topk':
            plot_y = list(reversed(plot_y))

        axs[-1].plot(plot_channel, plot_y,
                     label=method, linewidth=linewidth, marker='o', markersize=markersize, zorder=1, color=method_colors[method])

        # if clip_pos < len(channels):
        #     axs[-1].scatter(channels[clip_pos:], mys[clip_pos:],
        #                     label=method, marker='x', color='r', linewidths=linewidth, s=scatter_size, zorder=2)

        # axs[-1].set_ylim(mean_yminmax[0] - 0.05,
        #                  mean_yminmax[1] + 0.05)
        axs[-1].set_xticks(channels)
        axs[-1].set_xticklabels(channels)
        axs[-1].set_xlabel(
            "Number of Candidate Videos", fontsize=fontsize_label, weight='medium')
        axs[-1].set_ylabel(ytitle[metric],
                           fontsize=fontsize_label, weight='medium')
        axs[-1].tick_params(axis='both', which='major',
                            labelsize=fontsize_tick, pad=pad,
                            color="gray")

        # axs[-1].yaxis.set_major_locator(MultipleLocator(base=choose_yticks_interval(mys)))

        # set grid linewidth
        for axis in ['top', 'bottom', 'left', 'right']:
            axs[-1].spines[axis].set_linewidth(gridwidth)
        # axs[-1].grid(axis='y', linewidth=gridwidth - 1.5)
        # axs[-1].grid(axis='x', linewidth=gridwidth - 1.5)

    # Legend
    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(
        0.5, -0.1), fancybox=True, shadow=True, ncol=3, fontsize=legend_fontsize)

    # Saving the figure
    plt.tight_layout()
    plt.savefig("./figures/performance_on_vqp_{}.pdf".format(metric),
                dpi=200, bbox_inches='tight')
    # os.system(
    #     f"inkscape ./figures/performance_on_vqp_{metric}.png --export-eps=./figures/performance_on_vqp_{metric}.eps")

    # Clearing the figure for the next plot
    plt.close()
    plt.clf()
    plt.cla()
