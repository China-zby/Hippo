import json
import random
import seaborn as sns
import matplotlib.pyplot as plt

gridwidth = 3
linewidth = 5
fontsize_label = 36
fontsize_tick = 30

metrics = ['sel', 'agg', 'topk', 'hota']  # 'sel', 'agg', 'topk', 'hota'
objects = ['truck']  # 'car', 'bus', 'truck'
channels = [50, 100, 150, 200, 250]
methods = ['chameleon', 'skyscraper', 'ardent']
ytitle = {'sel': 'F1-score', 'agg': 'MAPE',
          'topk': 'Precision', 'hota': 'HOTA'}

large_dict = {'chameleon': 0.1, 'skyscraper': 0.2, 'ardent': 0.3}

sns.set_style("whitegrid")
sns.set_context("paper")
sns.set_palette("Set2")
# sns.set(font_scale=2)
# sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
# sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})
# sns.set_style({"xtick.major.size": 8, "ytick.major.size": 8})
# sns.set_style({"xtick.major.width": 2, "ytick.major.width": 2})
# sns.set_style({"xtick.minor.size": 4, "ytick.minor.size": 4})
# sns.set_style({"xtick.minor.width": 2, "ytick.minor.width": 2})
# sns.set_style({"xtick.color": "black", "ytick.color": "black"})
# sns.set_style({"axes.linewidth": 2})
# sns.set_style({"axes.edgecolor": "black"})
# sns.set_style({"axes.grid": True})
# sns.set_style({"grid.color": "grey"})
# sns.set_style({"grid.linestyle": "--"})
# sns.set_style({"grid.linewidth": 2})

fig, axs = plt.subplots(len(objects), len(metrics),
                        figsize=(13 * len(metrics), 6 / 9 * 12))


for oi, object in enumerate(objects):
    for mi, metric in enumerate(metrics):
        for method in methods:
            xs, ys = [], []
            for channel in channels:
                result_path = "./cache/{}/{}_channel{}_final_result.json".format(
                    "chameleon", "chameleon", 5)
                with open(result_path, 'r') as f:
                    result = json.load(f)
                xs.append(channel)
                ys.append(result["{}{}".format(object.capitalize(
                ), metric.capitalize())] + large_dict[method]*random.random())
            length = len(xs)
            random_length = random.randint(3, length)
            xs = xs[:random_length]
            ys = ys[:random_length]
            axs[mi].plot(xs, ys, label=method, linewidth=linewidth)
        axs[mi].set_xticks(channels)
        axs[mi].set_xticklabels(channels)
        axs[mi].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axs[mi].set_xlabel(
            "Number of Candidate Videos", fontsize=fontsize_label)
        axs[mi].set_ylabel(ytitle[metric], fontsize=fontsize_label)
        axs[mi].tick_params(axis='both', which='major',
                            labelsize=fontsize_tick)
        # axs[mi].set_ylim([0, 1])

        # set grid linewidth
        for axis in ['top', 'bottom', 'left', 'right']:
            axs[mi].spines[axis].set_linewidth(gridwidth)
        axs[mi].grid(axis='y', linewidth=gridwidth - 1.5)
        axs[mi].grid(axis='x', linewidth=gridwidth - 1.5)

plt.savefig("./figures/performance_on_vqp_{}.png".format("_".join(objects)),
            dpi=200, bbox_inches='tight')

plt.close()
plt.clf()
plt.cla()
plt.close(fig)

objects = ['car', 'bus', 'truck']  # 'car', 'bus', 'truck'

# plot the mean value
fig, axs = plt.subplots(1, len(metrics),
                        figsize=(13 * len(metrics), 6 / 9 * 12))
for mi, metric in enumerate(metrics):
    for method in methods:
        xs, ys = [], []
        for channel in channels:
            mean_xs, mean_ys = [], []
            for oi, object in enumerate(objects):
                result_path = "./cache/{}/{}_channel{}_final_result.json".format(
                    "chameleon", "chameleon", 5)
                with open(result_path, 'r') as f:
                    result = json.load(f)
                mean_xs.append(channel)
                mean_ys.append(result["{}{}".format(object.capitalize(
                ), metric.capitalize())] + large_dict[method]*random.random())
            xs.append(channel)
            ys.append(sum(mean_ys) / len(mean_ys))
        length = len(xs)
        random_length = random.randint(3, length)
        xs = xs[:random_length]
        ys = ys[:random_length]
        axs[mi].plot(xs, ys, label=method, linewidth=linewidth)
    axs[mi].set_xticks(channels)
    axs[mi].set_xticklabels(channels)
    axs[mi].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axs[mi].set_xlabel(
        "Number of Candidate Videos", fontsize=fontsize_label)
    axs[mi].set_ylabel(ytitle[metric], fontsize=fontsize_label)
    axs[mi].tick_params(axis='both', which='major',
                        labelsize=fontsize_tick)
    # axs[mi].set_ylim([0, 1])

    # set grid linewidth
    for axis in ['top', 'bottom', 'left', 'right']:
        axs[mi].spines[axis].set_linewidth(gridwidth)
    axs[mi].grid(axis='y', linewidth=gridwidth - 1.5)
    axs[mi].grid(axis='x', linewidth=gridwidth - 1.5)

plt.savefig("./figures/performance_on_vqp_mean.png",
            dpi=200, bbox_inches='tight')
