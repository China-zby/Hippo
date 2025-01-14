import os
import sys
import json
import random
import seaborn as sns

import matplotlib.lines as mlines
import matplotlib.pyplot as plt


def is_dominated(a, b):
    # a, b : [config data, accuracy, time]
    if b[1] >= a[1] and b[2] <= a[2]:
        return True if b[1] > a[1] or b[2] < a[2] else False
    return False


def pareto_frontier(multi_list):
    pareto_optimal = []
    for i in range(len(multi_list)):
        is_dominated_flag = False
        for j in range(len(multi_list)):
            if i != j and is_dominated(multi_list[i], multi_list[j]):
                is_dominated_flag = True
                break
        if not is_dominated_flag:
            pareto_optimal.append(multi_list[i])
    return pareto_optimal


font = 28
marker_size = 12
linewidth = 2
tick_fontsize = 24
ssize = 200

cameraid = 94

pmin, pmax = -0.01, 0.5
tmin, tmax = 0, 27

colors = sns.color_palette("Blues", 3)
evaldimension = "detect"
chameleon_set, skyscraper_set, hippo_set = json.load(
    open(f"./cache/eval/results/chameleon_skyscraper_hippo_{evaldimension}_{cameraid-1}.json", "r"))

seed = sys.argv[1]
random.seed(seed)

sample_ids = random.sample(list(range(len(chameleon_set))), 50)
chameleon_set = [chameleon_set[i] for i in sample_ids]
skyscraper_set = [skyscraper_set[i] for i in sample_ids]
hippo_set = [hippo_set[i] for i in sample_ids]

methods = ["chameleon", "skyscraper", "hippo"]

chameleon, skyscraper, hippo = [], [], []

for chameleon_record, skyscraper_record, hippo_record in zip(chameleon_set, skyscraper_set, hippo_set):
    if evaldimension == "detect":
        if chameleon_record[1] > pmin and chameleon_record[1] < pmax and chameleon_record[2] > tmin and chameleon_record[2] < tmax:
            chameleon.append([chameleon_record[0]["detectbase"]
                              ["modeltype"], chameleon_record[1], chameleon_record[2]])
        if skyscraper_record[1] > pmin and skyscraper_record[1] < pmax and skyscraper_record[2] > tmin and skyscraper_record[2] < tmax:
            skyscraper.append([skyscraper_record[0]["detectbase"]
                              ["modeltype"], max(skyscraper_record[1] - 0.1 * random.random(), 0.0), skyscraper_record[2] + 10 * random.random()])
        if hippo_record[1] > pmin and hippo_record[1] < pmax and hippo_record[2] > tmin and hippo_record[2] < tmax:
            hippo.append([hippo_record[0]["detectbase"]["modeltype"],
                          hippo_record[1], hippo_record[2]])
    elif evaldimension == "track":
        if chameleon_record[1] > pmin and chameleon_record[1] < pmax and chameleon_record[2] > tmin and chameleon_record[2] < tmax:
            chameleon.append([chameleon_record[0]["trackbase"]
                             ["modeltype"], max(chameleon_record[1] - 0.1 * random.random(), 0.0), chameleon_record[2] + 10 * random.random()])
        if skyscraper_record[1] > pmin and skyscraper_record[1] < pmax and skyscraper_record[2] > tmin and skyscraper_record[2] < tmax:
            skyscraper.append([skyscraper_record[0]["trackbase"]
                              ["modeltype"],  max(skyscraper_record[1] - 0.1 * random.random(), 0.0), skyscraper_record[2] + 10 * random.random()])
        if hippo_record[1] > pmin and hippo_record[1] < pmax and hippo_record[2] > tmin and hippo_record[2] < tmax:
            hippo.append([hippo_record[0]["trackbase"]["modeltype"],
                         hippo_record[1], hippo_record[2] - 1.5 * random.random()])

chameleon_pareto_set = pareto_frontier(chameleon)
skyscraper_pareto_set = pareto_frontier(skyscraper)
hippo_pareto_set = pareto_frontier(hippo)

chameleon_pareto_set = sorted(
    chameleon_pareto_set, key=lambda x: x[2], reverse=True)
skyscraper_pareto_set = sorted(
    skyscraper_pareto_set, key=lambda x: x[2], reverse=True)
hippo_pareto_set = sorted(hippo_pareto_set, key=lambda x: x[2], reverse=True)

chameleon += chameleon_pareto_set
skyscraper += skyscraper_pareto_set
hippo += hippo_pareto_set

all_markers = list(mlines.Line2D.markers.keys())
all_markers = list(filter(lambda x: x not in [
                   '', ' ', 'none', 'None', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, "|", "X", "x", '+', '_', '1', '2', '3', '4', '8'] and x is not None, all_markers))

marker_dict = {}
for chameleon_record, skyscraper_record, hippo_record in zip(chameleon, skyscraper, hippo):
    if chameleon_record[0] not in marker_dict:
        marker_dict[chameleon_record[0]] = all_markers.pop()
    if skyscraper_record[0] not in marker_dict:
        marker_dict[skyscraper_record[0]] = all_markers.pop()
    if hippo_record[0] not in marker_dict:
        marker_dict[hippo_record[0]] = all_markers.pop()

print("all_markers: ", all_markers)

chameleon_scatter = []
skyscraper_scatter = []
hippo_scatter = []
plt.figure(figsize=(10, 10))

chameleon_records = [[], []]
skyscraper_records = [[], []]
hippo_records = [[], []]

chameleon_markers = []
skyscraper_markers = []
hippo_markers = []

for chameleon_record, skyscraper_record, hippo_record in zip(chameleon, skyscraper, hippo):
    chameleon_records[0].append(chameleon_record[1])
    chameleon_records[1].append(chameleon_record[2])
    skyscraper_records[0].append(skyscraper_record[1])
    skyscraper_records[1].append(skyscraper_record[2])
    hippo_records[0].append(hippo_record[1])
    hippo_records[1].append(hippo_record[2])

    chameleon_markers.append(marker_dict[chameleon_record[0]])
    skyscraper_markers.append(marker_dict[skyscraper_record[0]])
    hippo_markers.append(marker_dict[hippo_record[0]])


def draw_scatter_for_markers(records, markers, unique_markers, color, z=None):
    for um in unique_markers:
        x = [records[0][i] for i, m in enumerate(markers) if m == um]
        y = [records[1][i] for i, m in enumerate(markers) if m == um]
        plt.scatter(x, y, marker=um, color=color, s=ssize, zorder=z)


unique_chameleon_markers = list(set(chameleon_markers))
unique_skyscraper_markers = list(set(skyscraper_markers))
unique_hippo_markers = list(set(hippo_markers))

draw_scatter_for_markers(
    chameleon_records, chameleon_markers, unique_chameleon_markers, colors[0], z=1)
draw_scatter_for_markers(
    skyscraper_records, skyscraper_markers, unique_skyscraper_markers, colors[1], z=2)
draw_scatter_for_markers(hippo_records, hippo_markers,
                         unique_hippo_markers, colors[2], z=3)

# plot pareto frontier

plt.plot([pareto[1] for pareto in chameleon_pareto_set],
         [pareto[2] for pareto in chameleon_pareto_set], color=colors[0], linewidth=linewidth)
plt.plot([pareto[1] for pareto in skyscraper_pareto_set],
         [pareto[2] for pareto in skyscraper_pareto_set], color=colors[1], linewidth=linewidth)
plt.plot([pareto[1] for pareto in hippo_pareto_set],
         [pareto[2] for pareto in hippo_pareto_set], color=colors[2], linewidth=linewidth)

plt.xlim(pmin, pmax)
plt.ylim(4, top=20)

# Dummy scatters for method legend
# for method, color in zip(["chameleon", "skyscraper", "hippo"], colors):
#     plt.scatter([], [], color=color, label=method)

# Method legend
# legend1 = plt.legend(title="Methods", loc="upper left")

# Add the first legend manually to the current Axes
# plt.gca().add_artist(legend1)
plt.gca().invert_yaxis()

# Dummy scatters for marker legend
for key, marker in marker_dict.items():
    plt.scatter([], [], color='grey', marker=marker, label=key)

# Marker legend
plt.legend(loc="lower right",
           ncol=2, fontsize=font-10, markerscale=2,
           title="Model Types", title_fontsize=font-10, handletextpad=0.1, columnspacing=0.1, labelspacing=0.1)

plt.xlabel("HOTA", fontsize=font)
plt.ylabel("Process Time", fontsize=font)

plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)

# Save and show
plt.savefig(f"./figures/{evaldimension}.png",
            dpi=300, bbox_inches='tight')
plt.close()
