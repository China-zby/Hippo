import os
import csv
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from prettytable import PrettyTable
from sklearn.decomposition import PCA
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import matplotlib as mpl

skyscraper_cluster_times = [1465.93, 2020.8, 2504.48, 2916.85, 3821.45]#[1177.8627610206604, 1465.2777922153473, 2271.0851831436157, 2776.58877836135489, 3321.0000000000000]
hippo_cluster_times = [11.17, 13.71, 17.49, 21.86, 26.06]#[11.949437379837036, 12.919046401977539, 17.260918378829956, 21.52518630027771, 25.922534465789795]

skyscraper_cluster_accuracies = [0.48,0.54,0.515,0.4240,0.43]
hippo_cluster_accuracies = [0.6,0.613,0.560,0.592,0.61]
# 启用 usetex 以使用系统 LaTeX 渲染
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times']
# 指定字体文件路径
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
font_prop = FontProperties(fname=font_path, size=17)

# plot times
sns.set_context("paper")
sns.set_palette("Set1")
sns.set_color_codes()
marker_size = 10
linewidth = 3
fontsize = 19
barwidth = 0.36

plt.figure()

# Add numerical values ​​with line chart colors.
skyscraper_color = sns.color_palette("Set1")[1]
plt.plot(skyscraper_cluster_times, label='Skyscraper', marker='v', color=skyscraper_color, markersize=marker_size, linewidth=linewidth)
for i, txt in enumerate(skyscraper_cluster_times):
    txt_data = "%.2f" % txt
    plt.annotate(txt_data, (i, txt), textcoords="offset points",
                 xytext=(10, -22), ha='center',color=skyscraper_color,
                 fontsize=fontsize - 6)

# Add numerical values ​​with line chart colors.
hippo_color = sns.color_palette("Set1")[3]
plt.plot(hippo_cluster_times, label='Hippo', marker='^', color=hippo_color, markersize=marker_size, linewidth=linewidth)
for i, txt in enumerate(hippo_cluster_times):
    txt_data = "%.2f" % txt
    plt.annotate(txt_data, (i, txt), textcoords="offset points",
                 xytext=(0, 12), ha='center',color=hippo_color,
                 fontsize=fontsize - 6)

plt.xlabel('Number of Videos', fontproperties=font_prop, fontsize=fontsize)
plt.ylabel('Cluster Vector Time (s)', fontproperties=font_prop, fontsize=fontsize)
xs = np.arange(len(skyscraper_cluster_times))
number_of_videos = [100, 150, 200, 250, 300]
plt.xticks(xs, number_of_videos, fontsize=fontsize - 4)
plt.yticks(fontsize=fontsize - 4)
plt.ylim(0, 5500)
plt.legend(fontsize=fontsize - 4)
plt.savefig('cluster_times.png', bbox_inches='tight', dpi=300)

plt.close()
plt.cla()
plt.clf()

# plot bar accuracies
plt.figure()
plt.bar(np.arange(len(skyscraper_cluster_accuracies)) - barwidth/2, skyscraper_cluster_accuracies, barwidth, label='Skyscraper', color='#b6ccd8')
plt.bar(np.arange(len(hippo_cluster_accuracies)) + barwidth/2, hippo_cluster_accuracies, barwidth, label='Hippo', color='#00668c')
plt.xlabel('Number of Videos', fontproperties=font_prop, fontsize=fontsize+3)
plt.ylabel('Contextual Feature Quality', fontproperties=font_prop, fontsize=fontsize+3)
xs = np.arange(len(skyscraper_cluster_accuracies))
number_of_videos = [100, 150, 200, 250, 300]
plt.xticks(xs, number_of_videos, fontproperties=font_prop, fontsize=fontsize )
plt.yticks(fontproperties=font_prop, fontsize=fontsize )
plt.ylim(0, 0.8)
plt.legend(loc='upper left',prop=font_prop, fontsize=fontsize+4)
plt.savefig('cluster_accuracies.png', bbox_inches='tight', dpi=300)

plt.close()
plt.cla()
plt.clf()

# plot cluster times
video_number = 100
load_context_vectors_dir = "./result/cluster_efficiency"
load_skyscraper_context_vectors_path = os.path.join(load_context_vectors_dir, f"skyscraper_context_vectors_{video_number}.npy")
load_hippo_context_vectors_path = os.path.join(load_context_vectors_dir, f"hippo_context_vectors_{video_number}.npy")
skyscraper_context_vectors = np.load(load_skyscraper_context_vectors_path)
hippo_context_vectors = np.load(load_hippo_context_vectors_path)
print(f"skyscraper_context_vectors shape: {skyscraper_context_vectors.shape}")
print(f"hippo_context_vectors shape: {hippo_context_vectors.shape}")

pca = PCA(n_components=2)
skyscraper_features_pca = pca.fit_transform(skyscraper_context_vectors)
hippo_features_pca = pca.fit_transform(hippo_context_vectors)

skyscraper_kmeans = KMeans(n_clusters=5)
hippo_kmeans = KMeans(n_clusters=5)
skyscraper_labels = skyscraper_kmeans.fit_predict(skyscraper_context_vectors)
hippo_labels = hippo_kmeans.fit_predict(hippo_context_vectors)

# plot pca
plt.figure()
plt.scatter(skyscraper_context_vectors[:, 0], skyscraper_context_vectors[:, 1], c=skyscraper_labels, cmap='viridis')
plt.savefig('skyscraper_pca.png', bbox_inches='tight', dpi=300)

plt.close()
plt.cla()
plt.clf()

plt.figure()
plt.scatter(hippo_features_pca[:, 0], hippo_features_pca[:, 1], c=hippo_labels, cmap='viridis')
plt.savefig('hippo_pca.png', bbox_inches='tight', dpi=300)

plt.close()
plt.cla()
plt.clf()