import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
import seaborn as sns

# 指定字体文件路径
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
font_prop = FontProperties(fname=font_path, size = 17)
# 设置数学公式字体
mpl.font_manager.fontManager.addfont(font_path)
# 启用 usetex 以使用系统 LaTeX 渲染
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['times']
# 数据
video_streams = [100, 150, 200, 250, 300]
method1_part1 = [11.17, 13.71, 17.49, 21.86, 26.06]
skyscraper_cluster_times = [1465.93, 2020.8, 2504.48, 2916.85, 3821.45]
hippo_cluster_times = method1_part1

# 对数计算
method1_part1_log = np.log10(method1_part1)
skyscraper_cluster_times_log = np.log10(np.array(skyscraper_cluster_times))
hippo_cluster_times_log = np.log10(hippo_cluster_times)

x = np.arange(len(video_streams))
width = 0.35

# 设置画布和风格
sns.set_context("paper")
sns.set_palette("Set1")
sns.set_color_codes()

# 绘图
plt.figure()
plt.bar(x - width/2, skyscraper_cluster_times_log, width, label='Skyscraper', color='#b6ccd8')
plt.bar(x + width/2, method1_part1_log, width, label='Hippo', color='#00668c')

# plt.plot(x, skyscraper_cluster_times_log, marker='v', color='b', markersize=8, linewidth=2)
# plt.plot(x, hippo_cluster_times_log, marker='^', color='m', markersize=8, linewidth=2)

# 添加文本标签
for i in range(len(video_streams)):
    plt.text(x[i] - width/2 + 0.05, skyscraper_cluster_times_log[i] + 0.05,
             f'{skyscraper_cluster_times[i]:.0f}s', ha='center', va='bottom',
             color='black', fontproperties=font_prop, fontsize=19)
    plt.text(x[i] + width/2 + 0.05, method1_part1_log[i] + 0.05,
             f'{method1_part1[i]:.0f}s', ha='center', va='bottom',
             color='black', fontproperties=font_prop, fontsize=19)

# 设置轴标签
plt.xlabel('Number of Videos', fontproperties=font_prop, fontsize=22)
plt.ylabel('Vector Generation Time (s)', fontproperties=font_prop, fontsize=22)

# 设置刻度字体
plt.xticks(x, video_streams, fontproperties=font_prop, fontsize=19)
y_ticks = [10**i for i in range(6)]  # [1, 10, 100, 1000, 10000, 100000]
y_labels = [r'$10^{{{}}}$'.format(i) for i in range(6)]
y = np.arange(len(y_ticks))
plt.yticks(y, y_labels, fontproperties=font_prop, fontsize=19)

# 添加图例
plt.legend(loc='upper left', prop=font_prop, fontsize=23)

# 保存并显示图形
plt.savefig('log_combined_cluster_time.png', bbox_inches='tight', dpi=300)
plt.show()
