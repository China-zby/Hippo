import matplotlib.pyplot as plt
import seaborn as sns

# *1 figure1 (a)
# 选择 seaborn 色板
colors = sns.color_palette("Blues")
bar_color = colors[1]   # 选择柱状图的颜色
line_color1 = colors[3]  # 选择第一条折线的颜色
line_color2 = colors[5]  # 选择第二条折线的颜色

title1 = "Detector Threshold"
values = ['0.1', '0.2', '0.3', '0.4', '0.5']
times = [19.1, 19.0, 18.0, 18.7, 19.0]
car_hotas = [33.3, 34.7, 36.1, 37.4, 37.8]
truck_hotas = [14.0, 14.3, 14.6, 11.8, 10.8]

sorted_ids = sorted(range(len(times)), key=lambda k: times[k])

values = [values[i] for i in sorted_ids]
times = [times[i] for i in sorted_ids]
car_hotas = [car_hotas[i] for i in sorted_ids]
truck_hotas = [truck_hotas[i] for i in sorted_ids]

# 创建图形和坐标轴对象
fig, ax1 = plt.subplots(figsize=(8, 7))

font = 36
marker_size = 26
linewidth = 6
tick_fontsize = 32

# 绘制 'times' 为柱状图
ax1.bar(values, times, width=0.2, label='Process Time (s)', color=bar_color)
# ax1.set_xlim(0.08, 0.52)
ax1.set_ylim(14.0, 26.1)
ax1.set_xticks(values)
ax1.set_xlabel('Threshold Values', fontsize=font)
ax1.set_ylabel('Time', color='black', fontsize=font)
ax1.tick_params(axis='both', labelsize=tick_fontsize)

# 创建第二个 y 轴，为 car_hotas 和 truck_hotas
ax2 = ax1.twinx()
ax2.plot(values, car_hotas, 's-', label='Car HOTA',
         color=line_color1, markersize=marker_size - 5, linewidth=linewidth)
ax2.plot(values, truck_hotas, marker='*', label='Truck HOTA',
         color=line_color2, markersize=marker_size, linewidth=linewidth)
ax2.set_ylim(4, 45)
ax2.set_yticks([4, 12, 20, 28, 36, 44])
ax2.set_ylabel('HOTA', color='black', fontsize=font)
ax2.tick_params(axis='both', labelsize=tick_fontsize)

# 添加标题和其他属性设置
# plt.title(title1, fontsize=font, fontweight='bold')
# fig.tight_layout()
# fig.legend(loc='upper left', fontsize=tick_fontsize)
plt.grid(True, which='both', axis='x')

plt.savefig("./figures/figure1_a.pdf", dpi=200, bbox_inches='tight')

# *2 figure1 (b)
# 选择 seaborn 色板
colors = sns.color_palette("Blues")
bar_color = colors[1]   # 选择柱状图的颜色
line_color1 = colors[3]  # 选择第一条折线的颜色
line_color2 = colors[5]  # 选择第二条折线的颜色

title1 = "Detector Architecture"
values = ['Y3', 'Y7', 'Y8', 'DE', 'FA', 'SP', 'RE', 'VF']
# values = list(range(1, 1 + len(values)))
times = [18.9, 18.5, 22.3, 26.4, 28.0, 32, 28, 34.0]

sorted_ids = sorted(range(len(times)), key=lambda k: times[k])

car_hotas = [35.1, 39.7, 48.5, 36.5, 48.6, 39.4, 42.2, 47.8]
truck_hotas = [14.4, 18.4, 23.8, 21.3, 21.9, 22.7, 18.3, 21.6]

values = [values[i] for i in sorted_ids]
times = [times[i] for i in sorted_ids]
car_hotas = [car_hotas[i] for i in sorted_ids]
truck_hotas = [truck_hotas[i] for i in sorted_ids]

# 创建图形和坐标轴对象
fig, ax1 = plt.subplots(figsize=(8, 7))

# font = 32
# marker_size = 22
# linewidth = 4
# tick_fontsize = 28

# 绘制 'times' 为柱状图
ax1.bar(values, times, width=0.32, label='Process Time (s)', color=bar_color)
# ax1.set_xlim(0.08, 0.52)
ax1.set_ylim(0, 55)
ax1.set_xticks(values)
ax1.set_xlabel('Architecture abbreviation', fontsize=font)
ax1.set_ylabel('Time', color='black', fontsize=font)
ax1.tick_params(axis='both', labelsize=tick_fontsize)

# 创建第二个 y 轴，为 car_hotas 和 truck_hotas
ax2 = ax1.twinx()
ax2.plot(values, car_hotas, 's-', label='Car HOTA',
         color=line_color1, markersize=marker_size - 5, linewidth=linewidth)
ax2.plot(values, truck_hotas, marker='*', label='Truck HOTA',
         color=line_color2, markersize=marker_size, linewidth=linewidth)
ax2.set_ylim(10, 55)
ax2.set_ylabel('HOTA', color='black', fontsize=font)
ax2.set_yticks([10, 18, 26, 34, 42, 50])
ax2.tick_params(axis='both', labelsize=tick_fontsize)

# 添加标题和其他属性设置
# plt.title(title1, fontsize=font, fontweight='bold')
# fig.tight_layout()
# fig.legend(loc='upper left', fontsize=tick_fontsize)
plt.grid(True, which='both', axis='x')

plt.savefig("./figures/figure1_b.pdf", dpi=200, bbox_inches='tight')
