import matplotlib.pyplot as plt
import seaborn as sns

# 设置样式
# sns.set_style("whitegrid")
sns.set_context("paper")
sns.set_palette("Set1")
sns.set_color_codes()

# 创建一个空白的figure对象和一个axes对象
fig, ax = plt.subplots()

# 假设methods是你的方法列表，这里模拟生成图例所需要的label和handle
methods = ['OTIF', 'Skyscraper', 'UniTune', 'Hippo']
handles = [plt.Rectangle((0,0),1,1, color=sns.color_palette("Set1")[i]) for i in range(len(methods))]

# 创建图例
legend = ax.legend(handles, methods, loc='center', frameon=False)

# 创建一个新的figure，用来单独保存图例
fig_leg = plt.figure(figsize=(6, 1))
ax_leg = fig_leg.add_subplot(111)
# 把已经创建的图例绘制在新的figure上
ax_leg.legend(handles, methods, loc='center', frameon=False, ncol=4)
ax_leg.axis('off')  # 关闭坐标轴

# 保存图例
fig_leg.savefig('./paretos/results/legend.png', bbox_inches='tight', dpi=300)

# 关闭图例的figure，释放资源
plt.close(fig_leg)
