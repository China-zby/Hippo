import numpy as np
import matplotlib.pyplot as plt
import math

def incentive_function(acc, delay):
    return (math.exp(acc) / math.exp(1) * math.exp(1.5 / delay) / math.exp(1))**0.2


fontsize_title = 28
fontsize_label = 24

# Define the grid
acc_values = np.linspace(0.0, 1.01, 100)
delay_values = np.linspace(2.0, 8.0, 100)
acc_grid, delay_grid = np.meshgrid(acc_values, delay_values)
z_values = np.vectorize(incentive_function)(acc_grid, delay_grid)

plt.figure(figsize=(10,8))
cp = plt.contourf(acc_grid, delay_grid, z_values, 50, cmap='Blues')
cbar = plt.colorbar(cp, label='Incentive Function Value')
cbar.ax.tick_params(labelsize=fontsize_label)
cbar.set_label('Incentive Function Value', fontsize=fontsize_label)

# Emphasize contour lines
contour = plt.contour(acc_grid, delay_grid, z_values, colors='black', linestyles='solid', linewidths=1.5, fontsize=fontsize_label)
plt.clabel(contour, inline=1, fontsize=fontsize_label)

# Get midpoints of contours and draw arrows
midpoints = []
for collection in contour.collections:
    for path in collection.get_paths():
        vertices = path.to_polygons()[0]
        midpoint = vertices[len(vertices) // 2]
        midpoints.append(midpoint)

start_points = [0, 2, 4, 6, 7]
end_points = [2, 4, 6, 7, 9]
# for i in range(1, len(midpoints) - 1, 1):
#     plt.annotate("", xy=midpoints[i + 1], xytext=midpoints[i], arrowprops=dict(arrowstyle="->", color="blue", lw=2))

shift_vector_1s = [[0.10845562673490687, 6.075757575757576], [0.3540644013018783, 5.563636363636363], [0.569281474940117, 4.8], [0.7532316074549265, 4.0], [0.8938495462713953, 3.212121212121212]]
shift_vector_2s = [[0.355644013018783, 5.563636363636363], [0.58281474940117, 4.77], [0.7532316074549265, 4.0], [0.8938495462713953, 3.212121212121212], [0.9794333786118915, 2.484848484848485]]

i = 0
for shift, end_points in zip(shift_vector_1s, shift_vector_2s):
    plt.annotate("", xy=end_points, xytext=shift, arrowprops=dict(arrowstyle="wedge,tail_width=0.9,shrink_factor=0.5", color="blue", lw=4), fontsize=fontsize_label)
    i += 1

# plt.title('Incentive Function with Contours and Directional Arrows')
plt.xlabel('Accuracy (acc)', fontsize=fontsize_label)
plt.ylabel('Delay (delay)', fontsize=fontsize_label)

plt.xticks(fontsize=fontsize_label)
plt.yticks(fontsize=fontsize_label)

plt.savefig('incentive_function.png', dpi=200, bbox_inches='tight')
