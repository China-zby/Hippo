import os
import json
import time
import random
import subprocess
import numpy as np
import seaborn as sns
from alive_progress import alive_bar
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.patches import PathPatch, Path


def read_json(json_file):
    data = json.load(open(json_file, 'r'))
    return data


def run_all_commands(commands):
    processes = []
    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True)
        #    , stdin=subprocess.DEVNULL,
        #    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(process)
    for p in processes:
        p.wait()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# Create a custom skull path
def create_skull_path():
    verts = [
        (0.5, 0.8),
        (0.65, 0.65),
        (0.7, 0.5),
        (0.65, 0.35),
        (0.5, 0.2),
        (0.35, 0.35),
        (0.3, 0.5),
        (0.35, 0.65),
        (0.5, 0.8),
    ]
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
    return Path(verts, codes)


sns.set_palette("husl")


def place_skull_on(ax, x, y, size=0.5):
    path = create_skull_path()
    offset_and_scale = transforms.Affine2D().scale(size).translate(x, y)
    skull = PathPatch(path, facecolor='black',
                      transform=offset_and_scale + ax.transData)
    ax.add_patch(skull)


parra_number = 3

time_rates = read_json("time_rates.json")

# Data extraction
x_data = np.array(time_rates)[:, 0][:98]
y_data = np.array(time_rates)[:, -1][:98]

fontsize_title = 24
fontsize_label = 22

# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 4))

ax.set_xlabel('Number of Video Candidates', fontsize=fontsize_label)
ax.set_ylabel('STR (%)', fontsize=fontsize_label)

ax.set_xticks([i for i in range(1, 102, 10)],
              minor=False)
ax.tick_params(axis='x', labelsize=fontsize_label - 2)
ax.tick_params(axis='y', labelsize=fontsize_label - 2)

# plot the horizontal line that y = 1.0
# ax.axhline(y=1.0, color='r', linestyle='--', lw=2)
# plot the vertical line that y(x) ≈ 1.0
bad_start_x = 0
bad_start_y = 0
bad_start_i = 0
bad_distance = 9999.9
for i, (videonumber, total_time, process_time, rate) in enumerate(time_rates):
    if abs(rate - 1.0) < bad_distance:
        bad_distance = abs(rate - 1.0)
        bad_start_x = videonumber
        bad_start_y = rate
        bad_start_i = i
ax.axvline(x=bad_start_x, color="#808080", linestyle='--', lw=2)

ax.plot(x_data[:bad_start_i+1], y_data[:bad_start_i+1], color='dodgerblue', marker='o',
        lw=2, label='Efficiency Before Threshold', markersize=5)

ax.plot(x_data[bad_start_i:], y_data[bad_start_i:], color='#8C8CA9', marker='o',
        lw=2, label='Efficiency After Threshold', markersize=5)


plt.grid(True, which='both', axis='y', linestyle='--', linewidth=1.5)

# Save the figure with higher resolution
plt.savefig("./figures/conflicts.png", dpi=200, bbox_inches='tight')

# Clear the current figure and axis
plt.close()
plt.clf()
plt.cla()

parra_number = 3
# Assuming you have a function to read the JSON file.
time_rates = read_json("time_rates.json")

# To make this example self-contained, let's assume the following dummy data
# time_rates = [(i, i*10, i*5, i*0.02) for i in range(100)]

# Data extraction
x_data = np.array(time_rates)[:, 0][:98]
y_data = np.array(time_rates)[:, -1][:98]

fontsize_title = 24
fontsize_label = 22

# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 4))

bad_start_x = 0
bad_start_y = 0
bad_start_i = 0
bad_distance = 9999.9
for i, (videonumber, total_time, process_time, rate) in enumerate(time_rates):
    if abs(rate - 1.0) < bad_distance:
        bad_distance = abs(rate - 1.0)
        bad_start_x = videonumber
        bad_start_y = rate
        bad_start_i = i

# ax.axvline(x=bad_start_x, color="#808080", linestyle='--', lw=2)
ax.axhline(y=1.0, color='r', linestyle='--', lw=2)

chunk_size = 10  # Chunk size
x_data_chunked = [chunk[0] for chunk in chunks(x_data, chunk_size)]
y_data_chunked = [chunk[0] for chunk in chunks(y_data, chunk_size)]

x_data_before, y_data_before = [], []
x_data_after, y_data_after = [], []

for i, (x, y) in enumerate(zip(x_data, y_data)):
    if i % chunk_size != 0:
        continue

    if i < bad_start_i:
        x_data_before.append(x)
        y_data_before.append(y)
    else:
        x_data_after.append(x)
        y_data_after.append(y)

colors = sns.color_palette("Blues")

# Plotting bars instead of lines
width = chunk_size * 0.4  # width of the bars
print(x_data_before)
print(y_data_before)
ax.bar(x_data_before, y_data_before,
       color=colors[1], width=width, label='STR < 100%')

ax.bar(x_data_after, y_data_after,
       color=colors[-1], width=width, label='STR > 100%')

ax.set_xlabel('Number of Video Candidates', fontsize=fontsize_label)
ax.set_ylabel('STR (%)', fontsize=fontsize_label)

ax.set_xticks([i + 1 for i in x_data if i % chunk_size == 0], [int(i)
              for i in x_data if i % chunk_size == 0], minor=False)
ax.tick_params(axis='x', labelsize=fontsize_label - 2)
ax.tick_params(axis='y', labelsize=fontsize_label - 2)

plt.grid(True, which='both', axis='y', linestyle='--', linewidth=1.5)
plt.legend(loc="upper left", fontsize=fontsize_label - 2)

# Save the figure with higher resolution
plt.savefig("./figures/conflicts_bar.pdf", dpi=200, bbox_inches='tight')

# Clear the current figure and axis
plt.close()
plt.clf()
plt.cla()

# with alive_bar(100, title="Running") as bar:
#     for videonumber in range(1, 101, 1):
#         if videonumber in [i[0] for i in time_rates]:
#             bar()
#             continue
#         cmds = []
#         for i in range(videonumber):
#             cmds.append(
#                 [f"go run step.go base.yaml ./conficts/base_{i}.json {i}", f"./conficts/base_{i}.json"])
#         total_time, process_time = 0, 0
#         for run_cmds in chunks(cmds, parra_number):
#             time_start = time.time()
#             run_all_commands([run_cmd[0] for run_cmd in run_cmds])
#             time_end = time.time()
#             process_time_one_step = time_end - time_start
#             total_time += process_time_one_step

#             max_latency = -9999
#             for run_cmd in run_cmds:
#                 ingestion_result = read_json(run_cmd[1])
#                 latency = ingestion_result["Latency"]
#                 if latency > max_latency:
#                     max_latency = latency

#             process_time += process_time_one_step - max_latency

#             os.system("bash rm_all.sh")

#         time_rates.append(
#             [videonumber, total_time, process_time, process_time / 60])
#         json.dump(time_rates, open("time_rates.json", "w"))
#         print(videonumber, total_time, process_time, process_time / 60)
#         bar()

#         # # Seaborn styling
#         # sns.set_style("whitegrid")

#         # # Data extraction
#         # x_data = np.array(time_rates)[:, 0]
#         # y_data = np.array(time_rates)[:, -1]

#         # # Create a figure and axis
#         # fig, ax = plt.subplots(figsize=(12, 6))

#         # # Plot data with a smooth line and better color
#         # ax.plot(x_data, y_data, '-o', color='dodgerblue',
#         #         lw=2, label='Efficiency Loss Ratio')

#         # # Title and labels
#         # ax.set_title('Efficiency Loss Ratio for Resolving Conflicts',
#         #              fontsize=fontsize_title, fontweight='bold')
#         # ax.set_xlabel('The Number of Video Streams', fontsize=fontsize_label)
#         # ax.set_ylabel('Efficiency Loss Ratio (%)', fontsize=fontsize_label)

#         # ax.set_xticks([i for i in range(0, 101, 10)],
#         #               minor=False)
#         # ax.tick_params(axis='x', labelsize=fontsize_label)
#         # ax.tick_params(axis='y', labelsize=fontsize_label)

#         # # Displaying the legend
#         # # ax.legend(loc="upper left")

#         # # Save the figure with higher resolution
#         # plt.savefig("conflicts.png", dpi=200, bbox_inches='tight')

#         # # Clear the current figure and axis
#         # plt.close()
#         # plt.clf()
#         # plt.cla()
