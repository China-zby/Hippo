import pandas as pd
import os
import re
import csv
import json

# Setting visual parameters
pad = 5
gridwidth = 2
linewidth = 4
markersize = 18
scatter_size = 150
fontsize_label = 24
fontsize_tick = 20
fontsize_title = 28
legend_fontsize = 18

# Data parameters
channel_range_step = 5
channel_range_gap = 50
time_constrain = 60
metrics = ['sel', 'agg', 'topk', 'hota']
objects = ['car', 'bus', 'truck']
channels = [50, 100, 150, 200, 250]
methods = ['chameleon', 'otif', 'skyscraper',
           'dynamicsafe', 'cdbtune', 'unitune', 'ardent']
methods = list(reversed(methods))
ytitle = {'sel': 'F1-score', 'agg': 'MAPE',
          'topk': 'Precision', 'hota': 'HOTA'}

write_file = open('./tables/safe_rate.csv', 'w')
csv_writer = csv.writer(write_file)

csv_writer.writerow(['Method'] + channels)

for method in methods:
    method_files = os.listdir(f"./cache/{method}")
    pattern = re.compile(
        f"{re.escape(method)}_channel(\\d+)_final_result.json")

    cache_channel_files = [f for f in method_files if pattern.match(f)]
    cache_channels = [int(pattern.match(f).group(1))
                      for f in cache_channel_files]

    safe_rates = [100] * len(channels)
    for ci, channel in enumerate(channels):
        filter_channel_files = []
        for cache_channel_file, cache_channel in zip(cache_channel_files, cache_channels):
            if cache_channel >= channel - channel_range_gap and cache_channel <= channel and (cache_channel - 1) % channel_range_step == 0:
                filter_channel_files.append(cache_channel_file)

        safe_number = 0
        for channel_file in filter_channel_files:
            channel_file = f"./cache/{method}/{channel_file}"
            with open(channel_file, 'r') as f:
                channel_result = json.load(f)
            if channel_result['Latency'] < time_constrain:
                safe_number += 1
        if len(filter_channel_files) > 0:
            safe_rates[ci] = safe_number / len(filter_channel_files) * 100
        else:
            safe_rates[ci] = 0

    csv_writer.writerow([method] + safe_rates)
write_file.close()
print('Done')

# print the table
df = pd.read_csv('./tables/safe_rate.csv')
print(df)
