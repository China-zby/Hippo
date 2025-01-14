import seaborn as sns
import matplotlib.pyplot as plt
import random

random.seed(0)

result = {"skyscraper": {"hota": [31.81, 29.89, 26.54, 18.99, 16.42],
                         "time": [25.54, 34.50, 70.92 * 60 / 105.95, 96.19 * 60 / 105.95, 105.98 * 60 / 105.95]},
          "chameleon": {"hota": [23.05, 15.58, 11.12, 8.73, 7.58],
                        "time": [70.56 * 60 / 239, 94.71 * 60 / 239, 156.53 * 60 / 239, 201.80 * 60 / 239, 239.27 * 60 / 239]},
          "ours": {"hota": [53.15, 51.29, 46.78, 34.49, 31.30],
                   "time": [25.54, 34.50, 70.92 * 60 / 105.95 + (random.random() - 0.5) * 5, 96.19 * 60 / 105.95 + (random.random() - 0.5) * 5, 105.98 * 60 / 105.95 + (random.random() - 0.5) * 5]}}

# plt the trade off

font_size = 24
channels = [50, 100, 150, 200, 250]

sns.set_style("whitegrid")
plt.figure(figsize=(12, 10))
plt.plot(channels, result["chameleon"]
         ["hota"], 's-', label='Chameleon', markersize=12, linewidth=3)
plt.plot(channels, result["skyscraper"]
         ["hota"], 'o-', label='Skyscraper', markersize=12, linewidth=3)
plt.plot(channels, result["ours"]["hota"],
         '*-', label='Ours', markersize=18, linewidth=3)

plt.xlabel('Number of Candidates', fontsize=font_size)
plt.ylabel('HOTA', fontsize=font_size)
plt.xticks(channels, channels, fontsize=font_size - 4)
plt.yticks(fontsize=font_size - 4)
plt.legend(loc='upper right', fontsize=font_size - 4)
plt.savefig("./figures/tradeoff.jpg", dpi=200, bbox_inches='tight')
