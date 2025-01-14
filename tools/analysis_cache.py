import os
import pickle as pkl
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == "__main__":
    cache_dir = "./cache/global/one_camera_with_config"
    save_dir = "./delete_figures"
    cache_files = os.listdir(cache_dir)
    
    video_cache_dict = {}
    for cache_file in cache_files:
        cache_infos = cache_file.split(".")[0].split("_")[1:]
        video_id, config_vector = int(cache_infos[0]), cache_infos[1:]
        config_str = "".join(config_vector)

        with open(os.path.join(cache_dir, cache_file), "rb") as f:
            cache = pkl.load(f)
            overallPerformance, ingestionvalue, record_latency, mot_metrics, save_record, \
                    cmetric, cmetricname = cache
        acc, eff = ingestionvalue, cmetric[-1]
        
        if video_id not in video_cache_dict:
            video_cache_dict[video_id] = []

        video_cache_dict[video_id].append([acc, eff])
        
    # 画横竖20 * 40 = 800张图
    # plt.figure(figsize=(40, 20))
    # 画图
    # video_ids = [84, 200, 206, 237, 267, 268, 304, 358, 364, 464, 471, 479, 489, 501, 543, 560, 601, 716, 769]
    video_ids = list(video_cache_dict.keys())
    for video_id in tqdm(video_ids):
        cache_list = video_cache_dict[video_id]
        cache_list = sorted(cache_list, key=lambda x: x[0])
        
        # ax = plt.subplot(20, 40, video_id + 1)
        # ax.scatter([x[0] for x in cache_list], [x[1] for x in cache_list])
        # ax.set_xlabel("Accuracy")
        # ax.set_ylabel("Efficiency")
        # ax.set_xlim(0, 1)
        # ax.set_ylim(0, 1)
        # ax.set_title(f"{video_id}")
        
        plt.figure()
        plt.scatter([x[0] for x in cache_list], [x[1] for x in cache_list])
        plt.xlabel("Accuracy")
        plt.ylabel("Efficiency")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.savefig(os.path.join(save_dir, f"{video_id}.png"))
        plt.close()
        plt.cla()
        plt.clf()

    # plt.savefig(os.path.join(save_dir, "all.png"))