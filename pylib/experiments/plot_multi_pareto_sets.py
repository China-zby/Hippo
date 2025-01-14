import os
import json
import numpy as np
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    method_names = ["hippo", "otif", "skyscraper", "unitune"]
    method_colors = sns.color_palette("hls", len(method_names))
    method_markers = ["o", "s", "D", "P"]

    title_fontsize = 12
    label_fontsize = 12
    data_dir = "./paretos"
    value_dir = os.path.join(data_dir, "values")
    result_dir = os.path.join(data_dir, "results")
    save_figure_dir = os.path.join(data_dir, "figures")

    # video_nums = [100, 150, 200, 250, 300]
    video_nums = [100]
    cluster_ids = list(range(5))

    value_path_names = os.listdir(value_dir)
    result_path_names = os.listdir(result_dir)

    for video_num in video_nums:
        for cluster_id in cluster_ids:
            plt.figure()
            for method_i, method_name in enumerate(method_names):
                path_prefix = f"{method_name}_cluster_{video_num}_{cluster_id}"
                filter_path_names = [
                    value_path_name for value_path_name in value_path_names if path_prefix in value_path_name]
                optimal_path_prefix = f"{method_name}_{video_num}_optimal_config_indices"
                filter_optimal_path_names = [
                    result_path_name for result_path_name in result_path_names if optimal_path_prefix in result_path_name]
                if len(filter_path_names) == 0:
                    print("filter_path_names: ", filter_path_names, "method_name: ",
                          method_name, "video_num: ", video_num, "cluster_id: ", cluster_id)
                    continue
                filter_path_name = filter_path_names[0]
                filter_optimal_path_name = filter_optimal_path_names[0]
                filter_optimal_path = os.path.join(
                    result_dir, filter_optimal_path_name)
                optimal_config_indices = json.load(
                    open(filter_optimal_path, "r"))
                in_cluster_number = int(
                    filter_path_name.rstrip(".pkl").split("_")[-1])
                filter_path = os.path.join(value_dir, filter_path_name)
                pareto_set = pkl.load(open(filter_path, "rb"))
                if method_name == "hippo":
                    print("pareto_set: ", filter_path, len(pareto_set))
                x = [pareto[0] for pareto in pareto_set]
                y = [pareto[1] for pareto in pareto_set]
                plt.scatter(x, y, color=method_colors[method_names.index(method_name)], marker=method_markers[method_names.index(
                    method_name)], label=method_name.capitalize(), alpha=0.5, s=100, edgecolors='black')
                # Use arrows to highlight the selected configuration points.
                if optimal_config_indices is not None:
                    method_optimal_config_indice = int(
                        optimal_config_indices[cluster_id])
                    selected_x = x[method_optimal_config_indice]
                    selected_y = y[method_optimal_config_indice]
                    plt.annotate(f"{method_name.capitalize()}", (selected_x, selected_y), textcoords="offset points",
                                 xytext=(10 * (method_i + 1), 10 * (method_i + 1)), ha='center', fontsize=8, arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
            plt.title(
                f"Video Number: {video_num},    Cluster Id: {cluster_id},    Cluster Number: {in_cluster_number}", fontsize=title_fontsize)
            plt.xlabel("Accuracy", fontsize=label_fontsize)
            plt.ylabel("Latency", fontsize=label_fontsize)
            plt.legend()
            plt.savefig(os.path.join(
                save_figure_dir, f"pareto_video_{video_num}_cluster_{cluster_id}_{in_cluster_number}.png"))
            plt.cla()
            plt.clf()
            plt.close()
