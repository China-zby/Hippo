import os
import pickle
from tqdm import tqdm

def is_dominated(x, y):
    return all(x[i] <= y[i] for i in range(2)) and any(x[i] < y[i] for i in range(2))

def identify_pareto(solutions):
    pareto_front = []
    for solution in solutions:
        if not any(is_dominated(solution, other) for other in solutions):
            if ",".join([str(int(res)) for res in solution[-1]]) not in [",".join([str(int(res)) for res in pareto[-1]]) for pareto in pareto_front]:
                pareto_front.append(solution)
    return pareto_front

def collocate_pareto_set(video_id, camera_type="train", cache_dir="./cache/global/one_camera_with_config"):
    ingestion_result_path_name_list = os.listdir(cache_dir)
    filtered_ingestion_result_path_name_list = [
        path_name for path_name in ingestion_result_path_name_list if f"{camera_type}_{video_id}_" in path_name]
    result_sets = []
    for ingestion_result_path_name in filtered_ingestion_result_path_name_list:
        ingestion_result_path = os.path.join(
            cache_dir, ingestion_result_path_name)
        with open(ingestion_result_path, "rb") as f:
            ingestion_result = pickle.load(f)
            cache_info = ingestion_result_path_name.rstrip(".pkl").split("_")[
                1:]
            video_id, config_vector = int(cache_info[0]), cache_info[1:]
            config_vector = [int(res) for res in config_vector]
            _, ingestionvalue, _, _, _, cmetric, _ = ingestion_result
            result_sets.append(
                [ingestionvalue, cmetric[-1], config_vector])
    pareto_set = identify_pareto(result_sets)
    pareto_set = sorted(pareto_set, key=lambda x: x[0])
    return pareto_set

if __name__ == "__main__":
    cache_data_dir = "./cache/global/one_camera_with_config/"
    filtered_video_ids = []
    for video_id in tqdm(range(800)):
        video_pareto_set = collocate_pareto_set(video_id, cache_dir=cache_data_dir)
        if len(video_pareto_set) < 5:
            filtered_video_ids.append(video_id)
    print(filtered_video_ids)