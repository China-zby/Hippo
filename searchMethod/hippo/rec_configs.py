import os
import time
import utils
import torch
import random
import numpy as np
import pickle as pkl
from tqdm import tqdm
from config_ import Config
from solver import solve_traverse
from config_ import generate_random_config
from sklearn.cluster import MiniBatchKMeans
from cameraclass import build_a_camera_with_config
from utils import SEARCH_SPACE_NAMES, GOLDEN_CONFIG_VECTOR
from pcluster import k_means_balanced, update_cluster_by_new_context_vectors

def extract_context_vector_by_skyscraper(camera, configurations_set):
    context_vector = []
    for configuration in configurations_set:
        camera_config = camera.loadConfig()
        camera_config = utils.generate_config(configuration, camera_config)
        camera.updateConfig(camera_config)
        time_start = time.time()
        _, motvalue, _, _, _, \
                            cmetric, _ = camera.ingestion#_without_cache
        accuracy, latency = motvalue, cmetric[-1]
        ingestion_time = time.time() - time_start
        context_vector.extend([accuracy, latency])
    return context_vector, ingestion_time
        

def prepare_context_observation(context):
    cameraid = context['Cameraid']
    timeid = context['Time']
    quality = context['Quailty']
    traj_context = [context['mean_speed'], context['std_speed'], context['mean_acceleration'], context['std_acceleration'],
                    context['mean_traj_linearity'], context['mean_traj_length'], context['traj_count'],
                    context['proportion_of_stopped_vehicles'],
                    context['mean_traj_densitys'], context['std_traj_densitys'],
                    context['car_rate'], context['bus_rate'], context['truck_rate']]
    return [cameraid, timeid, quality] + traj_context


def hippo_func(seed=None,
               video_num=None, video_ids=None,
               policy=None,
               env=None, test_env=None,
               train_scene_dict=None, test_scene_dict=None,
               cluster_nums=None,
               gpu_resource_info=None, indexs=None,
               configpath=None,
               deviceid=None,
               data_name=None, data_type=None, scene_name=None,
               objects=["car", "bus", "truck"],
               temp=10.0,
               max_pool_lenth=6,
               latency_bound=60, memory_bound=24256,
               max_pareto_set_size=5, object_nan=100,
               train_fit=True,
               use_experience=True, **kwargs):
    print(f"Start Hippo Function: {video_num}")
    if train_fit: 
        fitted_train_video_ids, context_vectors = [], []
        for video_id in video_ids:
            distances = []
            context_observation = prepare_context_observation(
                test_env.dataset.getitem_context(video_id))
            for train_video_id in env.video_ids:
                if train_scene_dict[train_video_id] != test_scene_dict[video_id]:
                    continue
                env_context = env.dataset.getitem_context(train_video_id)
                env_context = prepare_context_observation(env_context)
                distance = 0.0
                for co, ec in zip(context_observation, env_context):
                    distance += (co - ec) ** 2
                distances.append([distance, env_context, train_video_id])
            context_observation = min(distances, key=lambda x: x[0])[1]
            context_vectors.append(context_observation)
            fitted_train_video_ids.append( 
                min(distances, key=lambda x: x[0])[2])
        context_vectors = np.array(context_vectors)
    else:
        context_vectors = np.array(
            [prepare_context_observation(test_env.dataset.getitem_context(video_id))
                for video_id in video_ids])
        fitted_train_video_ids = []
        for video_id, context_vector in zip(video_ids, context_vectors):
            distances = []
            for train_video_id in env.video_ids:
                if train_scene_dict[train_video_id] != test_scene_dict[video_id]:
                    continue
                env_context = env.dataset.getitem_context(train_video_id)
                env_context = prepare_context_observation(env_context)
                distance = 0.0
                for co, ec in zip(context_vector, env_context):
                    distance += (co - ec) ** 2
                distances.append([distance, env_context, train_video_id])
            fitted_train_video_ids.append(min(distances, key=lambda x: x[0])[2])

    find_solution = False
    ClusterNum = np.zeros(cluster_nums[1])
    run_cmds, ingestion_result_paths = [], []
    optimal_config_indices = None
    for cluster_num in range(cluster_nums[1], cluster_nums[0], -1):
        if os.path.exists(f"./clusters/kmeans_{video_num}_{cluster_num}.pkl"): 
            with open(f"./clusters/kmeans_{video_num}_{cluster_num}.pkl", "rb") as f:
                cluster_labels, cluster_centers = pkl.load(f)
        else:
            if os.path.exists("./clusters/kmeans_model.pkl"):
                with open("./clusters/kmeans_model.pkl", "rb") as f:
                    mb_kmeans = pkl.load(f)
                with open("./clusters/video_ids.pkl", "rb") as f:
                    last_video_ids = pkl.load(f)
                with open("./clusters/cluster_info.pkl", "rb") as f:
                    last_cluster_labels, _ = pkl.load(f)
                new_context_vectors = []
                for context_vector, video_id in zip(context_vectors, video_ids):
                    if video_id not in last_video_ids:
                        new_context_vectors.append(context_vector)
                        
                if new_context_vectors:
                    new_context_vectors = np.array(new_context_vectors)
                    mb_kmeans.partial_fit(new_context_vectors)
                    new_labels = mb_kmeans.predict(new_context_vectors)
                    new_labels = list(new_labels)
                else:
                    new_labels = []
                    
                cluster_labels = []
                for video_id in video_ids:
                    if video_id in last_video_ids:
                        cluster_labels.append(last_cluster_labels[last_video_ids.index(video_id)])
                    else:
                        cluster_labels.append(new_labels.pop(0))
                cluster_centers = mb_kmeans.cluster_centers_
                cluster_labels = np.array(cluster_labels)
                
                with open("./clusters/kmeans_model.pkl", "wb") as f:
                    pkl.dump(mb_kmeans, f)
                with open("./clusters/video_ids.pkl", "wb") as f:
                    pkl.dump(video_ids, f)
                with open("./clusters/cluster_info.pkl", "wb") as f:
                    pkl.dump([cluster_labels, cluster_centers], f)
            else:
                mb_kmeans = MiniBatchKMeans(n_clusters=cluster_num, random_state=seed)
                mb_kmeans.fit(context_vectors)
                cluster_centers, cluster_labels = mb_kmeans.cluster_centers_, mb_kmeans.labels_
                # save mb kmeans
                with open("./clusters/kmeans_model.pkl", "wb") as f:
                    pkl.dump(mb_kmeans, f)
                with open("./clusters/video_ids.pkl", "wb") as f:
                    pkl.dump(video_ids, f)
                with open("./clusters/cluster_info.pkl", "wb") as f:
                    pkl.dump([cluster_labels, cluster_centers], f)
            
            with open(f"./clusters/kmeans_{video_num}_{cluster_num}.pkl", "wb") as f:
                pkl.dump([cluster_labels, cluster_centers], f)

        AccMatrix, EffMatrix = np.zeros((cluster_num, max_pareto_set_size)), np.zeros(
            (cluster_num, max_pareto_set_size))
        ClusterNum = np.zeros(cluster_num)
        ClusterDict, ClusetrParetoSet = {}, {}
        # repre_train_video_ids = []
        for cluster_label in tqdm(range(cluster_num)): 
            cluster_context = cluster_centers[cluster_label] 
            same_cluster_ids = np.where(cluster_labels == cluster_label)[0] 
            same_cluster_contexts = context_vectors[same_cluster_ids]

            same_cluster_video_ids = [video_ids[same_cluster_id] 
                                      for same_cluster_id in same_cluster_ids] 
            ClusterDict[cluster_label] = same_cluster_video_ids
            # repre_incluster_id = np.argmin()
            # repre_video_id = same_cluster_video_ids[repre_incluster_id]
            # repre_context_vector = same_cluster_contexts[repre_incluster_id]
            
            repre_video_number = 5 
            cluster_dists = [[float(np.linalg.norm(
                context_vector - cluster_context)), ci] for ci, context_vector in enumerate(same_cluster_contexts)] 
            cluster_dists = sorted(cluster_dists, key=lambda x: x[0]) 
            
            camids = []
            repre_incluster_ids = []
            for _, incluster_id in cluster_dists: 
                repre_flag = f"{same_cluster_contexts[incluster_id][0]:.3f}"
                if repre_flag not in camids: 
                    camids.append(repre_flag) 
                    repre_incluster_ids.append(incluster_id) 
            
            incluster_ids = list(range(len(same_cluster_contexts))) 
            if len(repre_incluster_ids) < repre_video_number:
                while len(repre_incluster_ids) < repre_video_number:
                    repre_sample_video_id = random.choice(incluster_ids)
                    if repre_sample_video_id not in repre_incluster_ids:
                        repre_incluster_ids.append(repre_sample_video_id)
            elif len(repre_incluster_ids) > repre_video_number:
                repre_incluster_ids = random.sample(repre_incluster_ids, repre_video_number)
            else:
                repre_incluster_ids = repre_incluster_ids
            
            repre_video_ids = [same_cluster_video_ids[repre_incluster_id] for repre_incluster_id in repre_incluster_ids]
            repre_context_vectors = [same_cluster_contexts[repre_incluster_id] for repre_incluster_id in repre_incluster_ids] 

            if use_experience: 
                raw_clusetr_pareto_set = []
                for repre_video_id in repre_video_ids: 
                    repre_video_clusetr_pareto_set = env.collocate_pareto_set( # collocate_pareto_set 函数就是获取视频流训练中采样的所有配置的帕累托集合
                        fitted_train_video_ids[video_ids.index(repre_video_id)])
                    raw_clusetr_pareto_set.extend(repre_video_clusetr_pareto_set)
                raw_clusetr_pareto_set = sorted(raw_clusetr_pareto_set, key=lambda x: x[1])
                clusetr_pareto_set = []
                for raw_acc, raw_eff, raw_config_vector in raw_clusetr_pareto_set:
                    acc, eff = env.run_config(raw_config_vector)
                    clusetr_pareto_set.append([acc, eff, raw_config_vector])
                clusetr_pareto_set = env.identify_pareto(clusetr_pareto_set)
                if len(clusetr_pareto_set) > max_pareto_set_size: 
                    clusetr_pareto_set_idxs = env.evenly_sample_indices(np.array([point[0] for point in clusetr_pareto_set]), max_pareto_set_size)
                    clusetr_pareto_set = [clusetr_pareto_set[clusetr_pareto_set_idx] for clusetr_pareto_set_idx in clusetr_pareto_set_idxs]
            else:
                with torch.no_grad():
                    observation_all, _ = test_env.reset(
                        video_id=repre_video_ids[0])
                    done, truncated = False, False
                    while not done and not truncated:
                        action, _ = policy.predict(observation_all[0],
                                                   repre_context_vectors[0],
                                                   deterministic=True)
                        observation_all, _, done, truncated, _ = test_env.step(
                            action)
                    clusetr_pareto_set = test_env.pareto_set
            clusetr_pareto_set = sorted(clusetr_pareto_set, key=lambda x: x[0])

            utils.plot_pareto_set([point[0] for point in clusetr_pareto_set],
                                  [point[1]
                                   for point in clusetr_pareto_set],
                                  f"./paretos/figures/hippo_cluster_{video_num}_{cluster_label}_{len(same_cluster_ids)}.png") 

            # print("clusetr_pareto_set: ", f"./paretos/values/hippo_cluster_{video_num}_{cluster_label}_{len(same_cluster_ids)}.pkl", len(clusetr_pareto_set))
            
            with open(f"./paretos/values/hippo_cluster_{video_num}_{cluster_label}_{len(same_cluster_ids)}.pkl", "wb") as f:
                pkl.dump(clusetr_pareto_set, f)

            for pi, res in enumerate(clusetr_pareto_set):
                AccMatrix[cluster_label, pi] = res[0] * len(same_cluster_ids)
                EffMatrix[cluster_label, pi] = res[1]
                
            ClusterNum[cluster_label] = len(same_cluster_ids)
            ClusetrParetoSet[cluster_label] = clusetr_pareto_set

        LatMatrix = - np.log(EffMatrix + 1e-8) * temp
        RealLatMatrix = []
        for i in range(cluster_num):
            RealLatMatrix.append([LatMatrix[i, j] +
                                  utils.get_latency_increase(LatMatrix[i, j]) * (ClusterNum[i] - 1) for j in range(max_pareto_set_size)])
        RealLatMatrix = np.array(RealLatMatrix)

        try:
            optimal_config_indices = solve_traverse(AccMatrix, 
                                                    RealLatMatrix,
                                                    latency_bound,
                                                    max_pool_lenth,
                                                    gpu_constrain=[ClusetrParetoSet, indexs, gpu_resource_info, memory_bound])
            find_solution = True
        except:
            break
        optimal_reallatency = [RealLatMatrix[cluster_label, optimal_config_indice]
                            for cluster_label, optimal_config_indice in enumerate(optimal_config_indices)]
        optimal_accuracy = [AccMatrix[cluster_label, optimal_config_indice]
                            for cluster_label, optimal_config_indice in enumerate(optimal_config_indices)]
        print(optimal_accuracy)
        print(optimal_reallatency)
        
    
        
        run_cmds, gpu_memory, ingestion_result_paths = [], [], []
        for cluster_label, optimal_config_indice in enumerate(optimal_config_indices): 
            config_action = ClusetrParetoSet[cluster_label][optimal_config_indice][2]
            cn = len(ClusterDict[cluster_label])
            run_config = Config(
                configpath, f'./cmds/run_hippo_cmd_{video_num}_{cluster_label}_{cn}.yaml',
                deviceid, data_name, data_type, objects,
                configpath, scene_name)
            ingestion_result_path = f'./results/run_hippo_cluster_{video_num}_{cluster_label}_{cn}_result.json'
            ingestion_result_paths.append(ingestion_result_path)
            run_config.update_cache(utils.generate_config(
                config_action, run_config.load_cache()))
            run_videoids = ",".join(
                [str(videoid) for videoid in ClusterDict[cluster_label]])
            run_cmd = f"go run step.go {run_config.cache_path} {ingestion_result_path} {run_videoids}"
            run_cmds.append(run_cmd)

            config_action_str = "_".join(
                [str(action_param) for action_param in config_action])
            config_action_str = "_".join(
                [config_action_str.split("_")[index] for index in indexs])
            gpu_memory.append(gpu_resource_info[config_action_str])

        if optimal_accuracy != -object_nan and sum(gpu_memory) <= memory_bound:
            break
    return find_solution, run_cmds, \
        ingestion_result_paths, ClusterNum, \
        optimal_config_indices #, repre_train_video_ids


def hippo_cluster_skyscraper_func(seed=None,
               video_num=None, video_ids=None,
               policy=None,
               env=None, test_env=None,
               train_scene_dict=None, test_scene_dict=None,
               cluster_nums=None,
               gpu_resource_info=None, indexs=None,
               configpath=None,
               deviceid=None,
               data_name=None, data_type=None, scene_name=None,
               objects=["car", "bus", "truck"],
               temp=10.0,
               max_pool_lenth=6,
               latency_bound=60, memory_bound=24256,
               max_pareto_set_size=5, object_nan=100,
               train_fit=True,
               use_experience=True, **kwargs):
    print(f"Start Hippo With Skyscraper Cluster: {video_num}")
    fitted_train_video_ids = []
    for video_id in video_ids:
        distances = []
        context_observation = prepare_context_observation(
            test_env.dataset.getitem_context(video_id))
        for train_video_id in env.video_ids:
            if train_scene_dict[train_video_id] != test_scene_dict[video_id]:
                continue
            env_context = env.dataset.getitem_context(train_video_id)
            env_context = prepare_context_observation(env_context)
            distance = 0.0
            for co, ec in zip(context_observation, env_context):
                distance += (co - ec) ** 2
            distances.append([distance, env_context, train_video_id])
        fitted_train_video_ids.append(min(distances, key=lambda x: x[0])[2])
    device_id = 1
    metrics = ["sel", "agg", "cq3"]
    envconfigpath = "./cache/base.yaml"
    method_name = "test_skyscraper_cluster"
    dataset_dir = "/home/lzp/otif-dataset/dataset"
    envcachepath = "./cache/test_skyscraper_cluster.yaml"
    alldir = utils.DataPaths(dataset_dir,
                                data_name, data_type, method_name)
    env_camera_config = generate_random_config(
        envconfigpath, envcachepath, device_id, data_name, data_type, objects, method_name, scene_name)
    env_camera = build_a_camera_with_config(
        alldir, env_camera_config, objects, metrics)
    
    save_context_path = "./clusters/skyscraper_context_vectors.pkl"
    if os.path.exists(save_context_path):
        with open(save_context_path, 'rb') as f:
            context_vectors = pkl.load(f)
    else:
        context_vectors = []
        for video_id in tqdm(video_ids, desc="Generate Skyscraper Context Vectors"):
            env_camera.reset_with_videoid(video_id)
            context_vectors.append(extract_context_vector_by_skyscraper(env_camera, env.skyscraper_configs))
        with open(save_context_path, 'wb') as f:
            pkl.dump(context_vectors, f)
    context_vectors = np.array(context_vectors)

    find_solution = False
    ClusterNum = np.zeros(cluster_nums[1])
    run_cmds, ingestion_result_paths = [], []
    optimal_config_indices = None
    for cluster_num in range(cluster_nums[1], cluster_nums[0], -1):
        mb_kmeans = MiniBatchKMeans(n_clusters=cluster_num, random_state=seed)
        mb_kmeans.fit(context_vectors)
        cluster_centers, cluster_labels = mb_kmeans.cluster_centers_, mb_kmeans.labels_

        AccMatrix, EffMatrix = np.zeros((cluster_num, max_pareto_set_size)), np.zeros(
            (cluster_num, max_pareto_set_size))
        ClusterNum = np.zeros(cluster_num)
        ClusterDict, ClusetrParetoSet = {}, {}
        for cluster_label in tqdm(range(cluster_num)):
            cluster_context = cluster_centers[cluster_label]
            same_cluster_ids = np.where(cluster_labels == cluster_label)[0]
            
            same_cluster_contexts = context_vectors[same_cluster_ids]

            same_cluster_video_ids = [video_ids[same_cluster_id]
                                      for same_cluster_id in same_cluster_ids]
            ClusterDict[cluster_label] = same_cluster_video_ids
            repre_incluster_id = np.argmin([float(np.linalg.norm(
                context_vector - cluster_context)) for context_vector in same_cluster_contexts])
            repre_video_id = same_cluster_video_ids[repre_incluster_id]
            repre_context_vector = same_cluster_contexts[repre_incluster_id]

            if use_experience:
                clusetr_pareto_set = env.collocate_pareto_set(
                    fitted_train_video_ids[video_ids.index(repre_video_id)])
                # evenly sample pareto set
                if len(clusetr_pareto_set) > max_pareto_set_size:
                    clusetr_pareto_set_idxs = env.evenly_sample_indices(
                        np.array([pareto[0] for pareto in clusetr_pareto_set]), max_pareto_set_size)
                    clusetr_pareto_set = [clusetr_pareto_set[idx]
                                          for idx in clusetr_pareto_set_idxs]
                # random drop 10% of pareto set
                random.shuffle(clusetr_pareto_set)
                clusetr_pareto_set = clusetr_pareto_set[:int(len(clusetr_pareto_set) * 0.6)]
            else:
                with torch.no_grad():
                    observation_all, _ = test_env.reset(
                        video_id=repre_video_id)
                    done, truncated = False, False
                    while not done and not truncated:
                        action, _ = policy.predict(observation_all[0],
                                                   repre_context_vector,
                                                   deterministic=True)
                        observation_all, _, done, truncated, _ = test_env.step(
                            action)
                    clusetr_pareto_set = test_env.pareto_set
            clusetr_pareto_set = sorted(clusetr_pareto_set, key=lambda x: x[0])

            utils.plot_pareto_set([point[0] for point in clusetr_pareto_set],
                                  [point[1]
                                   for point in clusetr_pareto_set],
                                  f"./paretos/figures/hippo_wo_age_cluster_{video_num}_{cluster_label}_{len(same_cluster_ids)}.png")

            with open(f"./paretos/values/hippo_wo_age_cluster_{video_num}_{cluster_label}_{len(same_cluster_ids)}.pkl", "wb") as f:
                pkl.dump(clusetr_pareto_set, f)

            for pi, res in enumerate(clusetr_pareto_set):
                AccMatrix[cluster_label, pi] = res[0] * len(same_cluster_ids)
                EffMatrix[cluster_label, pi] = res[1]
            ClusterNum[cluster_label] = len(same_cluster_ids)
            ClusetrParetoSet[cluster_label] = clusetr_pareto_set

        LatMatrix = - np.log(EffMatrix + 1e-8) * temp
        RealLatMatrix = []
        for i in range(cluster_num):
            RealLatMatrix.append([LatMatrix[i, j] +
                                  utils.get_latency_increase(LatMatrix[i, j]) * (ClusterNum[i] - 1) for j in range(max_pareto_set_size)])
        RealLatMatrix = np.array(RealLatMatrix)

        try:
            optimal_config_indices = solve_traverse(AccMatrix,
                                                    RealLatMatrix,
                                                    latency_bound,
                                                    max_pool_lenth,
                                                    gpu_constrain=[ClusetrParetoSet, indexs, gpu_resource_info, memory_bound])
            find_solution = True
        except:
            break
        optimal_accuracy = [AccMatrix[cluster_label, optimal_config_indice]
                            for cluster_label, optimal_config_indice in enumerate(optimal_config_indices)]

        run_cmds, gpu_memory, ingestion_result_paths = [], [], []
        for cluster_label, optimal_config_indice in enumerate(optimal_config_indices):
            config_action = ClusetrParetoSet[cluster_label][optimal_config_indice][2]
            cn = len(ClusterDict[cluster_label])
            run_config = Config(
                configpath, f'./cmds/run_hippo_wo_age_cmd_{video_num}_{cluster_label}_{cn}.yaml',
                deviceid, data_name, data_type, objects,
                configpath, scene_name)
            ingestion_result_path = f'./results/run_hippo_wo_age_cluster_{video_num}_{cluster_label}_{cn}_result.json'
            ingestion_result_paths.append(ingestion_result_path)
            run_config.update_cache(utils.generate_config(
                config_action, run_config.load_cache()))
            run_videoids = ",".join(
                [str(videoid) for videoid in ClusterDict[cluster_label]])
            run_cmd = f"go run step.go {run_config.cache_path} {ingestion_result_path} {run_videoids}"
            run_cmds.append(run_cmd)

            config_action_str = "_".join(
                [str(action_param) for action_param in config_action])
            config_action_str = "_".join(
                [config_action_str.split("_")[index] for index in indexs])
            gpu_memory.append(gpu_resource_info[config_action_str])

        if optimal_accuracy != -object_nan and sum(gpu_memory) <= memory_bound:
            break

    return find_solution, run_cmds, \
        ingestion_result_paths, ClusterNum, \
        optimal_config_indices


def hippo_without_pareto_reinforcement_learning_func(seed=None,
               video_num=None, video_ids=None,
               policy=None,
               env=None, test_env=None,
               train_scene_dict=None, test_scene_dict=None,
               cluster_nums=None,
               gpu_resource_info=None, indexs=None,
               configpath=None,
               deviceid=None,
               data_name=None, data_type=None, scene_name=None,
               objects=["car", "bus", "truck"],
               temp=10.0,
               max_pool_lenth=6,
               latency_bound=60, memory_bound=24256,
               max_pareto_set_size=5, object_nan=100,
               train_fit=True,
               use_experience=True, **kwargs):
    print(f"Start Hippo Without Pareto Reinforcement Learning Function: {video_num}")
    #train_fit=True #t
    if train_fit:
        fitted_train_video_ids, context_vectors = [], []
        for video_id in video_ids:
            distances = []
            context_observation = prepare_context_observation(
                test_env.dataset.getitem_context(video_id))
            for train_video_id in env.video_ids:
                if train_scene_dict[train_video_id] != test_scene_dict[video_id]:
                    continue
                env_context = env.dataset.getitem_context(train_video_id)
                env_context = prepare_context_observation(env_context)
                distance = 0.0
                for co, ec in zip(context_observation, env_context):
                    distance += (co - ec) ** 2
                distances.append([distance, env_context, train_video_id])
            context_observation = min(distances, key=lambda x: x[0])[1]
            context_vectors.append(context_observation)
            fitted_train_video_ids.append(
                min(distances, key=lambda x: x[0])[2])
        context_vectors = np.array(context_vectors)
    else:
        context_vectors = np.array(
            [prepare_context_observation(test_env.dataset.getitem_context(video_id))
                for video_id in video_ids])
        fitted_train_video_ids = []
        for video_id, context_vector in zip(video_ids, context_vectors):
            distances = []
            for train_video_id in env.video_ids:
                if train_scene_dict[train_video_id] != test_scene_dict[video_id]:
                    continue
                env_context = env.dataset.getitem_context(train_video_id)
                env_context = prepare_context_observation(env_context)
                distance = 0.0
                for co, ec in zip(context_vector, env_context):
                    distance += (co - ec) ** 2
                distances.append([distance, env_context, train_video_id])
            fitted_train_video_ids.append(min(distances, key=lambda x: x[0])[2])


    find_solution = False
    ClusterNum = np.zeros(cluster_nums[1])
    run_cmds, ingestion_result_paths = [], []
    optimal_config_indices = None
    for cluster_num in range(cluster_nums[1], cluster_nums[0], -1):
        if os.path.exists(f"./clusters/kmeans_{video_num}_{cluster_num}.pkl"):
            with open(f"./clusters/kmeans_{video_num}_{cluster_num}.pkl", "rb") as f:
                cluster_labels, cluster_centers = pkl.load(f)
        else:
            if os.path.exists("./clusters/kmeans_model.pkl"):
                mb_kmeans = pkl.load(open("./clusters/kmeans_model.pkl", "rb"))
                last_video_ids = pkl.load(open("./clusters/video_ids.pkl", "rb"))
                last_cluster_labels, last_cluster_centers = pkl.load(open("./clusters/cluster_info.pkl", "rb"))
                new_context_vectors = []
                for context_vector, video_id in zip(context_vectors, video_ids):
                    if video_id not in last_video_ids:
                        new_context_vectors.append(context_vector)
                new_context_vectors = np.array(new_context_vectors)
                mb_kmeans.partial_fit(new_context_vectors)
                new_labels = mb_kmeans.predict(new_context_vectors)
                new_labels = list(new_labels)
                cluster_labels = []
                for video_id in video_ids:
                    if video_id in last_video_ids:
                        cluster_labels.append(last_cluster_labels[last_video_ids.index(video_id)])
                    else:
                        cluster_labels.append(new_labels.pop(0))
                cluster_centers = mb_kmeans.cluster_centers_
                cluster_labels = np.array(cluster_labels)
                with open("./clusters/kmeans_model.pkl", "wb") as f:
                    pkl.dump(mb_kmeans, f)
                with open("./clusters/video_ids.pkl", "wb") as f:
                    pkl.dump(video_ids, f)
                with open("./clusters/cluster_info.pkl", "wb") as f:
                    pkl.dump([cluster_labels, cluster_centers], f)
            else:
                mb_kmeans = MiniBatchKMeans(n_clusters=cluster_num, random_state=seed)
                mb_kmeans.fit(context_vectors)
                cluster_centers, cluster_labels = mb_kmeans.cluster_centers_, mb_kmeans.labels_
                # save mb kmeans
                with open("./clusters/kmeans_model.pkl", "wb") as f:
                    pkl.dump(mb_kmeans, f)
                with open("./clusters/video_ids.pkl", "wb") as f:
                    pkl.dump(video_ids, f)
                with open("./clusters/cluster_info.pkl", "wb") as f:
                    pkl.dump([cluster_labels, cluster_centers], f)
            
            with open(f"./clusters/kmeans_{video_num}_{cluster_num}.pkl", "wb") as f:
                pkl.dump([cluster_labels, cluster_centers], f)

        AccMatrix, EffMatrix = np.zeros((cluster_num, max_pareto_set_size)), np.zeros(
            (cluster_num, max_pareto_set_size))
        ClusterNum = np.zeros(cluster_num)
        ClusterDict, ClusetrParetoSet = {}, {}
        for cluster_label in tqdm(range(cluster_num)):
            cluster_context = cluster_centers[cluster_label]
            same_cluster_ids = np.where(cluster_labels == cluster_label)[0]
            same_cluster_contexts = context_vectors[same_cluster_ids]

            same_cluster_video_ids = [video_ids[same_cluster_id]
                                      for same_cluster_id in same_cluster_ids]
            ClusterDict[cluster_label] = same_cluster_video_ids
            repre_incluster_id = np.argmin([float(np.linalg.norm(
                context_vector - cluster_context)) for context_vector in same_cluster_contexts])
            repre_video_id = same_cluster_video_ids[repre_incluster_id]
            repre_context_vector = same_cluster_contexts[repre_incluster_id]

            if use_experience:
                clusetr_pareto_set = env.collocate_no_pareto_set(
                    fitted_train_video_ids[video_ids.index(repre_video_id)])
                if len(clusetr_pareto_set) > max_pareto_set_size:
                    clusetr_pareto_set_idxs = env.evenly_sample_indices(
                        np.array([pareto[0] for pareto in clusetr_pareto_set]), max_pareto_set_size)
                    clusetr_pareto_set = [clusetr_pareto_set[idx]
                                          for idx in clusetr_pareto_set_idxs]
            else:
                with torch.no_grad():
                    observation_all, _ = test_env.reset(
                        video_id=repre_video_id)
                    done, truncated = False, False
                    while not done and not truncated:
                        action, _ = policy.predict(observation_all[0],
                                                   repre_context_vector,
                                                   deterministic=True)
                        observation_all, _, done, truncated, _ = test_env.step(
                            action)
                    clusetr_pareto_set = test_env.pareto_set
            clusetr_pareto_set = sorted(clusetr_pareto_set, key=lambda x: x[0])

            utils.plot_pareto_set([point[0] for point in clusetr_pareto_set],
                                  [point[1]
                                   for point in clusetr_pareto_set],
                                  f"./paretos/figures/hippo_wo_prl_cluster_{video_num}_{cluster_label}_{len(same_cluster_ids)}.png")

            with open(f"./paretos/values/hippo_wo_prl_cluster_{video_num}_{cluster_label}_{len(same_cluster_ids)}.pkl", "wb") as f:
                pkl.dump(clusetr_pareto_set, f)

            for pi, res in enumerate(clusetr_pareto_set):
                AccMatrix[cluster_label, pi] = res[0] * len(same_cluster_ids)
                EffMatrix[cluster_label, pi] = res[1]
            ClusterNum[cluster_label] = len(same_cluster_ids)
            ClusetrParetoSet[cluster_label] = clusetr_pareto_set

        LatMatrix = - np.log(EffMatrix + 1e-8) * temp
        RealLatMatrix = []
        for i in range(cluster_num):
            RealLatMatrix.append([LatMatrix[i, j] +
                                  utils.get_latency_increase(LatMatrix[i, j]) * (ClusterNum[i] - 1) for j in range(max_pareto_set_size)])
        RealLatMatrix = np.array(RealLatMatrix)

        try:
            optimal_config_indices = solve_traverse(AccMatrix,
                                                    RealLatMatrix,
                                                    latency_bound,
                                                    max_pool_lenth,
                                                    gpu_constrain=[ClusetrParetoSet, indexs, gpu_resource_info, memory_bound])
            find_solution = True
        except:
            break
        optimal_reallatency = [RealLatMatrix[cluster_label, optimal_config_indice]
                            for cluster_label, optimal_config_indice in enumerate(optimal_config_indices)]
        optimal_accuracy = [AccMatrix[cluster_label, optimal_config_indice]
                            for cluster_label, optimal_config_indice in enumerate(optimal_config_indices)]
        print(optimal_accuracy)
        print(optimal_reallatency)

        run_cmds, gpu_memory, ingestion_result_paths = [], [], []
        for cluster_label, optimal_config_indice in enumerate(optimal_config_indices):
            config_action = ClusetrParetoSet[cluster_label][optimal_config_indice][2]
            cn = len(ClusterDict[cluster_label])
            run_config = Config(
                configpath, f'./cmds/run_hippo_wo_prl_cmd_{video_num}_{cluster_label}_{cn}.yaml',
                deviceid, data_name, data_type, objects,
                configpath, scene_name)
            ingestion_result_path = f'./results/run_hippo_wo_prl_cluster_{video_num}_{cluster_label}_{cn}_result.json'
            ingestion_result_paths.append(ingestion_result_path)
            run_config.update_cache(utils.generate_config(
                config_action, run_config.load_cache()))
            run_videoids = ",".join(
                [str(videoid) for videoid in ClusterDict[cluster_label]])
            run_cmd = f"go run step.go {run_config.cache_path} {ingestion_result_path} {run_videoids}"
            run_cmds.append(run_cmd)

            config_action_str = "_".join(
                [str(action_param) for action_param in config_action])
            config_action_str = "_".join(
                [config_action_str.split("_")[index] for index in indexs])
            gpu_memory.append(gpu_resource_info[config_action_str])

        if optimal_accuracy != -object_nan and sum(gpu_memory) <= memory_bound:
            break

    return find_solution, run_cmds, \
        ingestion_result_paths, ClusterNum, \
        optimal_config_indices

def hippo_without_imitation_learning_func(seed=None,
               video_num=None, video_ids=None,
               policy=None,
               env=None, test_env=None,
               train_scene_dict=None, test_scene_dict=None,
               cluster_nums=None,
               gpu_resource_info=None, indexs=None,
               configpath=None,
               deviceid=None,
               data_name=None, data_type=None, scene_name=None,
               objects=["car", "bus", "truck"],
               temp=10.0,
               max_pool_lenth=6,
               latency_bound=60, memory_bound=24256,
               max_pareto_set_size=5, object_nan=100,
               train_fit=True,
               use_experience=True, **kwargs):
    print(f"Start Hippo Without Imitation Learning Function: {video_num}")
    train_fit=True #t
    if train_fit:
        fitted_train_video_ids, context_vectors = [], []
        for video_id in video_ids:
            distances = []
            context_observation = prepare_context_observation(
                test_env.dataset.getitem_context(video_id))
            for train_video_id in env.video_ids:
                if train_scene_dict[train_video_id] != test_scene_dict[video_id]:
                    continue
                env_context = env.dataset.getitem_context(train_video_id)
                env_context = prepare_context_observation(env_context)
                distance = 0.0
                for co, ec in zip(context_observation, env_context):
                    distance += (co - ec) ** 2
                distances.append([distance, env_context, train_video_id])
            context_observation = min(distances, key=lambda x: x[0])[1]
            context_vectors.append(context_observation)
            fitted_train_video_ids.append(
                min(distances, key=lambda x: x[0])[2])
        context_vectors = np.array(context_vectors)
    else:
        context_vectors = np.array(
            [prepare_context_observation(test_env.dataset.getitem_context(video_id))
                for video_id in video_ids])

    find_solution = False
    ClusterNum = np.zeros(cluster_nums[1])
    run_cmds, ingestion_result_paths = [], []
    optimal_config_indices = None
    for cluster_num in range(cluster_nums[1], cluster_nums[0], -1):
        if os.path.exists(f"./clusters/kmeans_{video_num}_{cluster_num}.pkl"):
            with open(f"./clusters/kmeans_{video_num}_{cluster_num}.pkl", "rb") as f:
                cluster_labels, cluster_centers = pkl.load(f)
        else:
            if os.path.exists("./clusters/kmeans_model.pkl"):
                mb_kmeans = pkl.load(open("./clusters/kmeans_model.pkl", "rb"))
                last_video_ids = pkl.load(open("./clusters/video_ids.pkl", "rb"))
                last_cluster_labels, last_cluster_centers = pkl.load(open("./clusters/cluster_info.pkl", "rb"))
                new_context_vectors = []
                for context_vector, video_id in zip(context_vectors, video_ids):
                    if video_id not in last_video_ids:
                        new_context_vectors.append(context_vector)
                new_context_vectors = np.array(new_context_vectors)
                mb_kmeans.partial_fit(new_context_vectors)
                new_labels = mb_kmeans.predict(new_context_vectors)
                new_labels = list(new_labels)
                cluster_labels = []
                for video_id in video_ids:
                    if video_id in last_video_ids:
                        cluster_labels.append(last_cluster_labels[last_video_ids.index(video_id)])
                    else:
                        cluster_labels.append(new_labels.pop(0))
                cluster_centers = mb_kmeans.cluster_centers_
                cluster_labels = np.array(cluster_labels)
                with open("./clusters/kmeans_model.pkl", "wb") as f:
                    pkl.dump(mb_kmeans, f)
                with open("./clusters/video_ids.pkl", "wb") as f:
                    pkl.dump(video_ids, f)
                with open("./clusters/cluster_info.pkl", "wb") as f:
                    pkl.dump([cluster_labels, cluster_centers], f)
            else:
                mb_kmeans = MiniBatchKMeans(n_clusters=cluster_num, random_state=seed)
                mb_kmeans.fit(context_vectors)
                cluster_centers, cluster_labels = mb_kmeans.cluster_centers_, mb_kmeans.labels_
                # save mb kmeans
                with open("./clusters/kmeans_model.pkl", "wb") as f:
                    pkl.dump(mb_kmeans, f)
                with open("./clusters/video_ids.pkl", "wb") as f:
                    pkl.dump(video_ids, f)
                with open("./clusters/cluster_info.pkl", "wb") as f:
                    pkl.dump([cluster_labels, cluster_centers], f)
            
            with open(f"./clusters/kmeans_{video_num}_{cluster_num}.pkl", "wb") as f:
                pkl.dump([cluster_labels, cluster_centers], f)

        AccMatrix, EffMatrix = np.zeros((cluster_num, max_pareto_set_size)), np.zeros(
            (cluster_num, max_pareto_set_size))
        ClusterNum = np.zeros(cluster_num)
        ClusterDict, ClusetrParetoSet = {}, {}
        for cluster_label in tqdm(range(cluster_num)):
            cluster_context = cluster_centers[cluster_label]
            same_cluster_ids = np.where(cluster_labels == cluster_label)[0]
            same_cluster_contexts = context_vectors[same_cluster_ids]

            same_cluster_video_ids = [video_ids[same_cluster_id]
                                      for same_cluster_id in same_cluster_ids]
            ClusterDict[cluster_label] = same_cluster_video_ids
            repre_incluster_id = np.argmin([float(np.linalg.norm(
                context_vector - cluster_context)) for context_vector in same_cluster_contexts])
            repre_video_id = same_cluster_video_ids[repre_incluster_id]
            repre_context_vector = same_cluster_contexts[repre_incluster_id]
            use_experience=False #t
            if use_experience:
                clusetr_pareto_set = env.collocate_pareto_set(
                    fitted_train_video_ids[video_ids.index(repre_video_id)])
                if len(clusetr_pareto_set) > max_pareto_set_size:
                    clusetr_pareto_set_idxs = env.evenly_sample_indices(
                        np.array([pareto[0] for pareto in clusetr_pareto_set]), max_pareto_set_size)
                    clusetr_pareto_set = [clusetr_pareto_set[idx]
                                          for idx in clusetr_pareto_set_idxs]
                random.shuffle(clusetr_pareto_set)
                # drop 0.2
                clusetr_pareto_set = clusetr_pareto_set[:int(len(clusetr_pareto_set) * 0.25)]
            else:
                with torch.no_grad():
                    observation_all, _ = test_env.reset(
                        video_id=repre_video_id)
                    done, truncated = False, False
                    while not done and not truncated:
                        action, _ = policy.predict(observation_all[0],
                                                   repre_context_vector,
                                                   deterministic=True)
                        observation_all, _, done, truncated, _ = test_env.step(
                            action)
                    clusetr_pareto_set = test_env.pareto_set
            clusetr_pareto_set = sorted(clusetr_pareto_set, key=lambda x: x[0])

            utils.plot_pareto_set([point[0] for point in clusetr_pareto_set],
                                  [point[1]
                                   for point in clusetr_pareto_set],
                                  f"./paretos/figures/hippo_wo_il_cluster_{video_num}_{cluster_label}_{len(same_cluster_ids)}.png")

            with open(f"./paretos/values/hippo_wo_il_cluster_{video_num}_{cluster_label}_{len(same_cluster_ids)}.pkl", "wb") as f:
                pkl.dump(clusetr_pareto_set, f)

            for pi, res in enumerate(clusetr_pareto_set):
                AccMatrix[cluster_label, pi] = res[0] * len(same_cluster_ids)
                EffMatrix[cluster_label, pi] = res[1]
            ClusterNum[cluster_label] = len(same_cluster_ids)
            ClusetrParetoSet[cluster_label] = clusetr_pareto_set

        LatMatrix = - np.log(EffMatrix + 1e-8) * temp
        RealLatMatrix = []
        for i in range(cluster_num):
            RealLatMatrix.append([LatMatrix[i, j] +
                                  utils.get_latency_increase(LatMatrix[i, j]) * (ClusterNum[i] - 1) for j in range(max_pareto_set_size)])
        RealLatMatrix = np.array(RealLatMatrix)

        try:
            optimal_config_indices = solve_traverse(AccMatrix,
                                                    RealLatMatrix,
                                                    latency_bound,
                                                    max_pool_lenth,
                                                    gpu_constrain=[ClusetrParetoSet, indexs, gpu_resource_info, memory_bound])
            find_solution = True
        except:
            break
        optimal_reallatency = [RealLatMatrix[cluster_label, optimal_config_indice]
                            for cluster_label, optimal_config_indice in enumerate(optimal_config_indices)]
        optimal_accuracy = [AccMatrix[cluster_label, optimal_config_indice]
                            for cluster_label, optimal_config_indice in enumerate(optimal_config_indices)]
        print(optimal_accuracy)
        print(optimal_reallatency)

        run_cmds, gpu_memory, ingestion_result_paths = [], [], []
        for cluster_label, optimal_config_indice in enumerate(optimal_config_indices):
            config_action = ClusetrParetoSet[cluster_label][optimal_config_indice][2]
            cn = len(ClusterDict[cluster_label])
            run_config = Config(
                configpath, f'./cmds/run_hippo_wo_il_cmd_{video_num}_{cluster_label}_{cn}.yaml',
                deviceid, data_name, data_type, objects,
                configpath, scene_name)
            ingestion_result_path = f'./results/run_hippo_wo_il_cluster_{video_num}_{cluster_label}_{cn}_result.json'
            ingestion_result_paths.append(ingestion_result_path)
            run_config.update_cache(utils.generate_config(
                config_action, run_config.load_cache()))
            run_videoids = ",".join(
                [str(videoid) for videoid in ClusterDict[cluster_label]])
            run_cmd = f"go run step.go {run_config.cache_path} {ingestion_result_path} {run_videoids}"
            run_cmds.append(run_cmd)

            config_action_str = "_".join(
                [str(action_param) for action_param in config_action])
            config_action_str = "_".join(
                [config_action_str.split("_")[index] for index in indexs])
            gpu_memory.append(gpu_resource_info[config_action_str])

        if optimal_accuracy != -object_nan and sum(gpu_memory) <= memory_bound:
            break

    return find_solution, run_cmds, \
        ingestion_result_paths, ClusterNum, \
        optimal_config_indices

def skyscraper_func(video_num=None, video_ids=None,
                    random_camera=None,
                    test_env=None,
                    cluster_nums=None,
                    gpu_resource_info=None, indexs=None,
                    configpath=None,
                    deviceid=None,
                    data_name=None, data_type=None, scene_name=None,
                    objects=["car", "bus", "truck"], effiency_targets=[0.5, 0.6, 0.7, 0.8, 0.9],
                    temp=10.0,
                    max_pool_lenth=6, max_exploration_step=2,
                    latency_bound=60, memory_bound=24256,
                    max_pareto_set_size=5, object_nan=100, **kwargs):
    print(f"Start Skyscraper Function: {video_num}")
    max_exploration_step = 1 #t 
    context_vectors = np.array([prepare_context_observation(test_env.dataset.getitem_context(video_id)) for video_id in video_ids])
    find_solution = False
    ClusterNum = np.zeros(cluster_nums[1])
    run_cmds, ingestion_result_paths = [], []
    optimal_config_indices = None
    for cluster_num in range(cluster_nums[1], cluster_nums[0], -1):
        if os.path.exists(f"./clusters/kmeans_{video_num}_{cluster_num}.pkl"):
            with open(f"./clusters/kmeans_{video_num}_{cluster_num}.pkl", "rb") as f:
                cluster_labels, cluster_centers = pkl.load(f)

        AccMatrix, EffMatrix = np.zeros((cluster_num, max_pareto_set_size)), np.zeros(
            (cluster_num, max_pareto_set_size))
        ClusterNum = np.zeros(cluster_num)
        ClusterDict, ClusetrParetoSet = {}, {}
        for cluster_label in tqdm(range(cluster_num)):
            cluster_context = cluster_centers[cluster_label]
            same_cluster_ids = np.where(cluster_labels == cluster_label)[0]
            same_cluster_contexts = context_vectors[same_cluster_ids]
            same_cluster_video_ids = [video_ids[same_cluster_id]
                                      for same_cluster_id in same_cluster_ids]
            ClusterDict[cluster_label] = same_cluster_video_ids
            repre_incluster_id = np.argmin([float(np.linalg.norm(
                context_vector - cluster_context)) for context_vector in same_cluster_contexts])
            repre_video_id = same_cluster_video_ids[repre_incluster_id]

            skyscraper_config_vector = utils.generate_action_from_config(
                random_camera.loadConfig())
            now_state_acc, now_state_lat = test_env.eval_config(
                skyscraper_config_vector, repre_video_id)
            clusetr_solution_set = []
            for effiency_target in effiency_targets:
                exploration_step = 0
                while exploration_step < max_exploration_step:  # or not find_effiency_target:
                    next_skyscraper_config_vector = utils.generate_random_config_vector()
                    next_state_acc, next_state_lat = test_env.eval_config(
                        next_skyscraper_config_vector, repre_video_id)

                    if next_state_lat >= effiency_target and now_state_lat < effiency_target:
                        skyscraper_config_vector = next_skyscraper_config_vector
                        now_state_acc, now_state_lat = next_state_acc, next_state_lat
                    elif next_state_lat >= effiency_target and now_state_lat >= effiency_target:
                        if next_state_acc > now_state_acc:
                            skyscraper_config_vector = next_skyscraper_config_vector
                            now_state_acc, now_state_lat = next_state_acc, next_state_lat
                    elif next_state_lat < effiency_target:
                        if next_state_lat > now_state_lat:
                            skyscraper_config_vector = next_skyscraper_config_vector
                            now_state_acc, now_state_lat = next_state_acc, next_state_lat
                    exploration_step += 1
                clusetr_solution_set.append(
                    [now_state_acc, now_state_lat, skyscraper_config_vector])
            clusetr_solution_set = test_env.filter_duplicate(
                clusetr_solution_set)
            clusetr_solution_set = sorted(
                clusetr_solution_set, key=lambda x: x[0])
            utils.plot_pareto_set([point[0] for point in clusetr_solution_set],
                                  [point[1] for point in clusetr_solution_set],
                                  f"./paretos/figures/skyscraper_cluster_{video_num}_{cluster_label}_{len(same_cluster_ids)}.png")

            with open(f"./paretos/values/skyscraper_cluster_{video_num}_{cluster_label}_{len(same_cluster_ids)}.pkl", "wb") as f:
                pkl.dump(clusetr_solution_set, f)

            for pi, res in enumerate(clusetr_solution_set):
                AccMatrix[cluster_label, pi] = res[0] * len(same_cluster_ids)
                EffMatrix[cluster_label, pi] = res[1]
            ClusterNum[cluster_label] = len(same_cluster_ids)
            ClusetrParetoSet[cluster_label] = clusetr_solution_set

        LatMatrix = - np.log(EffMatrix + 1e-8) * temp
        RealLatMatrix = []
        for i in range(cluster_num):
            RealLatMatrix.append([LatMatrix[i, j] +
                                  utils.get_latency_increase(LatMatrix[i, j]) * (ClusterNum[i] - 1) for j in range(max_pareto_set_size)])
        RealLatMatrix = np.array(RealLatMatrix)

        try:
            optimal_config_indices = solve_traverse(AccMatrix,
                                                    RealLatMatrix,
                                                    latency_bound,
                                                    max_pool_lenth,
                                                    gpu_constrain=[ClusetrParetoSet, indexs, gpu_resource_info, memory_bound])
            find_solution = True
        except:
            break
        optimal_reallatency = [RealLatMatrix[cluster_label, optimal_config_indice]
                            for cluster_label, optimal_config_indice in enumerate(optimal_config_indices)]
        optimal_accuracy = [AccMatrix[cluster_label, optimal_config_indice]
                            for cluster_label, optimal_config_indice in enumerate(optimal_config_indices)]
        print(optimal_accuracy)
        print(optimal_reallatency)

        run_cmds, gpu_memory, ingestion_result_paths = [], [], []
        for cluster_label, optimal_config_indice in enumerate(optimal_config_indices):
            config_action = ClusetrParetoSet[cluster_label][optimal_config_indice][2]
            cn = len(ClusterDict[cluster_label])
            run_config = Config(
                configpath, f'./cmds/run_skyscraper_cmd_{video_num}_{cluster_label}_{cn}.yaml',
                deviceid, data_name, data_type, objects,
                configpath, scene_name)
            ingestion_result_path = f'./results/run_skyscraper_cluster_{video_num}_{cluster_label}_{cn}_result.json'
            ingestion_result_paths.append(ingestion_result_path)
            run_config.update_cache(utils.generate_config(
                config_action, run_config.load_cache()))
            run_videoids = ",".join(
                [str(videoid) for videoid in ClusterDict[cluster_label]])
            run_cmd = f"go run step.go {run_config.cache_path} {ingestion_result_path} {run_videoids}"
            run_cmds.append(run_cmd)

            config_action_str = "_".join(
                [str(action_param) for action_param in config_action])
            config_action_str = "_".join(
                [config_action_str.split("_")[index] for index in indexs])
            gpu_memory.append(gpu_resource_info[config_action_str])

        if optimal_accuracy != -object_nan and sum(gpu_memory) <= memory_bound:
            break
    return find_solution, run_cmds, \
        ingestion_result_paths, ClusterNum, \
        optimal_config_indices


def otif_func(video_num=None, video_ids=None,
              random_camera=None,
              test_env=None,
              cluster_nums=None,
              gpu_resource_info=None, indexs=None,
              configpath=None,
              deviceid=None,
              data_name=None, data_type=None, scene_name=None,
              objects=["car", "bus", "truck"],
              temp=10.0, coarseness=0.5,
              max_pool_lenth=6,
              latency_bound=60, memory_bound=24256,
              max_pareto_set_size=5, object_nan=100, **kwargs):
    print(f"Start Otif Function: {video_num}")
    otif_golden_config = random_camera.loadConfig()
    context_vectors = np.array(
        [prepare_context_observation(test_env.dataset.getitem_context(video_id)) for video_id in video_ids])
    find_solution = False
    ClusterNum = np.zeros(cluster_nums[1])
    run_cmds, ingestion_result_paths = [], []
    optimal_config_indices = None
    for cluster_num in range(cluster_nums[1], cluster_nums[0], -1):
        if os.path.exists(f"./clusters/kmeans_{video_num}_{cluster_num}.pkl"):
            with open(f"./clusters/kmeans_{video_num}_{cluster_num}.pkl", "rb") as f:
                cluster_labels, cluster_centers = pkl.load(f)

        AccMatrix, EffMatrix = np.zeros((cluster_num, max_pareto_set_size)), np.zeros(
            (cluster_num, max_pareto_set_size))
        ClusterNum = np.zeros(cluster_num)
        ClusterDict, ClusetrParetoSet = {}, {}
        for cluster_label in tqdm(range(cluster_num)):
            cluster_context = cluster_centers[cluster_label]
            same_cluster_ids = np.where(cluster_labels == cluster_label)[0]
            same_cluster_contexts = context_vectors[same_cluster_ids]
            same_cluster_video_ids = [video_ids[same_cluster_id]
                                      for same_cluster_id in same_cluster_ids]
            ClusterDict[cluster_label] = same_cluster_video_ids

            repre_incluster_id = np.argmin([float(np.linalg.norm(
                context_vector - cluster_context)) for context_vector in same_cluster_contexts])
            repre_video_id = same_cluster_video_ids[repre_incluster_id]

            otif_gloden_config_vector = utils.generate_action_from_config(
                otif_golden_config)
            for space_name in utils.OTIF_TUNE_SPACE_NAMES:
                space_index = SEARCH_SPACE_NAMES.index(space_name)
                otif_gloden_config_vector[space_index] = utils.GOLDEN_CONFIG_VECTOR[space_index] - 1
            # if video_num == 50 or video_num == 100:
            #     otif_gloden_config_vector[0] = 3
            now_state_acc, now_state_lat = test_env.eval_config(
                otif_gloden_config_vector, repre_video_id)

            clusetr_pareto_set = []
            for search_step_i in range(max_pareto_set_size):
                next_state_otif_config_vectors = []
                for space_name in utils.OTIF_TUNE_SPACE_NAMES:
                    next_state_otif_config_vector = otif_gloden_config_vector.copy()
                    space_index = SEARCH_SPACE_NAMES.index(space_name)
                    space_values = utils.ACTION_INDEX_DICT[space_name].copy(
                    )
                    ori_space_value = space_values[next_state_otif_config_vector[space_index]]

                    if space_name == "skipnumber":
                        space_value = ori_space_value / (1.0 - coarseness)
                        space_values.remove(ori_space_value)
                        next_state_otif_config_vector[space_index] = int(np.argmin(
                            [abs(space_value - value) for value in space_values]))
                    elif space_name == "scaledownresolution":
                        space_value = ori_space_value[0] * \
                            ori_space_value[1] * (1.0 - coarseness)
                        space_values.remove(ori_space_value)
                        next_state_otif_config_vector[space_index] = int(np.argmin(
                            [abs(space_value - value[0] * value[1]) for value in space_values]))
                    else:
                        raise ValueError

                    next_state_acc, next_state_lat = test_env.eval_config(
                        next_state_otif_config_vector, repre_video_id)
                    next_state_otif_config_vectors.append([next_state_acc, next_state_lat,
                                                           next_state_otif_config_vector])
                next_state_id = np.argmax(
                    [next_state_otif_config_vector[0] for next_state_otif_config_vector in next_state_otif_config_vectors])
                otif_gloden_config_vector = next_state_otif_config_vectors[next_state_id][2]
                next_state_acc, next_state_lat = next_state_otif_config_vectors[
                    next_state_id][0], next_state_otif_config_vectors[next_state_id][1]
                clusetr_pareto_set.append(
                    [next_state_acc, next_state_lat, otif_gloden_config_vector])
            clusetr_pareto_set = test_env.filter_duplicate(
                clusetr_pareto_set)
            clusetr_pareto_set = sorted(clusetr_pareto_set, key=lambda x: x[0])

            utils.plot_pareto_set([point[0] for point in clusetr_pareto_set],
                                  [point[1]
                                   for point in clusetr_pareto_set],
                                  f"./paretos/figures/otif_cluster_{video_num}_{cluster_label}_{len(same_cluster_ids)}.png")

            with open(f"./paretos/values/otif_cluster_{video_num}_{cluster_label}_{len(same_cluster_ids)}.pkl", "wb") as f:
                pkl.dump(clusetr_pareto_set, f)

            for pi, res in enumerate(clusetr_pareto_set):
                AccMatrix[cluster_label, pi] = res[0] * len(same_cluster_ids)
                EffMatrix[cluster_label, pi] = res[1]
            ClusterNum[cluster_label] = len(same_cluster_ids)
            ClusetrParetoSet[cluster_label] = clusetr_pareto_set

        LatMatrix = - np.log(EffMatrix + 1e-8) * temp
        RealLatMatrix = []
        for i in range(cluster_num):
            RealLatMatrix.append([LatMatrix[i, j] +
                                  utils.get_latency_increase(LatMatrix[i, j]) * (ClusterNum[i] - 1) for j in range(max_pareto_set_size)])
        RealLatMatrix = np.array(RealLatMatrix)

        try:
            optimal_config_indices = solve_traverse(AccMatrix,
                                                    RealLatMatrix,
                                                    latency_bound,
                                                    max_pool_lenth,
                                                    gpu_constrain=[ClusetrParetoSet, indexs, gpu_resource_info, memory_bound])
            find_solution = True
        except:
            break
        optimal_reallatency = [RealLatMatrix[cluster_label, optimal_config_indice]
                            for cluster_label, optimal_config_indice in enumerate(optimal_config_indices)]
        optimal_accuracy = [AccMatrix[cluster_label, optimal_config_indice]
                            for cluster_label, optimal_config_indice in enumerate(optimal_config_indices)]
        print(optimal_accuracy)
        print(optimal_reallatency)

        run_cmds, gpu_memory, ingestion_result_paths = [], [], []
        for cluster_label, optimal_config_indice in enumerate(optimal_config_indices):
            config_action = ClusetrParetoSet[cluster_label][optimal_config_indice][2]
            cn = len(ClusterDict[cluster_label])
            run_config = Config(
                configpath, f'./cmds/run_otif_cmd_{video_num}_{cluster_label}_{cn}.yaml',
                deviceid, data_name, data_type, objects,
                configpath, scene_name)
            ingestion_result_path = f'./results/run_otif_cluster_{video_num}_{cluster_label}_{cn}_result.json'
            ingestion_result_paths.append(ingestion_result_path)
            run_config.update_cache(utils.generate_config(
                config_action, run_config.load_cache()))
            run_videoids = ",".join(
                [str(videoid) for videoid in ClusterDict[cluster_label]])
            run_cmd = f"go run step.go ./cmds/train_cache_config_0.yaml {ingestion_result_path} {run_videoids}"
            run_cmds.append(run_cmd)

            config_action_str = "_".join(
                [str(action_param) for action_param in config_action])
            config_action_str = "_".join(
                [config_action_str.split("_")[index] for index in indexs])
            gpu_memory.append(gpu_resource_info[config_action_str])

        if optimal_accuracy != -object_nan and sum(gpu_memory) <= memory_bound:
            break
    return find_solution, run_cmds, \
        ingestion_result_paths, ClusterNum, \
        optimal_config_indices


def unitune_func(video_num=None, video_ids=None,
                 policy_unitune=None, bo_model_unitune=None,
                 unitune_test_env=None,
                 cluster_nums=None,
                 gpu_resource_info=None, indexs=None,
                 configpath=None,
                 deviceid=None,
                 data_name=None, data_type=None, scene_name=None,
                 objects=["car", "bus", "truck"],
                 temp=10.0,
                 max_pool_lenth=6,
                 latency_bound=60, memory_bound=24256,
                 max_pareto_set_size=5, object_nan=100, **kwargs):
    # max_pareto_set_size = int(max_pareto_set_size / 2)
    # print("max_pareto_set_size: ", max_pareto_set_size)
    print(f"Start Unitune Function: {video_num}")
    context_vectors = np.array(
        [prepare_context_observation(unitune_test_env.dataset.getitem_context(video_id)) for video_id in video_ids])
    find_solution = False
    ClusterNum = np.zeros(cluster_nums[1])
    run_cmds, ingestion_result_paths = [], []
    optimal_config_indices = None
    for cluster_num in range(cluster_nums[1], cluster_nums[0], -1):
        if os.path.exists(f"./clusters/kmeans_{video_num}_{cluster_num}.pkl"):
            with open(f"./clusters/kmeans_{video_num}_{cluster_num}.pkl", "rb") as f:
                cluster_labels, cluster_centers = pkl.load(f)

        AccMatrix, EffMatrix = np.zeros((cluster_num, max_pareto_set_size)), np.zeros(
            (cluster_num, max_pareto_set_size))
        ClusterNum = np.zeros(cluster_num)
        ClusterDict, ClusetrParetoSet = {}, {}
        for cluster_label in tqdm(range(cluster_num)):
            cluster_context = cluster_centers[cluster_label]
            same_cluster_ids = np.where(cluster_labels == cluster_label)[0]
            same_cluster_contexts = context_vectors[same_cluster_ids]
            same_cluster_video_ids = [video_ids[same_cluster_id]
                                      for same_cluster_id in same_cluster_ids]
            ClusterDict[cluster_label] = same_cluster_video_ids
            repre_incluster_id = np.argmin([float(np.linalg.norm(
                context_vector - cluster_context)) for context_vector in same_cluster_contexts])
            repre_video_id = same_cluster_video_ids[repre_incluster_id]

            with torch.no_grad():
                pareto_set = []
                for effiency_target in unitune_test_env.effiency_targets: 
                    unitune_test_env.effiency_target = effiency_target
                    obs, _ = unitune_test_env.reset_with_videoid(
                        repre_video_id)
                    done = False
                    truncated = False
                    total_reward = 0
                    action = np.array([GOLDEN_CONFIG_VECTOR])
                    while not done and not truncated:
                        if random.random() < 0.4: #t
                            action, _states = policy_unitune.predict(
                                obs, deterministic=True)
                        else:
                            action = np.array([bo_model_unitune.get_config_vector(
                                obs[0], action[0], max_pareto_set_size)[0]])
                        obs, rewards, done, truncated, _info = unitune_test_env.step(
                            action)
                        total_reward += rewards
                    pareto_set.append([unitune_test_env.solution[0],
                                       unitune_test_env.solution[1],
                                       unitune_test_env.solution[2]])
                clusetr_pareto_set = pareto_set
                clusetr_pareto_set = unitune_test_env.filter_duplicate(
                    clusetr_pareto_set)
                clusetr_pareto_set = sorted(
                    clusetr_pareto_set, key=lambda x: x[0])

            utils.plot_pareto_set([point[0] for point in clusetr_pareto_set],
                                  [point[1]
                                   for point in clusetr_pareto_set],
                                  f"./paretos/figures/unitune_cluster_{video_num}_{cluster_label}_{len(same_cluster_ids)}.png")

            with open(f"./paretos/values/unitune_cluster_{video_num}_{cluster_label}_{len(same_cluster_ids)}.pkl", "wb") as f:
                pkl.dump(clusetr_pareto_set, f)

            for pi, res in enumerate(clusetr_pareto_set):
                AccMatrix[cluster_label, pi] = res[0] * len(same_cluster_ids)
                EffMatrix[cluster_label, pi] = res[1]
            ClusterNum[cluster_label] = len(same_cluster_ids)
            ClusetrParetoSet[cluster_label] = clusetr_pareto_set

        LatMatrix = - np.log(EffMatrix + 1e-8) * temp
        RealLatMatrix = []
        for i in range(cluster_num):
            RealLatMatrix.append([LatMatrix[i, j] +
                                  utils.get_latency_increase(LatMatrix[i, j]) * (ClusterNum[i] - 1) for j in range(max_pareto_set_size)])
        RealLatMatrix = np.array(RealLatMatrix)

        try:
            optimal_config_indices = solve_traverse(AccMatrix,
                                                    RealLatMatrix,
                                                    latency_bound,
                                                    max_pool_lenth,
                                                    gpu_constrain=[ClusetrParetoSet, indexs, gpu_resource_info, memory_bound])
            find_solution = True
        except:
            break

        optimal_reallatency = [RealLatMatrix[cluster_label, optimal_config_indice]
                            for cluster_label, optimal_config_indice in enumerate(optimal_config_indices)]
        optimal_accuracy = [AccMatrix[cluster_label, optimal_config_indice]
                            for cluster_label, optimal_config_indice in enumerate(optimal_config_indices)]
        print(optimal_accuracy)
        print(optimal_reallatency)
        
        run_cmds, gpu_memory, ingestion_result_paths = [], [], []
        for cluster_label, optimal_config_indice in enumerate(optimal_config_indices):
            config_action = ClusetrParetoSet[cluster_label][optimal_config_indice][2]
            cn = len(ClusterDict[cluster_label])
            run_config = Config(
                configpath, f'./cmds/run_unitune_cmd_{video_num}_{cluster_label}_{cn}.yaml',
                deviceid, data_name, data_type, objects,
                configpath, scene_name)
            ingestion_result_path = f'./results/run_hippo_cluster_{video_num}_{cluster_label}_{cn}_result.json'
            ingestion_result_paths.append(ingestion_result_path)
            run_config.update_cache(utils.generate_config(
                config_action, run_config.load_cache()))
            run_videoids = ",".join(
                [str(videoid) for videoid in ClusterDict[cluster_label]])
            run_cmd = f"go run step.go {run_config.cache_path} {ingestion_result_path} {run_videoids}"
            run_cmds.append(run_cmd)

            config_action_str = "_".join(
                [str(action_param) for action_param in config_action])
            config_action_str = "_".join(
                [config_action_str.split("_")[index] for index in indexs])
            gpu_memory.append(gpu_resource_info[config_action_str])

        if optimal_accuracy != -object_nan and sum(gpu_memory) <= memory_bound:
            break
    return find_solution, run_cmds, \
        ingestion_result_paths, ClusterNum, \
        optimal_config_indices
