import torch
import random
import numpy as np


def build_pareto_set(env, policy, vid):
    # with torch.no_grad():
    #     observation_all, _ = env.reset_with_videoid(vid)
    #     done, truncated = False, False
    #     while not done and not truncated:
    #         action, _ = policy.predict(observation_all[0],
    #                                     observation_all[1],
    #                                     deterministic=True)
    #         observation_all, _, done, truncated, _ = env.step(
    #             action)
    #     clusetr_pareto_set = [point_data[-1] for point_data in env.pareto_set]
    clusetr_pareto_set = [[random.random(),
                           random.random(),
                           [random.randint(0, 4) for _ in range(7)]] for _ in range(10)]
    return clusetr_pareto_set


def eval_if_change_pareto_set(pareto_set, config,
                              env, video_id, use_cache=True):
    # pareto_set_str = ["_".join(list(map(str, point_data[2]))) for point_data in pareto_set]
    # accuracy, latency = env.eval_config(config, video_id, use_cache=use_cache)
    # new_point = [accuracy, latency, config]
    # new_pareto_set = env.identify_pareto(pareto_set + [new_point])
    # new_pareto_set_str = ["_".join(list(map(str, point_data[2]))) for point_data in new_pareto_set]
    # if len(set(pareto_set_str) - set(new_pareto_set_str)) > 0:
    #     return True
    # else:
    #     return False
    if random.random() > 0.5:
        return True
    else:
        return False


def pareto_cluster(vids, feats,
                   env, policy,
                   max_cluster_num=6):
    repre_vids = []
    vid_paretos = {}
    dist_matrix = np.zeros((len(vids), len(vids)))
    for i in range(len(vids)):
        for j in range(len(vids)):
            dist_matrix[i, j] = np.linalg.norm(feats[i] - feats[j])

    # 1. find the centroid of the vids.
    repre_vids.append(random.choice(vids))
    vid_paretos[repre_vids[-1]] = build_pareto_set(env, policy, repre_vids[-1])
    sample_vids = []
    next_vid = repre_vids[-1]
    while next_vid not in sample_vids:
        sample_vids.append(next_vid)
        # 2. Construct a sampling probability distribution based on distance and randomly sample a point.
        dists, near_repre_vids = [], {}
        for vid in vids:
            video_order_id = vids.index(vid)
            if vid not in repre_vids:
                min_dist = float("inf")
                for repre_vid in repre_vids:
                    repre_order_id = vids.index(repre_vid)
                    if dist_matrix[video_order_id, repre_order_id] < min_dist:
                        min_dist = dist_matrix[video_order_id, repre_order_id]
                        near_repre_vids[vid] = repre_vid
                dists.append(min_dist)
            else:
                dists.append(0)
                near_repre_vids[vid] = vid
        dists = np.array(dists)
        dists = dists / np.sum(dists)
        next_vid = np.random.choice(vids, p=dists)
        next_vid_pareto = build_pareto_set(env, policy, next_vid)
        vid_paretos[next_vid] = next_vid_pareto

        # 3. If the Pareto sets coincide or a different configuration does not affect another video, discard the point.
        near_repre_vid = near_repre_vids[next_vid]

        next_vid_pareto = vid_paretos[next_vid]
        near_repre_vid_pareto = vid_paretos[near_repre_vid]

        next_vid_pareto_str = [
            "_".join(list(map(str, point_data[2]))) for point_data in next_vid_pareto]
        near_repre_vid_pareto_str = ["_".join(
            list(map(str, point_data[2]))) for point_data in near_repre_vid_pareto]

        minus_configs = list(set(next_vid_pareto_str) -
                             set(near_repre_vid_pareto_str))
        minus_configs = [list(map(int, minus_config.split('_')))
                         for minus_config in minus_configs][:1]

        use_this_config = True
        for minus_config in minus_configs:
            if eval_if_change_pareto_set(near_repre_vid_pareto, minus_config, env, near_repre_vid):
                use_this_config = False
        if not use_this_config:
            continue

        repre_vids.append(next_vid)

        if len(repre_vids) >= max_cluster_num:
            break

    large_weight = 1.01
    pcluster = {i: [repre_vids[i]] for i in range(len(repre_vids))}
    for vid in vids:
        if vid not in repre_vids:
            min_dist = float("inf")
            min_dist_id = -1
            for i in range(len(repre_vids)):
                if dist_matrix[vids.index(vid), vids.index(repre_vids[i])] < min_dist:
                    min_dist = dist_matrix[vids.index(
                        vid), vids.index(repre_vids[i])]
                    min_dist_id = i
            pcluster[min_dist_id].append(vid)
            # 4. Modify the distance of some categories. Since the number has increased, the distance from all points to them needs to be increased.
            for i in range(len(vids)):
                dist_matrix[i, vids.index(repre_vids[min_dist_id])] = dist_matrix[i, vids.index(
                    repre_vids[min_dist_id])] * large_weight
    return pcluster

# if __name__ == "__main__":
#     total_video_num = 200
#     video_num, feat_num = 50, 16
#     vids = random.sample(list(range(total_video_num)), video_num) # [random.randint(0, total_video_num-1) for _ in range(video_num)]
#     config_vectors = np.random.rand(video_num, feat_num)

#     print("vids: ", vids)

#     cluster_set = pareto_cluster(vids, config_vectors, None, None)

#     print("cluster_set: ", cluster_set)
#     for cluster_id in cluster_set:
#         print("cluster_id: ", cluster_id, "cluster_vids: ", cluster_set[cluster_id], "length: ", len(cluster_set[cluster_id]))


def initialize_centroids(X, k):

    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]


def compute_distances(X, centroids):
    return np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))


def initial_labels(X, centroids):
    distances = compute_distances(X, centroids)
    return np.argmin(distances, axis=0)


def balanced_cluster_assignment(distances, cluster_sizes, alpha):

    adjusted_distances = distances * (cluster_sizes[:, np.newaxis]**alpha)
    return np.argmin(adjusted_distances, axis=0)


def update_centroids(X, labels, k):
    new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    return new_centroids


def k_means_balanced(X, k, alpha=0.1, max_iter=100):
    centroids = initialize_centroids(X, k)
    labels = initial_labels(X, centroids)
    for _ in range(max_iter):
        distances = compute_distances(X, centroids)
        cluster_sizes = np.array([np.sum(labels == i) for i in range(k)])
        labels = balanced_cluster_assignment(distances, cluster_sizes, alpha)
        new_centroids = update_centroids(X, labels, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return labels, centroids

def update_cluster_by_new_context_vectors(last_cluster_labels, last_cluster_centers,
                                          new_context_vectors, alpha=0.1, max_iter=100):
    new_cluster_labels = last_cluster_labels
    new_cluster_centers = last_cluster_centers
    for _ in range(max_iter):
        distances = compute_distances(new_context_vectors, new_cluster_centers)
        cluster_sizes = np.array([np.sum(new_cluster_labels == i) for i in range(len(new_cluster_centers))])
        new_cluster_labels = balanced_cluster_assignment(distances, cluster_sizes, alpha)
        new_cluster_centers = update_centroids(new_context_vectors, new_cluster_labels, len(new_cluster_centers))
    return new_cluster_labels, new_cluster_centers