import math
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms


def TrackDistance2(track1, track2):
    maxDistance = 0.0
    lastIdx = 0
    for point2 in track2:
        best_d = 9999
        for idx, point1 in enumerate(track1):
            if idx < lastIdx:
                continue
            d = PointDist(point1, point2)
            if d < best_d:
                best_d = d
                lastIdx = idx
        if best_d > maxDistance:
            maxDistance = best_d
    return maxDistance


def PointDist(point1, point2):
    """
    Calculate the Euclidean distance between two points
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def SampleNormalizedPoints(track, sample_point=20):
    """
    Sample the points from the track and normalize the points to the range of [0, 1]
    """
    trackLength = 0
    for i in range(len(track) - 1):
        start_point, end_point = track[i], track[i + 1]
        start_center_point, \
            end_center_point = [(start_point[0] + start_point[2]) / 2, (start_point[1] + start_point[3]) / 2], \
            [(end_point[0] + end_point[2]) / 2,
             (end_point[1] + end_point[3]) / 2]
        trackLength += math.sqrt((end_center_point[0] - start_center_point[0]) ** 2 +
                                 (end_center_point[1] - start_center_point[1]) ** 2)
    pointFreq = trackLength / sample_point

    points = [[(track[0][0] + track[0][2]) / 2,
               (track[0][1] + track[0][3]) / 2]]
    remaining = pointFreq
    for i in range(len(track) - 1):
        segment = [[(track[i][0] + track[i][2])/2, (track[i][1] + track[i][3])/2],
                   [(track[i+1][0] + track[i+1][2])/2, (track[i+1][1] + track[i+1][3])/2]]
        while PointDist(segment[0], segment[1]) > remaining:
            vector = [segment[1][0] - segment[0]
                      [0], segment[1][1] - segment[0][1]]
            p = [vector[0] * remaining / PointDist(segment[0], segment[1]) + segment[0][0],
                 vector[1] * remaining / PointDist(segment[0], segment[1]) + segment[0][1]]
            points.append(p)
            segment[0] = p
            remaining = pointFreq
        remaining -= PointDist(segment[0], segment[1])
    while len(points) < sample_point:
        points.append([(track[-1][0] + track[-1][2]) / 2,
                      (track[-1][1] + track[-1][3]) / 2])

    return points[:sample_point]


def SampledPointDist(sampled_points_a, sampled_points_b):
    """
    Calculate the Euclidean distance between two sets of sampled points
    """
    max_dist = 0
    for i in range(len(sampled_points_a)):
        max_dist = max(max_dist, PointDist(
            sampled_points_a[i], sampled_points_b[i]))
    return max_dist


def ClusterTracks(scene_tracks, threshold=50):
    scene_clusters = {}
    for scene_id in scene_tracks:
        sceneid_tracks = scene_tracks[scene_id]
        sampled_tracks = [SampleNormalizedPoints(
            track) for track in sceneid_tracks]
        distances = []
        for i in range(len(sampled_tracks)):
            distance_vector = []
            for j in range(len(sampled_tracks)):
                distance_vector.append(SampledPointDist(
                    sampled_tracks[i], sampled_tracks[j]))
            distances.append(distance_vector)
        distances = np.array(distances)
        clusters = []
        for trackIdx, track in enumerate(sceneid_tracks):
            bestCluster, bestDistance = -1, 9999
            for clusterIdx, cluster_center_track in enumerate(clusters):
                d = distances[trackIdx][cluster_center_track[0]]
                if d >= threshold:
                    continue
                if bestCluster == -1 or d < bestDistance:
                    bestCluster, bestDistance = clusterIdx, d
            if bestCluster != -1:
                clusters[bestCluster][-1] += 1
                continue

            clusters.append([trackIdx, track, 1])

        scene_clusters[scene_id] = clusters
    return scene_clusters


def NextDetection(track, idx, direction, threshold=100):
    point = track[idx]
    if direction > 0:
        for i in range(idx + direction, len(track), direction):
            if PointDist(track[i], point) > threshold:
                return i
    else:
        for i in range(idx + direction, -1, direction):
            if PointDist(track[i], point) > threshold:
                return i
    return -1


def plot_track(track, name="", color='r'):
    plt.figure(figsize=(10, 6))
    plt.gca().set_facecolor('black')
    before_x, before_y = zip(*track)
    plt.scatter(before_x, before_y, 20, color)
    plt.xlim(0, 720)
    plt.ylim(0, 480)
    plt.savefig("./demo_tracks/{}.png".format(name))
    plt.close()
    plt.clf()
    plt.cla()


def Postprocess(before_tracks,
                clusters,
                maxFrame,
                large_threshold=50,
                track_length_threshold=10):
    # maxFrame = 0
    # for trackid in before_tracks:
    #     maxFrame = max(
    #         maxFrame, before_tracks[trackid]["frame_bound"]["end_frame_id"])

    postprocess_tracks = {}
    for trackid in before_tracks:
        postprocess_tracks[trackid] = deepcopy(before_tracks[trackid])
        closestCluster = -1
        bestDistance = -1
        for ci, (_, cluster_track, cluster_num) in enumerate(clusters):
            track_d = TrackDistance2([[(point[0] + point[2])/2,
                                       (point[1] + point[3])/2] for point in cluster_track], before_tracks[trackid]["position_list"])
            if bestDistance == -1 or track_d < bestDistance:
                closestCluster = clusters[ci]
                bestDistance = track_d

        if bestDistance == -1 or bestDistance > large_threshold or len(closestCluster[1]) < track_length_threshold:
            continue

        prefix = closestCluster[1][0]
        suffix = closestCluster[1][len(closestCluster[1]) - 1]

        # use_prefix, use_suffix = False, False

        pnext = NextDetection(before_tracks[trackid]["position_list"], 0, 1)
        if pnext != -1 and before_tracks[trackid]["frame_bound"]["start_frame_id"] > 1:
            vector1 = [(prefix[0] + prefix[2])/2 - before_tracks[trackid]["position_list"][0][0],
                       (prefix[1] + prefix[3])/2 - before_tracks[trackid]["position_list"][0][1]]
            vector2 = [before_tracks[trackid]["position_list"][0][0] - before_tracks[trackid]["position_list"][pnext][0],
                       before_tracks[trackid]["position_list"][0][1] - before_tracks[trackid]["position_list"][pnext][1]]
            angle = SignedAngle(vector1, vector2)
            if abs(angle) < math.pi / 4:
                # Find the closest point to the first point of the "before_tracks[trackid]["position_list"]" in the cluster.
                _, p_n_p_i = find_nearest_point_of_track(before_tracks[trackid]["position_list"][0],
                                                         [[(point[0] + point[2])/2,
                                                           (point[1] + point[3])/2] for point in closestCluster[1]])
                postprocess_tracks[trackid]["frame_bound"]['start_frame_id'] = max(
                    postprocess_tracks[trackid]["frame_bound"]['start_frame_id'] - p_n_p_i, 1)
                postprocess_tracks[trackid]["position_list"].insert(
                    0, [(prefix[0] + prefix[2])/2,
                        (prefix[1] + prefix[3])/2])
                # use_prefix = True

        snext = NextDetection(before_tracks[trackid]["position_list"],
                              len(before_tracks[trackid]["position_list"]) - 1, -1)
        if snext != -1 and before_tracks[trackid]["frame_bound"]["end_frame_id"] < maxFrame:
            vector1 = [(suffix[0] + suffix[2])/2 - before_tracks[trackid]["position_list"][-1][0],
                       (suffix[1] + suffix[3])/2 - before_tracks[trackid]["position_list"][-1][1]]
            vector2 = [before_tracks[trackid]["position_list"][-1][0] - before_tracks[trackid]["position_list"][snext][0],
                       before_tracks[trackid]["position_list"][-1][1] - before_tracks[trackid]["position_list"][snext][1]]
            angle = SignedAngle(vector1, vector2)
            if abs(angle) < math.pi / 4:
                # Find the closest point to the first point of the "before_tracks[trackid]["position_list"]" in the cluster.
                _, s_n_p_i = find_nearest_point_of_track(before_tracks[trackid]["position_list"][-1],
                                                         [[(point[0] + point[2])/2,
                                                           (point[1] + point[3])/2] for point in closestCluster[1]])
                postprocess_tracks[trackid]['frame_bound']['end_frame_id'] = min(postprocess_tracks[trackid]["frame_bound"]['end_frame_id'] + abs(len(closestCluster[1]) - s_n_p_i),
                                                                                 maxFrame)
                postprocess_tracks[trackid]["position_list"].append([(suffix[0] + suffix[2])/2,
                                                                     (suffix[1] + suffix[3])/2])
                # use_suffix = True

        # if use_prefix or use_suffix:
        #     plot_track(before_tracks[trackid]["position_list"], name="before_{}".format(trackid), color='g')
        #     plot_track(postprocess_tracks[trackid]["position_list"], name="after_{}".format(trackid), color='r')
        #     print("use prefix: ", use_prefix, "use suffix: ", use_suffix, "before frame bound: ", before_tracks[trackid]["frame_bound"], "after frame bound: ", postprocess_tracks[trackid]["frame_bound"], file=open("frame_bound.txt", "a"))
    return postprocess_tracks


def SignedAngle(vector1, vector2):
    return math.atan2(vector2[1], vector2[0]) - math.atan2(vector1[1], vector1[0])


def find_nearest_point_of_track(refer_point, track):
    min_dist = 9999
    nearest_point, nearest_point_i = None, None
    for i, point in enumerate(track):
        dist = PointDist(refer_point, point)
        if dist < min_dist:
            min_dist = dist
            nearest_point = point
            nearest_point_i = i
    return nearest_point, nearest_point_i
