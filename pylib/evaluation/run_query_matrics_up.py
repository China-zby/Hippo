import os
import cv2
import sys
import json
import struct
import pickle
import argparse
import cluster_utils
from query_matrics import QueryMatrics
from alive_progress import alive_bar


def read_json(stdin):
    buf = stdin.read(4)
    if not buf:
        return None
    (l,) = struct.unpack('>I', buf)
    buf = stdin.read(l)
    return json.loads(buf.decode('utf-8'))


parse = argparse.ArgumentParser("Run query metrics")
parse.add_argument("--data_root", default="./", help="data root")
parse.add_argument("--testmode", default="./", help="test mode")
parse.add_argument("--gtdir", default="./", help="ground truth data root")
parse.add_argument("--videoidlist", type=str, default="0-1-2-3-4")
parse.add_argument("--skipframelist", type=str, default="8-8-16-8-32")
parse.add_argument("--method_name", type=str, default="otif")
parse.add_argument("--filter_gt", action="store_true",
                   help="Filter ground truth")
parse.add_argument("--dataset_name", default="amsterdam",
                   help="Input dataset name")
parse.add_argument("--classes", type=str, default="car,bus,truck")
args = parse.parse_args()

classes = args.classes.split(',')
classesMap = {"car": 2, "bus": 5, "truck": 7}
classesMapReverse = {2: "car", 5: "bus", 7: "truck"}

Results = {}
for className in classes:
    Results[className] = {"recall": [], "precision": [], "accuracy": [],
                          "f1": [], "mae": [], "acc": [], "gt_count": [],
                          "pred_count": [], "acc_topk": []}

dataname = args.dataset_name
datatype = args.testmode
method_name = args.method_name
videoids = list(map(int, args.videoidlist.split("-")))
skipframes = list(map(int, args.skipframelist.split("-")))

if not os.path.exists(f"./result/{method_name}/{dataname}"):
    os.makedirs(f"./result/{method_name}/{dataname}")

post_process_flag = False

if post_process_flag:
    train_gap, test_gap = 100, 25
    use_scene_ids = [
        videoid // (train_gap if datatype == "train" else test_gap) for videoid in videoids]
    use_scene_ids = list(set(use_scene_ids))

    cluster_track_path = f"{args.data_root}/train/"
    cluster_tracks = {}
    un_use_scene_ids = []
    for scene_id in use_scene_ids:
        cluster_track_path = f"{args.data_root}/train/cluster_{scene_id}.pkl"
        if os.path.exists(cluster_track_path):
            cluster_tracks[scene_id] = pickle.load(
                open(cluster_track_path, "rb"))
        else:
            un_use_scene_ids.append(scene_id)

    if len(un_use_scene_ids) != 0:
        tracks_dir = f"{args.data_root}/train/tracks"
        track_path_names = os.listdir(tracks_dir)
        clusters = {use_scene_id: [] for use_scene_id in un_use_scene_ids}
        for track_path_name in track_path_names:
            if "json" in track_path_name or "txt" in track_path_name:
                continue
            video_id = int(track_path_name.split(".")[0])
            scene_id = video_id // train_gap
            if scene_id not in un_use_scene_ids:
                continue
            track_path = os.path.join(tracks_dir, track_path_name)
            track_datas = pickle.load(open(track_path, "rb"))
            sub_tracks = []
            for track_id in track_datas:
                track_data = track_datas[track_id]
                sub_tracks.append(track_data[1])
            clusters[scene_id].extend(sub_tracks)

        sub_cluster_tracks = cluster_utils.ClusterTracks(clusters)
        for scene_id in un_use_scene_ids:
            cluster_track_path = f"{args.data_root}/train/cluster_{scene_id}.pkl"
            pickle.dump(sub_cluster_tracks[scene_id], open(
                cluster_track_path, "wb"))
            cluster_tracks[scene_id] = sub_cluster_tracks[scene_id]

stdin = sys.stdin.detach()
while True:
    packet = read_json(stdin)
    if packet is None:
        break
    if packet['run_type'] != 'start':
        continue
    for videoid, skipframe in zip(videoids, skipframes):
        videoPath = os.path.join(
            args.data_root, f"{args.testmode}/video/{videoid}.mp4")
        framenumbers = int(cv2.VideoCapture(
            videoPath).get(cv2.CAP_PROP_FRAME_COUNT)) + 1
        framerate = int(cv2.VideoCapture(
            videoPath).get(cv2.CAP_PROP_FPS))
        videologo = f"{dataname}S{skipframe}-{datatype}"
        videopathname = f"{dataname}S{skipframe}-{videoid}.txt"
        trackdir = f"./TrackEval/data/trackers/videodb/{videologo}/{method_name}/data"
        trackpath = os.path.join(trackdir, videopathname)
        pred_tuple, gt_tuple = {}, {}
        with open(trackpath, "r") as reader:
            for dataline in reader:
                if len(dataline.split(",")) != 1:
                    frameid, trackid, x1, y1, width, height, score, classid, _, _, _ = dataline.split(
                        ",")
                    frameid, trackid = int(frameid), int(trackid)
                    x1, y1, width, height = float(x1), float(y1), float(width), float(height)
                    if trackid not in pred_tuple:
                        pred_tuple[trackid] = {"frame_bound": {"start_frame_id": frameid,
                                                               "end_frame_id": frameid},
                                               "class_id_list": [int(float(classid))],
                                               "position_list": [[x1 + width / 2, y1 + height / 2]]}
                    else:
                        pred_tuple[trackid]['frame_bound']["end_frame_id"] = frameid
                        pred_tuple[trackid]["class_id_list"].append(
                            int(float(classid)))
                        pred_tuple[trackid]["position_list"].append(
                            [x1 + width / 2, y1 + height / 2])

        new_pred_tuple = {}
        for trackid in pred_tuple.keys():
            pred_tuple[trackid]["class_id"] = max(set(pred_tuple[trackid]["class_id_list"]),
                                                  key=pred_tuple[trackid]["class_id_list"].count)
            new_pred_tuple[trackid] = pred_tuple[trackid]

        if post_process_flag:
            scene_id = videoid // (train_gap if datatype ==
                                   "train" else test_gap)
            new_pred_tuple = cluster_utils.Postprocess(new_pred_tuple, 
                                                       cluster_tracks[scene_id],
                                                       framenumbers) # skipframe,

        with open(os.path.join(args.gtdir, f"{videoid}.json"), "r") as reader:
            gt_data = json.load(reader)
            for frameid, trackdata in enumerate(gt_data):
                frameid += 1
                if trackdata == [] or trackdata is None:
                    continue
                for objectdata in trackdata:
                    x1, y1, x2, y2 = objectdata["left"], objectdata["top"], \
                        objectdata["right"], objectdata["bottom"]
                    trackid = int(objectdata["track_id"])
                    if trackid not in gt_tuple:
                        gt_tuple[trackid] = {"frame_bound": {"start_frame_id": frameid,
                                                             "end_frame_id": frameid},
                                             "class_id_list": [classesMap[objectdata["class"]]],
                                             "position_list": [[(x1 + x2) / 2, (y1 + y2) / 2]]}
                    else:
                        gt_tuple[trackid]["frame_bound"]["end_frame_id"] = frameid
                        gt_tuple[trackid]["class_id_list"].append(
                            classesMap[objectdata["class"]])
                        gt_tuple[trackid]["position_list"].append(
                            [(x1 + x2) / 2, (y1 + y2) / 2])

        new_gt_tuple = {}
        for trackid in gt_tuple.keys():
            gt_tuple[trackid]["class_id"] = max(set(gt_tuple[trackid]["class_id_list"]),
                                                key=gt_tuple[trackid]["class_id_list"].count)
            new_gt_tuple[trackid] = gt_tuple[trackid]

        for className in classes:
            new_gt_class_tuple, new_pred_class_tuple = {}, {}
            for trackid in new_gt_tuple.keys():
                if new_gt_tuple[trackid]["class_id"] == classesMap[className]:
                    new_gt_class_tuple[trackid] = new_gt_tuple[trackid]
            if len(new_gt_class_tuple) == 0:
                continue  # * if no gt, skip
            QM = QueryMatrics(classesMap[className], framenumbers, framerate)
            QM.preprocess(new_gt_tuple, new_pred_tuple)
            recall, precision, accuracy, f1, mae, acc, gt_vehicle, pred_vehicle, acc_topk = QM.matrics
            Results[className]["recall"].append(recall)
            Results[className]["precision"].append(precision)
            Results[className]["accuracy"].append(accuracy)
            Results[className]["f1"].append(f1)
            Results[className]["mae"].append(mae)
            Results[className]["acc"].append(acc)
            Results[className]["gt_count"].append(gt_vehicle)
            Results[className]["pred_count"].append(pred_vehicle)
            Results[className]["acc_topk"].append(acc_topk)
            del QM

    for className in classes:
        if len(Results[className]["recall"]) == 0:
            Results[className]["recall"] = None
            Results[className]["precision"] = None
            Results[className]["accuracy"] = None
            Results[className]["f1"] = None
            Results[className]["mae"] = None
            Results[className]["acc"] = None
            Results[className]["gt_count"] = 0
            Results[className]["pred_count"] = 0
            Results[className]["acc_topk"] = None
            Results[className]["recall_mean"] = 0
            Results[className]["precision_mean"] = 0
            Results[className]["accuracy_mean"] = 0
            Results[className]["f1_mean"] = 0
            Results[className]["mae_mean"] = 0
            Results[className]["acc_mean"] = 0
            Results[className]["gt_count_mean"] = 0
            Results[className]["pred_count_mean"] = 0
            Results[className]["acc_topk_mean"] = 0
            continue
        Results[className]["recall_mean"] = float(
            sum(Results[className]["recall"]) / len(Results[className]["recall"]))
        Results[className]["precision_mean"] = float(
            sum(Results[className]["precision"]) / len(Results[className]["precision"]))
        Results[className]["accuracy_mean"] = float(
            sum(Results[className]["accuracy"]) / len(Results[className]["accuracy"]))
        Results[className]["f1_mean"] = float(
            sum(Results[className]["f1"]) / len(Results[className]["f1"]))
        Results[className]["mae_mean"] = float(
            sum(Results[className]["mae"]) / len(Results[className]["mae"]))
        Results[className]["acc_mean"] = float(
            sum(Results[className]["acc"]) / len(Results[className]["acc"]))
        Results[className]["gt_count_mean"] = int(
            sum(Results[className]["gt_count"]))
        Results[className]["pred_count_mean"] = int(
            sum(Results[className]["pred_count"]))
        Results[className]["acc_topk_mean"] = float(
            sum(Results[className]["acc_topk"]) / len(Results[className]["acc_topk"]))

    print(json.dumps(Results, indent=4, sort_keys=True))
    pickle.dump(Results, open(os.path.join(
        f"./result/{method_name}/{dataname}", f"query_matrics.pkl"), "wb"))

    sys.stdout.write('json'+json.dumps(Results)+'\n')
    sys.stdout.flush()
