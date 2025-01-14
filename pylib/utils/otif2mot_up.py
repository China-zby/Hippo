import os
import cv2
import json
import pickle
import argparse
import configparser
import cluster_utils

def linear_interpolation_func(tracklet):
    frameids = list(tracklet.keys())
    frameids.sort()
    for i in range(len(frameids) - 1):
        start_frameid, end_frameid = frameids[i], frameids[i + 1]
        start_data, end_data = tracklet[start_frameid], tracklet[end_frameid]
        start_x1, start_y1, start_width, start_height, start_score = start_data
        start_xc, start_yc = start_x1 + start_width / 2, start_y1 + start_height / 2
        end_x1, end_y1, end_width, end_height, end_score = end_data
        end_xc, end_yc = end_x1 + end_width / 2, end_y1 + end_height / 2
        for interpolation_frameid in range(start_frameid + 1, end_frameid):
            ratio = (interpolation_frameid - start_frameid) / \
                (end_frameid - start_frameid)
            xc = int(start_xc + (end_xc - start_xc) * ratio)
            yc = int(start_yc + (end_yc - start_yc) * ratio)
            width = int(start_width + (end_width - start_width) * ratio)
            height = int(start_height + (end_height - start_height) * ratio)
            x1, y1 = xc - width / 2, yc - height / 2
            score = start_score + (end_score - start_score) * ratio
            tracklet[interpolation_frameid] = [x1, y1, width, height, score]
    return tracklet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./outputs/")
    parser.add_argument("--input_gt_dir", type=str, default="./")
    parser.add_argument("--data_root", type=str, default="./")
    parser.add_argument("--input_video_dir", type=str, default="./")
    parser.add_argument("--method_name", type=str, default="otif")
    parser.add_argument("--dataset_name", type=str, default="amsterdam")
    parser.add_argument("--videoidlist", type=str, default="0-1-2-3-4")
    parser.add_argument("--skipframelist", type=str, default="8-8-16-8-32")
    parser.add_argument("--framenumber", type=int, default=1800)
    parser.add_argument("--testmode", type=str, default="test_separate_video")
    parser.add_argument("--differentclass", action='store_true')
    parser.add_argument("--width", type=str, default='')
    parser.add_argument("--height", type=str, default='')
    parser.add_argument("--classes", type=str, default="car,bus,truck")
    parser.add_argument("--flag", type=str, default="xxx")
    args = parser.parse_args()

    if args.differentclass:
        class_map_dict = {"car": 2, "bus": 5, "truck": 7}
        class_list = args.classes.split(',')
        class_map_dict = {
            class_name: class_map_dict[class_name] for class_name in class_list}
    else:
        class_map_dict = {"car": 2, "bus": 2, "truck": 2}

    dataname = args.dataset_name
    datatype = args.testmode
    method_name = args.method_name
    videoids = list(map(int, args.videoidlist.split("-")))
    skipframes = list(map(int, args.skipframelist.split("-")))
    widths = list(map(int, args.width.split("-")))
    heights = list(map(int, args.height.split("-")))

    skipframesets = set(skipframes)
    seqmap_writer_dict = {}
    for skipframe in skipframesets:
        videologo = f"{dataname}S{skipframe}-{datatype}"
        save_gt_dir = f"./TrackEval/data/gt/videodb/{videologo}"

        seqmap_dir = os.path.join(os.path.dirname(save_gt_dir), "seqmaps")
        if not os.path.exists(seqmap_dir):
            os.makedirs(seqmap_dir)
        seqmap_path = os.path.join(os.path.dirname(
            save_gt_dir), f"seqmaps/{os.path.basename(save_gt_dir)}_{args.flag}.txt")

        seqmap_writer = open(seqmap_path, "w")
        seqmap_writer.write("name\n")
        if skipframe not in seqmap_writer_dict.keys():
            seqmap_writer_dict[skipframe] = [seqmap_writer, seqmap_path]
            

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

    for videoid, skipframe, resolutionwidth, resolutionheight in zip(videoids, skipframes, widths, heights):
        videologo = f"{dataname}S{skipframe}-{datatype}"
        save_dir = f"./TrackEval/data/trackers/videodb/{videologo}/{method_name}/data"
        save_gt_dir = f"./TrackEval/data/gt/videodb/{videologo}"

        trackid_dict, trackstartid = {}, 0

        # track txt
        output_txt_video = os.path.join(save_dir, f"{dataname}S{skipframe}-{videoid}.txt")
        if not os.path.exists(os.path.dirname(output_txt_video)):
            os.makedirs(os.path.dirname(output_txt_video))
        video_writer = open(output_txt_video, "w")

        input_json_path = os.path.join(args.input_dir, f"{videoid}.json")
        input_video_path = os.path.join(
            args.data_root, f"{args.testmode}/video/{videoid}.mp4")
        framenumbers = int(cv2.VideoCapture(
            input_video_path).get(cv2.CAP_PROP_FRAME_COUNT))
        frameid = 1
        with open(input_json_path, 'r') as freader:
            content = freader.read()
            raw_data = json.loads(content)
            TrackletData = {}
            if raw_data is not None:
                for data_line in raw_data:
                    if data_line is not None:
                        for detection in data_line:
                            x1, y1, x2, y2 = float(detection["left"]), float(detection["top"]), float(
                                detection["right"]), float(detection["bottom"])
                            width, height = x2 - x1, y2 - y1
                            if detection["class"] not in class_map_dict.keys():
                                continue
                            classid = int(class_map_dict[detection["class"]])
                            score = float(detection["score"])
                            videotrackid = int(
                                detection["track_id"]) + trackstartid
                            if videotrackid not in trackid_dict.keys():
                                trackid_dict[videotrackid] = len(
                                    trackid_dict.keys()) + 1
                            trackid = trackid_dict[videotrackid]

                            if trackid not in TrackletData.keys():
                                TrackletData[trackid] = [{}, []]
                            TrackletData[trackid][0][frameid] = [int(x1), int(y1), 
                                                                int(width), int(height), 
                                                                float(score)]
                            TrackletData[trackid][1].append(int(classid))
                    frameid += 1
                    
                if post_process_flag:
                    scene_id = videoid // (train_gap if datatype ==
                                   "train" else test_gap)
                    TrackletData = cluster_utils.Postprocess(
                        TrackletData, cluster_tracks[scene_id], framenumbers)

        if len(TrackletData.keys()) > 0:
            for trackid in TrackletData.keys():
                classid = max(TrackletData[trackid][1],
                            key=TrackletData[trackid][1].count)

                InterpolationTrackletData = linear_interpolation_func(
                    TrackletData[trackid][0])
                
                sorted_frameids = sorted(list(InterpolationTrackletData.keys())) 
                for frameid in sorted_frameids:
                    x1, y1, width, height, score = InterpolationTrackletData[frameid]
                    video_writer.write(
                        f"{frameid},{trackid},{int(x1)},{int(y1)},{int(width)},{int(height)},{score},{classid},-1,-1,-1\n")

        video_writer.close()

        seqmap_writer, seqmap_path = seqmap_writer_dict[skipframe]
        datalines = open(seqmap_path, "r").readlines()
        if f"{dataname}S{skipframe}-{videoid}\n" not in datalines:
            seqmap_writer.write(f"{dataname}S{skipframe}-{videoid}\n")

        # gt
        gt_trackid_dict = {}
        save_gt_path = os.path.join(
            save_gt_dir, f"{dataname}S{skipframe}-{videoid}/gt/gt.txt")
        if not os.path.exists(os.path.dirname(save_gt_path)):
            os.makedirs(os.path.dirname(save_gt_path))

        video_writer = open(save_gt_path, "w")

        input_gt_path = os.path.join(args.input_gt_dir, f"{videoid}.txt")
        with open(input_gt_path, "r") as reader:
            GTTrackletData = {}
            for dataline in reader.readlines():
                frameid, videotrackid, x1, y1, width, height, score, classid, _, _, _ = dataline.split(
                    ",")
                if videotrackid not in gt_trackid_dict.keys():
                    gt_trackid_dict[videotrackid] = len(gt_trackid_dict.keys()) + 1
                trackid = gt_trackid_dict[videotrackid]
                frameid = int(frameid) + 1
                x1, y1, width, height = float(x1), float(
                    y1), float(width), float(height)
                if int(classid) not in list(class_map_dict.values()):
                    continue
                classid = float(classid) if args.differentclass else 2
                if trackid not in GTTrackletData.keys():
                    GTTrackletData[trackid] = [[], []]
                GTTrackletData[trackid][0].append(
                    [frameid, int(x1), int(y1), int(width), int(height), float(score)])
                GTTrackletData[trackid][1].append(int(classid))

        for trackid in GTTrackletData.keys():
            classid = max(GTTrackletData[trackid][1],
                        key=GTTrackletData[trackid][1].count)
            for dataline in GTTrackletData[trackid][0]:
                frameid, x1, y1, width, height, score = dataline
                video_writer.write(
                    f"{frameid},{trackid},{int(x1)},{int(y1)},{int(width)},{int(height)},1,{classid},1\n")
        video_writer.close()

        # seginfo
        seginfo_path = os.path.join(
            save_gt_dir, f"{dataname}S{skipframe}-{videoid}/seqinfo.ini")
        config = configparser.ConfigParser()
        try:
            config.add_section("Sequence")
            config.set("Sequence", "name", dataname + f"{videoid}")
            config.set("Sequence", "imDir", "images")
            config.set("Sequence", "frameRate", "30")
            config.set("Sequence", "seqLength", f"{framenumbers}")
            config.set("Sequence", "imWidth", f"{resolutionwidth}")
            config.set("Sequence", "imHeight", f"{resolutionheight}")
            config.set("Sequence", "imExt", ".jpg")
        except configparser.DuplicateSectionError:
            print("Section 'Sequence' already exists")
        config.write(open(seginfo_path, "w"))
