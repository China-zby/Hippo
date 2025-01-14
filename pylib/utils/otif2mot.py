import os
import cv2
import json
import argparse
import configparser

parse = argparse.ArgumentParser()
parse.add_argument("--input_dir", type=str, default="./outputs/")
parse.add_argument("--input_gt_dir", type=str, default="./")
parse.add_argument("--data_root", type=str, default="./")
parse.add_argument("--input_video_dir", type=str, default="./")
parse.add_argument("--method_name", type=str, default="otif")
parse.add_argument("--dataset_name", type=str, default="amsterdam")
parse.add_argument("--videoidlist", type=str, default="0-1-2-3-4")
parse.add_argument("--skipframelist", type=str, default="8-8-16-8-32")
parse.add_argument("--framenumber", type=int, default=1800)
parse.add_argument("--testmode", type=str, default="test_separate_video")
parse.add_argument("--differentclass", action='store_true')
parse.add_argument("--width", type=str, default='')
parse.add_argument("--height", type=str, default='')
parse.add_argument("--classes", type=str, default="car,bus,truck")
parse.add_argument("--flag", type=str, default="xxx")
args = parse.parse_args()

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
videoflag = "".join(args.videoidlist.split("-"))
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
    # if not os.path.exists(seqmap_path):
    seqmap_writer = open(seqmap_path, "w")
    seqmap_writer.write("name\n")
    # else:
    #     seqmap_writer = open(seqmap_path, "a")
    if skipframe not in seqmap_writer_dict.keys():
        seqmap_writer_dict[skipframe] = [seqmap_writer, seqmap_path]

for videoid, skipframe, resolutionwidth, resolutionheight in zip(videoids, skipframes, widths, heights):
    # print(f"Processing: {videoid} --- Skipframe: {skipframe}")
    videologo = f"{dataname}S{skipframe}-{datatype}"
    save_dir = f"./TrackEval/data/trackers/videodb/{videologo}/{method_name}/data"
    save_gt_dir = f"./TrackEval/data/gt/videodb/{videologo}"

    input_raw_videos = os.listdir(args.input_video_dir)
    trackid_dict, trackstartid = {}, 0

    # track
    output_txt_video = os.path.join(
        save_dir, f"{dataname}S{skipframe}-{videoid}.txt")
    if not os.path.exists(os.path.dirname(output_txt_video)):
        os.makedirs(os.path.dirname(output_txt_video))
    video_writer = open(output_txt_video, "w")

    input_json_path = os.path.join(args.input_dir, f"{videoid}.json")
    input_video_path = os.path.join(
        args.data_root, f"{args.testmode}/video/{videoid}.mp4")
    frameid = 1
    with open(input_json_path, 'r') as freader:
        content = freader.read()
        data = json.loads(content)
        TrackletData = {}
        if data is not None:
            for data_line in data:
                if (frameid - 1) % skipframe == 0:
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
                            remapframeid = int((frameid - 1) / skipframe) + 1
                            if trackid not in TrackletData.keys():
                                TrackletData[trackid] = [[], []]
                            TrackletData[trackid][0].append(
                                [remapframeid, int(x1), int(y1), int(width), int(height), float(score)])
                            TrackletData[trackid][1].append(int(classid))
                frameid += 1

    if len(TrackletData.keys()) > 0:
        for trackid in TrackletData.keys():
            classid = max(TrackletData[trackid][1],
                          key=TrackletData[trackid][1].count)

            for dataline in TrackletData[trackid][0]:
                remapframeid, x1, y1, width, height, score = dataline
                video_writer.write(
                    f"{remapframeid},{trackid},{int(x1)},{int(y1)},{int(width)},{int(height)},{score},{classid},-1,-1,-1\n")
    # else:
    #     video_writer.write("\n")

    framenumbers = cv2.VideoCapture(
        input_video_path).get(cv2.CAP_PROP_FRAME_COUNT)
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
            if (frameid - 1) % skipframe == 0:
                remapframeid = int((frameid - 1) / skipframe) + 1
                if trackid not in GTTrackletData.keys():
                    GTTrackletData[trackid] = [[], []]
                GTTrackletData[trackid][0].append(
                    [remapframeid, int(x1), int(y1), int(width), int(height), float(score)])
                GTTrackletData[trackid][1].append(int(classid))

    # input_gt_path = os.path.join(args.input_gt_dir, f"{videoid}.json")
    # with open(input_gt_path, "r") as reader:
    #     GTTrackletData = {}
    #     gt_data = json.load(reader)
    #     for frameid, trackdata in enumerate(gt_data):
    #         frameid += 1
    #         if (frameid - 1) % skipframe == 0:
    #             remapframeid = int((frameid - 1) / skipframe) + 1
    #             if trackdata == [] or trackdata is None:
    #                 continue
    #             for objectdata in trackdata:
    #                 x1, y1, x2, y2 = objectdata["left"], objectdata["top"],\
    #                     objectdata["right"], objectdata["bottom"]
    #                 width, height = x2 - x1, y2 - y1
    #                 score = objectdata["score"]
    #                 videotrackid = int(objectdata["track_id"])
    #                 classid = class_map_dict[objectdata["class"]]
    #                 if videotrackid not in gt_trackid_dict.keys():
    #                     gt_trackid_dict[videotrackid] = len(
    #                         gt_trackid_dict.keys()) + 1
    #                 trackid = gt_trackid_dict[videotrackid]
    #                 if trackid not in GTTrackletData.keys():
    #                     GTTrackletData[trackid] = [[], []]
    #                 GTTrackletData[trackid][0].append(
    #                     [remapframeid, int(x1), int(y1), int(width), int(height), float(score)])
    #                 GTTrackletData[trackid][1].append(int(classid))

    for trackid in GTTrackletData.keys():
        classid = max(GTTrackletData[trackid][1],
                      key=GTTrackletData[trackid][1].count)
        for dataline in GTTrackletData[trackid][0]:
            remapframeid, x1, y1, width, height, score = dataline
            video_writer.write(
                f"{remapframeid},{trackid},{int(x1)},{int(y1)},{int(width)},{int(height)},1,{classid},1\n")
    # if len(GTTrackletData.keys()) == 0:
    #     video_writer.write("\n")
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
        config.set("Sequence", "seqLength",
                   f"{int((framenumbers - 1) / skipframe) + 1}")
        config.set("Sequence", "imWidth", f"{resolutionwidth}")
        config.set("Sequence", "imHeight", f"{resolutionheight}")
        config.set("Sequence", "imExt", ".jpg")
    except configparser.DuplicateSectionError:
        print("Section 'Sequence' already exists")
    config.write(open(seginfo_path, "w"))
