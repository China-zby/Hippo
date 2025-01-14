import os
import csv
import random
from copy import deepcopy

if __name__ == "__main__":
    datadir = "/home/lzp/otif-dataset/dataset/hippo/test"
    context_path = os.path.join(datadir, "context.csv")
    context_info = csv.reader(open(context_path, "r"))
    next(context_info)
    camera_dict = {}
    for context_row in context_info:
        vid, cid, _, _ = context_row
        vid, cid = int(vid), int(cid)
        if cid not in camera_dict:
            camera_dict[cid] = []
        camera_dict[cid].append(vid)
    camera_dict = {k: sorted(v) for k, v in camera_dict.items()}
    videodir = os.path.join(datadir, "video")
    video_path_name_list = os.listdir(videodir)
    sample_video_numbers = [100, 109, 118, 125, 127, 136, 145,
                            150, 159, 168, 175, 177, 186, 195,
                            200, 209, 218, 225, 227, 236, 245,
                            250, 259, 268, 275, 277, 286, 295,
                            300, 309, 318, 325, 327, 336, 345] # [100, 150, 200, 250, 300]
    video_idx_dict = {}
    for i in range(len(sample_video_numbers)):
        sample_video_number = sample_video_numbers[i]
        if i == 0:
            sum_sample_video_number = sample_video_number
        else:
            sum_sample_video_number = sample_video_number - \
                sample_video_numbers[i - 1]

        sample_camera_numbers = []
        for k in range(len(camera_dict)):
            if sum_sample_video_number - sum(sample_camera_numbers) >= sum_sample_video_number // len(camera_dict):
                sample_camera_numbers.append(
                    sum_sample_video_number // len(camera_dict))
            else:
                sample_camera_numbers.append(
                    sum_sample_video_number - sum(sample_camera_numbers))
        if sum(sample_camera_numbers) != sum_sample_video_number:
            for p in range(sum_sample_video_number - sum(sample_camera_numbers)):
                sample_camera_numbers[p] += 1

        assert sum_sample_video_number == sum(sample_camera_numbers)

        random.shuffle(sample_camera_numbers)
        if sample_video_number not in video_idx_dict:
            video_idx_dict[sample_video_number] = []
        if i == 0:
            video_idx_list = []
            for j, cid in enumerate(camera_dict):
                video_idx_list.append(random.sample(
                    camera_dict[cid], sample_camera_numbers[j]))
            video_idx_dict[sample_video_number] = video_idx_list
        else:
            video_idx_list = deepcopy(
                video_idx_dict[sample_video_numbers[i - 1]])
            for j, cid in enumerate(camera_dict):
                video_idx_list[j] += random.sample(set(camera_dict[cid]) - set(
                    video_idx_list[j]), sample_camera_numbers[j])
            video_idx_dict[sample_video_number] = video_idx_list
    for sample_video_number in sample_video_numbers:
        video_idxs = []
        for video_idx_list in video_idx_dict[sample_video_number]:
            video_idxs += video_idx_list
        video_idxs = sorted(video_idxs)
        video_idx_dict[sample_video_number] = video_idxs

    # for sample_video_number in video_idx_dict:
    #     print(video_idx_dict[sample_video_number], len(video_idx_dict[sample_video_number]))

    for i in range(len(sample_video_numbers) - 1):
        sample_video_number = sample_video_numbers[i]
        next_sample_video_number = sample_video_numbers[i + 1]
        all_in_flag = True
        for j in range(len(video_idx_dict[sample_video_number])):
            if video_idx_dict[sample_video_number][j] not in video_idx_dict[next_sample_video_number]:
                all_in_flag = False
                break
        assert all_in_flag

    print(video_idx_dict)
