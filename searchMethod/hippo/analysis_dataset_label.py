import os
from glob import glob

track_path_dir = "/mnt/data_ssd1/lzp/otif-dataset/dataset/hippo/train/tracks/*.txt"
track_path_list = glob(track_path_dir)

bus_path_num = 0
truck_path_num = 0
for track_path in track_path_list:
    with open(track_path, 'r') as file_reader:
        data_lines = file_reader.readlines()
        track_data = {}
        for data_line in data_lines:
            data_line = data_line.rstrip("\n").split(",")
            frameid, trackid, x, y, w, h, score, classid, _, _, _ = data_line
            frameid = int(frameid)
            trackid = int(trackid)
            classid = int(classid)
            if trackid not in track_data:
                track_data[trackid] = [classid,[]]
            track_data[trackid][1].append(int(frameid))
        class_num = {2:0, 5:0, 7:0}
        for trackid in track_data:
            class_num[track_data[trackid][0]] += 1

        if class_num[5] > 0:
            bus_path_num += 1
            print(track_path, class_num)
            
        if class_num[7] > 0:
            truck_path_num += 1
            print(track_path, class_num)

print("bus_path_num: ", bus_path_num)
print("truck_path_num: ", truck_path_num)