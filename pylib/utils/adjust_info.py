import os
import cv2

datadir = "/home/lzp/otif-dataset/dataset"
dataname = "camera_envs"
datatype = "streamline"

infodir = os.path.join(datadir, dataname, datatype, "info_raw")
newinfodir = os.path.join(datadir, dataname, datatype, "info")
videodir = os.path.join(datadir, dataname, datatype, "video")

if not os.path.exists(newinfodir):
    os.makedirs(newinfodir)

videolist = os.listdir(videodir)
videolist.sort()
for videopathname in videolist:
    videoinfopath = os.path.join(infodir, videopathname.replace(".mp4", ".txt"))
    newvideoinfopath = os.path.join(newinfodir, videopathname.replace(".mp4", ".txt"))
    videopath = os.path.join(videodir, videopathname)
    
    videoCapture = cv2.VideoCapture(videopath)
    fps = int(videoCapture.get(cv2.CAP_PROP_FPS))
    frame_count = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(frame_count / fps)
    width = videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    with open(videoinfopath, "r") as f:
        dataline = f.readline()
        dataname, datawidth, dataheight = dataline.split("-")
        datawidth, dataheight = int(datawidth), int(dataheight)
        assert datawidth == width and dataheight == height
        dataline += f"-{fps}-{frame_count}-{duration}"
    
    with open(newvideoinfopath, "w") as f:
        f.write(dataline)