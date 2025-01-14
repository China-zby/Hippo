import sys
import json
import numpy as np

import matching
from dataloader import read_json
from kalman_filter import KalmanFilter
from basetrack import BaseTrack, TrackState


data_root = sys.argv[1]
data_name = sys.argv[2]
skip_bound = int(sys.argv[3])
device_id = sys.argv[4]

hash_size = int(sys.argv[5])
low_feature_distance_threshold = float(sys.argv[6])
max_lost_time = int(sys.argv[7])
move_threshold = float(sys.argv[8])
keep_threshold = float(sys.argv[9])
min_threshold = float(sys.argv[10])
create_object_threshold = float(sys.argv[11])
iou_threshold = float(sys.argv[12])
unmatch_location_threshold = float(sys.argv[13])
match_visual_threshold = float(sys.argv[14])
kf_pos_weight = float(sys.argv[15])
kf_vel_weight = float(sys.argv[16])


class STrack(BaseTrack):
    shared_kalman = KalmanFilter(kf_pos_weight, kf_vel_weight)

    def __init__(self, tlwh, score, frame_gap):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.frame_gap = frame_gap

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
                multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        # if frame_id == 1:
        #     self.is_activated = True
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, frame_gap=1,
                 iou_threshold=0.3, track_thresh=0.3,
                 track_buffer=60):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.frame_gap = frame_gap
        self.track_thresh = track_thresh
        self.det_thresh = track_thresh + 0.1
        self.buffer_size = track_buffer
        self.match_thresh = iou_threshold
        self.match_thresh_second = self.match_thresh - 0.2
        self.match_thresh_unconfirmed = self.match_thresh - 0.1
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter(kf_pos_weight, kf_vel_weight)

    def update(self, output_results):
        self.frame_id += self.frame_gap
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        bboxes = output_results[:, :4]  # x1y1x2y2
        scores = output_results[:, 4]
        track_ids = [None] * len(bboxes)

        remain_inds = scores > self.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        remain_idxs = np.where(remain_inds)[0]
        idxs_second = np.where(inds_second)[0]
        hit_det_ids = []

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, self.frame_gap) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.ciou_distance(strack_pool, detections)
        # if not self.args.mot20:
        #     dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
            hit_det_ids.append([remain_idxs[idet], track.track_id])

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, self.frame_gap) for
                                 (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i]
                             for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.ciou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(
            dists, thresh=self.match_thresh_second)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
            hit_det_ids.append([idxs_second[idet], track.track_id])

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        remain_idxs = [remain_idxs[i] for i in u_detection]
        detections = [detections[i] for i in u_detection]
        dists = matching.ciou_distance(unconfirmed, detections)
        # if not self.args.mot20:
        #     dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(
            dists, thresh=self.match_thresh_unconfirmed)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
            hit_det_ids.append(
                [remain_idxs[idet], unconfirmed[itracked].track_id])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
            hit_det_ids.append([remain_idxs[inew], track.track_id])

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(
            self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(
            self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(
            self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(
            self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        # output_stracks = [
        #     track for track in self.tracked_stracks if track.is_activated]

        for hdi, tid in hit_det_ids:
            track_ids[hdi] = int(tid)

        return track_ids, 1.0


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb


def test_tracker():
    import cv2
    import random
    from ultralytics import YOLO
    detector = YOLO("yolov5x.pt")
    # stdin = sys.stdin.detach()
    trackers = {}

    # model = RNNModel().cuda()
    # model.load_state_dict(torch.load(model_path), strict=False)
    # model.eval()

    skip_number = 8
    frame_idx = 1

    videoPath = "/mnt/data_ssd1/lzp/otif-dataset/dataset/hippo/train/video/79.mp4"
    videoCap = cv2.VideoCapture(videoPath)
    orig_width, orig_height = videoCap.get(
        cv2.CAP_PROP_FRAME_WIDTH), videoCap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    color_dict = {}
    track_class_list = [2]
    video_writer = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(
        *'mp4v'), max(30 // skip_number, 1), (int(orig_width), int(orig_height)))

    while True:
        ret, im = videoCap.read()
        if not ret:
            break

        # if not ret:
        #     break
        # packet = read_json(stdin)
        # if packet is None:
        #     break
        # id = packet['id']
        id = 0
        # msg = packet['type']
        # frame_idx = packet['frame_idx']

        # if msg == 'end':
        #     # del trackers[id]
        #     continue

        if frame_idx % skip_number != 0:
            frame_idx += 1
            continue

        # im = read_im(stdin)
        # im = im[:, :, ::-1]
        results = detector(im, verbose=False, iou=0.45)
        detections = []
        for result in results:
            # print(result.boxes[0], result.boxes.shape)
            xyxys = result.boxes.xyxy
            confs = result.boxes.conf
            clses = result.boxes.cls
            for xyxy, conf, cls in zip(xyxys, confs, clses):
                classID = int(cls)
                # print(classID)
                if classID not in track_class_list:
                    continue
                if (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]) < 500:
                    continue
                detection = {"left": int(xyxy[0]), "top": int(xyxy[1]), "right": int(
                    xyxy[2]), "bottom": int(xyxy[3]), "score": float(conf), "class": int(cls)}
                detections.append(detection)

        # detections = packet['detections']
        # print("detections", detections, file=open(f"./demos/log+{id}.txt", "a"))
        if detections is None:
            detections = []

        if id not in trackers:
            trackers[id] = BYTETracker(frame_gap=skip_number,
                                       iou_threshold=0.3, track_thresh=0.3, track_buffer=60)

        detections_array = []
        for detection in detections:
            detections_array.append([detection['left'], detection['top'],
                                     detection['right'], detection['bottom'], detection['score']])
        detections_array = np.array(detections_array)

        if detections_array.size == 0:
            detections_array = np.empty((0, 5))
        # , packet['resolution'], packet['sceneid'])

        track_ids, frame_confidence = trackers[id].update(detections_array)

        # print("track_ids", track_ids, file=open(f"./demos/log+{id}.txt", "a"))
        d = {
            'outputs': track_ids,
            'conf': frame_confidence,
            't': {},
        }
        # print("d", d, file=open(f"./demos/log+{id}.txt", "a"))
        # sys.stdout.write('json'+json.dumps(d)+'\n')
        # sys.stdout.flush()

        for detection, track_id in zip(detections, track_ids):  # visualize
            if track_id is None:
                continue
            if track_id not in color_dict:
                color_dict[track_id] = (random.randint(
                    0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.rectangle(im, (detection['left'], detection['top']), (
                detection['right'], detection['bottom']), color_dict[track_id], 2)
            cv2.putText(im, str(track_id), (detection['left'], detection['top']),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color_dict[track_id], 2)
        video_writer.write(im)
        cv2.imwrite("./demo_tracker/"+str(frame_idx)+".jpg", im)

        # print(d)
        frame_idx += 1


def go_tracker():
    stdin = sys.stdin.detach()
    trackers = {}

    while True:
        packet = read_json(stdin)
        if packet is None:
            break
        id = packet['id']
        msg = packet['type']
        frame_idx = packet['frame_idx']

        if msg == 'end':
            # del trackers[id]
            continue

        detections = packet['detections']
        if detections is None:
            detections = []

        if id not in trackers:
            trackers[id] = BYTETracker(frame_gap=skip_bound,
                                       iou_threshold=iou_threshold, track_thresh=keep_threshold,
                                       track_buffer=max_lost_time)

        detections_array = []
        for detection in detections:
            detections_array.append([detection['left'], detection['top'],
                                     detection['right'], detection['bottom'], detection['score']])
        detections_array = np.array(detections_array)

        if detections_array.size == 0:
            detections_array = np.empty((0, 5))

        track_ids, frame_confidence = trackers[id].update(detections_array)

        d = {
            'outputs': track_ids,
            'conf': frame_confidence,
            't': {},
        }

        sys.stdout.write('json'+json.dumps(d)+'\n')
        sys.stdout.flush()


if __name__ == '__main__':
    # test_tracker()
    go_tracker()
