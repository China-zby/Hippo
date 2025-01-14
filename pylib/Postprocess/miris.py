import os
import sys
import json
import torch
import struct

from model import PSRNN


def read_json(stdin):
    buf = stdin.read(4)
    if not buf:
        return None
    (l,) = struct.unpack('>I', buf)
    buf = stdin.read(l)
    return json.loads(buf.decode('utf-8'))


def closest_value(original_frame, frame_gap):
    reminder = original_frame % frame_gap
    if reminder < frame_gap / 2:
        return original_frame - reminder
    else:
        return original_frame - reminder + frame_gap


if __name__ == "__main__":
    device_id = int(sys.argv[1])
    device = torch.device('cuda', device_id)
    scene2id = {
        "amsterdam": 0,
        "warsaw":    1, "shibuya": 2,
        "jackson": 3, "caldot1": 4,
        "caldot2": 5, "uav": 6,
    }
    model = PSRNN(scene2id)
    model.load_state_dict(torch.load('./weights/Postprocess/rnn.pth'))
    model.eval()
    model = model.cuda()
    frame_norm = 1800

    stdin = sys.stdin.detach()
    while True:
        packet = read_json(stdin)
        if packet is None:
            break

        msg = packet['type']

        if msg == 'end':
            continue

        # Tracks [][]TrackDetection `json:"tracks"`
        # Skip   int                `json:"skip"`
        # Width  int                `json:"width"`
        # Height int                `json:"height"`
        # Scene  string             `json:"scene"`
        # Type   string             `json:"type"`
        # FrameNumber int           `json:"frame_number"`

        tracks = packet['tracks']
        skip = packet['skip']
        width = packet['width']
        height = packet['height']
        scene = packet['scene']
        type = packet['type']
        frame_number = packet['frame_number']

        with torch.no_grad():
            new_tracks = []
            for track in tracks:
                mean_score = sum([point['score'] for point in track])
                if mean_score / len(track) < 0.6:
                    new_tracks.append(track)
                    continue

                track_tensor = []
                for point in track:
                    left, top, right, bottom = point['left'], point['top'], point['right'], point['bottom']
                    frameid = point['FrameIdx']
                    relative_frameid = frameid - track[0]['FrameIdx']
                    track_tensor.append([left / width, top / height,
                                        right / width, bottom / height,
                                        relative_frameid / frame_norm])
                track_tensor = torch.FloatTensor(
                    track_tensor).cuda().unsqueeze(0)
                sceneid = scene2id[scene]
                sceneid = torch.LongTensor([sceneid]).cuda()
                prefix, suffix = model.inference(track_tensor, sceneid)

                prefix = prefix.squeeze().cpu().numpy()
                suffix = suffix.squeeze().cpu().numpy()

                prefix_left, prefix_top, prefix_right, prefix_bottom, prefix_frame_gap = prefix
                suffix_left, suffix_top, suffix_right, suffix_bottom, suffix_frame_gap = suffix

                prefix_left = int(prefix_left * width)
                prefix_top = int(prefix_top * height)
                prefix_right = int(prefix_right * width)
                prefix_bottom = int(prefix_bottom * height)
                prefix_frame_gap = int(prefix_frame_gap * frame_norm)

                suffix_left = int(suffix_left * width)
                suffix_top = int(suffix_top * height)
                suffix_right = int(suffix_right * width)
                suffix_bottom = int(suffix_bottom * height)
                suffix_frame_gap = int(suffix_frame_gap * frame_norm)

                start_frame = track[0]['FrameIdx'] - prefix_frame_gap
                end_frame = track[-1]['FrameIdx'] + suffix_frame_gap

                start_frame = closest_value(start_frame, skip)
                end_frame = closest_value(end_frame, skip)

                new_start, new_end = [], []
                if (start_frame < track[0]['FrameIdx'] and start_frame >= 0) and (end_frame > track[-1]['FrameIdx'] and end_frame < frame_number):
                    new_start.append({'left': prefix_left, 'top': prefix_top, 'right': prefix_right, 'bottom': prefix_bottom, 'class': track[0]
                                      ['class'], 'score': track[0]['score'], 'track_id': track[0]['track_id'], 'FrameIdx': start_frame})
                    new_end.append({'left': suffix_left, 'top': suffix_top, 'right': suffix_right, 'bottom': suffix_bottom, 'class': track[-1]
                                    ['class'], 'score': track[-1]['score'], 'track_id': track[-1]['track_id'], 'FrameIdx': end_frame})
                elif (start_frame >= track[0]['FrameIdx'] or start_frame < 0) and (end_frame > track[-1]['FrameIdx'] and end_frame < frame_number):
                    new_end.append({'left': suffix_left, 'top': suffix_top, 'right': suffix_right, 'bottom': suffix_bottom, 'class': track[-1]
                                    ['class'], 'score': track[-1]['score'], 'track_id': track[-1]['track_id'], 'FrameIdx': end_frame})
                elif (start_frame < track[0]['FrameIdx'] and start_frame >= 0) and (end_frame < track[-1]['FrameIdx'] or end_frame >= frame_number):
                    new_start.append({'left': prefix_left, 'top': prefix_top, 'right': prefix_right, 'bottom': prefix_bottom, 'class': track[0]['class'],
                                      'score': track[0]['score'], 'track_id': track[0]['track_id'], 'FrameIdx': start_frame})
                new_track = new_start + track + new_end
                new_tracks.append(new_track)

        sys.stdout.write('json'+json.dumps(new_tracks)+'\n')
        sys.stdout.flush()
