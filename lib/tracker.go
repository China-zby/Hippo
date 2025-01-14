package lib

/*
for 1 fps

* threshold means we look at more frames when confidence is less than threshold
* threshold=0: don't look at any more frames
* threshold=1: don't track at this rate (just start tracking at next framerate)

pre-process:
(1) track over the video at 1 fps; but update RNN features using ground truth tracks
(2) for each track, it yields:
	(a) 1 - (2nd highest) / (1st highest): so if threshold is higher, then we would look at extra frame
	(b) minimum threshold needed to get it correct, i.e., 0 if correct option had highest prob, or 1 - (correct score) / (1st highest) otherwise
(3) using (b), compute minimum threshold needed to recover 80% contiguous segment of each track, call this (c)

setting threshold:
(1) iterate over alpha [0.99, 0.98, 0.96, 0.92, ...]
(2) pick alpha-percentile value in the (c) values
(3) estimate how many extra frames we would use based on the (a) values
*/

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
)

type Tracker struct {
	modeltype string
	cmd       *exec.Cmd
	stdin     io.WriteCloser
	rd        *bufio.Reader
	mu        sync.Mutex
}

type TrackerPacket struct {
	ID         int          `json:"id"`
	FrameIdx   int          `json:"frame_idx"`
	Detections []Detection  `json:"detections"`
	GT         map[int]*int `json:"gt"`
	Type       string       `json:"type"`
	Resolution [2]int       `json:"resolution"`
	SceneID    int          `json:"sceneid"`
}

type TrackerResponse struct {
	Outputs []int           `json:"outputs"`
	Conf    float64         `json:"conf"`
	T       map[int]float64 `json:"t"`
}

var TRACKTYPES = []string{"ByteTrack", "DeepSORT", "SORT", "OTIF", "GNNCNN"}

func NewTracker(
	DeviceID int,
	SkipNumber int,
	ModelType string,
	WeightRoot string,
	DataName string,
	HashSize int,
	LowFeatureDistanceThreshold float64,
	MaxLostTime int,
	MoveThreshold float64,
	KeepThreshold float64,
	MinThreshold float64,
	CreateObjectThreshold float64,
	MatchLocationThreshold float64,
	UnmatchLocationThreshold float64,
	VisualThreshold float64,
	KFPosWeight float64,
	KFVelWeight float64,
) *Tracker {
	var cmd *exec.Cmd
	deviceID := strconv.Itoa(DeviceID)
	SkipBound := -1
	if SkipNumber < 16 {
		SkipBound = 16
	} else if SkipNumber < 32 {
		SkipBound = 32
	} else if SkipNumber < 64 {
		SkipBound = 64
	} else if SkipNumber < 128 {
		SkipBound = 128
	} else if SkipNumber < 256 {
		SkipBound = 256	
	} else {
		SkipBound = 256
	}

	supported := false
	for _, tracktype := range TRACKTYPES {
		if strings.EqualFold(ModelType, tracktype) {
			supported = true
			break
		}
	}
	if !supported {
		panic("unsupported tracker type: " + ModelType)
	}

	cmd = exec.Command(
		"python", "-W ignore", "./pylib/Trackers/"+strings.ToLower(ModelType)+".py",
		WeightRoot,
		DataName, strconv.Itoa(SkipBound),
		deviceID,
		strconv.Itoa(HashSize),
		strconv.FormatFloat(LowFeatureDistanceThreshold, 'f', -1, 64),
		strconv.Itoa(MaxLostTime),
		strconv.FormatFloat(MoveThreshold, 'f', -1, 64),
		strconv.FormatFloat(KeepThreshold, 'f', -1, 64),
		strconv.FormatFloat(MinThreshold, 'f', -1, 64),
		strconv.FormatFloat(CreateObjectThreshold, 'f', -1, 64),
		strconv.FormatFloat(MatchLocationThreshold, 'f', -1, 64),
		strconv.FormatFloat(UnmatchLocationThreshold, 'f', -1, 64),
		strconv.FormatFloat(VisualThreshold, 'f', -1, 64),
		strconv.FormatFloat(KFPosWeight, 'f', -1, 64),
		strconv.FormatFloat(KFVelWeight, 'f', -1, 64),
	)

	// fmt.Println(
	// 	"python", "-W ignore", "./pylib/Trackers/"+strings.ToLower(ModelType)+".py",
	// 	WeightRoot,
	// 	DataName, strconv.Itoa(SkipBound),
	// 	deviceID,
	// 	strconv.Itoa(HashSize),
	// 	strconv.FormatFloat(LowFeatureDistanceThreshold, 'f', -1, 64),
	// 	strconv.Itoa(MaxLostTime),
	// 	strconv.FormatFloat(MoveThreshold, 'f', -1, 64),
	// 	strconv.FormatFloat(KeepThreshold, 'f', -1, 64),
	// 	strconv.FormatFloat(MinThreshold, 'f', -1, 64),
	// 	strconv.FormatFloat(CreateObjectThreshold, 'f', -1, 64),
	// 	strconv.FormatFloat(MatchLocationThreshold, 'f', -1, 64),
	// 	strconv.FormatFloat(UnmatchLocationThreshold, 'f', -1, 64),
	// 	strconv.FormatFloat(VisualThreshold, 'f', -1, 64),
	// )

	stdin, err := cmd.StdinPipe()
	if err != nil {
		panic(err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		panic(err)
	}
	if true {
		cmd.Stderr = os.Stderr
	}
	if err := cmd.Start(); err != nil {
		panic(err)
	}
	rd := bufio.NewReader(stdout)
	return &Tracker{
		modeltype: strings.ToLower(ModelType),
		cmd:       cmd,
		stdin:     stdin,
		rd:        rd,
	}
}

func (tracker *Tracker) do(id int, frameIdx int, im Image, detections []Detection, gt map[int]*int, Resolution [2]int, sceneid int) TrackerResponse {
	packet := TrackerPacket{
		ID:         id,
		FrameIdx:   frameIdx,
		Detections: detections,
		GT:         gt,
		Type:       "job",
		Resolution: Resolution,
		SceneID:    sceneid,
	}
	bytes := JsonMarshal(packet)

	tracker.mu.Lock()
	defer tracker.mu.Unlock()

	header := make([]byte, 4)
	binary.BigEndian.PutUint32(header, uint32(len(bytes)))
	tracker.stdin.Write(header)
	tracker.stdin.Write(bytes)

	if tracker.modeltype == "deepsort" || tracker.modeltype == "gnncnn" {
		header = make([]byte, 12)
		binary.BigEndian.PutUint32(header[0:4], uint32(len(im.Bytes)))
		binary.BigEndian.PutUint32(header[4:8], uint32(im.Width))
		binary.BigEndian.PutUint32(header[8:12], uint32(im.Height))
		tracker.stdin.Write(header)
		tracker.stdin.Write(im.Bytes)
	}

	var line string
	for {
		var err error
		line, err = tracker.rd.ReadString('\n')

		if err != nil {
			fmt.Println("error:", err)
			panic(err)
		}
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "json") {
			continue
		}
		break
	}

	jsonBytes := []byte(line[4:])
	var response TrackerResponse
	if err := json.Unmarshal(jsonBytes, &response); err != nil {
		panic(err)
	}
	return response
}

func (tracker *Tracker) Infer(id int, frameIdx int, im Image, detections []Detection, Resolution [2]int, sceneid int) ([]Detection, float64) {
	response := tracker.do(id, frameIdx, im, detections, nil, Resolution, sceneid)

	var outputs []Detection
	for i, d := range detections {
		d.TrackID = new(int)
		*d.TrackID = response.Outputs[i]
		if *d.TrackID == 0 {
			continue
		}
		outputs = append(outputs, d)
	}
	return outputs, response.Conf
}

func (tracker *Tracker) End(id int) {
	packet := TrackerPacket{
		ID:   id,
		Type: "end",
	}
	bytes := JsonMarshal(packet)

	tracker.mu.Lock()
	header := make([]byte, 4)
	binary.BigEndian.PutUint32(header, uint32(len(bytes)))
	tracker.stdin.Write(header)
	tracker.stdin.Write(bytes)
	tracker.mu.Unlock()
}

type TrackerProfile struct {
	NumFrames  int
	Thresholds []float64
}

func (prof TrackerProfile) MaxGap() int {
	if len(prof.Thresholds) < 2 {
		return 1
	}
	gap := 1
	for _, threshold := range prof.Thresholds[1:] {
		if threshold == 1 {
			return gap
		}
		gap *= 2
	}
	return gap
}

func (tracker *Tracker) Close() {
	tracker.stdin.Close()
	tracker.cmd.Wait()
}

// car:
//   recall: 0.996
//   precision: 0.964
//   accuracy: 0.976
//   f1: 0.976
//   mae: 1.586
//   acc: 0.346
//   acc_topk: 0.553
//   HOTA_car: 0.569
// bus: {}
// truck:
//   recall: 0.788
//   precision: 0.664
//   accuracy: 0.894
//   f1: 0.711
//   mae: 0.322
//   acc: 0.75
//   acc_topk: 0.642
//   HOTA_truck: 0.231
