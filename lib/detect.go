package lib

import (
	"bufio"
	"encoding/binary"
	"encoding/json"

	// "fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"

	"github.com/mitroadmaps/gomapinfer/common"
)

type Detection struct {
	Left    int     `json:"left"`
	Top     int     `json:"top"`
	Right   int     `json:"right"`
	Bottom  int     `json:"bottom"`
	Class   string  `json:"class"`
	Score   float64 `json:"score"`
	TrackID *int    `json:"track_id,omitempty"`
}

func (d *Detection) GetCenter() [2]int {
	return [2]int{(d.Left + d.Right) / 2, (d.Top + d.Bottom) / 2}
}

type Point struct {
	X int
	Y int
}

func RescaleDetections(detections [][]Detection, origDims [2]int, newDims [2]int) {
	for frameIdx := range detections {
		for i := range detections[frameIdx] {
			detections[frameIdx][i].Left = detections[frameIdx][i].Left * newDims[0] / origDims[0]
			detections[frameIdx][i].Right = detections[frameIdx][i].Right * newDims[0] / origDims[0]
			detections[frameIdx][i].Top = detections[frameIdx][i].Top * newDims[1] / origDims[1]
			detections[frameIdx][i].Bottom = detections[frameIdx][i].Bottom * newDims[1] / origDims[1]
		}
	}
}

func FilterDetectionsByClass(detections [][]Detection, cls []string) [][]Detection {
	ndetections := make([][]Detection, len(detections))
	for frameIdx, dlist := range detections {
		for _, d := range dlist {
			findClass := false
			for _, c := range cls {
				if d.Class == c {
					findClass = true
					break
				}
			}
			if findClass {
				ndetections[frameIdx] = append(ndetections[frameIdx], d)
			}
		}
	}
	return ndetections
}

type TrackDetection struct {
	Detection
	FrameIdx int
}

func GetTracks(detections [][]Detection) [][]TrackDetection {
	tracks := make(map[int][]TrackDetection)
	for frameIdx, dlist := range detections {
		for _, d := range dlist {
			tracks[*d.TrackID] = append(tracks[*d.TrackID], TrackDetection{
				Detection: d,
				FrameIdx:  frameIdx,
			})
		}
	}
	var trackList [][]TrackDetection
	for _, track := range tracks {
		trackList = append(trackList, track)
	}
	return trackList
}

func DetectionsFromTracks(tracks [][]TrackDetection) [][]Detection {
	var detections [][]Detection
	for _, track := range tracks {
		for _, d := range track {
			for len(detections) <= d.FrameIdx {
				detections = append(detections, nil)
			}
			detections[d.FrameIdx] = append(detections[d.FrameIdx], d.Detection)
		}
	}
	return detections
}

func (d Detection) Center() Point {
	return Point{
		X: (d.Left + d.Right) / 2,
		Y: (d.Top + d.Bottom) / 2,
	}
}

func (d Detection) Rectangle() common.Rectangle {
	return common.Rectangle{
		Min: common.Point{X: float64(d.Left), Y: float64(d.Top)},
		Max: common.Point{X: float64(d.Right), Y: float64(d.Bottom)},
	}
}

func (p Point) Distance(o Point) float64 {
	dx := p.X - o.X
	dy := p.Y - o.Y
	return math.Sqrt(float64(dx*dx + dy*dy))
}

type Detector struct {
	cmd       *exec.Cmd
	stdin     io.WriteCloser
	rd        *bufio.Reader // stdout of the python process
	mu        sync.Mutex
	batchSize int
	origDims  [2]int
	zeroImage []byte
}

// paramDims indicate the uncropped image dimensions (which determines the model parameters that we load)
func NewDetector(
	ModelType string,
	BatchSize int,
	ModelSize string,
	Threshold float64,
	Classes []string,
	DeviceID int,
	ScaleDownResolution [2]int,
	DataRoot string,
	DataName string) *Detector {
	var classList []string
	classList = append(classList, Classes...)
	deviceID := strconv.Itoa(DeviceID)
	detectName := ModelType

	// {'YOLOV3': ['X'],
	// 'YOLOV5': ['N', 'S', 'M', 'L', 'X'],
	// 'YOLOVM': ["N", "S", "M", "L"],
	// 'YOLOV7': ["L", "X"],
	// 'YOLOV8': ['N', 'S', 'M', 'L', 'X'],
	// 'DETR': ['50', '101'],
	// 'FASTERRCNN': ['r50', 'r101', 'x101'],
	// 'SPARSERCNN': ['r50100', 'r50300', 'r101100', 'r101300'],
	// 'RETINANET': ['r50', 'r101', 'x101'],
	// 'VFNET': ['r50', 'r101', 'x101']}

	model_size_remap := map[string]map[string]string{
		"YOLOV3":    {"S": "X", "M": "X", "L": "X", "XL": "X", "XXL": "X"},
		"YOLOV5":    {"S": "N", "M": "S", "L": "M", "XL": "L", "XXL": "X"},
		"YOLOVM":    {"S": "N", "M": "S", "L": "M", "XL": "L", "XXL": "L"},
		"YOLOV7":    {"S": "L", "M": "L", "L": "L", "XL": "X", "XXL": "X"},
		"YOLOV8":    {"S": "N", "M": "S", "L": "M", "XL": "L", "XXL": "X"},
		"DETR":      {"S": "50", "M": "50", "L": "101", "XL": "101", "XXL": "101"},
		"FASTERRCNN": {"S": "r50", "M": "r50", "L": "r101", "XL": "r101", "XXL": "x101"},
		"SPARSERCNN": {"S": "r50100", "M": "r50100", "L": "r101100", "XL": "r101100", "XXL": "r101300"},
		"RETINANET":  {"S": "r50", "M": "r50", "L": "r101", "XL": "r101", "XXL": "x101"},
		"VFNET":      {"S": "r50", "M": "r50", "L": "r101", "XL": "r101", "XXL": "x101"},
	}

	ModelSize = model_size_remap[ModelType][ModelSize]

	cmd := exec.Command(
		"python", "-W ignore", "./pylib/Detectors/"+strings.ToLower(detectName)+".py",
		DataRoot,
		strconv.Itoa(BatchSize),
		strconv.Itoa(ScaleDownResolution[0]), strconv.Itoa(ScaleDownResolution[1]),
		strconv.FormatFloat(Threshold, 'f', 4, 64), strings.Join(classList, ","),
		DataName, ModelSize, deviceID,
	)
	if strings.ToLower(detectName) == "yolov8" {
		cmd.Env = append(os.Environ(), "CUDA_VISIBLE_DEVICES="+deviceID)
	}
	// fmt.Println(
	// 	"python", "-W ignore", "./pylib/Detectors/"+strings.ToLower(detectName)+".py",
	// 	DataRoot,
	// 	strconv.Itoa(BatchSize),
	// 	strconv.Itoa(ScaleDownResolution[0]), strconv.Itoa(ScaleDownResolution[1]),
	// 	strconv.FormatFloat(Threshold, 'f', 4, 64), strings.Join(classList, ","),
	// 	DataName, ModelSize, deviceID,
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
	return &Detector{
		cmd:       cmd,
		stdin:     stdin,
		rd:        rd,
		batchSize: BatchSize,
		origDims:  [2]int{ScaleDownResolution[0], ScaleDownResolution[1]},
		zeroImage: make([]byte, ScaleDownResolution[0]*ScaleDownResolution[1]*3),
	}
}

func (yolo *Detector) Detect(images []Image) [][]Detection {
	yolo.mu.Lock()

	header := make([]byte, 16) // 4 bytes for each int32: bytes size; width; height; batch size
	binary.BigEndian.PutUint32(header[0:4], uint32(len(images[0].Bytes)*yolo.batchSize))
	binary.BigEndian.PutUint32(header[4:8], uint32(images[0].Width))
	binary.BigEndian.PutUint32(header[8:12], uint32(images[0].Height))
	binary.BigEndian.PutUint32(header[12:16], uint32(yolo.batchSize))
	yolo.stdin.Write(header)

	for _, im := range images {
		yolo.stdin.Write(im.Bytes)
	}
	for i := len(images); i < yolo.batchSize; i++ {
		yolo.stdin.Write(yolo.zeroImage)
	}

	var line string
	for {
		var err error
		line, err = yolo.rd.ReadString('\n')
		if err != nil {
			// fmt.Println("Error reading from detector")
			// fmt.Println("line", line)
			// fmt.Println(err)
			panic(err)
		}
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "json") {
			continue
		}
		break
	}
	yolo.mu.Unlock()
	jsonBytes := []byte(line[4:])
	var detections [][]Detection
	if err := json.Unmarshal(jsonBytes, &detections); err != nil {
		panic(err)
	}
	// fmt.Println("detections", detections)

	return detections[0:len(images)]
}

func (yolo *Detector) Close() {
	yolo.stdin.Close()
	yolo.cmd.Wait()
}
