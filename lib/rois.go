package lib

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
)

// if im.Width != Resolution[0] || im.Height != Resolution[1] {
// 	fmt.Println("ImagePreprocess: resize")
// 	im = *im.Resize(Resolution[0], Resolution[1])
// }

type RoiModel struct {
	cmd       *exec.Cmd
	stdin     io.WriteCloser
	rd        *bufio.Reader
	mu        sync.Mutex
	batchSize int
	zeroImage []byte
}

type Window struct {
	Bounds [4]int
	Cells  [][4]int
}

func NewRoiModel(
	BatchSize int,
	Resolution [2]int,
	ScaleDownResolution [2]int,
	Threshold float64,
	ModelType string,
	RawWindowSizes [][2]int,
	DeviceID int,
	WeightRoot string,
	DataName string) *RoiModel {
	var cmd *exec.Cmd
	deviceID := strconv.Itoa(DeviceID)
	if ModelType == "CNN" {
		WindowSizes := make([][2]int, len(RawWindowSizes))
		for i, windowSize := range RawWindowSizes {
			reshap_window := [2]int{}
			reshap_window[0] = windowSize[0] * ScaleDownResolution[0] / 640 / 32 * 32
			reshap_window[1] = windowSize[1] * ScaleDownResolution[1] / 352 / 32 * 32
			WindowSizes[i] = reshap_window
		}
		cmd = exec.Command(
			"python", "-W ignore", "./pylib/Rois/CNN/inference.py",
			strconv.Itoa(BatchSize),
			strconv.Itoa(Resolution[0]),
			strconv.Itoa(Resolution[1]),
			strconv.FormatFloat(Threshold, 'f', 4, 64),
			strconv.Itoa(ScaleDownResolution[0]),
			strconv.Itoa(ScaleDownResolution[1]),
			string(JsonMarshal(WindowSizes)),
			filepath.Join(WeightRoot, "Rois", DataName, fmt.Sprintf("CNN_%d_%d.pth", Resolution[0], Resolution[1])),
			deviceID,
		)
	}
	stdin, err := cmd.StdinPipe()
	if err != nil {
		panic(err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		panic(err)
	}
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		panic(err)
	}
	rd := bufio.NewReader(stdout)
	return &RoiModel{
		cmd:       cmd,
		stdin:     stdin,
		rd:        rd,
		batchSize: BatchSize,
		zeroImage: make([]byte, Resolution[0]*Resolution[1]*3),
	}
}

func (m *RoiModel) GetWindows(images []Image) [][]Window {
	m.mu.Lock()
	for _, im := range images {
		m.stdin.Write(im.Bytes)
	}
	for i := len(images); i < m.batchSize; i++ {
		m.stdin.Write(m.zeroImage)
	}
	var line string
	for {
		var err error
		line, err = m.rd.ReadString('\n')
		if err != nil {
			panic(err)
		}
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "json") {
			continue
		}
		break
	}
	m.mu.Unlock()
	jsonBytes := []byte(line[4:])
	var windows [][]Window
	if err := json.Unmarshal(jsonBytes, &windows); err != nil {
		panic(err)
	}
	return windows[0:len(images)]
}

func (m *RoiModel) Close() {
	m.stdin.Close()
	m.cmd.Wait()
}

func ImagePreprocess(im Image, EnhanceTools []string, EnhanceToolInfos []float64, Resolution [2]int) Image {
	var enhancetools = make(map[string]bool)
	var enhancetoolinfos = EnhanceToolInfos
	for _, enhancetoolname := range EnhanceTools {
		enhancetools[strings.ToLower(enhancetoolname)] = true
	}
	im.Enhance(enhancetools, enhancetoolinfos)

	return im
}
