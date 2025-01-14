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

type FrameFilterModel struct {
	// FrameFilterModel is the model of the frame filter.
	cmd       *exec.Cmd
	stdin     io.WriteCloser
	rd        *bufio.Reader
	mu        sync.Mutex
	batchSize int
	zeroImage []byte
}

func NewFrameFilterModel(
	ModelType string,
	BatchSize int,
	ScaleDownResolution [2]int,
	Resolution [2]int,
	Threshold float64,
	DeviceID int,
	WeightRoot string,
	DataName string,
) *FrameFilterModel {
	// NewFrameFilterModel creates a new FrameFilterModel.
	var cmd *exec.Cmd
	deviceID := strconv.Itoa(DeviceID)
	if ModelType == "CNN" {
		cmd = exec.Command(
			"python", "-W ignore", "./pylib/Filters/CNN/inference.py",
			strconv.Itoa(BatchSize),
			strconv.Itoa(Resolution[0]),
			strconv.Itoa(Resolution[1]),
			strconv.Itoa(ScaleDownResolution[0]),
			strconv.Itoa(ScaleDownResolution[1]),
			strconv.FormatFloat(Threshold, 'f', 4, 64),
			filepath.Join(WeightRoot, "Filters", DataName, fmt.Sprintf("CNN_%d_%d.pth", Resolution[0], Resolution[1])),
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
	return &FrameFilterModel{
		cmd:       cmd,
		stdin:     stdin,
		rd:        rd,
		batchSize: 1,
		zeroImage: make([]byte, ScaleDownResolution[0]*ScaleDownResolution[1]*3),
	}
}

func (m *FrameFilterModel) Filter(images []Image) []int {
	// Filter filters the images.
	m.mu.Lock()
	var err error
	for _, image := range images {
		// fmt.Println(image.Width, image.Height)
		if _, err = m.stdin.Write(image.Bytes); err != nil {
			panic(err)
		}
	}
	for i := len(images); i < m.batchSize; i++ {
		if _, err = m.stdin.Write(m.zeroImage); err != nil {
			panic(err)
		}
	}
	// fmt.Println(images[0].Width, images[0].Height)
	var line string
	for {
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
	var result []int
	if err := json.Unmarshal(jsonBytes, &result); err != nil {
		panic(err)
	}
	// fmt.Println(result)
	return result
}

func (m *FrameFilterModel) Close() {
	// Close closes the model.
	m.stdin.Close()
	m.cmd.Wait()
}
