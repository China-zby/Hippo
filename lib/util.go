package lib

import (
	"bytes"
	"crypto/md5"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

	"gopkg.in/yaml.v2"
)

func IsContain(items []string, item string) bool {
	for _, eachItem := range items {
		if eachItem == item {
			return true
		}
	}
	return false
}

func Sum(x []float64) float64 {
	var sum float64
	for _, x := range x {
		sum += x
	}
	return sum
}

func Mean(x []float64) float64 {
	var sum float64
	for _, x := range x {
		sum += x
	}
	return sum / float64(len(x))
}

func MaxInt(x, y int) int {
	if x > y {
		return x
	}
	return y
}

func MinInt(x, y int) int {
	if x < y {
		return x
	}
	return y
}

func ParseInt(s string) int {
	x, err := strconv.Atoi(s)
	if err != nil {
		panic(err)
	}
	return x
}

func JsonMarshal(x interface{}) []byte {
	bytes, err := json.Marshal(x)
	if err != nil {
		panic(err)
	}
	return bytes
}

func JsonUnmarshal(bytes []byte, x interface{}) {
	err := json.Unmarshal(bytes, x)
	if err != nil {
		panic(err)
	}
}

func ReadJsonFile(fname string, x interface{}) {
	bytes, err := os.ReadFile(fname)
	if err != nil {
		panic(err)
	}
	JsonUnmarshal(bytes, x)
}

func FloatsMean(floats []float64) float64 {
	var sum float64
	for _, x := range floats {
		sum += x
	}
	return sum / float64(len(floats))
}

// Returns the sample standard deviation.
func FloatsStddev(floats []float64) float64 {
	mean := FloatsMean(floats)
	var sqdevSum float64
	for _, x := range floats {
		sqdevSum += (x - mean) * (x - mean)
	}
	return math.Sqrt(sqdevSum / float64(len(floats)-1))
}

func FloatsStderr(floats []float64) float64 {
	return FloatsStddev(floats) / math.Sqrt(float64(len(floats)))
}

func FloatsMax(floats []float64) float64 {
	max := floats[0]
	for _, x := range floats {
		if x > max {
			max = x
		}
	}
	return max
}

func FloatsMin(floats []float64) float64 {
	min := floats[0]
	for _, x := range floats {
		if x < min {
			min = x
		}
	}
	return min
}

func FloatsMaxIndex(floats []float64) int {
	max := floats[0]
	maxIndex := 0
	for i, x := range floats {
		if x > max {
			max = x
			maxIndex = i
		}
	}
	return maxIndex
}

func FloatsMinIndex(floats []float64) int {
	min := floats[0]
	minIndex := 0
	for i, x := range floats {
		if x < min {
			min = x
			minIndex = i
		}
	}
	return minIndex
}

func RandomIndex(length int) int {
	return rand.Intn(length)
}

func RandomMAXMINIndex(min, max int) int {
	return rand.Intn(max-min) + min
}

func GetFloatList(length int,
	value float64) []float64 {
	floatlists := make([]float64, length)
	for i := range floatlists {
		floatlists[i] = value
	}
	return floatlists
}

func GetBoolList(length int,
	value bool) []bool {
	boolLists := make([]bool, length)
	for i := range boolLists {
		boolLists[i] = value
	}
	return boolLists
}

func SaveYaml(config Config, save_path string) {
	// Marshal Config struct into YAML data
	yamlData, err := yaml.Marshal(config)
	if err != nil {
		log.Fatalf("Error marshaling YAML data: %v", err)
	}

	// Write YAML data to file
	err = os.WriteFile(save_path, yamlData, 0644)
	if err != nil {
		log.Fatalf("Error writing YAML file: %v", err)
	}
}

func GetArgMin(x []float64) int {
	min := x[0]
	minIndex := 0
	for i, v := range x {
		if v < min {
			min = v
			minIndex = i
		}
	}
	return minIndex
}

// generatePoints creates n equally spaced points between start and end (inclusive).
func generatePoints(start, end float64, n int) []float64 {
	if n <= 1 {
		return []float64{start}
	}

	step := (end - start) / float64(n-1)
	points := make([]float64, n)
	for i := 0; i < n; i++ {
		points[i] = start + float64(i)*step
	}

	return points
}

func cleandir(dir string) {
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// 如果是一个文件则删除
		if !info.IsDir() {
			err := os.Remove(path)
			if err != nil {
				log.Println(err)
			}
		}

		return nil
	})

	if err != nil {
		log.Println(err)
	}
}

func Set(list []int) []int {
	uniqueElements := make(map[int]bool)
	for _, num := range list {
		uniqueElements[num] = true
	}
	uniqueElementsList := []int{}
	for k := range uniqueElements {
		uniqueElementsList = append(uniqueElementsList, k)
	}
	return uniqueElementsList
}

type CameraInfo struct {
	SceneName   string
	Width       int
	Height      int
	Fps         int
	FrameNumber int
	Duration    int
}

func SaveJsonFile(pathName string, jsondata []byte) {
	err := os.WriteFile(pathName, jsondata, 0644)
	if err != nil {
		fmt.Println("Error writing to file:", err)
	}
}

func extractElements(list []int, q int) [][]int {
	var result [][]int

	for i := 0; i < len(list); i += q {
		end := i + q
		if end > len(list) {
			end = len(list)
		}

		result = append(result, list[i:end])
	}

	return result
}

func getPythonGPUMemory(pid int) (string, error) {
	cmd := exec.Command("nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv")
	var out bytes.Buffer
	cmd.Stdout = &out
	err := cmd.Run()
	if err != nil {
		return "", err
	}

	pidStr := fmt.Sprintf("%d", pid)
	for _, line := range strings.Split(out.String(), "\n") {
		if strings.Contains(line, pidStr) {
			parts := strings.Split(line, ",")
			if len(parts) > 1 {
				return strings.TrimSpace(parts[1]), nil
			}
		}
	}
	return "0 MiB", nil
}

func replaceSuffix(s, oldSuffix, newSuffix string) string {
	if strings.HasSuffix(s, oldSuffix) {
		return strings.TrimSuffix(s, oldSuffix) + newSuffix
	}
	return s
}

func md5Hash(input string) string {
	h := md5.New()
	io.WriteString(h, input)
	return fmt.Sprintf("%x", h.Sum(nil))
}
