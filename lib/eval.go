package lib

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/fatih/color"
	"github.com/k0kubun/go-ansi"
	"github.com/mitchellh/colorstring"
	"github.com/schollz/progressbar/v3"
)

type EvalResult struct {
	Car struct {
		Recall      float64 `json:"recall_mean"`
		Precision   float64 `json:"precision_mean"`
		Accuracy    float64 `json:"accuracy_mean"`
		F1          float64 `json:"f1_mean"`
		MAE         float64 `json:"mae_mean"`
		ACC         float64 `json:"acc_mean"`
		GT_COUNT    int     `json:"gt_count_mean"`
		PRED_COUNT  int     `json:"pred_count_mean"`
		TOPK        float64 `json:"acc_topk_mean"`
		LIMIT_F1    float64 `json:"cardinality_f1_mean"`
		LIMIT2_F1   float64 `json:"cardinality2_f1_mean"`
		LIMIT3_ACC  float64 `json:"cardinality3_acc_mean"`
		CRecall     float64 `json:"crecall_mean"`
		CPrecision  float64 `json:"cprecision_mean"`
		CAccuracy   float64 `json:"caccuracy_mean"`
		CF1         float64 `json:"cf1_mean"`
		CMAE        float64 `json:"cmae_mean"`
		CACC        float64 `json:"cacc_mean"`
		CGT_COUNT   int     `json:"cgt_count_mean"`
		CPRED_COUNT int     `json:"cpred_count_mean"`
		CTOPK       float64 `json:"cacc_topk_mean"`
		CLIMIT_F1   float64 `json:"ccardinality_f1_mean"`
	} `json:"car"`
	Bus struct {
		Recall      float64 `json:"recall_mean"`
		Precision   float64 `json:"precision_mean"`
		Accuracy    float64 `json:"accuracy_mean"`
		F1          float64 `json:"f1_mean"`
		MAE         float64 `json:"mae_mean"`
		ACC         float64 `json:"acc_mean"`
		GT_COUNT    int     `json:"gt_count_mean"`
		PRED_COUNT  int     `json:"pred_count_mean"`
		TOPK        float64 `json:"acc_topk_mean"`
		LIMIT_F1    float64 `json:"cardinality_f1_mean"`
		LIMIT2_F1   float64 `json:"cardinality2_f1_mean"`
		LIMIT3_ACC  float64 `json:"cardinality3_acc_mean"`
		CRecall     float64 `json:"crecall_mean"`
		CPrecision  float64 `json:"cprecision_mean"`
		CAccuracy   float64 `json:"caccuracy_mean"`
		CF1         float64 `json:"cf1_mean"`
		CMAE        float64 `json:"cmae_mean"`
		CACC        float64 `json:"cacc_mean"`
		CGT_COUNT   int     `json:"cgt_count_mean"`
		CPRED_COUNT int     `json:"cpred_count_mean"`
		CTOPK       float64 `json:"cacc_topk_mean"`
		CLIMIT_F1   float64 `json:"ccardinality_f1_mean"`
	} `json:"bus"`
	Truck struct {
		Recall      float64 `json:"recall_mean"`
		Precision   float64 `json:"precision_mean"`
		Accuracy    float64 `json:"accuracy_mean"`
		F1          float64 `json:"f1_mean"`
		MAE         float64 `json:"mae_mean"`
		ACC         float64 `json:"acc_mean"`
		GT_COUNT    int     `json:"gt_count_mean"`
		PRED_COUNT  int     `json:"pred_count_mean"`
		TOPK        float64 `json:"acc_topk_mean"`
		LIMIT_F1    float64 `json:"cardinality_f1_mean"`
		LIMIT2_F1   float64 `json:"cardinality2_f1_mean"`
		LIMIT3_ACC  float64 `json:"cardinality3_acc_mean"`
		CRecall     float64 `json:"crecall_mean"`
		CPrecision  float64 `json:"cprecision_mean"`
		CAccuracy   float64 `json:"caccuracy_mean"`
		CF1         float64 `json:"cf1_mean"`
		CMAE        float64 `json:"cmae_mean"`
		CACC        float64 `json:"cacc_mean"`
		CGT_COUNT   int     `json:"cgt_count_mean"`
		CPRED_COUNT int     `json:"cpred_count_mean"`
		CTOPK       float64 `json:"cacc_topk_mean"`
		CLIMIT_F1   float64 `json:"ccardinality_f1_mean"`
	} `json:"truck"`
}

type EvalHotaResult struct {
	Hota_Car   float64 `json:"hota_car"`
	Hota_Bus   float64 `json:"hota_bus"`
	Hota_Truck float64 `json:"hota_truck"`

	Mota_Car   float64 `json:"mota_car"`
	Mota_Bus   float64 `json:"mota_bus"`
	Mota_Truck float64 `json:"mota_truck"`

	Idf1_Car   float64 `json:"idf1_car"`
	Idf1_Bus   float64 `json:"idf1_bus"`
	Idf1_Truck float64 `json:"idf1_truck"`

	Motp_Car   float64 `json:"motp_car"`
	Motp_Bus   float64 `json:"motp_bus"`
	Motp_Truck float64 `json:"motp_truck"`
}

type Vehicle struct {
	trackid     int
	classid     int
	framebounds [2]int
}

type VehicleNumberResult struct {
	trackids       []int
	vehicleNumbers []int
}

func EvalFilter(FilterResult map[[2]int]int, gtDir string) float64 {
	// read gt
	FilterGroundTruth := make(map[[2]int]int)
	fnames, err := ioutil.ReadDir(gtDir)

	if err != nil {
		fmt.Printf("Error 4")
		panic(err)
	}

	bar := progressbar.NewOptions(len(fnames),
		progressbar.OptionSetWriter(ansi.NewAnsiStdout()),
		progressbar.OptionEnableColorCodes(true),
		progressbar.OptionShowBytes(true),
		progressbar.OptionSetWidth(15),
		progressbar.OptionSetDescription("[cyan][EF][reset] Evaluate Filter"),
		progressbar.OptionSetTheme(progressbar.Theme{
			Saucer:        "[green]=[reset]",
			SaucerHead:    "[green]>[reset]",
			SaucerPadding: " ",
			BarStart:      "[",
			BarEnd:        "]",
		}))

	for _, fi := range fnames {
		if !strings.HasSuffix(fi.Name(), ".json") {
			bar.Add(1)
			continue
		}
		videoID, _ := strconv.ParseInt(strings.Split(fi.Name(), ".")[0], 10, 0)
		bytes, err := ioutil.ReadFile(path.Join(gtDir, fi.Name()))
		if err != nil {
			fmt.Printf("Error 2")
			panic(err)
		}
		var detections [][]Detection
		if err := json.Unmarshal(bytes, &detections); err != nil {
			fmt.Printf("Error 3")
			panic(err)
		}
		for frameID, frameDetections := range detections {
			if frameDetections == nil {
				FilterGroundTruth[[2]int{int(videoID), frameID}] = 0
				continue
			}
			if len(frameDetections) == 0 {
				FilterGroundTruth[[2]int{int(videoID), frameID}] = 0
			} else {
				FilterGroundTruth[[2]int{int(videoID), frameID}] = 1
			}
		}
		bar.Add(1)
	}
	// Calculate the number of true positives, false positives, and false negatives.
	var tp, tn, fp, fn int
	for key, value := range FilterResult {
		if FilterGroundTruth[key] == 1 {
			if value == 1 {
				tp++
			} else {
				fn++
			}
		} else {
			if value == 1 {
				fp++
			} else {
				tn++
			}
		}
	}
	color.Output = ansi.NewAnsiStdout()
	// color.Cyan("fatih/color")
	colorstring.Printf("\nTrue positives: [green]%d, ", tp)
	colorstring.Printf("False negatives: [red]%d, ", fn)
	colorstring.Printf("True negatives: [green]%d, ", tn)
	colorstring.Printf("False positives: [red]%d", fp)

	// colorstring.Fprintln(ansi.NewAnsiStdout(), "True positives: ") // [green]mitchellh
	// Output the results with red and green colors.
	// output := fmt.Sprintf("\nTrue positives: @g{(%d)} True negatives: @g{(%d)} False positives: @r{(%d)} False negatives: @r{(%d)}", tp, tn, fp, fn)
	// ansi.Printf("%s", output)
	// Calculate precision, recall, and F1 score.
	if tp+fp == 0 {
		return 0
	}
	if tp+fn == 0 {
		return 0
	}
	beta := 2.0
	precision := float64(tp) / float64(tp+fp)
	recall := float64(tp) / float64(tp+fn)
	accuracy := float64(tp+tn) / float64(tp+tn+fp+fn)
	// f1 := 2 * precision * recall / (precision + recall)
	fbeta := (1 + beta*beta) * precision * recall / ((beta*beta)*precision + recall)
	colorstring.Printf("\nPrecision: [green]%f, ", precision)
	colorstring.Printf("Recall: [green]%f, ", recall)
	colorstring.Printf("F1: [green]%f, ", fbeta)
	colorstring.Printf("Accuracy: [green]%f\n", accuracy)

	return fbeta
}

// func EvalRoi(RoiResult map[[2]int][][4]int, gtDir string, cfg Config) float64 {
// 	// read gt
// 	RoiGroundTruth := make(map[[2]int][][2]int)
// 	fnames, err := ioutil.ReadDir(gtDir)

// 	if err != nil {
// 		fmt.Printf("Error 4")
// 		panic(err)
// 	}

// 	bar := progressbar.NewOptions(len(fnames),
// 		progressbar.OptionSetWriter(ansi.NewAnsiStdout()),
// 		progressbar.OptionEnableColorCodes(true),
// 		progressbar.OptionShowBytes(true),
// 		progressbar.OptionSetWidth(15),
// 		progressbar.OptionSetDescription("[cyan][ER][reset] Evaluate Roi Model"),
// 		progressbar.OptionSetTheme(progressbar.Theme{
// 			Saucer:        "[green]=[reset]",
// 			SaucerHead:    "[green]>[reset]",
// 			SaucerPadding: " ",
// 			BarStart:      "[",
// 			BarEnd:        "]",
// 		}))

// 	for _, fi := range fnames {
// 		if !strings.HasSuffix(fi.Name(), ".json") {
// 			bar.Add(1)
// 			continue
// 		}
// 		videoID, _ := strconv.ParseInt(strings.Split(fi.Name(), ".")[0], 10, 0)
// 		bytes, err := ioutil.ReadFile(path.Join(gtDir, fi.Name()))
// 		if err != nil {
// 			fmt.Printf("Error 2")
// 			panic(err)
// 		}
// 		var detections [][]Detection
// 		if err := json.Unmarshal(bytes, &detections); err != nil {
// 			fmt.Printf("Error 3")
// 			panic(err)
// 		}
// 		for frameID, frameDetections := range detections {
// 			if frameDetections == nil {
// 				continue
// 			}
// 			if len(frameDetections) == 0 {
// 				continue
// 			} else {
// 				for _, detection := range frameDetections {
// 					original_center_point := detection.GetCenter()
// 					center_point := [2]int{int(float64(original_center_point[0]) / float64(cfg.VideoBase.Resolution[0]) * float64(cfg.VideoBase.ScaleDownResolution[0])), int(float64(original_center_point[1]) / float64(cfg.VideoBase.Resolution[1]) * float64(cfg.VideoBase.ScaleDownResolution[1]))}
// 					RoiGroundTruth[[2]int{int(videoID), frameID}] = append(RoiGroundTruth[[2]int{int(videoID), frameID}], center_point)
// 				}
// 			}
// 		}
// 		bar.Add(1)
// 	}
// 	// Calculate the hit number of roi model
// 	var tp, fp, fn int
// 	for key, value := range RoiResult {
// 		// fmt.Println(key, value)
// 		if len(RoiGroundTruth[key]) == 0 {
// 			fp += len(value)
// 			continue
// 		}
// 		hit_gt_idx := make([]bool, len(RoiGroundTruth[key]))
// 		hit_window_idx := make([]bool, len(value))
// 		for gi, gt := range RoiGroundTruth[key] {
// 			for vi, roi := range value {
// 				if roi[0] <= gt[0] && roi[2] >= gt[0] && roi[1] <= gt[1] && roi[3] >= gt[1] && !hit_window_idx[vi] {
// 					hit_window_idx[vi] = true
// 					hit_gt_idx[gi] = true
// 					break
// 				}
// 			}
// 		}
// 		for _, hit := range hit_window_idx {
// 			if hit {
// 				tp++
// 			} else {
// 				fp++
// 			}
// 		}
// 		for _, hit := range hit_gt_idx {
// 			if !hit {
// 				fn++
// 			}
// 		}
// 	}

// 	colorstring.Printf("\nTrue positives: [green]%d, ", tp)
// 	colorstring.Printf("False negatives: [red]%d, ", fn)
// 	colorstring.Printf("False positives: [red]%d", fp)

// 	if tp+fp == 0 {
// 		return 0
// 	}
// 	if tp+fn == 0 {
// 		return 0
// 	}
// 	beta := 2.0
// 	precision := float64(tp) / float64(tp+fp)
// 	recall := float64(tp) / float64(tp+fn)
// 	// f1 := 2 * precision * recall / (precision + recall)
// 	// f2 := (1 + 2*2) * precision * recall / (2*2*precision + recall)
// 	fbeta := (1 + beta*beta) * precision * recall / ((beta*beta)*precision + recall)

// 	colorstring.Printf("\nPrecision: [green]%f, ", precision)
// 	colorstring.Printf("Recall: [green]%f, ", recall)
// 	colorstring.Printf("F1: [green]%f\n", fbeta)
// 	return fbeta
// }

func FilterNoiseTracks(detections [][]Detection) [][]TrackDetection {
	tracks := GetTracks(detections)

	// filter noise tracks
	// Create a regular expression pattern to match the desired format
	// pattern := regexp.MustCompile(`S(\d+)>(\d+)S\|T(\d+)>(\d+)T`)
	// matches := pattern.FindAllStringSubmatch(NoiseType, -1)[0]

	// fmt.Println("matches:", matches)
	// if len(matches) != 5 {
	// 	err := fmt.Errorf("input does not match the expected format")
	// 	panic(err)
	// }
	// fmt.Println("matches:", matches)

	// Convert matched strings to integers and append to the respective slices
	// spatialBound := []float64{100.0, 99999999.0}
	temporalBound := []float64{60.0, 99999999.0}
	// for i := 1; i < len(matches); i++ {
	// 	value, err := strconv.ParseFloat(matches[i], 64)
	// 	if err != nil {
	// 		panic(err)
	// 	}

	// 	if i < 3 {
	// 		spatialBound = append(spatialBound, value)
	// 	} else {
	// 		temporalBound = append(temporalBound, value)
	// 	}
	// }
	// fmt.Println("Frame Spatial Bound:", spatialBound, "Frame Temporal Bound:", temporalBound)

	goodTracks := make([][]TrackDetection, 0)
	for _, track := range tracks {
		duration := float64(track[len(track)-1].FrameIdx - track[0].FrameIdx)
		// distance := track[0].Center().Distance(track[len(track)-1].Center())
		// if distance < spatialBound[0] || distance > spatialBound[1] {
		// 	continue
		// }
		if duration < temporalBound[0] || duration > temporalBound[1] {
			continue
		}
		goodTracks = append(goodTracks, track)
	}
	return goodTracks
}

func EvalMOTA(dataRoot string, cfg Config, validtest string, path string) (float64, float64) {
	getMOTA := func(fname1 string, fname2 string) float64 {
		cmd := exec.Command("python", "../data-scripts/compute-mota.py", fname1, fname2, "16")
		bytes, err := cmd.CombinedOutput()
		fmt.Println(string(bytes))
		if err != nil {
			panic(err)
		}
		var acc float64
		for _, line := range strings.Split(string(bytes), "\n") {
			parts := strings.Fields(line)
			if parts[0] == "acc" {
				acc, _ = strconv.ParseFloat(parts[1], 64)
				break
			}
		}
		return acc
	}

	var accs []float64
	for _, idx := range []int{3, 7, 8} {
		fname1 := path + fmt.Sprintf("%d.json", idx)
		fname2 := filepath.Join(dataRoot, "dataset", cfg.DataBase.DataName, validtest+"-mota", fmt.Sprintf("%d.json", idx))
		acc := getMOTA(fname1, fname2)
		accs = append(accs, acc)
	}
	return FloatsMean(accs), FloatsStderr(accs)
}

func MakeGtsandPreds(cfg Config, videoid int, frameNumber int, outDetections [][]Detection) (map[int]VehicleNumberResult, map[int]VehicleNumberResult) {
	// get ground truth
	var classids = [3]int{2, 5, 7}
	var classid_dict = map[string]int{"car": 2, "bus": 5, "truck": 7}
	var gt_vehicles = make(map[int]Vehicle)
	var gt_vehicleNumberResults = make(map[int]VehicleNumberResult)

	for _, classid := range classids {
		if gt_vehicleNumberResult, ok := gt_vehicleNumberResults[classid]; !ok {
			gt_vehicleNumberResult = VehicleNumberResult{trackids: make([]int, 0), vehicleNumbers: make([]int, frameNumber)}
			gt_vehicleNumberResults[classid] = gt_vehicleNumberResult
		} // gt_vehicleNumberResults[classid].vehicleNumbers[i] = 0
	}

	gtFile := filepath.Join(cfg.DataBase.DataRoot, "dataset", cfg.MethodName, cfg.DataBase.DataName, "labels", strconv.Itoa(videoid)+".txt")
	bytes, _ := ioutil.ReadFile(gtFile)
	for _, line := range strings.Split(string(bytes), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			// fmt.Println("line", line, gtFile, videoid)
			// os.Exit(1)
			continue
		}
		// fmt.Println("line", line, gtFile)
		parts := strings.Split(line, ",") // frameid,trackid,Left,Top,Width,Height,Score,ClassID,_,_,_
		// fmt.Println("parts", parts[0], parts[1], parts[7], parts[7][0:1])
		frameid := ParseInt(parts[0])
		trackid := ParseInt(parts[1])
		ClassID := ParseInt(parts[7][0:1])
		// _, ok := gt_vehicles[frameid]

		if gt_vehicle, ok := gt_vehicles[trackid]; ok {
			gt_vehicle.framebounds[0] = MinInt(gt_vehicle.framebounds[0], frameid)
			gt_vehicle.framebounds[1] = MaxInt(gt_vehicle.framebounds[1], frameid)
			gt_vehicles[trackid] = gt_vehicle
		} else {
			gt_vehicle = Vehicle{trackid: trackid, classid: ClassID, framebounds: [2]int{frameid, frameid}}
			gt_vehicles[trackid] = gt_vehicle
		}
	}

	for key, value := range gt_vehicles {
		if gt_vehicleNumberResult, ok := gt_vehicleNumberResults[value.classid]; ok {
			gt_vehicleNumberResult.trackids = append(gt_vehicleNumberResult.trackids, key)
			for j := value.framebounds[0]; j <= value.framebounds[1]; j++ {
				gt_vehicleNumberResult.vehicleNumbers[j] += 1
			}
			gt_vehicleNumberResults[value.classid] = gt_vehicleNumberResult
		}
	}

	// get predictions
	// var pred_vehicles map[int][]Vehicle
	var pred_vehicleNumberResults = make(map[int]VehicleNumberResult)
	for _, classid := range classids {
		if pred_vehicleNumberResult, ok := pred_vehicleNumberResults[classid]; !ok {
			pred_vehicleNumberResult = VehicleNumberResult{trackids: make([]int, 0), vehicleNumbers: make([]int, frameNumber)}
			pred_vehicleNumberResults[classid] = pred_vehicleNumberResult
		}
	}
	// fmt.Println("outDetections", len(outDetections))
	tracks := GetTracks(outDetections)
	for _, track := range tracks {
		if len(track) >= 1 {
			if pred_vehicleNumberResult, ok := pred_vehicleNumberResults[classid_dict[track[0].Class]]; ok {
				pred_vehicleNumberResult.trackids = append(pred_vehicleNumberResult.trackids, *track[0].TrackID)
				for j := track[0].FrameIdx; j <= track[len(track)-1].FrameIdx; j++ {
					pred_vehicleNumberResult.vehicleNumbers[j] += 1
				}
				pred_vehicleNumberResults[classid_dict[track[0].Class]] = pred_vehicleNumberResult
			}
		}
	}
	// fmt.Println("pred_vehicleNumberResults", pred_vehicleNumberResults)
	return gt_vehicleNumberResults, pred_vehicleNumberResults
}

func EvalQuery(cfg Config, videoid int, frameNumber int, outDetections [][]Detection) ([]float64, []float64, []float64, []int, []int) {
	gt_vehicleNumberResults, pred_vehicleNumberResults := MakeGtsandPreds(cfg, videoid, frameNumber, outDetections)
	var select_Q1_accs []float64
	var aggregation_Q1_maes []float64
	var aggregation_Q1_accs []float64
	var aggregation_Q3_gts []int
	var aggregation_Q3_preds []int
	for _, classid := range [3]int{2, 5, 7} {
		gt_vehicleNumberResult := gt_vehicleNumberResults[classid]
		pred_vehicleNumberResult := pred_vehicleNumberResults[classid]
		s_q1_acc := EvalSelectQ1(gt_vehicleNumberResult, pred_vehicleNumberResult)
		a_q1_mae, a_q1_acc := EvalAggregationQ1(gt_vehicleNumberResult, pred_vehicleNumberResult)
		a_q3_gt, a_q3_pred := EvalAggregationQ3(gt_vehicleNumberResult, pred_vehicleNumberResult)
		select_Q1_accs = append(select_Q1_accs, s_q1_acc)
		aggregation_Q1_maes = append(aggregation_Q1_maes, a_q1_mae)
		aggregation_Q1_accs = append(aggregation_Q1_accs, a_q1_acc)
		aggregation_Q3_gts = append(aggregation_Q3_gts, a_q3_gt)
		aggregation_Q3_preds = append(aggregation_Q3_preds, a_q3_pred)
	}
	return select_Q1_accs, aggregation_Q1_maes, aggregation_Q1_accs, aggregation_Q3_gts, aggregation_Q3_preds
}

func EvalSelectQ1(gt_vehicleNumberResult VehicleNumberResult, pred_vehicleNumberResult VehicleNumberResult) float64 {
	var TP, FP, TN, FN int = 0, 0, 0, 0
	for i := 0; i < len(gt_vehicleNumberResult.vehicleNumbers); i++ {
		if gt_vehicleNumberResult.vehicleNumbers[i] >= 1 && pred_vehicleNumberResult.vehicleNumbers[i] >= 1 {
			TP += 1
		} else if gt_vehicleNumberResult.vehicleNumbers[i] >= 1 && pred_vehicleNumberResult.vehicleNumbers[i] == 0 {
			FN += 1
		} else if gt_vehicleNumberResult.vehicleNumbers[i] == 0 && pred_vehicleNumberResult.vehicleNumbers[i] >= 1 {
			FP += 1
		} else if gt_vehicleNumberResult.vehicleNumbers[i] == 0 && pred_vehicleNumberResult.vehicleNumbers[i] == 0 {
			TN += 1
		}
	}
	// acc := float64(TP+TN)/float64(TP+TN+FP+FN)
	var F1 float64
	if 2*TP+FP+FN == 0 {
		F1 = 1.0
	} else {
		F1 = float64(2*TP) / float64(2*TP+FP+FN)
	}
	return F1
}

func EvalAggregationQ1(gt_vehicleNumberResult VehicleNumberResult, pred_vehicleNumberResult VehicleNumberResult) (float64, float64) {
	var mae, acc float64 = 0.0, 0.0
	for i := 0; i < len(gt_vehicleNumberResult.vehicleNumbers); i++ {
		mae += math.Abs(float64(gt_vehicleNumberResult.vehicleNumbers[i]) - float64(pred_vehicleNumberResult.vehicleNumbers[i]))
		if gt_vehicleNumberResult.vehicleNumbers[i] != 0 {
			acc += 1.0 - (math.Abs(float64(pred_vehicleNumberResult.vehicleNumbers[i])-float64(gt_vehicleNumberResult.vehicleNumbers[i])) / float64(gt_vehicleNumberResult.vehicleNumbers[i]))
		} else {
			if pred_vehicleNumberResult.vehicleNumbers[i] != 0 {
				acc += 0.0
			} else {
				acc += 1.0
			}
		}
	}
	mae = mae / float64(len(gt_vehicleNumberResult.vehicleNumbers))
	acc = acc / float64(len(gt_vehicleNumberResult.vehicleNumbers))
	return mae, acc
}

func EvalAggregationQ3(gt_vehicleNumberResult VehicleNumberResult, pred_vehicleNumberResult VehicleNumberResult) (int, int) {
	var gt_count, pred_count int = len(gt_vehicleNumberResult.trackids), len(pred_vehicleNumberResult.trackids)
	return gt_count, pred_count
}

func EvalAllMetric(cfg Config, outDir string, videoids []int, pipelineids []int, InferenceTime float64) EvalResult {
	var skipnumberstrs []string
	var videoidstrs []string
	for idx, vid := range videoids {
		videoidstrs = append(videoidstrs, strconv.Itoa(vid))
		skipnumberstrs = append(skipnumberstrs, strconv.Itoa(cfg.VideoBase.SkipNumber[pipelineids[idx]]))
	}
	pipeline_number := len(cfg.VideoBase.SkipNumber)
	skipframelist := strings.Join(skipnumberstrs, "-")
	videoidlist := strings.Join(videoidstrs, "-")

	videoPath := filepath.Join(cfg.DataBase.DataRoot, "dataset", cfg.DataBase.DataName, cfg.DataBase.DataType, "video/")

	result_dir := fmt.Sprintf("./result/%s/%s", cfg.MethodName, cfg.DataBase.DataName)
	input_gt_dir := filepath.Join(cfg.DataBase.DataRoot, "dataset", cfg.DataBase.DataName, cfg.DataBase.DataType, "tracks")
	data_dir := filepath.Join(cfg.DataBase.DataRoot, "dataset", cfg.DataBase.DataName)

	cmd := exec.Command("python", "./pylib/utils/otif2mot.py",
		"--input_dir", outDir,
		"--input_gt_dir", input_gt_dir,
		"--data_root", data_dir,
		"--input_video_dir", videoPath,
		"--method_name", cfg.MethodName,
		"--dataset_name", cfg.DataBase.DataName,
		"--videoidlist", videoidlist,
		"--skipframelist", skipframelist,
		"--testmode", cfg.DataBase.DataType,
		"--width", fmt.Sprintf("%d", cfg.VideoBase.Resolution[0]),
		"--height", fmt.Sprintf("%d", cfg.VideoBase.Resolution[1]),
		"--differentclass",
		"--classes", strings.Join(cfg.DetectBase.Classes, ","))

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout // 标准输出
	cmd.Stderr = &stderr // 标准错误
	cmd.Run()

	// fmt.Printf("python ./pylib/utils/otif2mot.py out:\n%s\npython ./pylib/utils/otif2mot.py err:\n%s", stdout.String(), stderr.String())

	var hResult EvalHotaResult
	var Result EvalResult
	for pipej := 0; pipej < pipeline_number; pipej++ {
		skipframe := cfg.VideoBase.SkipNumber[pipej]
		evalcmd := exec.Command("python", "./TrackEval/scripts/run_videodb_chellenge.py",
			"--BENCHMARK", cfg.DataBase.DataName+"S"+strconv.Itoa(skipframe),
			"--PRINT_RESULTS", "False",
			"--PRINT_CONFIG", "False",
			"--TIME_PROGRESS", "False",
			"--OUTPUT_DETAILED", "True",
			"--OUTPUT_SUMMARY", "True",
			"--SPLIT_TO_EVAL", cfg.DataBase.DataType,
			"--classes", strings.Join(cfg.DetectBase.Classes, ","))

		stdin_rqm, err := evalcmd.StdinPipe()
		if err != nil {
			panic(err)
		}
		packet := map[string]string{
			"run_type": "start",
		}
		bytes := JsonMarshal(packet)
		header := make([]byte, 4)
		binary.BigEndian.PutUint32(header, uint32(len(bytes)))
		stdin_rqm.Write(header)
		stdin_rqm.Write(bytes)
		stdout_rqm, err := evalcmd.StdoutPipe()
		if err != nil {
			panic(err)
		}
		evalcmd.Stderr = os.Stderr
		result_output := bufio.NewReader(stdout_rqm)
		evalcmd.Start()

		var line string
		for {
			var err error
			line, err = result_output.ReadString('\n')

			if err != nil {
				panic(err)
			}
			line = strings.TrimSpace(line)
			if !strings.HasPrefix(line, "json") {
				continue
			}
			break
		}

		jsonBytes := []byte(line[4:])
		if err := json.Unmarshal(jsonBytes, &hResult); err != nil {
			panic(err)
		}

		// fmt.Println("python ./TrackEval/scripts/run_videodb_chellenge.py out:\n%s\npython ./TrackEval/scripts/run_videodb_chellenge.py err:\n%s", stdout.String(), stderr.String())
	}

	evalcmd2 := exec.Command("python", "./pylib/evaluation/run_query_matrics.py",
		"--data_root", data_dir,
		"--testmode", cfg.DataBase.DataType,
		"--gtdir", input_gt_dir,
		"--videoidlist", videoidlist,
		"--skipframelist", skipframelist,
		"--dataset_name", cfg.DataBase.DataName,
		"--method_name", cfg.MethodName,
		"--filter_gt",
		"--classes", strings.Join(cfg.DetectBase.Classes, ","),
	)

	stdin_rqm, err := evalcmd2.StdinPipe()
	if err != nil {
		panic(err)
	}
	packet := map[string]string{
		"run_type": "start",
	}
	bytes := JsonMarshal(packet)
	header := make([]byte, 4)
	binary.BigEndian.PutUint32(header, uint32(len(bytes)))
	stdin_rqm.Write(header)
	stdin_rqm.Write(bytes)
	stdout_rqm, err := evalcmd2.StdoutPipe()
	if err != nil {
		panic(err)
	}
	if true {
		cmd.Stderr = os.Stderr
	}
	result_output := bufio.NewReader(stdout_rqm)
	evalcmd2.Start()

	var line string
	for {
		var err error
		line, err = result_output.ReadString('\n')

		if err != nil {
			panic(err)
		}
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "json") {
			continue
		}
		break
	}

	jsonBytes := []byte(line[4:])
	if err := json.Unmarshal(jsonBytes, &Result); err != nil {
		panic(err)
	}

	printcmd := exec.Command("python", "./pylib/utils/print_results.py",
		"--resultdir", result_dir,
		"--classes", strings.Join(cfg.DetectBase.Classes, ","),
		"--methodname", cfg.MethodName,
		"--datasetname", cfg.DataBase.DataName,
		"--videoidlist", videoidlist,
		"--skipframelist", skipframelist,
		"--testmode", cfg.DataBase.DataType,
		"--qeurytime", fmt.Sprintf("%f", InferenceTime),
	)

	printcmd.Stdout = &stdout // 标准输出
	printcmd.Stderr = &stderr // 标准错误
	err = printcmd.Run()
	outStr, errStr := stdout.String(), stderr.String()
	if errStr != "" {
		fmt.Printf("python ./pylib/utils/print_results.py out:\n%s\npython ./pylib/utils/print_results.py err:\n%s", outStr, errStr)
	} else {
		fmt.Printf("%s", outStr)
	}
	if err != nil {
		log.Fatalf("printcmd.Run() failed with %s\n", err)
	}
	return Result
}
