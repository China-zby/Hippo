package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
	"videotune/lib"
)

type StepOutput struct {
	Latency float64

	CarSel        float64
	CarAgg        float64
	CarGtCount    int
	CarPredCount  int
	CarTopk       float64
	CarCQ1        float64
	CarCQ2        float64
	CarCQ3        float64
	CarCSel       float64
	CarCAgg       float64
	CarCGtCount   int
	CarCPredCount int
	CarCTopk      float64
	CarCCQ1       float64

	BusSel        float64
	BusAgg        float64
	BusGtCount    int
	BusPredCount  int
	BusTopk       float64
	BusCQ1        float64
	BusCQ2        float64
	BusCQ3        float64
	BusCSel       float64
	BusCAgg       float64
	BusCGtCount   int
	BusCPredCount int
	BusCTopk      float64
	BusCCQ1       float64

	TruckSel        float64
	TruckAgg        float64
	TruckGtCount    int
	TruckPredCount  int
	TruckTopk       float64
	TruckCQ1        float64
	TruckCQ2        float64
	TruckCQ3        float64
	TruckCSel       float64
	TruckCAgg       float64
	TruckCGtCount   int
	TruckCPredCount int
	TruckCTopk      float64
	TruckCCQ1       float64

	CarHota   float64
	BusHota   float64
	TruckHota float64

	CarMota   float64
	BusMota   float64
	TruckMota float64

	CarIdf1   float64
	BusIdf1   float64
	TruckIdf1 float64

	CarMotp   float64
	BusMotp   float64
	TruckMotp float64
}

func main() {
	bytes, err := io.ReadAll(os.Stdin)
	if err != nil {
		fmt.Println("Read Failed:", err)
		return
	}
	var config lib.JsonConfig
	if err := json.Unmarshal(bytes, &config); err != nil {
		fmt.Println("Encoded Failed:", err)
		return
	}

	fmt.Println("Read configuration successfully:", config)
	save_path := os.Args[1]
	videoidstr := os.Args[2]

	parts := strings.Split(videoidstr, ",")

	var videoids []int

	for _, part := range parts {
		i, err := strconv.Atoi(part)
		if err != nil {
			fmt.Println(err)
			return
		}
		videoids = append(videoids, i)
	}

	outDir := fmt.Sprintf("%s/%s/%s/", config.LogBase.SaveRoot, config.DataBase.DataName, config.MethodName)
	logDir := fmt.Sprintf("%s/%s/%s/", config.LogBase.LogRoot, config.DataBase.DataName, config.MethodName)

	os.MkdirAll(outDir, 0755)
	os.MkdirAll(logDir, 0755)

	unixTimestamp := time.Now().Unix()
	timestampTime := time.Unix(unixTimestamp, 0)
	formattedTime := timestampTime.Format(time.RFC3339)
	logPath := filepath.Join(logDir, fmt.Sprintf("%s.log", formattedTime))
	config.EvaluateBase.EvaluateState = false
	config.Visualize = false
	query_time, mot_result, query_result, _, GPURecord := lib.Step(config, outDir, logPath, videoids, []int{0})

	step_output := StepOutput{
		query_time,

		query_result.Car.F1,
		query_result.Car.ACC,
		query_result.Car.GT_COUNT,
		query_result.Car.PRED_COUNT,
		query_result.Car.TOPK,
		query_result.Car.LIMIT_F1,
		query_result.Car.LIMIT2_F1,
		query_result.Car.LIMIT3_ACC,

		query_result.Car.CF1,
		query_result.Car.CACC,
		query_result.Car.CGT_COUNT, query_result.Car.CPRED_COUNT,
		query_result.Car.CTOPK,
		query_result.Car.CLIMIT_F1,

		query_result.Bus.F1,
		query_result.Bus.ACC,
		query_result.Bus.GT_COUNT, query_result.Bus.PRED_COUNT,
		query_result.Bus.TOPK,
		query_result.Bus.LIMIT_F1,
		query_result.Bus.LIMIT2_F1,
		query_result.Bus.LIMIT3_ACC,
		query_result.Bus.CF1,
		query_result.Bus.CACC,
		query_result.Bus.CGT_COUNT, query_result.Bus.CPRED_COUNT,
		query_result.Bus.CTOPK,
		query_result.Bus.CLIMIT_F1,

		query_result.Truck.F1,
		query_result.Truck.ACC,
		query_result.Truck.GT_COUNT, query_result.Truck.PRED_COUNT,
		query_result.Truck.TOPK,
		query_result.Truck.LIMIT_F1,
		query_result.Truck.LIMIT2_F1,
		query_result.Truck.LIMIT3_ACC,
		query_result.Truck.CF1,
		query_result.Truck.CACC,
		query_result.Truck.CGT_COUNT, query_result.Truck.CPRED_COUNT,
		query_result.Truck.CTOPK,
		query_result.Truck.CLIMIT_F1,

		mot_result.Hota_Car,
		mot_result.Hota_Bus,
		mot_result.Hota_Truck,

		mot_result.Mota_Car,
		mot_result.Mota_Bus,
		mot_result.Mota_Truck,

		mot_result.Idf1_Car,
		mot_result.Idf1_Bus,
		mot_result.Idf1_Truck,

		mot_result.Motp_Car,
		mot_result.Motp_Bus,
		mot_result.Motp_Truck,
	}

	jsonData, err := json.Marshal(step_output)
	if err != nil {
		fmt.Println("Error marshaling data:", err)
		return
	}

	err = os.WriteFile(save_path, jsonData, 0644)
	if err != nil {
		log.Fatalf("Error writing JSON data to file: %v", err)
	}
	fmt.Println("Saved stepRecords as JSON to:", save_path)

	// save GPURecord
	if config.Record {
		jsonData, err = json.Marshal(GPURecord)
		if err != nil {
			fmt.Println("Error marshaling data:", err)
			return
		}

		save_record_path := filepath.Dir(save_path) + "/GPURecord.json"
		err = os.WriteFile(save_record_path, jsonData, 0644)
		if err != nil {
			log.Fatalf("Error writing JSON data to file: %v", err)
		}
	}
}
