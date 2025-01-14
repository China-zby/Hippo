package main

import (
	"encoding/json"
	"fmt"
	"os"
	"otifpipeline/lib"
	"path/filepath"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

func main() {
	configRoot := os.Args[1]
	saveRoot := os.Args[2]

	if _, err := os.Stat(saveRoot); os.IsNotExist(err) {
		os.Mkdir(saveRoot, 0777)
	}

	cfg := lib.GetConfig(configRoot)
	var filtermodelsize [2]int
	filtermodelsizes := [][2]int{
		{160, 96},
		{224, 128},
		{320, 192},
		{416, 256},
		{640, 352},
	}
	thresholds := []float64{0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99} //
	max_descent_round, threadCount := 10, 5
	FindResult := make(map[string]float64)
	for _, filtermodelsize = range filtermodelsizes {
		best_f1, best_threshold, descent_round := 0.0, -1.0, 0
		xs, ys := make([]float64, 0), make([]float64, 0)
		cfg.FilterBase.Flag = true
		cfg.VideoBase.SkipNumber = 32
		cfg.FilterBase.Resolution = filtermodelsize
		for _, threshold := range thresholds {
			cfg.FilterBase.Threshold = threshold
			f1 := lib.FilterPart(cfg, threadCount)
			if f1 >= best_f1 {
				best_f1 = f1
				best_threshold = threshold
				descent_round = 0
			} else {
				descent_round += 1
			}
			xs = append(xs, threshold)
			ys = append(ys, f1)
			if descent_round > max_descent_round {
				break
			}
		}

		p := plot.New()

		p.Title.Text = "Find the best threshold for " + fmt.Sprintf("%dx%d", filtermodelsize[0], filtermodelsize[1])
		p.X.Label.Text = "Threshold"
		p.Y.Label.Text = "F1"

		data := make(plotter.XYs, len(xs))
		for i := range data {
			data[i].X = xs[i]
			data[i].Y = ys[i]
		}

		err := plotutil.AddLinePoints(p, "Sample Data", data)
		if err != nil {
			fmt.Println("Error:", err)
			return
		}

		savePath := filepath.Join(saveRoot, fmt.Sprintf("%dx%d", filtermodelsize[0], filtermodelsize[1])+"_threshold.png") // saveRoot + fmt.Sprintf("%dx%d", filtermodelsize[0], filtermodelsize[1]) + "_threshold.png"

		if err := p.Save(12*vg.Inch, 10*vg.Inch, savePath); err != nil {
			fmt.Println("Error:", err)
			return
		}

		fmt.Printf("Line graph saved to '%s'.", fmt.Sprintf("%dx%d", filtermodelsize[0], filtermodelsize[1])+"_threshold.png")
		FindResult[fmt.Sprintf("%d_%d", filtermodelsize[0], filtermodelsize[1])] = best_threshold
	}
	jsonData, _ := json.Marshal(FindResult)
	fmt.Println(string(jsonData))
	saveResultPath := filepath.Join(saveRoot, "find_result.json")
	file, err := os.Create(saveResultPath)
	if err != nil {
		fmt.Println("Error creating file:", err)
		return
	}
	defer file.Close()

	_, err = file.Write(jsonData)
	if err != nil {
		fmt.Println("Error writing to file:", err)
		return
	}

	fmt.Println("JSON data successfully written to " + saveResultPath)
}
