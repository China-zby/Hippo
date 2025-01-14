package lib

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

func Step(cfg JsonConfig, outDir string, logPath string, cameraIds []int, pipelineids []int) (float64, EvalHotaResult, EvalResult, [][][14]float64, map[int]string) {
	cleandir(outDir)
	scene2id := map[string]int{
		"amsterdam": 0,
		"warsaw":    1, "shibuya": 2,
		"jackson": 3, "caldot1": 4,
		"caldot2": 5, "uav": 6,
	}

	FilterResult := make(map[[2]int]int)
	RoiResult := make(map[[2]int][][4]int)
	WindowResult := make(map[int]int)

	MethodName := cfg.MethodName
	DataName := cfg.DataBase.DataName
	DataRoot := cfg.DataBase.DataRoot
	DataType := cfg.DataBase.DataType
	WeightRoot := cfg.LogBase.WeightRoot
	SceneName := cfg.LogBase.Scene
	GPURecord := make(map[int]string)

	clusters := make(map[string][]Cluster, 0)
	if cfg.PostProcessBase.PostProcessType == "OTIF" {
		trainSet := make(map[string][][]TrackDetection)
		trainDir := filepath.Join(DataRoot, "dataset", DataName, "train/tracks/")
		infoDir := filepath.Join(DataRoot, "dataset", DataName, "train/info/")

		files, err := os.ReadDir(trainDir)
		if err != nil {
			fmt.Printf("Error 1")
			panic(err)
		}
		var jsonFiles []string
		for _, filename := range files {
			if strings.HasSuffix(filename.Name(), ".json") {
				jsonFiles = append(jsonFiles, filename.Name())
			}
		}

		// filenumber := len(jsonFiles)
		// clusterbar := progressbar.Default(int64(filenumber))
		for _, fi := range jsonFiles {
			if !strings.HasSuffix(fi, ".json") {
				continue
			}
			videoid := strings.Split(fi, ".json")[0]
			infofi := filepath.Join(infoDir, fmt.Sprintf("%s.txt", videoid))
			bytes, err := os.ReadFile(filepath.Join(trainDir, fi))
			if err != nil {
				fmt.Printf("Error 2")
				panic(err)
			}
			var detections [][]Detection
			if err := json.Unmarshal(bytes, &detections); err != nil {
				fmt.Printf("Error 3")
				panic(err)
			}

			infofile, err := os.Open(infofi)
			if err != nil {
				panic(err)
			}
			defer infofile.Close()
			scanner := bufio.NewScanner(infofile)
			scanner.Scan()
			line := scanner.Text()
			parts := strings.Split(line, "-")
			scenename := parts[0]

			detections = FilterDetectionsByClass(detections, cfg.DetectBase.Classes)
			tracks := GetTracks(detections)
			trainSet[scenename] = append(trainSet[scenename], tracks...)

			// clusterbar.Add(1)
		}
		for scenename, tracks := range trainSet {
			clusters[scenename], _ = ClusterTracks(tracks)
		}
	}

	var miris *Miris
	if cfg.PostProcessBase.PostProcessType == "MIRIS" {
		miris = NewMiris(cfg)
		fmt.Println("Miris model loaded")
		if cfg.Record {
			GPURecord[miris.cmd.Process.Pid] = "0 MiB"
		}
	}

	var durationSamples []float64

	videoPath := filepath.Join(DataRoot, "dataset", DataName, DataType, "video/")
	gtPath := filepath.Join(DataRoot, "dataset", DataName, DataType, "tracks/")

	var videoFnames []string
	for _, cameraId := range cameraIds {
		videoFnames = append(videoFnames, fmt.Sprintf("%d.mp4", cameraId))
	}

	var mu sync.Mutex
	respCond := sync.NewCond(&mu)
	type ModelMeta struct {
		Size      [3]int
		Cond      *sync.Cond
		BatchSize int
	}

	type PendingJob struct {
		// video ID and frame index
		ID       int
		FrameIdx int
		// cropped window
		Image    Image
		OriImage Image
		// for detect job: offset from topleft (to correct detection coordinates)
		Offset [2]int
		// for detect job: windows that should contain the detection
		Cells   [][4]int
		windows []Window
	}

	type DetectorOutput struct {
		// how many detector jobs are still needed
		Needed int
		// flat detection list
		Detections []Detection
		// enhanced image
		EnhanceImage Image
	}
	// model size -> pending jobs
	pendingJobs := make(map[[3]int][]PendingJob) // store pending jobs
	// (video ID, frame idx) -> priority
	priorities := make(map[[2]int]int) // store priority
	// model outputs
	detectOutputs := make(map[[2]int]*DetectorOutput) // store detection outputs
	// number of videos that are waiting for some frame to be processed
	var numWaiting int = 0     // the number of videos that are waiting for some frame to be processed
	done := false              // whether all videos have been processed
	var modelWg sync.WaitGroup // wait for all models to finish
	var modelMetas []ModelMeta // store model metas
	wakeupModels := func() {   // wake up models
		for _, meta := range modelMetas {
			if len(pendingJobs[meta.Size]) < meta.BatchSize && (numWaiting < 1 || len(pendingJobs[meta.Size]) == 0) && !done {
				continue
			}
			meta.Cond.Broadcast()
		}
	}
	modelLoop := func(sz [3]int, batchSize int, f func([]PendingJob), cleanup func()) {
		cond := sync.NewCond(&mu)
		modelMetas = append(modelMetas, ModelMeta{
			Size:      sz,
			Cond:      cond,
			BatchSize: batchSize,
		})
		modelWg.Add(1)
		go func() {
			defer modelWg.Done()
			defer cleanup()
			for {
				mu.Lock()
				for len(pendingJobs[sz]) < batchSize && (numWaiting < 1 || len(pendingJobs[sz]) == 0) && !done {
					cond.Wait()
				}
				if done {
					mu.Unlock()
					break
				}

				// pick the jobs with lowest (most) priority
				sort.Slice(pendingJobs[sz], func(i, j int) bool {
					job1 := pendingJobs[sz][i]
					job2 := pendingJobs[sz][j]
					prio1 := priorities[[2]int{job1.ID, job1.FrameIdx}]
					prio2 := priorities[[2]int{job2.ID, job2.FrameIdx}]
					return prio1 < prio2 || (prio1 == prio2 && job1.FrameIdx < job2.FrameIdx)
				})
				var jobs []PendingJob
				if len(pendingJobs[sz]) < batchSize {
					jobs = pendingJobs[sz]
					pendingJobs[sz] = nil
				} else {
					jobs = append([]PendingJob{}, pendingJobs[sz][0:batchSize]...)
					n := copy(pendingJobs[sz][0:], pendingJobs[sz][batchSize:])
					pendingJobs[sz] = pendingJobs[sz][0:n]
				}

				mu.Unlock()
				f(jobs)
			}
		}()
	}

	RoiFlag := cfg.RoiBase.Flag
	FilterFlag := cfg.FilterBase.Flag
	EnhanceTools := []string{}
	if cfg.RoiBase.DenoisingFlag {
		EnhanceTools = append(EnhanceTools, "denoisingflag")
	}
	if cfg.RoiBase.EqualizationFlag {
		EnhanceTools = append(EnhanceTools, "equalizationflag")
	}
	if cfg.RoiBase.SharpeningFlag {
		EnhanceTools = append(EnhanceTools, "sharpeningflag")
	}
	if cfg.RoiBase.SaturationFlag {
		EnhanceTools = append(EnhanceTools, "saturationflag")
	}
	EnhanceToolInfos := cfg.RoiBase.EnhanceToolInfos
	SkipNumber := cfg.VideoBase.SkipNumber

	ScaleDownResolution := cfg.VideoBase.ScaleDownResolution

	jobChannels := getJobChannels(0, FilterFlag, RoiFlag)

	if FilterFlag {
		FilterModelType := cfg.FilterBase.ModelType
		FilterBatchSize := cfg.FilterBase.BatchSize
		FilterThreshold := cfg.FilterBase.Threshold
		FilterResolution := cfg.FilterBase.Resolution
		EvaluateFilterFlag := cfg.EvaluateBase.EvaluateFilter
		filtermodel := NewFrameFilterModel(
			FilterModelType,
			FilterBatchSize,
			ScaleDownResolution,
			FilterResolution,
			FilterThreshold,
			cfg.DeviceID,
			WeightRoot,
			SceneName)
		if cfg.Record {
			GPURecord[filtermodel.cmd.Process.Pid] = "0 MiB"
		}
		filterprocess := func(jobs []PendingJob) {
			var images []Image
			for _, job := range jobs {
				images = append(images, job.Image)
			}

			filterflags := filtermodel.Filter(images)
			mu.Lock()
			for i, filterflag := range filterflags {
				if EvaluateFilterFlag {
					FilterResult[[2]int{jobs[i].ID, jobs[i].FrameIdx}] = filterflag
				}

				if filterflag == 1 {
					job := jobs[i]
					pendingJobs[jobChannels.framefilterOutJobName] = append(pendingJobs[jobChannels.framefilterOutJobName], PendingJob{
						ID:       job.ID,
						FrameIdx: job.FrameIdx,
						Image:    job.Image,
					})
				} else {
					detectOutputs[[2]int{jobs[i].ID, jobs[i].FrameIdx}] = &DetectorOutput{
						Needed:       0,
						Detections:   []Detection{},
						EnhanceImage: jobs[i].Image,
					}
				}
			}
			wakeupModels()
			respCond.Broadcast()
			mu.Unlock()
		}
		filtercleanup := func() { filtermodel.Close() }
		modelLoop(jobChannels.framefilterInJobName, FilterBatchSize, filterprocess, filtercleanup)
	}

	if RoiFlag {
		RoiBatchSize := cfg.RoiBase.BatchSize
		RoiThreshold := cfg.RoiBase.Threshold
		RoiModelType := cfg.RoiBase.ModelType
		RoiWindowSizes := cfg.RoiBase.WindowSizes
		RoiResolution := cfg.RoiBase.Resolution
		EvaluateRoiFlag := cfg.EvaluateBase.EvaluateRoi
		seg := NewRoiModel(
			RoiBatchSize,
			RoiResolution,
			ScaleDownResolution,
			RoiThreshold,
			RoiModelType,
			RoiWindowSizes,
			cfg.DeviceID,
			WeightRoot,
			SceneName)
		if cfg.Record {
			GPURecord[seg.cmd.Process.Pid] = "0 MiB"
		}
		segprocess := func(jobs []PendingJob) {
			var images []Image
			for _, job := range jobs {
				if job.Image.Width != RoiResolution[0] || job.Image.Height != RoiResolution[1] {
					RjobImage := job.Image.Resize(RoiResolution[0], RoiResolution[1])
					images = append(images, *RjobImage)
				} else {
					images = append(images, job.Image)
				}
			}

			windows := seg.GetWindows(images)

			mu.Lock()
			for i, job := range jobs {
				if EvaluateRoiFlag {
					RoiBounds := make([][4]int, 0)
					for _, window := range windows[i] {
						RoiBounds = append(RoiBounds, window.Cells...)
					}
					RoiResult[[2]int{job.ID, job.FrameIdx}] = RoiBounds
					WindowResult[len(windows[i])] += 1
				}
				pendingJobs[jobChannels.roiOutJobName] = append(pendingJobs[jobChannels.roiOutJobName], PendingJob{
					ID:       job.ID,
					FrameIdx: job.FrameIdx,
					Image:    job.Image,
					windows:  windows[i],
				})
			}
			wakeupModels()
			respCond.Broadcast() // needed in case any jobs had no windows output
			mu.Unlock()
		}

		segcleanup := func() { seg.Close() }
		modelLoop(jobChannels.roiInJobName, RoiBatchSize, segprocess, segcleanup) // segprocess is called in a goroutine
	}

	enhanceprocess := func(jobs []PendingJob) {
		mu.Lock()
		for _, job := range jobs {
			if !RoiFlag {
				detectOutputs[[2]int{job.ID, job.FrameIdx}] = &DetectorOutput{
					Needed: 1,
				}
				enhanceImage := ImagePreprocess(job.Image, EnhanceTools, EnhanceToolInfos, ScaleDownResolution)
				pendingJobs[jobChannels.enhanceOutJobName] = append(pendingJobs[jobChannels.enhanceOutJobName], PendingJob{
					ID:       job.ID,
					FrameIdx: job.FrameIdx,
					Image:    enhanceImage,
					Offset:   [2]int{0, 0},
					Cells:    nil,
				})
			} else {
				windows := job.windows
				detectOutputs[[2]int{job.ID, job.FrameIdx}] = &DetectorOutput{
					Needed:     len(windows),
					Detections: []Detection{},
				}
				enhanceImage := ImagePreprocess(job.Image, EnhanceTools, EnhanceToolInfos, ScaleDownResolution)
				for _, window := range windows {
					crop := enhanceImage.Crop(window.Bounds[0], window.Bounds[1], window.Bounds[2], window.Bounds[3])
					pendingJobs[jobChannels.enhanceOutJobName] = append(pendingJobs[jobChannels.enhanceOutJobName], PendingJob{
						ID:       job.ID,
						FrameIdx: job.FrameIdx,
						Image:    crop,
						OriImage: enhanceImage,
						Offset:   [2]int{window.Bounds[0], window.Bounds[1]},
						Cells:    window.Cells,
					})
				}
			}
		}
		wakeupModels()
		respCond.Broadcast()
		mu.Unlock()
	}
	enhancecleanup := func() {}
	modelLoop(jobChannels.enhanceInJobName, cfg.RoiBase.BatchSize, enhanceprocess, enhancecleanup)

	DetectClasses := cfg.DetectBase.Classes
	DetectModelType := cfg.DetectBase.ModelType
	DetectBatchSize := cfg.DetectBase.BatchSize
	DetectModelSize := cfg.DetectBase.ModelSize
	DetectThreshold := cfg.DetectBase.Threshold
	detector := NewDetector(
		DetectModelType,
		DetectBatchSize,
		DetectModelSize,
		DetectThreshold,
		DetectClasses,
		cfg.DeviceID,
		ScaleDownResolution,
		DataRoot,
		SceneName)
	if cfg.Record {
		GPURecord[detector.cmd.Process.Pid] = "0 MiB"
	}
	detector_process := func(jobs []PendingJob) {
		var images []Image
		for _, job := range jobs {
			images = append(images, job.Image)
		}

		outputs := detector.Detect(images)

		mu.Lock()
		for i, job := range jobs {
			dlist := []Detection{}
			for _, d := range outputs[i] {
				d.Left = d.Left + job.Offset[0]
				d.Top = d.Top + job.Offset[1]
				d.Right = d.Right + job.Offset[0]
				d.Bottom = d.Bottom + job.Offset[1]

				if job.Cells != nil && len(job.Cells) > 0 {
					good := false
					cx, cy := (d.Left+d.Right)/2, (d.Top+d.Bottom)/2
					for _, cell := range job.Cells {
						if cx < cell[0] || cx >= cell[2] || cy < cell[1] || cy >= cell[3] {
							continue
						}
						good = true
						break
					}
					if !good {
						continue
					}
				}
				dlist = append(dlist, d)
			}

			output := detectOutputs[[2]int{job.ID, job.FrameIdx}]
			output.Detections = append(output.Detections, dlist...)
			if len(job.OriImage.Bytes) == 0 {
				output.EnhanceImage = job.Image
			} else {
				output.EnhanceImage = job.OriImage
			}
			output.Needed--
		}
		respCond.Broadcast()
		mu.Unlock()
	}
	detector_cleanup := func() { detector.Close() }
	modelLoop(jobChannels.detectInJobName, DetectBatchSize, detector_process, detector_cleanup)

	TrackModelType := cfg.TrackBase.ModelType
	TrackHashSize := cfg.TrackBase.HashSize
	TrackLowFeatureDistanceThreshold := cfg.TrackBase.LowFeatureDistanceThreshold
	TrackMaxLostTime := cfg.TrackBase.MaxLostTime
	TrackMoveThreshold := cfg.TrackBase.MoveThreshold
	TrackKeepThreshold := cfg.TrackBase.KeepThreshold
	TrackMinThreshold := cfg.DetectBase.Threshold
	TrackCreateObjectThreshold := cfg.TrackBase.CreateObjectThreshold
	TrackMatchLocationThreshold := cfg.TrackBase.MatchLocationThreshold
	TrackUnmatchLocationThreshold := cfg.TrackBase.UnmatchLocationThreshold
	TrackVisualThreshold := cfg.TrackBase.VisualThreshold
	TrackKFPosWeight := cfg.TrackBase.KFPosWeight
	TrackKFVelWeight := cfg.TrackBase.KFVelWeight
	tracker := NewTracker(
		cfg.DeviceID,
		SkipNumber,
		TrackModelType,
		WeightRoot,
		DataName,
		TrackHashSize,
		TrackLowFeatureDistanceThreshold,
		TrackMaxLostTime,
		TrackMoveThreshold,
		TrackKeepThreshold,
		TrackMinThreshold,
		TrackCreateObjectThreshold,
		TrackMatchLocationThreshold,
		TrackUnmatchLocationThreshold,
		TrackVisualThreshold,
		TrackKFPosWeight,
		TrackKFVelWeight)

	if cfg.Record {
		GPURecord[tracker.cmd.Process.Pid] = "0 MiB"
	}

	var wg sync.WaitGroup
	var videostates [][][14]float64
	var metrics [3]map[string][]float64
	var skipnumbers []string
	var videoids []string

	for i := 0; i < len(metrics); i++ {
		metrics[i] = make(map[string][]float64)
	}

	camerainfos := make([]CameraInfo, 0)
	// for _, fname := range videoFnames {
	for i := 0; i < len(videoFnames); i++ {
		// fname := filepath.Join(videoPath, fname)
		// infofname := strings.ReplaceAll(fname, "video", "info")
		// infofname = strings.ReplaceAll(infofname, "mp4", "txt")
		// infofile, err := os.Open(infofname)
		// if err != nil {
		// 	panic(err)
		// }
		// defer infofile.Close()
		// scanner := bufio.NewScanner(infofile)
		// scanner.Scan()
		// line := scanner.Text()
		// parts := strings.Split(line, "-")
		scenename := "hippo" // parts[0]
		width := 720         // ParseInt(parts[1])
		height := 480        // ParseInt(parts[2])
		fps := 30            // ParseInt(parts[3])
		framenumber := 1800  // ParseInt(parts[4])
		duration := 60       // ParseInt(parts[5])

		camerainfos = append(camerainfos, CameraInfo{
			SceneName:   scenename,
			Width:       width,
			Height:      height,
			Fps:         fps,
			FrameNumber: framenumber,
			Duration:    duration,
		})
	}

	for vi, fname := range videoFnames {
		submitOutJobName := [3]int{0, 1, 1}
		skipnumbers = append(skipnumbers, strconv.Itoa(cfg.VideoBase.SkipNumber))
		videoids = append(videoids, strings.Split(fname, ".mp4")[0])
		Resolution := [2]int{camerainfos[vi].Width,
			camerainfos[vi].Height}

		sceneid := scene2id[camerainfos[vi].SceneName]
		wg.Add(1)
		go func(fname string, absoluteid int) {
			defer wg.Done()
			id := ParseInt(strings.Split(fname, ".mp4")[0])
			submitted := make(map[int]bool)
			detectorFunc := func(frameIdx int, im Image, extras []int, extraImages []Image) ([]Detection, Image) {
				submit := func(idx int, im Image, priority int) {
					priorities[[2]int{id, idx}] = priority
					if submitted[idx] {
						return
					}
					pendingJobs[submitOutJobName] = append(pendingJobs[submitOutJobName], PendingJob{
						ID:       id,
						FrameIdx: idx,
						Image:    im,
					})
					submitted[idx] = true
				}

				mu.Lock()
				defer mu.Unlock()

				submit(frameIdx, im, 0)
				for i, idx := range extras {
					submit(idx, extraImages[i], 1+i)
				}

				numWaiting++
				wakeupModels()

				k := [2]int{id, frameIdx}
				for detectOutputs[k] == nil || detectOutputs[k].Needed > 0 {
					respCond.Wait()
				}

				numWaiting--

				xFactor := float64(Resolution[0]) / float64(ScaleDownResolution[0])
				yFactor := float64(Resolution[1]) / float64(ScaleDownResolution[1])

				filteredDetectOutputs := []Detection{}
				for di := range detectOutputs[k].Detections {
					detection := &detectOutputs[k].Detections[di]
					resizeDetectionLeft := float64(detection.Left) * xFactor
					resizeDetectionTop := float64(detection.Top) * yFactor
					resizeDetectionRight := float64(detection.Right) * xFactor
					resizeDetectionBottom := float64(detection.Bottom) * yFactor
					detection.Left = int(resizeDetectionLeft)
					detection.Top = int(resizeDetectionTop)
					detection.Right = int(resizeDetectionRight)
					detection.Bottom = int(resizeDetectionBottom)
					if (detection.Right-detection.Left)*(detection.Bottom-detection.Top) < 500 {
						continue
					}
					filteredDetectOutputs = append(filteredDetectOutputs, *detection)
				}
				detectOutputs[k].Detections = filteredDetectOutputs

				return detectOutputs[k].Detections, detectOutputs[k].EnhanceImage
			}

			detections, states, start_time := StepLoop(id, tracker, filepath.Join(videoPath, fname), cfg,
				cfg.VideoBase.SkipNumber, Resolution,
				cfg.VideoBase.ScaleDownResolution, EnhanceTools, cfg.RoiBase.EnhanceToolInfos,
				detectorFunc, sceneid)
			if cfg.EvaluateBase.EvaluateState {
				videostates = append(videostates, states)
			}

			mu.Lock()
			numWaiting++
			wakeupModels()
			durationSamples = append(durationSamples, float64(time.Since(start_time).Seconds()))
			for pid := range GPURecord {
				old_gpu_record, _ := strconv.ParseInt(GPURecord[pid][:len(GPURecord[pid])-4], 10, 0)
				new_gpu_memory, _ := getPythonGPUMemory(pid)
				new_gpu_record, _ := strconv.ParseInt(new_gpu_memory[:len(new_gpu_memory)-4], 10, 0)
				if new_gpu_record > old_gpu_record {
					GPURecord[pid] = new_gpu_memory
				}
			}
			mu.Unlock()

			// filter out bad tracks
			// if cfg.NoiseFilterBase.Flag {
			goodTracks := FilterNoiseTracks(detections)
			detections = DetectionsFromTracks(goodTracks)
			// }

			// refine using clusters computed earlier if desired
			if cfg.PostProcessBase.PostProcessType == "OTIF" {
				tracks := GetTracks(detections)
				tracks = Postprocess(clusters[camerainfos[absoluteid].SceneName], tracks, cfg.VideoBase.SkipNumber)
				detections = DetectionsFromTracks(tracks)
			} else if cfg.PostProcessBase.PostProcessType == "MIRIS" {
				tracks := GetTracks(detections)
				if len(tracks) > 0 {
					tracks = PostprocessMIRIS(tracks, cfg.VideoBase.SkipNumber, camerainfos[absoluteid], miris)
				}
				detections = DetectionsFromTracks(tracks)
			}

			// save all of the detections
			bytes := JsonMarshal(detections)
			outFname := fmt.Sprintf("%s/%d.json", outDir, id)
			if err := os.WriteFile(outFname, bytes, 0644); err != nil {
				fmt.Printf("Error 5")
				panic(err)
			}
		}(fname, vi)
	}
	wg.Wait()

	if cfg.Record {
		fmt.Println("GPURecord: ", GPURecord)
	}

	mu.Lock()
	done = true
	wakeupModels()
	mu.Unlock()
	modelWg.Wait()
	tracker.Close()
	if cfg.PostProcessBase.PostProcessType == "MIRIS" {
		miris.Close()
	}
	var durationSum float64 = 0
	for _, duration := range durationSamples {
		durationSum += duration
	}

	processTime := float64(durationSum) / float64(len(durationSamples))

	// // 打开或创建输出文件
	// fileName := fmt.Sprintf("/home/lzp/go-work/src/videotune/outputs/hippo/test_time/duration_output.txt")
	// file1, err1 := os.OpenFile(fileName, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	// if err1 != nil {
	// 	fmt.Println("Error opening file:", err1)
	// }
	// defer file1.Close()

	// // 写入每个视频的处理时间到文件
	// for i, duration_ := range durationSamples {
	// 	output := fmt.Sprintf("Video %d processed in %.2f seconds\n", i+1, duration_)
	// 	if _, err := file1.WriteString(output); err != nil {
	// 		fmt.Println("Error writing to file:", err)
	// 	}
	// } //t

	if cfg.EvaluateBase.EvaluateFilter {
		EvalFilter(FilterResult, gtPath)
	}

	logStr := fmt.Sprintf("%v\t%v", cfg, processTime)
	file, err := os.OpenFile(logPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		panic(err)
	}
	if _, err := file.Write([]byte(logStr + "\n")); err != nil {
		panic(err)
	}
	file.Close()

	saveLogDir := fmt.Sprintf("./result/%s/%s", MethodName, DataName)
	os.MkdirAll(saveLogDir, 0755)

	skipframelist := strings.Join(skipnumbers, "-")
	videoidlist := strings.Join(videoids, "-")
	videoFlag := strings.Join(videoids, "")
	widthlist := []string{}
	heightlist := []string{}

	for vi, _ := range videoFnames {
		widthlist = append(widthlist, strconv.Itoa(camerainfos[vi].Width))
		heightlist = append(heightlist, strconv.Itoa(camerainfos[vi].Height))
	}

	resolutionwidthlist := strings.Join(widthlist, "-")
	resolutionheightlist := strings.Join(heightlist, "-")

	hashFlag := md5Hash(videoFlag)

	result_dir := fmt.Sprintf("./result/%s/%s", cfg.MethodName, cfg.DataBase.DataName)
	input_gt_dir := filepath.Join(DataRoot, "dataset", DataName, DataType, "tracks")
	data_dir := filepath.Join(DataRoot, "dataset", DataName)

	var cmd *exec.Cmd
	if len(DetectClasses) == 1 {
		cmd = exec.Command("python", "./pylib/utils/otif2mot.py",
			"--input_dir", outDir,
			"--input_gt_dir", input_gt_dir,
			"--data_root", data_dir,
			"--input_video_dir", videoPath,
			"--method_name", MethodName,
			"--dataset_name", DataName,
			"--videoidlist", videoidlist,
			"--skipframelist", skipframelist,
			"--testmode", cfg.DataBase.DataType,
			"--width", resolutionwidthlist,
			"--height", resolutionheightlist,
			"--classes", strings.Join(cfg.DetectBase.Classes, ","),
			"--flag", hashFlag)
	} else {
		cmd = exec.Command("python", "./pylib/utils/otif2mot.py",
			"--input_dir", outDir,
			"--input_gt_dir", input_gt_dir,
			"--data_root", data_dir,
			"--input_video_dir", videoPath,
			"--method_name", MethodName,
			"--dataset_name", DataName,
			"--videoidlist", videoidlist,
			"--skipframelist", skipframelist,
			"--testmode", cfg.DataBase.DataType,
			"--width", resolutionwidthlist,
			"--height", resolutionheightlist,
			"--differentclass",
			"--classes", strings.Join(cfg.DetectBase.Classes, ","),
			"--flag", hashFlag)
	}

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout // 标准输出
	cmd.Stderr = &stderr // 标准错误
	cmd.Run()

	var hResult EvalHotaResult
	var Result EvalResult
	skipframe := cfg.VideoBase.SkipNumber
	evalcmd := exec.Command("python", "./TrackEval/scripts/run_videodb_chellenge.py",
		"--BENCHMARK", DataName+"S"+strconv.Itoa(skipframe),
		"--PRINT_RESULTS", "False",
		"--PRINT_CONFIG", "False",
		"--TIME_PROGRESS", "False",
		"--OUTPUT_DETAILED", "True",
		"--OUTPUT_SUMMARY", "True",
		"--SPLIT_TO_EVAL", DataType,
		"--classes", strings.Join(cfg.DetectBase.Classes, ","),
		"--methodname", MethodName,
		"--TRACKERS_TO_EVAL", MethodName,
		"--SEQMAP_FILE", "./TrackEval/data/gt/videodb/seqmaps/"+DataName+"S"+strconv.Itoa(skipframe)+"-"+DataType+"_"+hashFlag+".txt")

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

	evalcmd2 := exec.Command("python", "./pylib/evaluation/run_query_matrics.py",
		"--data_root", data_dir,
		"--testmode", DataType,
		"--gtdir", input_gt_dir,
		"--videoidlist", videoidlist,
		"--skipframelist", skipframelist,
		"--dataset_name", DataName,
		"--method_name", MethodName,
		"--classes", strings.Join(cfg.DetectBase.Classes, ","),
	)

	stdin_rqm, err = evalcmd2.StdinPipe()
	if err != nil {
		panic(err)
	}
	packet = map[string]string{
		"run_type": "start",
	}
	bytes = JsonMarshal(packet)
	header = make([]byte, 4)
	binary.BigEndian.PutUint32(header, uint32(len(bytes)))
	stdin_rqm.Write(header)
	stdin_rqm.Write(bytes)
	stdout_rqm, err = evalcmd2.StdoutPipe()
	if err != nil {
		panic(err)
	}
	evalcmd2.Stderr = os.Stderr
	result_output = bufio.NewReader(stdout_rqm)
	evalcmd2.Start()

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

	jsonBytes = []byte(line[4:])
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
		"--testmode", DataType,
		"--qeurytime", fmt.Sprintf("%f", processTime),
	)

	printcmd.Stdout = &stdout // 标准输出
	printcmd.Stderr = &stderr // 标准错误
	err = printcmd.Run()
	outStr, errStr := stdout.String(), stderr.String()
	if errStr != "" {
		fmt.Printf("python ./pylib/utils/print_results.py out:\n%s\npython ./pylib/utils/print_results.py err:\n%s", outStr, errStr)
	} else {
		fmt.Printf("%s", outStr)
		file, err := os.OpenFile(logPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			panic(err)
		}
		if _, err := file.Write([]byte(outStr + "\n")); err != nil {
			panic(err)
		}
		file.Close()
	}
	if err != nil {
		log.Fatalf("printcmd.Run() failed with %s\n", err)
	}

	// if cfg.Visualize {
	// 	skipframe := cfg.VideoBase.SkipNumber
	// 	save_dir := fmt.Sprintf("./TrackEval/data/trackers/videodb/%s/%s/data", DataName+"S"+strconv.Itoa(skipframe)+"-"+DataType, MethodName)
	// 	save_gt_dir := fmt.Sprintf("./TrackEval/data/gt/videodb/%s", DataName+"S"+strconv.Itoa(skipframe)+"-"+DataType)
	// 	viscmd := exec.Command("python", "./pylib/utils/visualizeotiftracker.py",
	// 		"--testmode", DataType,
	// 		"--gtpath", save_gt_dir,
	// 		"--trackerpath", save_dir,
	// 		"--dataset_name", DataName,
	// 		"--skipframe", fmt.Sprintf("%d", skipframe),
	// 		"--dataroot", data_dir)
	// 	viscmd.Run()
	// 	viscmd.Stdout = &stdout
	// 	viscmd.Stderr = &stderr
	// 	fmt.Printf("Save Video To : %s/video_data", save_dir)
	// }
	return processTime, hResult, Result, videostates, GPURecord
}

func StepLoop(id int,
	tracker *Tracker,
	videoFname string,
	cfg JsonConfig, SkipNumber int, Resolution [2]int, ScaleDownResolution [2]int, EnhanceTools []string, EnhanceToolInfos []float64,
	detectorFunc func(frameIdx int, im Image, extras []int, extraImages []Image) ([]Detection, Image), sceneid int) ([][]Detection, [][14]float64, time.Time) {
	defer tracker.End(id + 1)

	skipFrame := SkipNumber
	minSkip, maxSkip := skipFrame, skipFrame

	var outDetections [][]Detection
	setDetections := func(frameIdx int, dlist []Detection) {
		for frameIdx >= len(outDetections) {
			outDetections = append(outDetections, []Detection{})
		}
		outDetections[frameIdx] = dlist
	}

	vreader := ReadFfmpeg(videoFname, ScaleDownResolution[0], ScaleDownResolution[1])
	vreader.Skip = minSkip
	defer vreader.Close()
	bfr := NewBufferedFfmpegReader(vreader, 192)

	im, done := bfr.GetFrame(0)
	start_time := time.Now()
	if done {
		return nil, nil, start_time
	}

	dlist, enhanceIm := detectorFunc(0, im, nil, nil)
	tracked, _ := tracker.Infer(id, 0, enhanceIm, dlist, Resolution, sceneid)
	setDetections(0, tracked)

	var states [][14]float64
	var last_tracked []Detection
	if cfg.EvaluateBase.EvaluateState {
		last_tracked = tracked
	}

	lastFrame := 0

	getDetections := func(frameIdx int, im Image) ([]Detection, Image) {
		var extras []int
		var extraImages []Image
		buffer, offset := bfr.GetBuffer()
		for i, extraIm := range buffer {
			idx := (offset + i) * minSkip
			if idx == frameIdx || idx%maxSkip != 0 {
				continue
			}
			extras = append(extras, idx)
			extraImages = append(extraImages, extraIm)
		}
		return detectorFunc(frameIdx, im, extras, extraImages)
	}

	for {
		updated := false
		curFrame := lastFrame + skipFrame

		im, done := bfr.GetFrame(curFrame / skipFrame)
		if done {
			break
		}

		dlist, enhanceIm := getDetections(curFrame, im)

		tracked, _ := tracker.Infer(id, curFrame, enhanceIm, dlist, Resolution, sceneid)
		if cfg.EvaluateBase.EvaluateState {
			var state [14]float64
			state[0] = float64(len(dlist))
			state[1] = float64(len(tracked))
			min_change, max_change, mean_change, change_number := [4]float64{9999.0, 9999.0, 9999.0, 9999.0}, [4]float64{0.0, 0.0, 0.0, 0.0}, [4]float64{0.0, 0.0, 0.0, 0.0}, 0.0

			for last_track := range last_tracked {
				for track := range tracked {
					if *last_tracked[last_track].TrackID == *tracked[track].TrackID {
						change := [4]float64{float64(tracked[track].Left - last_tracked[last_track].Left),
							float64(tracked[track].Top - last_tracked[last_track].Top),
							float64(tracked[track].Right - last_tracked[last_track].Right),
							float64(tracked[track].Bottom - last_tracked[last_track].Bottom)}
						min_change_amplitude := min_change[0]*min_change[0] + min_change[1]*min_change[1] + min_change[2]*min_change[2] + min_change[3]*min_change[3]
						max_change_amplitude := max_change[0]*max_change[0] + max_change[1]*max_change[1] + max_change[2]*max_change[2] + max_change[3]*max_change[3]
						change_amplitude := change[0]*change[0] + change[1]*change[1] + change[2]*change[2] + change[3]*change[3]
						if change_amplitude < min_change_amplitude && change_amplitude > 0.0 {
							min_change = change
						}
						if change_amplitude > max_change_amplitude {
							max_change = change
						}
						mean_change[0] += float64(change[0])
						mean_change[1] += float64(change[1])
						mean_change[2] += float64(change[2])
						mean_change[3] += float64(change[3])
						change_number++
					}
				}
			}
			if change_number > 0 {
				mean_change[0] /= float64(change_number)
				mean_change[1] /= float64(change_number)
				mean_change[2] /= float64(change_number)
				mean_change[3] /= float64(change_number)
			}
			for i := 0; i < 4; i++ {
				state[i+2] = min_change[i]
				state[i+6] = max_change[i]
				state[i+10] = mean_change[i]
			}
			states = append(states, state)
			last_tracked = tracked
		}

		// update our state
		setDetections(curFrame, tracked)
		lastFrame = curFrame
		bfr.Discard(lastFrame / skipFrame)
		updated = true

		if !updated {
			break
		}
	}

	return outDetections, states, start_time
}
