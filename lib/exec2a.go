package lib

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/schollz/progressbar/v3"
)

func Exec2(cfg Config, outDir string, logPath string, cameraids []int, pipelineids []int) float64 {
	scene2id := map[string]int{
		"amsterdam": 0,
		"warsaw":    1, "shibuya": 2,
		"jackson": 3, "caldot1": 4,
		"caldot2": 5, "uav": 6,
	}

	FilterResult := make(map[[2]int]int)
	RoiResult := make(map[[2]int][][4]int)
	WindowResult := make(map[int]int)

	pipelineset := Set(pipelineids)
	pipeline_number := len(pipelineset)

	clusters := make(map[string][]Cluster, 0)
	var needClusters bool

	for pipek := 0; pipek < pipeline_number; pipek++ {
		if cfg.PostProcessBase.PostProcessType[pipek] == "OTIF" && cfg.PostProcessBase.Flag[pipek] {
			needClusters = true
		}
	}

	MethodName := cfg.MethodName
	DataName := cfg.DataBase.DataName
	DataRoot := cfg.DataBase.DataRoot
	DataType := cfg.DataBase.DataType
	WeightRoot := cfg.LogBase.WeightRoot
	SceneName := cfg.LogBase.Scene

	if needClusters {
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

		filenumber := len(jsonFiles)
		clusterbar := progressbar.Default(int64(filenumber))
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

			clusterbar.Add(1)
		}
		for scenename, tracks := range trainSet {
			clusters[scenename], _ = ClusterTracks(tracks)
		}
	}

	t0 := time.Now()
	var durationSamples []int

	videoPath := filepath.Join(DataRoot, "dataset", DataName, DataType, "video/")
	gtPath := filepath.Join(DataRoot, "dataset", DataName, DataType, "tracks/")

	var videoFnames []string
	// for start_videoid := 0; start_videoid < cameraNumber; start_videoid++ {
	// 	videoFnames = append(videoFnames, fmt.Sprintf("%d.mp4", start_videoid))
	// }
	for _, cameraid := range cameraids {
		videoFnames = append(videoFnames, fmt.Sprintf("%d.mp4", cameraid))
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
		Image Image
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
	}
	// model size -> pending jobs
	pendingJobs := make(map[[3]int][]PendingJob) // 保存待处理的任务
	// (video ID, frame idx) -> priority
	priorities := make(map[[2]int]int) // 保存每个视频的优先级
	// model outputs
	detectOutputs := make(map[[2]int]*DetectorOutput) // 保存检测结果
	trackers := make([]*Tracker, pipeline_number)
	// number of videos that are waiting for some frame to be processed
	var numWaiting int = 0     // 保存等待处理的视频数
	done := false              // 是否处理完所有视频
	var modelWg sync.WaitGroup // 保存所有模型的线程, 用于等待所有模型处理完毕
	var modelMetas []ModelMeta // 保存模型的信息
	wakeupModels := func() {   // 匿名函数，用于唤醒所有模型
		for _, meta := range modelMetas {
			if len(pendingJobs[meta.Size]) < meta.BatchSize && (numWaiting < len(cameraids) || len(pendingJobs[meta.Size]) == 0) && !done {
				continue
			}
			meta.Cond.Broadcast()
		}
	}
	modelLoop := func(sz [3]int, batchSize int, f func([]PendingJob), cleanup func()) { // 匿名函数，用于处理模型
		cond := sync.NewCond(&mu)                  // 条件变量，用于通知等待的线程
		modelMetas = append(modelMetas, ModelMeta{ // 保存模型的信息
			Size:      sz,
			Cond:      cond,
			BatchSize: batchSize,
		})
		modelWg.Add(1) // 保存所有模型的线程
		go func() {    // 开启一个 goroutine (协程)
			defer modelWg.Done() // 线程结束时，减少一个线程
			defer cleanup()      // 清理
			for {                // 无限循环
				mu.Lock()
				for len(pendingJobs[sz]) < batchSize && (numWaiting < len(cameraids) || len(pendingJobs[sz]) == 0) && !done { // 如果待处理的任务数小于batchSize，且等待处理的视频数小于总视频数，且没有处理完所有视频
					cond.Wait() // 等待
				}
				if done { // 如果处理完所有视频
					mu.Unlock() // 释放锁
					break       // 退出循环
				}

				// pick the jobs with lowest (most) priority
				sort.Slice(pendingJobs[sz], func(i, j int) bool { // 对待处理的任务进行排序
					job1 := pendingJobs[sz][i]
					job2 := pendingJobs[sz][j]
					prio1 := priorities[[2]int{job1.ID, job1.FrameIdx}]
					prio2 := priorities[[2]int{job2.ID, job2.FrameIdx}]
					return prio1 < prio2 || (prio1 == prio2 && job1.FrameIdx < job2.FrameIdx)
				})
				var jobs []PendingJob                 // 保存待处理的任务
				if len(pendingJobs[sz]) < batchSize { // 如果待处理的任务数小于batchSize
					jobs = pendingJobs[sz] // 保存待处理的任务
					pendingJobs[sz] = nil  // 清空待处理的任务
				} else {
					jobs = append([]PendingJob{}, pendingJobs[sz][0:batchSize]...) // 保存待处理的任务
					n := copy(pendingJobs[sz][0:], pendingJobs[sz][batchSize:])    // 保存待处理的任务
					pendingJobs[sz] = pendingJobs[sz][0:n]                         // 保存待处理的任务
				}

				mu.Unlock() // 释放锁
				f(jobs)     // 处理任务
			}
		}()
	}

	pipeRemap := make(map[int]int)
	for idx, pipei := range pipelineset {
		pipeRemap[pipei] = idx
		RoiFlag := cfg.RoiBase.Flag[pipei]
		FilterFlag := cfg.FilterBase.Flag[pipei]
		EnhanceTools := []string{}
		EnhanceToolInfos := cfg.RoiBase.EnhanceToolInfos
		SkipNumber := cfg.VideoBase.SkipNumber[pipei]
		Resolution := cfg.VideoBase.Resolution
		SceneName := SceneName
		ScaleDownResolution := cfg.VideoBase.ScaleDownResolution[pipei]

		jobChannels := getJobChannels(pipeRemap[pipei], FilterFlag, RoiFlag)

		if FilterFlag {
			FilterModelType := cfg.FilterBase.ModelType[pipei]
			FilterBatchSize := cfg.FilterBase.BatchSize[pipei]
			FilterThreshold := cfg.FilterBase.Threshold[pipei]
			FilterResolution := cfg.FilterBase.Resolution[pipei]
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
			filterprocess := func(jobs []PendingJob) {
				var images []Image
				for _, job := range jobs {
					if FilterResolution[0] != job.Image.Width || FilterResolution[1] != job.Image.Height {
						FilterImage := job.Image.Resize(FilterResolution[0], FilterResolution[1])
						images = append(images, *FilterImage)
					} else {
						images = append(images, job.Image)
					}
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
							Needed:     0,
							Detections: []Detection{},
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

		if RoiFlag { // 如果需要分割
			RoiBatchSize := cfg.RoiBase.BatchSize[pipei]
			RoiThreshold := cfg.RoiBase.Threshold[pipei]
			RoiModelType := cfg.RoiBase.ModelType[pipei]
			RoiWindowSizes := cfg.RoiBase.WindowSizes[pipei]
			RoiResolution := cfg.RoiBase.Resolution[pipei]
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
				SceneName) // 创建分割模型
			segprocess := func(jobs []PendingJob) { // 调用匿名函数，处理分割模型
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
			modelLoop(jobChannels.roiInJobName, RoiBatchSize, segprocess, segcleanup) // 调用匿名函数，处理分割模型
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
		modelLoop(jobChannels.enhanceInJobName, cfg.RoiBase.BatchSize[pipei], enhanceprocess, enhancecleanup)

		DetectClasses := cfg.DetectBase.Classes
		DetectModelType := cfg.DetectBase.ModelType[pipei]
		DetectBatchSize := cfg.DetectBase.BatchSize[pipei]
		DetectModelSize := cfg.DetectBase.ModelSize[pipei]
		DetectThreshold := cfg.DetectBase.Threshold[pipei]
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
					d.Left = d.Left * Resolution[0] / ScaleDownResolution[0]
					d.Top = d.Top * Resolution[1] / ScaleDownResolution[1]
					d.Right = d.Right * Resolution[0] / ScaleDownResolution[0]
					d.Bottom = d.Bottom * Resolution[1] / ScaleDownResolution[1]
					dlist = append(dlist, d)
				}

				output := detectOutputs[[2]int{job.ID, job.FrameIdx}]
				output.Detections = append(output.Detections, dlist...)
				output.Needed--
			}
			respCond.Broadcast()
			mu.Unlock()
		}
		detector_cleanup := func() { detector.Close() }
		modelLoop(jobChannels.detectInJobName, DetectBatchSize, detector_process, detector_cleanup)

		TrackModelType := cfg.TrackBase.ModelType[pipei]
		TrackHashSize := cfg.TrackBase.HashSize[pipei]
		TrackLowFeatureDistanceThreshold := cfg.TrackBase.LowFeatureDistanceThreshold[pipei]
		TrackMaxLostTime := cfg.TrackBase.MaxLostTime[pipei]
		TrackMoveThreshold := cfg.TrackBase.MoveThreshold[pipei]
		TrackKeepThreshold := cfg.TrackBase.KeepThreshold[pipei]
		TrackMinThreshold := cfg.DetectBase.Threshold[pipei]
		TrackCreateObjectThreshold := cfg.TrackBase.CreateObjectThreshold[pipei]
		TrackMatchLocationThreshold := cfg.TrackBase.MatchLocationThreshold[pipei]
		TrackUnmatchLocationThreshold := cfg.TrackBase.UnmatchLocationThreshold[pipei]
		TrackVisualThreshold := cfg.TrackBase.VisualThreshold[pipei]
		TrackKFPosWeight := cfg.TrackBase.KFPosWeight[pipei]
		TrackKFVelWeight := cfg.TrackBase.KFVelWeight[pipei]
		trackers[idx] = NewTracker(
			cfg.DeviceID,
			SkipNumber,
			TrackModelType,
			WeightRoot,
			SceneName,
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
	}

	var wg sync.WaitGroup
	var videostates [][][14]float64
	var metrics [3]map[string][]float64

	camerainfos := make([]CameraInfo, 0)

	for _, fname := range videoFnames {
		fname := filepath.Join(videoPath, fname)
		infofname := strings.ReplaceAll(fname, "video", "info")
		infofname = strings.ReplaceAll(infofname, "mp4", "txt")
		infofile, err := os.Open(infofname)
		if err != nil {
			panic(err)
		}
		defer infofile.Close()
		// just read one line
		scanner := bufio.NewScanner(infofile)
		scanner.Scan()
		line := scanner.Text()
		parts := strings.Split(line, "-")
		scenename := parts[0]
		width := ParseInt(parts[1])
		height := ParseInt(parts[2])
		camerainfos = append(camerainfos, CameraInfo{
			SceneName: scenename,
			Width:     width,
			Height:    height,
		})
	}

	for i := 0; i < len(metrics); i++ {
		metrics[i] = make(map[string][]float64)
	}

	for vi, fname := range videoFnames {
		pipeidx := pipeRemap[pipelineids[vi]]
		EnhanceTools := []string{}
		Resolution := [2]int{camerainfos[vi].Width, camerainfos[vi].Height}
		submitOutJobName := [3]int{pipeidx, 1, 1}
		sceneid := scene2id[camerainfos[vi].SceneName]
		wg.Add(1)
		go func(fname string, absoluteid int, EnhanceTools []string) {
			defer wg.Done()
			id := ParseInt(strings.Split(fname, ".mp4")[0])
			tracker := trackers[pipeidx]

			submitted := make(map[int]bool)

			detectorFunc := func(frameIdx int, im Image, extras []int, extraImages []Image) []Detection {
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
				return detectOutputs[k].Detections
			}

			detections, states := execTrackerLoop2(id, tracker, filepath.Join(videoPath, fname), cfg,
				cfg.VideoBase.SkipNumber[pipeidx], Resolution,
				cfg.VideoBase.ScaleDownResolution[pipeidx], EnhanceTools, cfg.RoiBase.EnhanceToolInfos,
				detectorFunc, sceneid)
			if cfg.EvaluateBase.EvaluateState {
				videostates = append(videostates, states)
			}

			mu.Lock()
			numWaiting++
			wakeupModels()
			durationSamples = append(durationSamples, int(time.Since(t0)/time.Second))
			mu.Unlock()

			// filter out bad tracks
			if cfg.NoiseFilterBase.NoiseFilterType[pipeidx] != "None" && cfg.NoiseFilterBase.Flag[pipeidx] {
				goodTracks := FilterNoiseTracks(detections)
				detections = DetectionsFromTracks(goodTracks)
			}

			// refine using clusters computed earlier if desired
			if cfg.PostProcessBase.PostProcessType[pipeidx] == "OTIF" && cfg.PostProcessBase.Flag[pipeidx] {
				tracks := GetTracks(detections)
				tracks = Postprocess(clusters[camerainfos[absoluteid].SceneName], tracks, cfg.VideoBase.SkipNumber[pipeidx])
				detections = DetectionsFromTracks(tracks)
			}

			// save all of the detections
			bytes := JsonMarshal(detections)
			outFname := fmt.Sprintf("%s/%d.json", outDir, id)
			if err := os.WriteFile(outFname, bytes, 0644); err != nil {
				fmt.Printf("Error 5")
				panic(err)
			}
		}(fname, vi, EnhanceTools)
	}
	wg.Wait()

	mu.Lock()
	done = true
	wakeupModels()
	mu.Unlock()
	modelWg.Wait()
	for _, tracker := range trackers {
		tracker.Close()
	}

	processTime := float64(time.Since(t0)) / float64(time.Second)

	var durationSum int = 0
	for _, duration := range durationSamples {
		durationSum += duration
	}

	if cfg.EvaluateBase.EvaluateFilter {
		EvalFilter(FilterResult, gtPath)
	}

	logStr := fmt.Sprintf("%v\t%v", cfg, float64(durationSum)/float64(len(durationSamples)))
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
	data_dir := filepath.Join(DataRoot, "dataset", DataName)

	if cfg.Visualize {
		var skipnumberset []int
		stdout := bytes.Buffer{}
		stderr := bytes.Buffer{}
		for _, skipnumber := range cfg.VideoBase.SkipNumber {
			for _, skipnumber2 := range skipnumberset {
				if skipnumber == skipnumber2 {
					continue
				}
			}
			skipnumberset = append(skipnumberset, skipnumber)
		}
		for _, skipframe := range skipnumberset {
			save_dir := fmt.Sprintf("./TrackEval/data/trackers/videodb/%s/%s/data", DataName+"S"+strconv.Itoa(skipframe)+"-"+DataType, MethodName)
			save_gt_dir := fmt.Sprintf("./TrackEval/data/gt/videodb/%s", DataName+"S"+strconv.Itoa(skipframe)+"-"+DataType)
			viscmd := exec.Command("python", "./pylib/utils/visualizeotiftracker.py",
				"--testmode", DataType,
				"--gtpath", save_gt_dir,
				"--trackerpath", save_dir,
				"--dataset_name", DataName,
				"--skipframe", fmt.Sprintf("%d", skipframe),
				"--dataroot", data_dir)
			viscmd.Run()
			viscmd.Stdout = &stdout
			viscmd.Stderr = &stderr
			fmt.Printf("Save Video To : %s/video_data", save_dir)
		}
	}
	return processTime
}

type BufferedFfmpegReader struct {
	mu     sync.Mutex // 进程锁
	cond   *sync.Cond // 条件变量的指针
	buffer []Image    // 缓冲区
	offset int        // 缓冲区偏移量
	extras []Image    // 额外的缓冲区
	done   bool       // 是否结束
}

func NewBufferedFfmpegReader(vreader *FfmpegReader, size int) *BufferedFfmpegReader {
	bfr := &BufferedFfmpegReader{}
	bfr.cond = sync.NewCond(&bfr.mu)
	for i := 0; i < size; i++ {
		bfr.extras = append(bfr.extras, NewImage(vreader.Width, vreader.Height))
	}

	go func() {
		bfr.mu.Lock()
		for {
			for len(bfr.extras) == 0 {
				bfr.cond.Wait()
			}
			im := bfr.extras[len(bfr.extras)-1]
			bfr.extras = bfr.extras[0 : len(bfr.extras)-1]
			bfr.mu.Unlock()

			err := vreader.ReadInto(im)
			if err == io.EOF {
				bfr.mu.Lock()
				bfr.done = true
				bfr.cond.Broadcast()
				bfr.mu.Unlock()
				return
			} else if err != nil {
				fmt.Printf("Error 6")
				panic(err)
			}

			bfr.mu.Lock()
			bfr.buffer = append(bfr.buffer, im)
			bfr.cond.Broadcast()
		}
	}()

	return bfr
}

// Returns (image, false), or if EOF then (..., true)
// Blocks until EOF or image is available.
func (bfr *BufferedFfmpegReader) GetFrame(frameIdx int) (Image, bool) {
	bfr.mu.Lock()
	defer bfr.mu.Unlock()

	for !bfr.done && bfr.offset+len(bfr.buffer) <= frameIdx {
		bfr.cond.Wait()
	}

	if frameIdx < bfr.offset+len(bfr.buffer) {
		return bfr.buffer[frameIdx-bfr.offset], false
	}

	return Image{}, true
}

// Discard frames below frameIdx 丢弃低于 frameIdx 的帧
func (bfr *BufferedFfmpegReader) Discard(frameIdx int) {
	bfr.mu.Lock()
	defer bfr.mu.Unlock()

	if frameIdx <= bfr.offset {
		return
	}

	// first index in buffer that will NOT be discarded
	pos := frameIdx - bfr.offset

	discarded := bfr.buffer[0:pos]
	bfr.extras = append(bfr.extras, discarded...)
	n := copy(bfr.buffer[0:], bfr.buffer[pos:])
	bfr.buffer = bfr.buffer[0:n]
	bfr.offset = frameIdx

	bfr.cond.Broadcast()
}

// Buffer is valid until the next Discard call. 缓冲区在下一次丢弃调用之前有效
func (bfr *BufferedFfmpegReader) GetBuffer() ([]Image, int) {
	bfr.mu.Lock()
	defer bfr.mu.Unlock()

	return bfr.buffer, bfr.offset
}

func execTrackerLoop2(id int,
	tracker *Tracker,
	videoFname string,
	cfg Config, SkipNumber int, Resolution [2]int, ScaleDownResolution [2]int, EnhanceTools []string, EnhanceToolInfos []float64,
	detectorFunc func(frameIdx int, im Image, extras []int, extraImages []Image) []Detection, sceneid int) ([][]Detection, [][14]float64) {
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

	vreader := ReadFfmpeg(videoFname, Resolution[0], Resolution[1])
	vreader.Skip = minSkip
	defer vreader.Close()
	bfr := NewBufferedFfmpegReader(vreader, 192)

	im, done := bfr.GetFrame(0)
	if done {
		return nil, nil
	}

	enhanceIm := ImagePreprocess(im, EnhanceTools, EnhanceToolInfos, ScaleDownResolution)

	dlist := detectorFunc(0, im, nil, nil)
	tracked, _ := tracker.Infer(id, 0, enhanceIm, dlist, Resolution, sceneid)
	setDetections(0, tracked)

	var states [][14]float64
	var last_tracked []Detection
	if cfg.EvaluateBase.EvaluateState {
		last_tracked = tracked
	}

	lastFrame := 0

	getDetections := func(frameIdx int, im Image) []Detection {
		var extras []int
		var extraImages []Image
		buffer, offset := bfr.GetBuffer()
		for i, extraIm := range buffer {
			idx := (offset + i) * minSkip
			if idx == frameIdx || idx%maxSkip != 0 {
				continue
			}
			extras = append(extras, idx)               // 存储额外的帧索引
			extraImages = append(extraImages, extraIm) // 存储额外的帧图像
		}
		return detectorFunc(frameIdx, im, extras, extraImages)
	}

	for {
		updated := false
		curFrame := lastFrame + skipFrame

		// t0 := time.Now()
		im, done := bfr.GetFrame(curFrame / skipFrame)
		// timeMu.Lock()
		// decodeNumber++
		// decodeTime += time.Now().Sub(t0)
		// timeMu.Unlock()
		if done {
			break
		}

		enhanceIm := ImagePreprocess(im, EnhanceTools, EnhanceToolInfos, ScaleDownResolution)
		dlist := getDetections(curFrame, im)

		// t0 = time.Now()
		tracked, _ := tracker.Infer(id, curFrame, enhanceIm, dlist, Resolution, sceneid)
		// timeMu.Lock()
		// trackNumber++
		// trackTime += time.Now().Sub(t0)
		// timeMu.Unlock()
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

	return outDetections, states
}
