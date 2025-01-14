package lib

type Channel struct {
	framefilterInJobName  [3]int
	framefilterOutJobName [3]int
	roiInJobName          [3]int
	roiOutJobName         [3]int
	enhanceInJobName      [3]int
	enhanceOutJobName     [3]int
	detectInJobName       [3]int
}

func getJobChannels(pipei int, FilterFlag bool, RoiFlag bool) Channel {
	var (
		framefilterInJobName  [3]int
		framefilterOutJobName [3]int
		roiInJobName          [3]int
		roiOutJobName         [3]int
		enhanceInJobName      [3]int
		enhanceOutJobName     [3]int
		detectInJobName       [3]int
	)
	if FilterFlag && RoiFlag {
		framefilterInJobName = [3]int{pipei, 1, 1}
		framefilterOutJobName = [3]int{pipei, 2, 2}

		roiInJobName = [3]int{pipei, 2, 2}
		roiOutJobName = [3]int{pipei, 3, 3}

		enhanceInJobName = [3]int{pipei, 3, 3}
		enhanceOutJobName = [3]int{pipei, 4, 4}

		detectInJobName = [3]int{pipei, 4, 4}
	} else if !FilterFlag && RoiFlag {
		framefilterInJobName = [3]int{pipei, -1, -1}
		framefilterOutJobName = [3]int{pipei, -1, -1}

		roiInJobName = [3]int{pipei, 1, 1}
		roiOutJobName = [3]int{pipei, 2, 2}

		enhanceInJobName = [3]int{pipei, 2, 2}
		enhanceOutJobName = [3]int{pipei, 3, 3}

		detectInJobName = [3]int{pipei, 3, 3}
	} else if FilterFlag && !RoiFlag {
		framefilterInJobName = [3]int{pipei, 1, 1}
		framefilterOutJobName = [3]int{pipei, 2, 2}

		roiInJobName = [3]int{pipei, -1, -1}
		roiOutJobName = [3]int{pipei, -1, -1}

		enhanceInJobName = [3]int{pipei, 2, 2}
		enhanceOutJobName = [3]int{pipei, 3, 3}

		detectInJobName = [3]int{pipei, 3, 3}
	} else {
		framefilterInJobName = [3]int{pipei, -1, -1}
		framefilterOutJobName = [3]int{pipei, -1, -1}

		roiInJobName = [3]int{pipei, -1, -1}
		roiOutJobName = [3]int{pipei, -1, -1}

		enhanceInJobName = [3]int{pipei, 1, 1}
		enhanceOutJobName = [3]int{pipei, 2, 2}

		detectInJobName = [3]int{pipei, 2, 2}
	}

	jobChannel := Channel{
		framefilterInJobName:  framefilterInJobName,
		framefilterOutJobName: framefilterOutJobName,
		roiInJobName:          roiInJobName,
		roiOutJobName:         roiOutJobName,
		enhanceInJobName:      enhanceInJobName,
		enhanceOutJobName:     enhanceOutJobName,
		detectInJobName:       detectInJobName,
	}

	return jobChannel
}
