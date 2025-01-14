package main

import (
	"image/png"
	"os"
	"videotune/lib"
)

func main() {
	skipFrame := 16
	resolution := [2]int{720, 480}
	videoFname := "./demo_videos/demo7.mp4" // demo2: [960, 540]
	vreader := lib.ReadFfmpeg(videoFname, resolution[0], resolution[1])
	vreader.Skip = skipFrame

	bfr := lib.NewBufferedFfmpegReader(vreader, 192)
	im, _ := bfr.GetFrame(0)

	file, err := os.Create("original.png")
	if err != nil {
		panic(err)
	}

	nrgba_im := lib.ImagetoNRGBA(&im)
	enhancetool := "sharpening"

	png.Encode(file, nrgba_im)
	enhancetools := []string{
		enhancetool,
	}
	lib.ImagePreprocess(im, enhancetools, []float64{3.0, 1.0, 10.0, 25.0}, resolution)
	file, err = os.Create(enhancetool + ".png")
	if err != nil {
		panic(err)
	}

	nrgba_im = lib.ImagetoNRGBA(&im)

	png.Encode(file, nrgba_im)
}
