package lib

import (
	"fmt"
	"io"
	"os"
	"os/exec"
)

type FfmpegReader struct {
	Cmd    *exec.Cmd
	Stdout io.ReadCloser
	Width  int
	Height int
	Skip   int

	skipBuf []byte
}

func ReadFfmpeg(fname string, width int, height int) *FfmpegReader {
	cmd := exec.Command(
		"ffmpeg",
		"-threads", "2", // 控制cpu使用率
		"-i", fname, // 输入文件
		"-loglevel", "quiet", // 静默模式
		"-hide_banner",     // 隐藏banner
		"-c:v", "rawvideo", // 编码设置 rawvideo: 指定将视频转成原始数据
		"-pix_fmt", "rgb24", // 显示可用的像素格式 一个像素用24个bit == 3个字节来表示，R, G， B分量分别用8个bit来表示
		"-f", "rawvideo", // 指定输出格式 rawvideo
		"-vf", fmt.Sprintf("scale=%dx%d", width, height), // 加一个过滤器，将输入的1920x1080缩小到width x height输出:
		"-",
	)
	if true {
		cmd.Stderr = os.Stderr
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		panic(err)
	}
	if err := cmd.Start(); err != nil {
		panic(err)
	}

	return &FfmpegReader{
		Cmd:     cmd,
		Stdout:  stdout,
		Width:   width,
		Height:  height,
		Skip:    1,
		skipBuf: make([]byte, width*height*3),
	}
}

func (rd *FfmpegReader) Read() (Image, error) {
	buf := make([]byte, rd.Width*rd.Height*3)
	_, err := io.ReadFull(rd.Stdout, buf)
	if err != nil {
		return Image{}, err
	}

	// Skip over rd.Skip-1 more frames.
	for i := 1; i < rd.Skip; i++ {
		io.ReadFull(rd.Stdout, rd.skipBuf)
	}

	im := ImageFromBytes(rd.Width, rd.Height, buf)
	return im, nil
}

func (rd *FfmpegReader) ReadInto(im Image) error {
	_, err := io.ReadFull(rd.Stdout, im.Bytes)
	if err != nil {
		return err
	}

	// Skip over rd.Skip-1 more frames.
	for i := 1; i < rd.Skip; i++ {
		io.ReadFull(rd.Stdout, rd.skipBuf)
	}

	return nil
}

func (rd *FfmpegReader) Close() {
	rd.Stdout.Close()
	rd.Cmd.Wait()
}
