package lib

import (
	"bytes"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"io"
	"math"
	"os"

	"github.com/disintegration/imaging"
)

type Image struct {
	Width  int
	Height int
	Bytes  []byte
}

func NewImage(width int, height int) Image {
	return Image{
		Width:  width,
		Height: height,
		Bytes:  make([]byte, 3*width*height),
	}
}

func ImageFromBytes(width int, height int, bytes []byte) Image {
	return Image{
		Width:  width,
		Height: height,
		Bytes:  bytes,
	}
}

func ImageFromJPGReader(rd io.Reader) Image {
	im, err := jpeg.Decode(rd)
	if err != nil {
		panic(err)
	}
	rect := im.Bounds()
	width := rect.Dx()
	height := rect.Dy()
	bytes := make([]byte, width*height*3)
	for i := 0; i < width; i++ {
		for j := 0; j < height; j++ {
			r, g, b, _ := im.At(i+rect.Min.X, j+rect.Min.Y).RGBA()
			bytes[(j*width+i)*3+0] = uint8(r >> 8)
			bytes[(j*width+i)*3+1] = uint8(g >> 8)
			bytes[(j*width+i)*3+2] = uint8(b >> 8)
		}
	}
	return Image{
		Width:  width,
		Height: height,
		Bytes:  bytes,
	}
}

func ImageFromFile(fname string) Image {
	file, err := os.Open(fname)
	if err != nil {
		panic(err)
	}
	im := ImageFromJPGReader(file)
	file.Close()
	return im
}

func (im Image) AsImage() image.Image {
	pixbuf := make([]byte, im.Width*im.Height*4)
	j := 0
	channels := 0
	for i := range im.Bytes {
		pixbuf[j] = im.Bytes[i]
		j++
		channels++
		if channels == 3 {
			pixbuf[j] = 255
			j++
			channels = 0
		}
	}
	img := &image.RGBA{
		Pix:    pixbuf,
		Stride: im.Width * 4,
		Rect:   image.Rect(0, 0, im.Width, im.Height),
	}
	return img
}

func (im Image) AsJPG() []byte {
	buf := new(bytes.Buffer)
	if err := jpeg.Encode(buf, im.AsImage(), nil); err != nil {
		panic(err)
	}
	return buf.Bytes()
}

func (im Image) AsPNG() []byte {
	buf := new(bytes.Buffer)
	if err := png.Encode(buf, im.AsImage()); err != nil {
		panic(err)
	}
	return buf.Bytes()
}

func (im Image) ToBytes() []byte {
	return im.Bytes
}

func (im Image) SetRGB(i int, j int, color [3]uint8) {
	if i < 0 || i >= im.Width || j < 0 || j >= im.Height {
		return
	}
	for channel := 0; channel < 3; channel++ {
		im.Bytes[(j*im.Width+i)*3+channel] = color[channel]
	}
}

func (im Image) GetRGB(i int, j int) [3]uint8 {
	var color [3]uint8
	for channel := 0; channel < 3; channel++ {
		color[channel] = im.Bytes[(j*im.Width+i)*3+channel]
	}
	return color
}

func (im Image) FillRectangle(left, top, right, bottom int, color [3]uint8) {
	for i := left; i < right; i++ {
		for j := top; j < bottom; j++ {
			im.SetRGB(i, j, color)
		}
	}
}

func (im Image) Copy() Image {
	bytes := make([]byte, len(im.Bytes))
	copy(bytes, im.Bytes)
	return Image{
		Width:  im.Width,
		Height: im.Height,
		Bytes:  bytes,
	}
}

func (im Image) DrawRectangle(left, top, right, bottom int, width int, color [3]uint8) {
	im.FillRectangle(left-width, top, left+width, bottom, color)
	im.FillRectangle(right-width, top, right+width, bottom, color)
	im.FillRectangle(left, top-width, right, top+width, color)
	im.FillRectangle(left, bottom-width, right, bottom+width, color)
}

func (im Image) DrawImage(left int, top int, other Image) {
	for i := 0; i < other.Width; i++ {
		for j := 0; j < other.Height; j++ {
			im.SetRGB(left+i, top+j, other.GetRGB(i, j))
		}
	}
}

func (im Image) Crop(sx int, sy int, ex int, ey int) Image {
	other := NewImage(ex-sx, ey-sy)
	for i := 0; i < other.Width; i++ {
		for j := 0; j < other.Height; j++ {
			other.SetRGB(i, j, im.GetRGB(sx+i, sy+j))
		}
	}
	return other
}

// func (img *Image) Resize(newWidth, newHeight int) *Image {
// 	newBytes := make([]byte, newWidth*newHeight*3)
// 	newImage := &Image{
// 		Width:  newWidth,
// 		Height: newHeight,
// 		Bytes:  newBytes,
// 	}

// 	xRatio := float64(img.Width) / float64(newWidth)
// 	yRatio := float64(img.Height) / float64(newHeight)

// 	for newY := 0; newY < newHeight; newY++ {
// 		for newX := 0; newX < newWidth; newX++ {
// 			x := int(math.Floor(float64(newX) * xRatio))
// 			y := int(math.Floor(float64(newY) * yRatio))
// 			xDiff := (float64(newX) * xRatio) - float64(x)
// 			yDiff := (float64(newY) * yRatio) - float64(y) // Get the neighboring pixels
// 			aR, aG, aB := img.GetPixel(x, y)
// 			bR, bG, bB := img.GetPixel(x+1, y)
// 			cR, cG, cB := img.GetPixel(x, y+1)
// 			dR, dG, dB := img.GetPixel(x+1, y+1)

// 			// Bilinear interpolation
// 			r := float64(aR)*(1-xDiff)*(1-yDiff) + float64(bR)*xDiff*(1-yDiff) + float64(cR)*(1-xDiff)*yDiff + float64(dR)*xDiff*yDiff
// 			g := float64(aG)*(1-xDiff)*(1-yDiff) + float64(bG)*xDiff*(1-yDiff) + float64(cG)*(1-xDiff)*yDiff + float64(dG)*xDiff*yDiff
// 			b := float64(aB)*(1-xDiff)*(1-yDiff) + float64(bB)*xDiff*(1-yDiff) + float64(cB)*(1-xDiff)*yDiff + float64(dB)*xDiff*yDiff

// 			newImage.SetPixel(newX, newY, byte(r), byte(g), byte(b))

// 		}
// 	}

// 	return newImage
// }

func (img *Image) Resize(newWidth, newHeight int) *Image {
	newBytes := make([]byte, newWidth*newHeight*3)
	newImage := &Image{
		Width:  newWidth,
		Height: newHeight,
		Bytes:  newBytes,
	}

	xRatio := float64(img.Width) / float64(newWidth)
	yRatio := float64(img.Height) / float64(newHeight)

	for newY := 0; newY < newHeight; newY++ {
		for newX := 0; newX < newWidth; newX++ {
			x := int(math.Floor(float64(newX) * xRatio))
			y := int(math.Floor(float64(newY) * yRatio))
			xDiff := (float64(newX) * xRatio) - float64(x)
			yDiff := (float64(newY) * yRatio) - float64(y)

			// Get the neighboring pixels
			aR, aG, aB := img.GetPixel(clamp(x, 0, img.Width-1), clamp(y, 0, img.Height-1))
			bR, bG, bB := img.GetPixel(clamp(x+1, 0, img.Width-1), clamp(y, 0, img.Height-1))
			cR, cG, cB := img.GetPixel(clamp(x, 0, img.Width-1), clamp(y+1, 0, img.Height-1))
			dR, dG, dB := img.GetPixel(clamp(x+1, 0, img.Width-1), clamp(y+1, 0, img.Height-1))

			// Bilinear interpolation
			r := float64(aR)*(1-xDiff)*(1-yDiff) + float64(bR)*xDiff*(1-yDiff) + float64(cR)*(1-xDiff)*yDiff + float64(dR)*xDiff*yDiff
			g := float64(aG)*(1-xDiff)*(1-yDiff) + float64(bG)*xDiff*(1-yDiff) + float64(cG)*(1-xDiff)*yDiff + float64(dG)*xDiff*yDiff
			b := float64(aB)*(1-xDiff)*(1-yDiff) + float64(bB)*xDiff*(1-yDiff) + float64(cB)*(1-xDiff)*yDiff + float64(dB)*xDiff*yDiff

			newImage.SetPixel(newX, newY, byte(r), byte(g), byte(b))
		}
	}

	return newImage
}

func clamp(value, min, max int) int {
	if value < min {
		return min
	} else if value > max {
		return max
	} else {
		return value
	}
}

// for image.Image

func (im Image) Set(i int, j int, c color.Color) {
	r, g, b, _ := c.RGBA()
	r = r >> 8
	g = g >> 8
	b = b >> 8
	im.SetRGB(i, j, [3]uint8{uint8(r), uint8(g), uint8(b)})
}

func (im Image) At(i int, j int) color.Color {
	c := im.GetRGB(i, j)
	return color.RGBA{c[0], c[1], c[2], 255}
}

func (im Image) ColorModel() color.Model {
	return color.RGBAModel
}

func (im Image) Bounds() image.Rectangle {
	return image.Rectangle{image.Point{0, 0}, image.Point{im.Width, im.Height}}
}

func (img *Image) GetPixel(x, y int) (byte, byte, byte) {
	offset := (y*img.Width + x) * 3
	return img.Bytes[offset], img.Bytes[offset+1], img.Bytes[offset+2]
}

func (img *Image) SetPixel(x, y int, r, g, b byte) {
	offset := (y*img.Width + x) * 3
	img.Bytes[offset] = r
	img.Bytes[offset+1] = g
	img.Bytes[offset+2] = b
}

func ImagetoNRGBA(img *Image) *image.NRGBA {
	rgba := image.NewNRGBA(image.Rect(0, 0, img.Width, img.Height))

	idx := 0
	for y := 0; y < img.Height; y++ {
		for x := 0; x < img.Width; x++ {
			r := img.Bytes[idx]
			g := img.Bytes[idx+1]
			b := img.Bytes[idx+2]
			rgba.SetNRGBA(x, y, color.NRGBA{r, g, b, 255})
			idx += 3
		}
	}
	return rgba
}

func fromImage(img image.Image) *Image {
	width := img.Bounds().Dx()
	height := img.Bounds().Dy()
	bytes := make([]byte, width*height*3)

	idx := 0
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			bytes[idx] = byte(r >> 8)
			bytes[idx+1] = byte(g >> 8)
			bytes[idx+2] = byte(b >> 8)
			idx += 3
		}
	}
	return &Image{Width: width, Height: height, Bytes: bytes}
}

func (img *Image) processImage(processFunc func(*image.NRGBA) *image.NRGBA) {
	nrgba := ImagetoNRGBA(img)
	processed := processFunc(nrgba)
	newImage := fromImage(processed)
	img.Bytes = newImage.Bytes
}

func (img *Image) Saturation(factor float64) {
	img.processImage(func(nrgba *image.NRGBA) *image.NRGBA {
		return imaging.AdjustSaturation(nrgba, factor)
	})
}

func (img *Image) Denoising(radius float64) {
	img.processImage(func(nrgba *image.NRGBA) *image.NRGBA {
		return imaging.Blur(nrgba, radius)
	})
}

func (img *Image) Equalization() {
	img.processImage(func(nrgba *image.NRGBA) *image.NRGBA {
		bounds := nrgba.Bounds()
		width, height := bounds.Dx(), bounds.Dy()

		hist := make([]int, 256)
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				_, _, _, a := nrgba.At(x, y).RGBA()
				hist[a>>8]++
			}
		}

		cdf := make([]uint32, 256)
		cdf[0] = uint32(hist[0])
		for i := 1; i < 256; i++ {
			cdf[i] = cdf[i-1] + uint32(hist[i])
		}

		min, max := cdf[0], cdf[255]
		scale := 255.0 / float64(max-min)

		out := image.NewNRGBA(bounds)
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				r, g, b, a := nrgba.At(x, y).RGBA()
				eq := uint8((float64(cdf[a>>8])-float64(min))*scale + 0.5)
				out.SetNRGBA(x, y, color.NRGBA{
					R: uint8(r >> 8),
					G: uint8(g >> 8),
					B: uint8(b >> 8),
					A: eq,
				})
			}
		}

		return out
	})
}

func (img *Image) Sharpening(factor float64) {
	img.processImage(func(nrgba *image.NRGBA) *image.NRGBA {
		return imaging.Sharpen(nrgba, factor)
	})
}

func (img *Image) Enhance(enhancetools map[string]bool, enhancetoolinfos []float64) {
	if enhancetools["denoising"] {
		img.Denoising(enhancetoolinfos[0])
	}
	if enhancetools["equalization"] {
		img.Equalization()
	}
	if enhancetools["sharpening"] {
		img.Sharpening(enhancetoolinfos[2])
	}
	if enhancetools["saturation"] {
		img.Saturation(enhancetoolinfos[3])
	}

}
