package lib

import (
	// "os"

	"fmt"
	"io/ioutil"
	"log"
	"os"
	"strconv"
	"strings"

	"gopkg.in/yaml.v2"
)

func ParseDims(dims string) [2]int {
	parts := strings.Split(dims, "x")
	if len(parts) != 2 {
		panic(fmt.Errorf("bad dims %v", dims))
	}
	return [2]int{
		ParseInt(parts[0]),
		ParseInt(parts[1]),
	}
}

type Config struct {
	DeviceID   int    `yaml:"deviceid"`
	MethodName string `yaml:"methodname"`
	Pipeline   int    `yaml:"pipeline"`
	Visualize  bool   `yaml:"visualize"`
	DataBase   struct {
		DataRoot string `yaml:"dataroot"`
		DataName string `yaml:"dataname"`
		DataType string `yaml:"datatype"`
	} `yaml:"database"`
	LogBase struct {
		WeightRoot string `yaml:"weightroot"`
		Scene      string `yaml:"scene"`
		SaveRoot   string `yaml:"saveroot"`
		LogRoot    string `yaml:"logroot"`
	}
	VideoBase struct {
		Resolution          [2]int   `yaml:"resolution"`
		SkipNumber          []int    `yaml:"skipnumber"`
		ScaleDownResolution [][2]int `yaml:"scaledownresolution"`
	} `yaml:"videobase"`
	FilterBase struct {
		Flag       []bool    `yaml:"flag"`
		BatchSize  []int     `yaml:"batchsize"`
		Resolution [][2]int  `yaml:"resolution"`
		ModelType  []string  `yaml:"modeltype"`
		Threshold  []float64 `yaml:"threshold"`
	} `yaml:"filterbase"`
	RoiBase struct {
		Flag             []bool     `yaml:"flag"`
		DenoisingFlag    bool       `json:"denoisingflag"`
		EqualizationFlag bool       `json:"equalizationflag"`
		SharpeningFlag   bool       `json:"sharpeningflag"`
		SaturationFlag   bool       `json:"saturationflag"`
		BatchSize        []int      `yaml:"batchsize"`
		Resolution       [][2]int   `yaml:"resolution"`
		WindowSizes      [][][2]int `yaml:"windowsizes"`
		ModelType        []string   `yaml:"modeltype"`
		Threshold        []float64  `yaml:"threshold"`
		EnhanceToolInfos []float64  `yaml:"enhancetoolinfos"`
	} `yaml:"roibase"`
	DetectBase struct {
		BatchSize []int     `yaml:"batchsize"`
		ModelType []string  `yaml:"modeltype"`
		ModelSize []string  `yaml:"modelsize"`
		Classes   []string  `yaml:"classes"`
		Threshold []float64 `yaml:"threshold"`
	} `yaml:"detectbase"`
	TrackBase struct {
		ModelType                   []string  `yaml:"modeltype"`
		VisualModule                []string  `yaml:"visualmodule"`
		HashSize                    []int     `yaml:"hashsize"`
		LowFeatureDistanceThreshold []float64 `yaml:"lowfeaturedistancethreshold"`
		MaxLostTime                 []int     `yaml:"maxlosttime"`
		MoveThreshold               []float64 `yaml:"movethreshold"`
		KeepThreshold               []float64 `yaml:"keepthreshold"`
		CreateObjectThreshold       []float64 `yaml:"createobjectthreshold"`
		MatchLocationThreshold      []float64 `yaml:"matchlocationthreshold"`
		UnmatchLocationThreshold    []float64 `yaml:"unmatchlocationthreshold"`
		VisualThreshold             []float64 `yaml:"visualthreshold"`
		KFPosWeight                 []float64 `yaml:"kfposweight"`
		KFVelWeight                 []float64 `yaml:"kfvelweight"`
	} `yaml:"trackbase"`
	PostProcessBase struct {
		Flag            []bool   `yaml:"flag"`
		PostProcessType []string `yaml:"postprocesstype"`
	} `yaml:"postprocessbase"`
	NoiseFilterBase struct {
		Flag            []bool   `yaml:"flag"`
		NoiseFilterType []string `yaml:"noisefiltertype"`
	} `yaml:"noisefilterbase"`
	EvaluateBase struct {
		EvaluateState  bool `yaml:"evaluatestate"`
		EvaluateFilter bool `yaml:"evaluatefilter"`
		EvaluateRoi    bool `yaml:"evaluateroi"`
	} `yaml:"evaluatebase"`
}

type JsonConfig struct {
	DeviceID   int    `json:"deviceid"`
	MethodName string `json:"methodname"`
	Pipeline   int    `json:"pipeline"`
	Visualize  bool   `json:"visualize"`
	Record     bool   `json:"record"`
	DataBase   struct {
		DataRoot string `json:"dataroot"`
		DataName string `json:"dataname"`
		DataType string `json:"datatype"`
	} `json:"database"`
	LogBase struct {
		WeightRoot string `json:"weightroot"`
		Scene      string `json:"scene"`
		SaveRoot   string `json:"saveroot"`
		LogRoot    string `json:"logroot"`
	} `json:"logbase"`
	VideoBase struct {
		Resolution          [2]int `json:"resolution"`
		SkipNumber          int    `json:"skipnumber"`
		ScaleDownResolution [2]int `json:"scaledownresolution"`
	} `json:"videobase"`
	FilterBase struct {
		Flag       bool    `json:"flag"`
		BatchSize  int     `json:"batchsize"`
		Resolution [2]int  `json:"resolution"`
		ModelType  string  `json:"modeltype"`
		Threshold  float64 `json:"threshold"`
	} `json:"filterbase"`
	RoiBase struct {
		Flag             bool      `json:"flag"`
		DenoisingFlag    bool      `json:"denoisingflag"`
		EqualizationFlag bool      `json:"equalizationflag"`
		SharpeningFlag   bool      `json:"sharpeningflag"`
		SaturationFlag   bool      `json:"saturationflag"`
		BatchSize        int       `json:"batchsize"`
		Resolution       [2]int    `json:"resolution"`
		WindowSizes      [][2]int  `json:"windowsizes"`
		ModelType        string    `json:"modeltype"`
		Threshold        float64   `json:"threshold"`
		EnhanceToolInfos []float64 `json:"enhancetoolinfos"`
	} `json:"roibase"`
	DetectBase struct {
		BatchSize int      `json:"batchsize"`
		ModelType string   `json:"modeltype"`
		ModelSize string   `json:"modelsize"`
		Classes   []string `json:"classes"`
		Threshold float64  `json:"threshold"`
	} `json:"detectbase"`
	TrackBase struct {
		ModelType                   string  `json:"modeltype"`
		VisualModule                string  `json:"visualmodule"`
		HashSize                    int     `json:"hashsize"`
		LowFeatureDistanceThreshold float64 `json:"lowfeaturedistancethreshold"`
		MaxLostTime                 int     `json:"maxlosttime"`
		MoveThreshold               float64 `json:"movethreshold"`
		KeepThreshold               float64 `json:"keepthreshold"`
		CreateObjectThreshold       float64 `json:"createobjectthreshold"`
		MatchLocationThreshold      float64 `json:"matchlocationthreshold"`
		UnmatchLocationThreshold    float64 `json:"unmatchlocationthreshold"`
		VisualThreshold             float64 `json:"visualthreshold"`
		KFPosWeight                 float64 `json:"kfposweight"`
		KFVelWeight                 float64 `json:"kfvelweight"`
	} `json:"trackbase"`
	PostProcessBase struct {
		PostProcessType string `json:"postprocesstype"`
	} `json:"postprocessbase"`
	NoiseFilterBase struct {
		Flag bool `json:"flag"`
	} `json:"noisefilterbase"`
	EvaluateBase struct {
		EvaluateState  bool `json:"evaluatestate"`
		EvaluateFilter bool `json:"evaluatefilter"`
		EvaluateRoi    bool `json:"evaluateroi"`
	} `json:"evaluatebase"`
}

type SearchSpaceConfig struct {
	FrameSelection struct {
		FrameFilter struct {
			Flag      bool     `yaml:"flag"`
			KnobSpace [][2]int `yaml:"knobspace"`
		} `yaml:"framefilter"`
		SamplingRate struct {
			Flag      bool  `yaml:"flag"`
			KnobSpace []int `yaml:"knobspace"`
		} `yaml:"samplingrate"`
	} `yaml:"frameselection"`
	FramePreprocess struct {
		ScaleDown struct {
			Flag      bool     `yaml:"flag"`
			KnobSpace [][2]int `yaml:"knobspace"`
		} `yaml:"scaledown"`
		Enhancement struct {
			Flag      bool     `yaml:"flag"`
			KnobSpace []string `yaml:"knobspace"`
		} `yaml:"enhancement"`
		ROI struct {
			Flag      bool     `yaml:"flag"`
			KnobSpace [][2]int `yaml:"knobspace"`
		} `yaml:"roi"`
	} `yaml:"framepreprocess"`
	Detect struct {
		ModelTypeSelection struct {
			Flag      bool     `yaml:"flag"`
			KnobSpace []string `yaml:"knobspace"`
		} `yaml:"modeltypeselection"`
		ModelSizeSelection struct {
			Flag      bool                `yaml:"flag"`
			KnobSpace map[string][]string `yaml:"knobspace"`
		} `yaml:"modelsizeselection"`
	} `yaml:"detect"`
	Tracking struct {
		ModelSelection struct {
			Flag      bool     `yaml:"flag"`
			KnobSpace []string `yaml:"knobspace"`
		} `yaml:"modelselection"`
	} `yaml:"tracking"`
	Postprocessing struct {
		ModelSelection struct {
			Flag      bool     `yaml:"flag"`
			KnobSpace []string `yaml:"knobspace"`
		} `yaml:"modelselection"`
		NoiseFilter struct {
			Flag      bool     `yaml:"flag"`
			KnobSpace []string `yaml:"knobspace"`
		} `yaml:"noisefilter"`
	} `yaml:"postprocessing"`
}

func GetConfig(configRoot string) Config {
	config := &Config{}

	data, err := ioutil.ReadFile(configRoot)
	if err != nil {
		log.Fatalf("error: %v", err)
	}

	err = yaml.Unmarshal(data, config)
	if err != nil {
		log.Fatalf("error: %v", err)
	}
	return *config
}

func GetJsonConfig(configRoot string) JsonConfig {
	config := &JsonConfig{}

	data, err := os.ReadFile(configRoot)
	if err != nil {
		log.Fatalf("error: %v", err)
	}

	err = yaml.Unmarshal(data, config)
	if err != nil {
		log.Fatalf("error: %v", err)
	}
	return *config
}

func GetSearchSpaceConfig(configRoot string) SearchSpaceConfig {
	config := &SearchSpaceConfig{}

	data, err := ioutil.ReadFile(configRoot)
	if err != nil {
		log.Fatalf("error: %v", err)
	}

	err = yaml.Unmarshal(data, config)
	if err != nil {
		log.Fatalf("error: %v", err)
	}
	return *config
}

type DetectorConfig struct {
	Name      string
	Dims      [2]int
	Sizes     [][2]int
	Threshold float64
}

func (cfg DetectorConfig) Dir() string {
	return fmt.Sprintf("%s-%dx%d", cfg.Name, cfg.Dims[0], cfg.Dims[1])
}

func (cfg DetectorConfig) String() string {
	return string(JsonMarshal(cfg))
}

type SegmentationConfig struct {
	Dims      [2]int
	Threshold float64 // int64
}

func ParseSegmentationConfig(s string) SegmentationConfig {
	var cfg SegmentationConfig
	parts := strings.Split(s, "_")
	cfg.Dims[0] = ParseInt(parts[0])
	cfg.Dims[1] = ParseInt(parts[1])
	if len(parts) >= 3 {
		// cfg.Threshold, _ = strconv.ParseInt(parts[2], 10, 32)
		cfg.Threshold, _ = strconv.ParseFloat(parts[2], 64)
	}
	return cfg
}

func (cfg SegmentationConfig) Dir() string {
	return fmt.Sprintf("%d_%d", cfg.Dims[0], cfg.Dims[1])
}

func (cfg SegmentationConfig) String() string {
	return fmt.Sprintf("%d_%d_%v", cfg.Dims[0], cfg.Dims[1], cfg.Threshold)
}

type DetectorCountConfig struct {
	Counts [5]int
}

func ParseDetectorCountConfig(s string) DetectorCountConfig {
	var cfg DetectorCountConfig
	var flag [5]bool
	var hit_count int
	var average_count int = 0
	parts := strings.Split(s, "_")
	for i := 0; i < 5; i++ {
		cfg.Counts[i] = 1
		flag[i] = false
	}
	for i := 0; i < 5; i++ {
		flag[i] = ParseInt(parts[2][i+1:i+2]) > 0
		if flag[i] {
			cfg.Counts[i] = ParseInt(parts[2][i+1 : i+2])
			hit_count += cfg.Counts[i]
			average_count += 1
		}
	}
	surplus_count := hit_count - average_count
	for i := 4; i >= 0; i-- {
		if surplus_count <= 0 {
			break
		}
		if !flag[i] {
			cfg.Counts[i] -= 1
			surplus_count -= 1
		}
	}

	return cfg
}

func (cfg DetectorCountConfig) Dir() string {
	return fmt.Sprintf("%v", cfg.Counts)
}

func (cfg DetectorCountConfig) String() string {
	return fmt.Sprintf("%v", cfg.Counts)
}

type TrackerConfig struct {
	Profile TrackerProfile
}

func ParseTrackerConfig(s string) TrackerConfig {
	var profile TrackerProfile
	JsonUnmarshal([]byte(s), &profile)
	return TrackerConfig{
		Profile: profile,
	}
}

func (cfg TrackerConfig) String() string {
	return string(JsonMarshal(cfg.Profile))
}
