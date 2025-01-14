SCALEDOWN_RESOLUTIONS = [[128, 128], [256, 256],
                         [320, 192], [416, 256],
                         [640, 352], [736, 416]]

SKIPS = [256, 128, 64, 32, 16, 8]
OTIFSKIPS = [64, 32, 16, 8]
SKYSKIPS = [60, 30, 5, 1]

FILTER_RESOLUTIONS = [[160, 96], [224, 128],
                      [320, 192], [416, 256], [640, 352]]

FILTER_THRESHOLDS = [1e-2, 1e-1, 0.2, 0.3, 0.4, 0.5]

ROI_RESOLUTIONS = [[160, 96], [224, 128],
                   [320, 192], [416, 256], [640, 352]]

ROI_THRESHOLDS = [1e-2, 1e-1, 0.2, 0.3, 0.4, 0.5]

ENHANCE_TOOLS = ["Saturation", "Denoising", "Equalization", "Sharpening"]

DETECT_NAMES = ['YOLOV3', 'YOLOV5', 'YOLOVM', 'YOLOV7', 'YOLOV8',
                "DETR", "FASTERRCNN", "SPARSERCNN", "RETINANET", "VFNET"]

DETECT_SIZES = ['S', 'M', 'L', 'XL', 'XXL']

DETECT_THRESHOLDS = [0.5, 0.4,
                     0.3, 0.2, 0.1]

TRACK_NAMES = ['OTIF', 'ByteTrack', 'SORT', 'DeepSORT']

DETECT_NUMBER = len(DETECT_NAMES)
DETECT_SIZE_NUMBERS = [len(DETECT_SIZES) for name in DETECT_NAMES]
TRACK_NUMBER = len(TRACK_NAMES)

TRACK_MAX_LOST_TIMES = [32, 64, 128, 256, 512]

TRACK_KEEP_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]

TRACK_CREATE_OBJECT_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]

TRACK_MATCH_LOCATION_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]

TRACK_KF_POS_WEIGHT = [1, 2, 4, 8, 16]

TRACK_KF_VEL_WEIGHT = [1, 2, 4, 8, 16]

POSTPROCESS_NAMES = ['None', 'OTIF', 'MIRIS']

NOISEFILTER_NAMES = ['S0>99999999S|T32>99999999T',
                     'S100>99999999S|T0>99999999T',
                     'S100>99999999S|T32>99999999T']

SEARCH_SPACE_DICT = {"skipnumber": len(SKIPS),
                     "otifskipnumber": len(OTIFSKIPS),
                     "skyskipnumber": len(SKYSKIPS),
                     "filterflag": 2,
                     "filterresolution": len(FILTER_RESOLUTIONS),
                     "filterthreshold": len(FILTER_THRESHOLDS),
                     "scaledownresolution": len(SCALEDOWN_RESOLUTIONS),
                     "roiflag": 2,
                     "roiresolution": 2,
                     "roithreshold": 2,
                     "denoisingflag": 2,
                     "equalizationflag": 2,
                     "sharpeningflag": 2,
                     "saturationflag": 2,
                     "detectname": len(DETECT_NAMES),
                     "detectsize": len(DETECT_SIZES),
                     "detectthreshold": len(DETECT_THRESHOLDS),
                     "trackname": len(TRACK_NAMES),
                     "trackmaxlosttime": len(TRACK_MAX_LOST_TIMES),
                     "trackkeepthreshold": len(TRACK_KEEP_THRESHOLDS),
                     "trackcreateobjectthreshold": len(TRACK_CREATE_OBJECT_THRESHOLDS),
                     "trackmatchlocationthreshold": len(TRACK_MATCH_LOCATION_THRESHOLDS),
                     "trackkfposweight": len(TRACK_KF_POS_WEIGHT),
                     "trackkfvelweight": len(TRACK_KF_VEL_WEIGHT),
                     "postprocessstrategy": len(POSTPROCESS_NAMES),
                     "noisefilterflag": 2}

PREDICT_TYPE_DICT = {"skipnumber": "R",
                     "otifskipnumber": "R",
                     "skyskipnumber": "R",
                     "scaledownresolution": "R",
                     "detectname": "C",
                     "detectsize": "R",
                     "detectthreshold": "R",
                     "trackname": "C",
                     "trackmaxlosttime": "R",
                     "trackkeepthreshold": "R",
                     "trackmatchlocationthreshold": "R",
                     "trackkfposweight": "R",
                     "trackkfvelweight": "R"}

BOUND_DICT = {"skipnumber": 256,
              "otifskipnumber": 64,
              "skyskipnumber": 60,
              "scaledownresolution": 736,
              "detectname": 9,
              "detectsize": 5,
              "detectthreshold": 1,
              "trackname": 4,
              "trackmaxlosttime": 512,
              "trackkeepthreshold": 1,
              "trackmatchlocationthreshold": 2,
              "trackkfposweight": 24,
              "trackkfvelweight": 24}

# BOOL: 0; INT:1; ENUM: 2; FLOAT: 3;
SEARCH_SPACE_TYPE_DICT = {"skipnumber": 1,
                          "otifskipnumber": 1,
                          "skyskipnumber": 1,
                          "filterflag": 0, "filterresolution": 1, "filterthreshold": 3,
                          "scaledownresolution": 1,
                          "roiflag": 0,
                          "roiresolution": 1,
                          "roithreshold": 3,
                          "denoisingflag": 0,
                          "equalizationflag": 0,
                          "sharpeningflag": 0,
                          "saturationflag": 0,
                          "detectname": 2,
                          "detectsize": 1,
                          "detectthreshold": 3,
                          "trackname": 2,
                          "trackmaxlosttime": 1,
                          "trackkeepthreshold": 3,
                          "trackcreateobjectthreshold": 3,
                          "trackmatchlocationthreshold": 3,
                          "trackkfposweight": 3,
                          "trackkfvelweight": 3,
                          "postprocessstrategy": 2,
                          "noisefilterflag": 0}

TYPE_GRAPH = {"0_0", "0_1", "0_2", "0_3",
              "1_1", "1_2", "1_3",  # "1_0",
              "2_1", "2_2", "2_3",  # "2_0",
              "3_2", "3_3"}         # "3_0", "3_1",

# Frame Selection: 0
# Frame Preprocessing: 1
# Object Detection: 2
# Object Tracking: 3
# Result Refinement: 4
SEARCH_SPACE_MODULE_DICT = {"skipnumber": 0,
                            "otifskipnumber": 0,
                            "skyskipnumber": 0,
                            "filterflag": 0, "filterresolution": 0, "filterthreshold": 0,
                            "scaledownresolution": 1,
                            "roiflag": 1,
                            "roiresolution": 1,
                            "roithreshold": 1,
                            "denoisingflag": 1,
                            "equalizationflag": 1,
                            "sharpeningflag": 1,
                            "saturationflag": 1,
                            "detectname": 2,
                            "detectsize": 2,
                            "detectthreshold": 2,
                            "trackname": 3,
                            "trackmaxlosttime": 3,
                            "trackkeepthreshold": 3,
                            "trackcreateobjectthreshold": 3,
                            "trackmatchlocationthreshold": 3,
                            "trackkfposweight": 3,
                            "trackkfvelweight": 3,
                            "postprocessstrategy": 4,
                            "noisefilterflag": 4}

MODULE_GRAPH = {"skipnumber_scaledownresolution", "skipnumber_trackname", "skipnumber_trackmaxlosttime", "skipnumber_trackkeepthreshold", "skipnumber_trackmatchlocationthreshold",
                "scaledownresolution_skipnumber", "scaledownresolution_detectname", "scaledownresolution_detectsize", "scaledownresolution_detectthreshold", "scaledownresolution_trackname", "scaledownresolution_trackkeepthreshold",
                "detectname_scaledownresolution", "detectname_detectsize", "detectname_detectthreshold", "detectname_trackname", "detectname_trackkeepthreshold",
                "detectsize_scaledownresolution", "detectsize_detectthreshold", "detectsize_trackkeepthreshold",
                "trackname_skipnumber", "trackname_scaledownresolution", "trackname_detectname", "trackname_trackmaxlosttime", "trackname_trackkeepthreshold", "trackname_trackmatchlocationthreshold",
                "trackmaxlosttime_skipnumber", "trackmaxlosttime_trackname", "trackmaxlosttime_trackkeepthreshold", "trackmaxlosttime_trackmatchlocationthreshold",
                "trackkeepthreshold_skipnumber", "trackkeepthreshold_scaledownresolution", "trackkeepthreshold_detectname", "trackkeepthreshold_detectsize", "trackkeepthreshold_detectthreshold", "trackkeepthreshold_trackname",
                "trackmatchlocationthreshold_skipnumber", "trackmatchlocationthreshold_trackname", "trackmatchlocationthreshold_trackmaxlosttime"}

# the parameter space of the search method
KNOB_TYPES_DICT = {"skipnumber": 0, "otifskipnumber": 0, "skyskipnumber": 0,
                   "filterflag": 0, "filterresolution": 0, "filterthreshold": 0,
                   "scaledownresolution": 1, "roiflag": 1, "roiresolution": 1, "roithreshold": 1,
                   "denoisingflag": 1, "equalizationflag": 1, "sharpeningflag": 1, "saturationflag": 1,
                   "detectname": 2, "detectsize": 2, "detectthreshold": 2,
                   "trackname": 3, "trackmaxlosttime": 3, "trackkeepthreshold": 3,
                   "trackcreateobjectthreshold": 3, "trackmatchlocationthreshold": 3,
                   "trackkfposweight": 3, "trackkfvelweight": 3,
                   "postprocessstrategy": 4, "noisefilterflag": 4}

# the best config
GOLDEN_CONFIG_DICT = {"skipnumber": len(SKIPS) - 1,
                      "otifskipnumber": len(OTIFSKIPS) - 1,
                      "skyskipnumber": len(SKYSKIPS) - 1,
                      "filterflag": 1,
                      "filterresolution": len(FILTER_RESOLUTIONS) - 1,
                      "filterthreshold": -1,
                      "scaledownresolution": len(SCALEDOWN_RESOLUTIONS) - 1,
                      "roiflag": 1,
                      "roiresolution": len(ROI_RESOLUTIONS) - 1,
                      "roithreshold": -1,
                      "denoisingflag": 1,
                      "equalizationflag": 1,
                      "sharpeningflag": 1,
                      "saturationflag": 1,
                      "detectname": 1,
                      "detectsize": len(DETECT_SIZES) - 1,
                      "detectthreshold": len(DETECT_THRESHOLDS) - 1,
                      "trackname": 1,
                      "trackmaxlosttime": 1,
                      "trackkeepthreshold": 2,
                      "trackcreateobjectthreshold": 7,
                      "trackmatchlocationthreshold": len(TRACK_MATCH_LOCATION_THRESHOLDS) - 1,
                      "trackkfposweight": 0,
                      "trackkfvelweight": 0,
                      "postprocessstrategy": -1,
                      "noisefilterflag": 0}

# the cheapest config
KMINUS_CONFIG_DICT = {"skipnumber": 0,
                      "otifskipnumber": 0,
                      "skyskipnumber": 0,
                      "filterflag": 0,
                      "filterresolution": -1,
                      "filterthreshold": -1,
                      "scaledownresolution": 0,
                      "roiflag": 0,
                      "roiresolution": -1,
                      "roithreshold": -1,
                      "denoisingflag": 0,
                      "equalizationflag": 0,
                      "sharpeningflag": 0,
                      "saturationflag": 0,
                      "detectname": 1,
                      "detectsize": 0,
                      "detectthreshold": 0,
                      "trackname": 1,
                      "trackmaxlosttime": -1,
                      "trackkeepthreshold": 0,
                      "trackcreateobjectthreshold": 0,
                      "trackmatchlocationthreshold": 0,
                      "trackkfposweight": 0,
                      "trackkfvelweight": 0,
                      "postprocessstrategy": 0,
                      "noisefilterflag": 0}

# the qualitative config
KPLUS_CONFIG_DICT = {"skipnumber": -1,
                     "otifskipnumber": -1,
                     "skyskipnumber": -1,
                     "filterflag": 0,
                     "filterresolution": -1,
                     "filterthreshold": -1,
                     "scaledownresolution": -1,
                     "roiflag": 0,
                     "roiresolution": -1,
                     "roithreshold": -1,
                     "denoisingflag": 0,
                     "equalizationflag": 0,
                     "sharpeningflag": 0,
                     "saturationflag": 0,
                     "detectname": 1,
                     "detectsize": -1,
                     "detectthreshold": 2,
                     "trackname": 1,
                     "trackmaxlosttime": 2,
                     "trackkeepthreshold": 5,
                     "trackcreateobjectthreshold": -3,
                     "trackmatchlocationthreshold": 0,
                     "trackkfposweight": 0,
                     "trackkfvelweight": 0,
                     "postprocessstrategy": 2,
                     "noisefilterflag": 0}

# config index
CONFIG_INDEX_DICT = {"skipnumber": ['videobase', 'skipnumber'],
                     "otifskipnumber": ['videobase', 'skipnumber'],
                     "skyskipnumber": ['videobase', 'skipnumber'],
                     "filterflag": ['filterbase', 'flag'],
                     "filterresolution": ['filterbase', 'resolution'],
                     "filterthreshold": ['filterbase', 'threshold'],
                     "scaledownresolution": ['videobase', 'scaledownresolution'],
                     "roiflag": ['roibase', 'flag'],
                     "roiresolution": ['roibase', 'resolution'],
                     "roithreshold": ['roibase', 'threshold'],
                     "denoisingflag": ['roibase', 'denoising'],
                     "equalizationflag": ['roibase', 'equalization'],
                     "sharpeningflag": ['roibase', 'sharpening'],
                     "saturationflag": ['roibase', 'saturation'],
                     "detectname": ['detectbase', 'modeltype'],
                     "detectsize": ['detectbase', 'modelsize'],
                     "detectthreshold": ['detectbase', 'threshold'],
                     "trackname": ['trackbase', 'modeltype'],
                     "trackmaxlosttime": ['trackbase', 'maxlosttime'],
                     "trackkeepthreshold": ['trackbase', 'keepthreshold'],
                     "trackcreateobjectthreshold": ['trackbase', 'createobjectthreshold'],
                     "trackmatchlocationthreshold": ['trackbase', 'matchlocationthreshold'],
                     "trackkfposweight": ['trackbase', 'kfposweight'],
                     "trackkfvelweight": ['trackbase', 'kfvelweight'],
                     "postprocessstrategy": ['postprocessbase', 'postprocesstype'],
                     "noisefilterflag": ['noisefilterbase', 'flag']}

# action index
ACTION_INDEX_DICT = {"skipnumber": SKIPS,
                     "otifskipnumber": OTIFSKIPS,
                     "skyskipnumber": SKYSKIPS,
                     "filterflag": [0, 1],
                     "filterresolution": FILTER_RESOLUTIONS,
                     "filterthreshold": FILTER_THRESHOLDS,
                     "scaledownresolution": SCALEDOWN_RESOLUTIONS,
                     "roiflag": [0, 1],
                     "roiresolution": ROI_RESOLUTIONS,
                     "roithreshold": ROI_THRESHOLDS,
                     "denoisingflag": [0, 1],
                     "equalizationflag": [0, 1],
                     "sharpeningflag": [0, 1],
                     "saturationflag": [0, 1],
                     "detectname": DETECT_NAMES,
                     "detectsize": DETECT_SIZES,
                     "detectthreshold": DETECT_THRESHOLDS,
                     "trackname": TRACK_NAMES,
                     "trackmaxlosttime": TRACK_MAX_LOST_TIMES,
                     "trackkeepthreshold": TRACK_KEEP_THRESHOLDS,
                     "trackcreateobjectthreshold": TRACK_CREATE_OBJECT_THRESHOLDS,
                     "trackmatchlocationthreshold": TRACK_MATCH_LOCATION_THRESHOLDS,
                     "trackkfposweight": TRACK_KF_POS_WEIGHT,
                     "trackkfvelweight": TRACK_KF_VEL_WEIGHT,
                     "postprocessstrategy": POSTPROCESS_NAMES,
                     "noisefilterflag": [0, 1]}

GREEDY_HILL_SET = [0, 2, 4, 6, 13]
NOGREEDY_HILL_SET = [1, 3, 5, 7, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
