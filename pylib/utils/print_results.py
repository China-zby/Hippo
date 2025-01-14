import os
import yaml
import pickle
import argparse
import pandas as pd


def mean(datalist):
    return sum(datalist) / len(datalist)


parse = argparse.ArgumentParser()
parse.add_argument("--resultdir", default="./", help="query data root")
parse.add_argument("--classes", type=str, default="car,bus,truck")
parse.add_argument("--methodname", type=str, default="otif")
parse.add_argument("--datasetname", type=str, default="amsterdam")
parse.add_argument("--videoidlist", type=str, default="0-1-2-3-4")
parse.add_argument("--skipframelist", type=str, default="8-8-16-8-32")
parse.add_argument("--testmode", type=str, default="test_separate_video")
parse.add_argument("--qeurytime", type=float)
args = parse.parse_args()

methodname, datasetname, datasettype = args.methodname, args.datasetname, args.testmode
result_dir = args.resultdir
Classes = args.classes.split(',')
MOTMetrics = ["HOTA", "MOTA", "IDF1"]
videoids = list(map(int, args.videoidlist.split("-")))
skipframes = list(map(int, args.skipframelist.split("-")))
skipframesets = set(skipframes)

Results = pickle.load(
    open(os.path.join(result_dir, f"query_matrics.pkl"), "rb"))

PrintResults = {"process time": args.qeurytime}

for object_class in Classes:
    Results[object_class][f"HOTA_{object_class}"] = []
    PrintResults[object_class] = {}

for skipframe in skipframesets:
    videologo = f"{datasetname}S{skipframe}-{datasettype}"
    save_dir = f"./TrackEval/data/trackers/videodb/{videologo}/{methodname}"
    for object_class in Classes:
        if not os.path.exists(os.path.join(save_dir, f"{object_class}_detailed.csv")):
            continue
        objectPath = os.path.join(save_dir, f"{object_class}_detailed.csv")
        df = pd.read_csv(objectPath)
        df = df[df['GT_IDs'] > 0]
        HOTAlist = list(df['HOTA___AUC'])

        Results[object_class][f"HOTA_{object_class}"].extend(HOTAlist)

for object_class in Classes:
    if len(Results[object_class][f"HOTA_{object_class}"]) == 0:
        Results[object_class][f"HOTA_{object_class}"] = None
    else:
        Results[object_class][f"HOTA_{object_class}_mean"] = mean(
            Results[object_class][f"HOTA_{object_class}"])

for object_class in Classes:
    for metric in list(Results[object_class].keys()):
        if "mean" not in metric:
            continue
        # isinstance(Results[object_class][metric], int) or
        if Results[object_class][metric] is None:
            # !! 注意这里～～～～～～
            PrintResults[object_class][metric[:-len("_mean")]] = 0.0
            continue
        #     PrintResults[object_class][metric[:-len("_mean")]] = 0.0
        # else:
        PrintResults[object_class][metric[:-len("_mean")]] = round(
            Results[object_class][metric], 3)

print(yaml.dump(PrintResults, sort_keys=False, default_flow_style=False))
if not os.path.exists(f"./result/{methodname}/{datasetname}"):
    os.makedirs(f"./result/{methodname}/{datasetname}")

with open(f"./result/{methodname}/{datasetname}/results.yaml", "w") as f:
    yaml.dump(PrintResults, f, sort_keys=False, default_flow_style=False)
