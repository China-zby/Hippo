import os
import sys
import pickle

methodname = sys.argv[1]
datasetname = sys.argv[2]
boundline = float(sys.argv[3])
chosen_object = sys.argv[4]
chosen_metric = sys.argv[5]
resultDir = "./result/"

query_result = pickle.load(open(os.path.join(
    resultDir, methodname, datasetname, f"query_matrics.pkl"), "rb"))

if chosen_metric == 'sel':
    chosen_result = query_result[chosen_object]['f1']
    safenum = len([x for x in chosen_result if x > boundline])
    print(f"Safe Rate is {safenum}/{len(chosen_result)} = {safenum/len(chosen_result)}")