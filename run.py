import json
import subprocess
import numpy as np

def save_json(data, json_file):
    with open(json_file, 'w') as file:
        json.dump(data, file)

def run_all_commands(run_commands):
    for commands in run_commands:
        processes = []

        for cmd in commands:
            process = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            processes.append(process)

        for p in processes:
            p.wait()

def read_json(json_file):
    data = json.load(open(json_file, 'r'))
    return data

def combine_result(cluster, channel, cacheresultdir, method_name, objects, run_ids):
    final_result = {"Latency": {j: -9999 for j, _ in enumerate(run_ids)}}
    camera_id_number = {objectname: 0 for objectname in objects}
    for i, cluster_id in enumerate(cluster):
        batch_i = -1
        for j, ids in enumerate(run_ids):
            if i in ids:
                batch_i = j
                break

        if isinstance(cluster[cluster_id], dict):
            if "ids" in cluster[cluster_id]:
                camera_ids = cluster[cluster_id]['ids']
            else:
                camera_ids = cluster[cluster_id]['points']
            camera_ids = [pi for pi in camera_ids]
        else:
            camera_ids = [pi for pi in cluster[cluster_id]]
        ingestion_result_path = f'{cacheresultdir}/{method_name}_channel{channel}_cluster_{cluster_id}.json'
        ingestion_result = read_json(ingestion_result_path)

        for objectname in objects:
            if ingestion_result[f"{objectname.capitalize()}GtCount"] > 0:
                camera_id_number[objectname] += len(camera_ids)

        if i == 0:
            for key in ingestion_result:
                if "latency" == key.lower():
                    final_result[key][batch_i] = max(
                        final_result[key][batch_i], ingestion_result[key])
                else:
                    hitobjectname = None
                    for objectname in objects:
                        if objectname.lower() in key.lower():
                            hitobjectname = objectname
                            break
                    if hitobjectname is None:
                        continue
                    if ingestion_result[f"{hitobjectname.capitalize()}GtCount"] <= 0:
                        final_result[key] = 0
                    else:
                        if "count" in key.lower():
                            final_result[key] = ingestion_result[key]
                        else:
                            final_result[key] = ingestion_result[key] * \
                                len(camera_ids)
        else:
            for key in ingestion_result:
                if "latency" == key.lower():
                    final_result[key][batch_i] = max(
                        final_result[key][batch_i], ingestion_result[key])
                else:
                    hitobjectname = None
                    for objectname in objects:
                        if objectname.lower() in key.lower():
                            hitobjectname = objectname
                            break
                    if hitobjectname is None:
                        continue
                    if ingestion_result[f"{hitobjectname.capitalize()}GtCount"] <= 0:
                        final_result[key] += 0
                    else:
                        if "count" in key.lower():
                            final_result[key] += ingestion_result[key]
                        else:
                            final_result[key] += ingestion_result[key] * \
                                len(camera_ids)
    for key in final_result:
        hitobjectname = None
        for objectname in objects:
            if objectname.lower() in key.lower():
                hitobjectname = objectname
                break
        if key.lower() == "latency":
            print(final_result[key].values())
            final_result[key] = sum(final_result[key].values())
            continue
        if key.lower() != "latency" and "count" not in key.lower():
            if camera_id_number[hitobjectname] <= 0:
                final_result[key] = 0
            else:
                final_result[key] /= camera_id_number[hitobjectname]

    return final_result

method_name = "test_parrallel"
channel = 777
objects = ["car", "bus", "truck"]

run_ids = [[0]]
pareto_config_paths = ['./cases/metric/Car_hotas_aggs_bottom_394_0.00_0.70.yaml']

camera_number = 100

clusters = {}
run_cmds = []
for ids in run_ids:
    cmds = []
    for ri in ids:
        pareto_config_path = pareto_config_paths[ri]
        ingestion_result_path = f'./{method_name}_channel{channel}_cluster_{ri}.json'

        camera_ids = [ci for ci in range(camera_number) if ci % (ri + 1) == 0]
        camera_ids = ",".join([str(camera_id) for camera_id in camera_ids])
        clusters[ri] = {"ids": camera_ids}
        cmds.append(
            f"go run step.go {pareto_config_path} {ingestion_result_path} {camera_ids}")
    run_cmds.append(cmds)

run_all_commands(run_cmds)

final_result = combine_result(
    clusters, channel, "./", method_name, objects, run_ids)
print(json.dumps(final_result, indent=4))
save_json(final_result, f"./{method_name}_channel{channel}_final_result.json")

COUNTs, HOTAs, MOTAs, IDF1s = [], [], [], []
for objectname in objects:
    count = final_result[f"{objectname.capitalize()}GtCount"]
    hota = final_result[f"{objectname.capitalize()}Hota"]
    mota = final_result[f"{objectname.capitalize()}Mota"]
    idf1 = final_result[f"{objectname.capitalize()}Idf1"]
    HOTAs.append(hota * count)
    MOTAs.append(mota * count)
    IDF1s.append(idf1 * count)
    COUNTs.append(count)
final_output_path = f"./{method_name}_channel{channel}_final_result.txt"

print(json.dumps(final_result), file=open(final_output_path, "w"))
print(f"COUNT: {COUNTs}", file=open(final_output_path, "a"))
print(f"HOTA: {HOTAs}", file=open(final_output_path, "a"))
print(f"MOTA: {MOTAs}", file=open(final_output_path, "a"))
print(f"IDF1: {IDF1s}", file=open(final_output_path, "a"))
if sum(COUNTs) > 0:
    print("Mean HOTA: {:.4f}".format(
        np.sum(HOTAs) / np.sum(COUNTs)), file=open(final_output_path, "a"))
    print("Mean MOTA: {:.4f}".format(
        np.sum(MOTAs) / np.sum(COUNTs)), file=open(final_output_path, "a"))
    print("Mean IDF1: {:.4f}".format(
        np.sum(IDF1s) / np.sum(COUNTs)), file=open(final_output_path, "a"))
    print(f"Latency: {final_result['Latency']}", file=open(
        final_output_path, "a"))
else:
    print("Mean HOTA: 0", file=open(final_output_path, "a"))
    print("Mean MOTA: 0", file=open(final_output_path, "a"))
    print("Mean IDF1: 0", file=open(final_output_path, "a"))
    print(f"Latency: {final_result['Latency']}", file=open(
        final_output_path, "a"))