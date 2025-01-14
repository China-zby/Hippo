import json
import itertools
from tqdm import tqdm
from config_ import Config
from cameraclass import build_a_camera_with_config
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from utils import SEARCH_SPACE_NAMES, EFFE_GPU_SPACE_NAMES, SEARCH_SPACE, DataPaths, \
    generate_action_from_config, generate_config, makedirs

if __name__ == "__main__":
    scene_name = "hippo"
    method_name = "gpu"
    configpath = "./cache/info.yaml"
    deviceid, data_name, data_type, objects, metrics = 1, 'hippo', 'train', [
        "car", "bus", "truck"], ["sel", "agg", "topk"]
    alldir = DataPaths('/home/lzp/otif-dataset/dataset',
                       data_name, data_type, method_name)
    makedirs([alldir.visdir,
              alldir.cachedir,
              alldir.graphdir,
              alldir.recorddir,
              alldir.framedir,
              alldir.storedir,
              alldir.cacheconfigdir,
              alldir.cacheresultdir])
    cameraConfig = Config(
        configpath, 'gpu.yaml',
        deviceid, data_name, data_type, objects,
        configpath, scene_name)
    random_camera = build_a_camera_with_config(
        alldir, cameraConfig, objects, metrics)
    config_vector = generate_action_from_config(cameraConfig.load_cache())
    GPU_DICT = {}
    all_config_list = list(itertools.product(
        *[list(range(SEARCH_SPACE[SEARCH_SPACE_NAMES.index(name)])) for name in EFFE_GPU_SPACE_NAMES]))
    custom_columns = [
        "[progress.description]{task.description}",
        BarColumn(bar_width=30),  
        "[progress.percentage]{task.percentage:>3.0f}%",  
        "(", TextColumn(
            "[progress.completed]{task.completed}"), "/", TextColumn("{task.total}"), ")",
        TimeRemainingColumn(),  
    ]
    with Progress(*custom_columns) as progress:
        task = progress.add_task("[cyan]Processing...", total=len
                                 (all_config_list))
        for combination in all_config_list:
            for effei, name in enumerate(EFFE_GPU_SPACE_NAMES):
                config_vector[SEARCH_SPACE_NAMES.index(
                    name)] = combination[effei]
            random_camera.updateConfig(generate_config(config_vector,
                                                       random_camera.loadConfig()))
            random_camera.ingestion_without_cache
            gpu_record = json.loads(
                open(random_camera.config.cache_dir + "GPURecord.json").read())
            max_gpu_memory = 0
            for pid in gpu_record:
                max_gpu_memory += int(gpu_record[pid].rstrip("MiB")[:-1])

            GPU_DICT["_".join(map(str, config_vector))] = max_gpu_memory
            progress.update(task, advance=1)
    with open("./GPUInfo.json", "w") as f:
        f.write(json.dumps(GPU_DICT))
