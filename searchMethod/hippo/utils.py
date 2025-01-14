import os
import re
import cv2
import time
import yaml
import json
import math
import torch
import struct
import pickle
import random
import hashlib
import colorsys
import threading
import subprocess
import numpy as np
from PIL import Image
import multiprocessing
from pathlib import Path
from copy import deepcopy
import matplotlib.pyplot as plt
from collections import defaultdict
from alive_progress import config_handler

from css import *

ACCURACY_MASK_VALUE = 0.5

OTIF_SEARCH_SPACE_NAMES = ["otifskipnumber", "scaledownresolution",
                           "detectname", "detectthreshold"]

SKY_SEARCH_SPACE_NAMES = ["skyskipnumber", "scaledownresolution",
                          "detectsize", "trackmaxlosttime"]

MED_SEARCH_SPACE_NAMES = ["skipnumber", "scaledownresolution",
                      "detectname", "detectsize",
                      "detectthreshold", "trackname",
                      "trackmaxlosttime", "trackkeepthreshold"]

SEARCH_SPACE_NAMES = ["skipnumber", "scaledownresolution",
                      "detectname", "detectsize",
                      "detectthreshold", "trackname",
                      "trackmaxlosttime", "trackkeepthreshold",
                      "trackmatchlocationthreshold",
                      "trackkfposweight",
                      "trackkfvelweight"]

EFFE_GPU_SPACE_NAMES = ["scaledownresolution",
                        "detectname", "detectsize", "trackname"]

OTIF_TUNE_SPACE_NAMES = ["skipnumber", "scaledownresolution"]

SEARCH_SPACE_PREDICT_TYPE = [PREDICT_TYPE_DICT[search_space_name]
                             for search_space_name in SEARCH_SPACE_NAMES]

SEARCH_SPACE_BOUND = [BOUND_DICT[search_space_name]
                      for search_space_name in SEARCH_SPACE_NAMES]

SEARCH_SPACE = [SEARCH_SPACE_DICT[search_space_name]
                for search_space_name in SEARCH_SPACE_NAMES]
OTIF_SEARCH_SPACE = [SEARCH_SPACE_DICT[search_space_name]
                    for search_space_name in OTIF_SEARCH_SPACE_NAMES]
SKY_SEARCH_SPACE = [SEARCH_SPACE_DICT[search_space_name]
                    for search_space_name in SKY_SEARCH_SPACE_NAMES]
MED_SEARCH_SPACE = [SEARCH_SPACE_DICT[search_space_name]
                    for search_space_name in MED_SEARCH_SPACE_NAMES]

SEARCH_SPACE_TYPE = [SEARCH_SPACE_TYPE_DICT[search_space_name]
                     for search_space_name in SEARCH_SPACE_NAMES]
SEARCH_SPACE_MODULE = [SEARCH_SPACE_MODULE_DICT[search_space_name
                                                ] for search_space_name in SEARCH_SPACE_NAMES]
GOLDEN_CONFIG_VECTOR = [GOLDEN_CONFIG_DICT[search_space_name]
                        for search_space_name in SEARCH_SPACE_NAMES]
SKY_GOLDEN_CONFIG_VECTOR = [GOLDEN_CONFIG_DICT[search_space_name]
                            for search_space_name in SKY_SEARCH_SPACE_NAMES]
MED_GOLDEN_CONFIG_VECTOR = [GOLDEN_CONFIG_DICT[search_space_name]
                            for search_space_name in MED_SEARCH_SPACE_NAMES]
OTIF_GOLDEN_CONFIG_VECTOR = [GOLDEN_CONFIG_DICT[search_space_name]
                             for search_space_name in OTIF_SEARCH_SPACE_NAMES]
KMINUS_CONFIG_VECTOR = [KMINUS_CONFIG_DICT[search_space_name]
                        for search_space_name in SEARCH_SPACE_NAMES]
KPLUS_CONFIG_VECTOR = [KPLUS_CONFIG_DICT[search_space_name]
                       for search_space_name in SEARCH_SPACE_NAMES]
KNOB_TYPES = [KNOB_TYPES_DICT[search_space_name]
              for search_space_name in SEARCH_SPACE_NAMES]
POS_TYPES = list(range(len(SEARCH_SPACE)))


def load_parrallel_latency(latency_paths):
    parallel_latency = []
    for latency_path in latency_paths:
        with open(latency_path, 'r') as file:
            ingestion_result = json.load(file)
        ingestion_latency = ingestion_result["Latency"]
        parallel_latency.append(ingestion_latency)
    return parallel_latency


def search_file(directory, filename):
    file_path = Path(directory) / filename
    return file_path.exists()


def distance_matrix(tensor):
    # Compute the square of the tensor
    tensor_sq = torch.sum(tensor ** 2, dim=1, keepdim=True)

    # Compute the distance matrix
    dist_matrix = tensor_sq - 2 * \
        torch.matmul(tensor, tensor.t()) + tensor_sq.t()

    # Ensure the distance matrix is non-negative and take the square root
    dist_matrix = torch.sqrt(torch.clamp(dist_matrix, min=0.0))

    return dist_matrix


def discrete_sampling(data, num_samples=10):
    length = len(data)
    if length <= num_samples:
        return data
    else:
        # 计算采样间隔
        interval = length / num_samples
        # 离散采样
        return [data[int(i * interval)] for i in range(num_samples)]


def recursive_defaultdict():
    return defaultdict(recursive_defaultdict)


def print_golden_config():
    print("Golden config:")
    print(json.dumps(generate_config(GOLDEN_CONFIG_VECTOR, None)))


def load_record(recordpath, objects, metrics):
    with open(recordpath) as file:
        record = json.load(file)
    hotas, counts = [], []
    motas, idf1s = [], []
    for objectname in objects:
        if record[f'{objectname.capitalize()}GtCount'] > 0:
            hotas.append(
                max(record[f'{objectname.capitalize()}Hota'], 0.0))
            motas.append(
                max(record[f'{objectname.capitalize()}Mota'], 0.0))
            idf1s.append(
                max(record[f'{objectname.capitalize()}Idf1'], 0.0))
        else:
            hotas.append(0.0)
            motas.append(0.0)
            idf1s.append(0.0)
        counts.append(record[f'{objectname.capitalize()}GtCount'])
    total_counts = sum(counts)
    weighted_avg_sum = sum(hota * count + mota * count + idf1 * count
                           for hota, mota, idf1, count in zip(hotas, motas, idf1s, counts))
    motvalue = weighted_avg_sum / \
        (3 * total_counts) if total_counts > 0 else ACCURACY_MASK_VALUE

    cmetric, cmetricname = [], []
    for objectname in objects:
        gtcount = record[objectname.capitalize()+"GtCount"]
        predcount = record[objectname.capitalize()+"PredCount"]
        record[objectname.capitalize()+"CountRate"] = 1 - abs(gtcount -
                                                              predcount) / gtcount if gtcount != 0 else float(predcount == 0)
        record[objectname.capitalize()+"Sel"] = record[objectname.capitalize() +
                                                       "Sel"] if gtcount != 0 else float(predcount == 0)
        record[objectname.capitalize()+"Agg"] = record[objectname.capitalize() +
                                                       "Agg"] if gtcount != 0 else float(predcount == 0)
    for objectname in objects:
        del record[objectname.capitalize()+"GtCount"]
        del record[objectname.capitalize()+"PredCount"]
    objects_lower = [x.lower() for x in objects]
    metrics_lower = [x.lower() for x in metrics]
    for qn in record:
        qn_lower = qn.lower()
        if any(classname in qn_lower for classname in objects_lower):
            if any(metricname in qn_lower for metricname in metrics_lower):
                cmetric.append(record[qn])
                cmetricname.append(qn)
    cmetric.append(record['Latency'])
    os.remove(recordpath)
    return cmetric, motvalue


def plot_from_xy(x, y, save_path, xlabel=None, ylabel=None):
    plt.scatter(x, y, s=25, c='b', marker="x")
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    k, b = np.polyfit(x, y, 1)
    plt.plot(x, k * np.array(x) + b, c='r')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    plt.clf()
    plt.cla()
    return k, b


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# class DataDir():
#     def __init__(self, ):


class DataPaths:
    def __init__(self, datadir, data_name, data_type, method_name):
        self.weightdir = f"{datadir}/{data_name}/train"
        self.cameradir = f"{datadir}/{data_name}/{data_type}"
        self.cachedir = f"./cache/{method_name}"
        self.videodir = f"{self.cameradir}/video"
        self.trackdir = f"{self.cameradir}/tracks"
        self.cacheconfigdir = f"{self.cachedir}/configs"
        self.cacheresultdir = f"{self.cachedir}/results"
        self.framedir = f"./{self.cameradir}/cframes"
        self.plandir = f"./plans/{method_name}"
        self.graphdir = f"{self.cachedir}/graphs"
        self.recorddir = f"{self.cachedir}/records"
        self.visdir = f"{self.cachedir}/vis"
        self.storedir = f"{self.cachedir}/store"


def makedirs(dirlist):
    if isinstance(dirlist, list):
        for ddir in dirlist:
            if not os.path.exists(ddir):
                os.makedirs(ddir, exist_ok=True)
    else:
        n_dirlist = [dirlist.visdir, dirlist.cachedir,
                     dirlist.graphdir, dirlist.recorddir,
                     dirlist.framedir, dirlist.storedir,
                     dirlist.cacheconfigdir, dirlist.cacheresultdir]
        for nddir in n_dirlist:
            if not os.path.exists(nddir):
                os.makedirs(nddir, exist_ok=True)


def mean(x):
    return sum(x) / len(x)


def softmax(x, t=1.0):
    e_x = [math.exp(i * t) for i in x]
    return [i/sum(e_x) for i in e_x]


def summax(x):
    sum_x = sum(x)
    return [i/sum_x for i in x]


class OUProcess(object):
    def __init__(self, n_actions, mu=0, theta=0.15, sigma=0.2, dt=1e-2):
        self.n_actions = n_actions
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.state = np.ones(n_actions) * mu

    def reset(self, sigma):
        self.sigma = sigma
        self.state = np.ones(self.n_actions) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) * self.dt + self.sigma * \
            np.sqrt(self.dt) * np.random.randn(self.n_actions)
        self.state += dx
        return self.state


def get_state(video_states, context_vector, num_objects=5):
    video_object_number = []
    video_chosen_states = []
    for video_state in video_states:
        video_state_array = np.array(video_state)
        frame_object_number = np.sum(np.abs(video_state_array[:, :2]), axis=1)
        frame_idx = np.argmax(frame_object_number)
        video_object_number.append(frame_object_number[frame_idx])
        video_chosen_states.append(video_state[frame_idx])
    video_object_number = np.array(video_object_number)
    topk = np.argsort(video_object_number)[-num_objects:]
    states = []
    for i in topk:
        states += video_chosen_states[i]
    return states + context_vector


def reward_func(delta_value_0t, delta_value_tm1t):
    if delta_value_0t > 0:
        r = ((1+delta_value_0t)**2 - 1) * abs(1+delta_value_tm1t)
    else:
        r = -((1-delta_value_0t)**2 - 1) * abs(1-delta_value_tm1t)
    if r > 0 and delta_value_tm1t < 0:
        r = 0
    return r


def delta_latency_func(latency_base, latency_t):
    return (-latency_t + latency_base) / latency_base


def delta_accuracy_func(accuracy_base, accuracy_t):
    return (accuracy_t - accuracy_base) / accuracy_base


def get_reward(step_records,
               classes=['Car', 'Bus', 'Truck'],
               latency_weight=0.4,
               query_number=3):
    scores, score_weights = [
        math.exp(-step_records['Latency'] / 100.0)], [latency_weight]
    for class_name in classes:
        scores.append(step_records[f"{class_name}Sel"])
        score_weights.append((1 - latency_weight) /
                             (query_number * len(classes)))
        scores.append(step_records[f"{class_name}Agg"])
        score_weights.append((1 - latency_weight) /
                             (query_number * len(classes)))
        if step_records[f'{class_name}GtCount'] == 0:
            scores.append(0.0)
        else:
            scores.append(1.0 - abs(step_records[f'{class_name}GtCount'] -
                          step_records[f'{class_name}PredCount']) / step_records[f'{class_name}GtCount'])
        score_weights.append((1 - latency_weight) /
                             (query_number * len(classes)))

    weight_score = 0.0
    for score, weight in zip(scores, score_weights):
        weight_score += score * weight
    return weight_score


def update_target(target, source, tau):
    for (target_param, param) in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1-tau) + param.data * tau
        )


def abs_func(x, y):
    if isinstance(y, list):
        return sum([abs_func(x[i], y[i]) for i in range(len(x))])
    elif isinstance(y, int):
        return abs(x - y)
    else:
        raise ValueError("The type of y is not supported.")


def add_sample(state, action,
               reward, next_state,
               terminate,
               actor, critic,
               target_actor, target_critic,
               replay_memory, discount_factor,
               normalizer):

    batch_state = torch.FloatTensor(normalizer(state)).unsqueeze(0).cuda()
    batch_next_state = torch.FloatTensor(
        normalizer(next_state)).unsqueeze(0).cuda()
    batch_action = torch.FloatTensor(action).unsqueeze(0).cuda()
    batch_reward = torch.FloatTensor([reward]).unsqueeze(0).cuda()
    batch_terminate = torch.FloatTensor(
        [0 if x else 1 for x in [terminate]]).unsqueeze(0).cuda()

    critic.eval()
    actor.eval()
    target_critic.eval()
    target_actor.eval()

    with torch.no_grad():
        current_value = critic(batch_action, batch_state)
        target_action = target_actor(batch_next_state)
        target_value = batch_reward + batch_terminate * \
            target_critic(target_action, batch_next_state) * discount_factor
        error = float(torch.abs(current_value - target_value).cpu().numpy()[0])

    target_actor.train()
    actor.train()
    critic.train()
    target_critic.train()

    replay_memory.add(error, (state, action, reward, next_state, terminate))
    return replay_memory


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_state_actions(state_action, filename):
    f = open(filename, 'wb')
    pickle.dump(state_action, f)
    f.close()


def save_model(actor, critic, model_dir, title):
    torch.save(
        actor.state_dict(),
        '{}/{}_actor.pth'.format(model_dir, title)
    )

    torch.save(
        critic.state_dict(),
        '{}/{}_critic.pth'.format(model_dir, title)
    )


def lighten_color(hex_color, percentage=0.2):
    # Convert hex color to RGB
    rgb_color = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]

    # Convert RGB to HLS
    hls_color = colorsys.rgb_to_hls(
        rgb_color[0]/255.0, rgb_color[1]/255.0, rgb_color[2]/255.0)

    # Increase lightness
    hls_color = (hls_color[0], min(1, hls_color[1] +
                 hls_color[1]*percentage), hls_color[2])

    # Convert back to RGB
    rgb_color = colorsys.hls_to_rgb(hls_color[0], hls_color[1], hls_color[2])

    # Convert RGB to hex and return
    return '#%02x%02x%02x' % (int(rgb_color[0]*255), int(rgb_color[1]*255), int(rgb_color[2]*255))


def add_node(graph, node_id, label, shape='box', style='filled', color=None):
    if color is None:
        color_dict = {
            "Scarlet": "#FF2400",
            "Cerulean": "#007BA7",
            "Forest Green": "#228B22",
            "Magenta": "#FF00FF",
            "Amethyst": "#996622",
            "Burnt Orange": "#CC5500",
            "Coral Pink": "#F88300",
            "Sky Blue": "#87CE22",
            "Turquoise": "#E7FF00",
            "Plum Purple": "#40E0D0"
        }
        if 'SkipNumber' in label:
            color = color_dict['Scarlet']
        elif 'ScaleDown' in label:
            color = color_dict['Cerulean']
        elif 'Filter' in label and 'Noise' not in label:
            color = color_dict['Forest Green']
        elif 'ROI' in label:
            color = color_dict['Magenta']
        elif label in ENHANCE_TOOLS:
            color = color_dict['Amethyst']
        elif 'Detect' in label:
            color = color_dict['Burnt Orange']
        elif 'Track' in label:
            color = color_dict['Coral Pink']
        elif 'PostProcess' in label:
            color = color_dict['Sky Blue']
        elif 'NoiseFilter' in label:
            color = color_dict['Turquoise']
        else:
            color = color_dict['Plum Purple']
        color = lighten_color(color[1:], 0.75)

    graph.add_node(
        node_id, label=label, color='black',
        fillcolor=color, fontsize=30,
        shape=shape, style=style, fontname='Times New Roman',
    )


def read_json(json_file):
    data = json.load(open(json_file, 'r'))
    return data


def read_yaml(yaml_file):
    data = yaml.safe_load(open(yaml_file, 'r'))
    return data


def save_json(data, json_file):
    with open(json_file, 'w') as file:
        json.dump(data, file)


def save_yaml(data, yaml_file):
    with open(yaml_file, 'w') as file:
        yaml.dump(data, file)


def iterate_with_bar(total, bar):
    config_handler.set_global(receipt_text=True, dual_line=True)
    for i in range(total):
        yield i
        bar()


def read_image(image_path):
    image = Image.open(image_path)
    return image


def generate_config(action, config, search_space_names=None):
    if search_space_names is None:
        search_space_names = SEARCH_SPACE_NAMES
    # initialize config if not provided
    if config is None:
        config = recursive_defaultdict()
    config['logbase']['scene'] = config['logbase']['scene'][0] if isinstance(
        config['logbase']['scene'], list) else config['logbase']['scene']

    for search_space_name in search_space_names:
        if search_space_name in CONFIG_INDEX_DICT:
            level_1, level_2 = CONFIG_INDEX_DICT[search_space_name]
            #print(search_space_names)
            #print(search_space_name)
            # print(action)
            # print(ACTION_INDEX_DICT)
            # print(level_1)
            # print(level_2)
            # print(config)
            # print("xxxxxxxxxxxxxxxxxxxxxxxx")
            config[level_1][level_2] = ACTION_INDEX_DICT[search_space_name][int(
                action[search_space_names.index(search_space_name)])]

    return config

def generate_config_without_spac(action, config):
    # initialize config if not provided
    if config is None:
        config = recursive_defaultdict()
    config['logbase']['scene'] = config['logbase']['scene'][0] if isinstance(
        config['logbase']['scene'], list) else config['logbase']['scene']

    for search_space_name in SEARCH_SPACE_NAMES:
        if search_space_name in CONFIG_INDEX_DICT:
            level_1, level_2 = CONFIG_INDEX_DICT[search_space_name]
            # print(search_space_name, level_1, level_2, int(action[SEARCH_SPACE_NAMES.index(search_space_name)]))
            config[level_1][level_2] = action[SEARCH_SPACE_NAMES.index(
                search_space_name)]

    return config


def generate_action_from_config(config, search_space_name=None):
    if search_space_name is None:
        search_space_name = SEARCH_SPACE_NAMES
    
    action = [None] * len(search_space_name)

    for ssname in search_space_name:
        if ssname in CONFIG_INDEX_DICT:
            level_1, level_2 = CONFIG_INDEX_DICT[ssname]
            parameter_space = ACTION_INDEX_DICT[ssname]
            # print(parameter_space, ssname, level_1, level_2, config[level_1][level_2])
            action[search_space_name.index(ssname)] = parameter_space.index(
                config[level_1][level_2])

    return action


def generate_random_config_vector(spaces=None):
    if spaces is None:
        config_vector = []
        for i in range(len(SEARCH_SPACE)):
            config_vector.append(random.randint(0, SEARCH_SPACE[i] - 1))
        return config_vector
    else:
        config_vector = []
        for i in range(len(spaces)):
            config_vector.append(random.randint(0, spaces[i] - 1))
        return config_vector


def generate_random_set_config_vector(config_vector, r=1):
    config_vectors = []
    for i in range(len(SEARCH_SPACE)):
        next_config_vector = deepcopy(config_vector)
        candidate_knobs = []
        min_candidate_knob = max(config_vector[i] - r, 0)
        max_candidate_knob = min(
            config_vector[i] + r + 1, SEARCH_SPACE[i])
        for j in range(min_candidate_knob, max_candidate_knob):
            if j != config_vector[i]:
                candidate_knobs.append(j)
        if len(candidate_knobs) == 0:
            next_config_vector[i] = config_vector[i]
        else:
            next_config_vector[i] = random.choice(candidate_knobs)
        config_vectors.append(next_config_vector)
    return config_vectors


def generate_redius_config_vectors(config_vector, spaces, r=1):
    config_vectors = []
    for i in range(len(spaces)):
        next_config_vector = deepcopy(config_vector)
        candidate_knobs = []
        min_candidate_knob = max(config_vector[i] - r, 0)
        max_candidate_knob = min(
            config_vector[i] + r + 1, spaces[i])
        for j in range(min_candidate_knob, max_candidate_knob):
            if j != config_vector[i]:
                candidate_knobs.append(j)
        if len(candidate_knobs) == 0:
            next_config_vector[i] = config_vector[i]
        else:
            next_config_vector[i] = random.choice(candidate_knobs)
        config_vectors.append(next_config_vector)
    return config_vectors


def generate_golden_config_vector():
    return GOLDEN_CONFIG_VECTOR, \
        GREEDY_HILL_SET, \
        NOGREEDY_HILL_SET, SEARCH_SPACE


def generate_multichannel_config(config_vectors, config_data=None):
    if config_data is not None:
        configs = [generate_config(config_vector, config_data)
                   for config_vector in config_vectors]
    else:
        configs = config_vectors
    configids = list(range(len(configs)))
    multichannel_config = deepcopy(configs[0])
    multichannel_config['videobase']['skipnumber'] = [
        configs[i]['videobase']['skipnumber'] for i in configids]
    multichannel_config['videobase']['scaledownresolution'] = [
        configs[i]['videobase']['scaledownresolution'] for i in configids]

    # frame filter based on object number
    multichannel_config['filterbase']['flag'] = [
        configs[i]['filterbase']['flag'] for i in configids]
    multichannel_config['filterbase']['resolution'] = [
        configs[i]['filterbase']['resolution'] for i in configids]
    multichannel_config['filterbase']['threshold'] = [
        configs[i]['filterbase']['threshold'] for i in configids]
    multichannel_config['filterbase']['batchsize'] = [
        configs[i]['filterbase']['batchsize'] for i in configids]
    multichannel_config['filterbase']['modeltype'] = [
        configs[i]['filterbase']['modeltype'] for i in configids]

    # roi based on object number
    multichannel_config['roibase']['flag'] = [
        configs[i]['roibase']['flag'] for i in configids]
    multichannel_config['roibase']['resolution'] = [
        configs[i]['roibase']['resolution'] for i in configids]
    multichannel_config['roibase']['threshold'] = [
        configs[i]['roibase']['threshold'] for i in configids]
    multichannel_config['roibase']['batchsize'] = [
        configs[i]['roibase']['batchsize'] for i in configids]
    multichannel_config['roibase']['modeltype'] = [
        configs[i]['roibase']['modeltype'] for i in configids]
    multichannel_config['roibase']['windowresolution'] = [
        configs[i]['roibase']['windowresolution'] for i in configids]
    multichannel_config['roibase']['windowsizes'] = [
        configs[i]['roibase']['windowsizes'] for i in configids
    ]

    multichannel_config['roibase']['enhancetools'] = [
        configs[i]['roibase']['enhancetools'] for i in configids
    ]

    # detection based on object number
    multichannel_config['detectbase']['modeltype'] = [
        configs[i]['detectbase']['modeltype'] for i in configids]
    multichannel_config['detectbase']['modelsize'] = [
        configs[i]['detectbase']['modelsize'] for i in configids]
    multichannel_config['detectbase']['threshold'] = [
        configs[i]['detectbase']['threshold'] for i in configids]
    multichannel_config['detectbase']['batchsize'] = [
        configs[i]['detectbase']['batchsize'] for i in configids]

    # tracking based on object number
    multichannel_config['trackbase']['modeltype'] = [
        configs[i]['trackbase']['modeltype'] for i in configids]
    multichannel_config['trackbase']['maxlosttime'] = [
        configs[i]['trackbase']['maxlosttime'] for i in configids]
    multichannel_config['trackbase']['keepthreshold'] = [
        configs[i]['trackbase']['keepthreshold'] for i in configids]
    multichannel_config['trackbase']['createobjectthreshold'] = [
        configs[i]['trackbase']['createobjectthreshold'] for i in configids]
    multichannel_config['trackbase']['unmatchlocationthreshold'] = [
        configs[i]['trackbase']['unmatchlocationthreshold'] for i in configids]
    multichannel_config['trackbase']['matchlocationthreshold'] = [
        configs[i]['trackbase']['matchlocationthreshold'] for i in configids]
    multichannel_config['trackbase']['visualthreshold'] = [
        configs[i]['trackbase']['visualthreshold'] for i in configids]
    multichannel_config['trackbase']['visualmodule'] = [
        configs[i]['trackbase']['visualmodule'] for i in configids]

    # post processing based on object number
    multichannel_config['postprocessbase']['flag'] = [
        configs[i]['postprocessbase']['flag'] for i in configids]
    multichannel_config['postprocessbase']['postprocesstype'] = [
        configs[i]['postprocessbase']['postprocesstype'] for i in configids]

    # noise filter based on object number
    multichannel_config['noisefilterbase']['flag'] = [
        configs[i]['noisefilterbase']['flag'] for i in configids]
    multichannel_config['noisefilterbase']['noisefiltertype'] = [
        configs[i]['noisefilterbase']['noisefiltertype'] for i in configids]

    return multichannel_config


def save_multichannel_config(multichannel_config, v2pdict,
                             multichannel_config_path):
    save_yaml(multichannel_config, multichannel_config_path)
    deployment_path = multichannel_config_path.replace('yaml', 'txt')
    with open(deployment_path, 'w') as f:
        for key in v2pdict.keys():
            for vid in v2pdict[key]:
                f.write(f"{vid} {key}\n")
    return multichannel_config_path, deployment_path


def generate_gp_config_vector(context_vector, config_vector, gp_model, boundline, step=2):
    candidate_config_vectors = []
    for i in range(len(config_vector)):
        knob_pos = config_vector[i]
        forward_step, backward_step = knob_pos + step//2, knob_pos - step//2
        if forward_step >= SEARCH_SPACE[i]:
            forward_step = SEARCH_SPACE[i] - 1
        if backward_step < 0:
            backward_step = 0
        change_knob_space = [backward_step, forward_step, knob_pos]
        change_knob_space = list(set(change_knob_space))
        for change_knob in change_knob_space:
            config_vector_copy = deepcopy(config_vector)
            config_vector_copy[i] = change_knob
            candidate_config_vectors.append(config_vector_copy)

    # filter out the invalid config vectors
    filtered_config_vectors = []
    for candidate_config_vector in candidate_config_vectors:
        candidate_config_vector = np.asarray(
            candidate_config_vector).reshape(1, -1)
        pmean, pvariance, lmean, lvariance = gp_model.predict(
            candidate_config_vector, context_vector)
        if pmean - pvariance/2 < boundline:
            continue
        filtered_config_vectors.append([candidate_config_vector,
                                        lmean + lvariance/2])

    fastest_config_vector = max(filtered_config_vectors, key=lambda x: x[1])[0]

    return fastest_config_vector


def replace_invalid_values(tensor, search_space_size):
    mask = torch.isnan(tensor) | torch.isinf(tensor)
    new_tensor = torch.where(mask, torch.tensor(
        1.0 / search_space_size, dtype=tensor.dtype, device=tensor.device), tensor)
    return new_tensor


def eval_multichannel_perfomance(cluster_cameras, save_multiresult_path,
                                 config_vector, configpath=None):
    if isinstance(config_vector, list):
        config = generate_multichannel_config(
            [config_vector], configpath)
        config_data = json.dumps(config)
    elif isinstance(config_vector, dict):
        config_data = json.dumps(config_vector)
    else:
        config_data = config_vector

    if isinstance(cluster_cameras[0], int):
        videoids = cluster_cameras
    else:
        videoids = [camera.id for camera in cluster_cameras]
    with subprocess.Popen(['go', 'run', './searchMethod/hippo/step.go',
                           save_multiresult_path,  ','.join([str(videoid) for videoid in videoids])],
                          stdin=subprocess.PIPE,
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL, text=True) as proc:
        _, _ = proc.communicate(input=config_data)
        # print(out, err)


def eval_perfomance(cameraid, save_path,
                    config_vector, configpath=None):
    if isinstance(config_vector, list):
        config = generate_multichannel_config(
            [config_vector], configpath)
        config_data = json.dumps(config)
    elif isinstance(config_vector, dict):
        config_data = json.dumps(config_vector)
    else:
        config_data = config_vector
    with subprocess.Popen(['go', 'run', './searchMethod/hippo/step.go',
                           save_path,  f"{cameraid}"],
                          stdin=subprocess.PIPE,
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL, text=True) as proc:
        _, _ = proc.communicate(input=config_data)
        # print(out, err)


def eval_performance_a_thread(cameraid, save_path, config_vector, configpath=None):
    if isinstance(config_vector, list):
        config = generate_multichannel_config(
            [config_vector], configpath)
        config_data = json.dumps(config)
    elif isinstance(config_vector, dict):
        config_data = json.dumps(config_vector)
        # print(config_data)
    else:
        config_data = config_vector

    proc = subprocess.Popen(['go', 'run', './searchMethod/hippo/step.go', save_path, f"{cameraid}"],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, text=True)

    proc.stdin.write(config_data)
    proc.stdin.flush()

    try:
        while proc.poll() is None:
            print(proc.poll())
            nvidia_smi_output = subprocess.check_output(
                ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv']).decode('utf-8')

            print(nvidia_smi_output, proc.pid)

            for line in nvidia_smi_output.split('\n'):
                if str(proc.pid) in line:
                    print(
                        f"GPU memory used by go process (PID {proc.pid}): {line.split(',')[1].strip()}")

            time.sleep(1)

        stdout, stderr = proc.communicate()
        if stderr:
            print(f"Errors from Go program:\n{stderr}")
    except Exception as e:
        print(f"An error occurred: {e}")


def generate_mask(masked_dimension):
    mask_dict = {
        "skipnumber": [[0, SKIPS.index(32)]],
        "filterbase": [[1, 0]],
        "filterbaseresolution": [[2, FILTER_RESOLUTIONS.index([320, 192])]],
        "filterbasethreshold": [[3, FILTER_THRESHOLDS.index(0.1)]],
        "scaledownresolution": [[4, SCALEDOWN_RESOLUTIONS.index([640, 352])]],
        "roibase": [[5, 1], [6, ROI_RESOLUTIONS.index([224, 128])], [7, ROI_THRESHOLDS.index(0.1)]],
        "roibaseresolution": [[6, ROI_RESOLUTIONS.index([224, 128])]],
        "roibasethreshold": [[7, ROI_THRESHOLDS.index(0.1)]],
        "saturation": [[8, 0]],
        "denoising": [[9, 0]],
        "equalization": [[10, 0]],
        "sharpening": [[11, 0]],
        "detectbasemodeltype": [[12, DETECT_NAMES.index("YOLOV3")]],
        "detectbasemodelsize": [[13, -1]],
        "detectbasethreshold": [[14, DETECT_THRESHOLDS.index(0.25)]],
        "trackbasemodeltype": [[15, TRACK_NAMES.index("OTIF")]],
        "trackbasemaxlosttime": [16, TRACK_MAX_LOST_TIMES.index(128)],
        "trackbasekeepthreshold": [[17, TRACK_KEEP_THRESHOLDS.index(0.6)]],
        "trackbasecreateobjectthreshold": [[18, TRACK_CREATE_OBJECT_THRESHOLDS.index(0.5)]],
        "trackbaseunmatchlocationthreshold": [[19, TRACK_UNMATCH_LOCATION_THRESHOLDS]],
        "postprocessbase": [[20, 1]],
        "noisefilterbase": [[22, 1]]
    }

    return mask_dict[masked_dimension]


def generate_kpluskminus_config_vector():
    return KMINUS_CONFIG_VECTOR, KPLUS_CONFIG_VECTOR


def update_config_datainfo(config_data, dataname,
                           datatype, deviceid, objects):
    config_data['database']['dataname'] = dataname
    config_data['database']['datatype'] = datatype
    config_data['deviceid'] = deviceid
    config_data['detectbase']['classes'] = list(
        map(lambda x: x.lower(), objects))
    return config_data


def run_parral_commands(run_commands, max_retries=5):
    # print("run_commands: ", run_commands)
    for retry in range(max_retries + 1):
        processes = [subprocess.Popen(
            cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for cmd in run_commands]
        exit_codes = [p.wait() for p in processes]

        if all(code == 0 for code in exit_codes):
            print("All commands succeeded.")
            break
        else:
            if retry < max_retries:
                print(
                    f"One or more commands failed. Retrying all commands... (Attempt {retry + 1}/{max_retries})")
            else:
                print("All commands failed after maximum retries.")
                print(run_commands)


def run_all_commands(run_commands):
    for commands in run_commands:
        processes = []

        for cmd in commands:
            process = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            processes.append(process)

        for p in processes:
            p.wait()


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


def combine_parallel_result(ingestion_result_paths, objects, ClusterNum):
    final_result = {"Latency": -9999}
    camera_id_number = {objectname: 0 for objectname in objects}
    for cluster_label, ingestion_result_path in enumerate(ingestion_result_paths):
        ingestion_result = read_json(ingestion_result_path)

        for objectname in objects:
            if ingestion_result[f"{objectname.capitalize()}GtCount"] > 0:
                camera_id_number[objectname] += ClusterNum[cluster_label]

        if cluster_label == 0:
            for key in ingestion_result:
                if "latency" == key.lower():
                    final_result[key] = max(
                        final_result[key], ingestion_result[key])
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
                                ClusterNum[cluster_label]
        else:
            for key in ingestion_result:
                if "latency" == key.lower():
                    final_result[key] = max(
                        final_result[key], ingestion_result[key])
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
                                ClusterNum[cluster_label]
    for key in final_result:
        hitobjectname = None
        for objectname in objects:
            if objectname.lower() in key.lower():
                hitobjectname = objectname
                break
        if key.lower() != "latency" and "count" not in key.lower():
            if camera_id_number[hitobjectname] <= 0:
                final_result[key] = 0
            else:
                final_result[key] /= camera_id_number[hitobjectname]

    return final_result


def load_latency(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data['Latency']


def run_cmd(cmd):
    p = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _, _ = p.communicate()
    # print(out, err)


def plot_pareto_set(acc, lat, save_path):
    fig, ax = plt.subplots()
    ax.scatter(acc, lat, c='r', marker='x', label='Pareto Set')
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Latency')
    ax.set_title('Pareto Set')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    plt.clf()
    plt.cla()


def crowding_distance_assignment(front, index_acc=1, index_time=2):
    if len(front) == 0:
        return []

    distances = [0 for _ in front]

    for index in [index_acc, index_time]:
        sorted_indices = sorted(
            range(len(front)), key=lambda x: front[x][index])

        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')

        for i in range(1, len(front) - 1):
            distances[sorted_indices[i]] += (front[sorted_indices[i+1]]
                                             [index] - front[sorted_indices[i-1]][index])

    return distances


def select_k_solutions(pareto_solutions, K):
    if K >= len(pareto_solutions):
        return pareto_solutions

    distances = crowding_distance_assignment(pareto_solutions)

    selected_indices = sorted(range(len(distances)),
                              key=lambda x: distances[x], reverse=True)[:K]

    return [pareto_solutions[i] for i in selected_indices]


def find_files(cachedir, method_name):
    """Search for files in a directory with a specific pattern and channel value greater than min_channel."""
    matches = []

    pattern = re.compile(
        f"{re.escape(method_name)}_channel(\\d+)_cluster.json")

    for root, _, filenames in os.walk(cachedir):
        for filename in filenames:
            match = pattern.match(filename)
            if match:
                matches.append(os.path.join(root, filename))

    return matches


def load_cachechannel(cameras, cache_skyscraper_cluster):
    skyscraper_cluster = {}
    for ci in range(len(cameras)):
        camera = cameras[ci]
        cluster_id = -1
        for cache_cluster_id, cache_cluster in cache_skyscraper_cluster.items():
            if camera.id in cache_cluster:
                cluster_id = cache_cluster_id
                break
        if cluster_id not in skyscraper_cluster:
            skyscraper_cluster[cluster_id] = []
        skyscraper_cluster[cluster_id].append(camera.id)

    return skyscraper_cluster


def generate_sorted_config_dimensions():
    pass


def md5_hash(input_str: str) -> str:
    return hashlib.md5(input_str.encode()).hexdigest()


def write_json_to_stdin(data):
    encoded = json.dumps(data).encode('utf-8')
    length = len(encoded)

    packed_length = struct.pack('>I', length)

    return packed_length + encoded


def plot_compound_list(compound_list, label_name="Pareto Set", save_path=None, latency_ticks=[0.5, 0.6, 0.7, 0.8, 0.9]):
    if isinstance(compound_list, list):
        marker_list = ['x', 'o']
        color_list = ['r', 'b']
        accs_list, lats_list = [], []
        # plt.figure(figsize=(10, 10))
        cid = 0
        for compound_l, label_n in zip(compound_list, label_name):
            accs = [x[0] for x in compound_l]
            lats = [x[1] for x in compound_l]
            plt.scatter(accs, lats,
                        c=color_list[cid],
                        marker=marker_list[cid],
                        label=label_n)
            accs_list.append(accs)
            lats_list.append(lats)
            cid += 1
        # for latency_tick in latency_ticks:
        #     plt.axhline(y=latency_tick, color='gray', linestyle='--')
        # plt.xticks(fontsize=20)
        # plt.yticks(latency_ticks, fontsize=20)
        # plt.xlabel('Accuracy', fontsize=20)
        # plt.ylabel('Latency', fontsize=20)
        # plt.title(
        #     f'MA: {mean(accs_list[0]):.2f}, ML: {mean(lats_list[0]):.2f}; RMA: {mean(accs_list[1]):.2f}, RML: {mean(lats_list[1]):.2f}', fontsize=20)
        plt.legend()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        plt.clf()
        plt.cla()
        return [mean(accs) for accs in accs_list], \
            [mean(lats) for lats in lats_list]
    else:
        plt.figure(figsize=(10, 10))
        accs = [x[0] for x in compound_list]
        lats = [x[1] for x in compound_list]
        plt.scatter(accs, lats, c='r', marker='x', label=label_name, s=25)
        plt.xlabel('Accuracy', fontsize=20)
        plt.ylabel('Latency', fontsize=20)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xticks(fontsize=20)
        plt.yticks(latency_ticks, fontsize=20)
        for latency_tick in latency_ticks:
            plt.axhline(y=latency_tick, color='gray', linestyle='--')
        plt.title(
            f'Mean Acc: {mean(accs):.2f}, Mean Lat: {mean(lats):.2f}', fontsize=20)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        plt.clf()
        plt.cla()
        return mean(accs), mean(lats)

 
def get_latency_increase(init_latency, weight=0.85):
    if init_latency < 3.526:
        return 0.324187 * weight
    elif init_latency < 4.135:
        return 0.468390 * weight
    elif init_latency < 5.216:
        return 1.017852 * weight
    elif init_latency < 6.337:
        return 0.864507 * weight
    else:
        return 3.752959 * weight


def generate_random_action_change(config_vector):
    random_change_dimension = random.randint(0, len(config_vector) - 1)
    random_change_value = random.randint(
        0, SEARCH_SPACE[random_change_dimension] - 1)
    config_vector[random_change_dimension] = random_change_value
    return config_vector

def extract_frames(video_path, frame_number, frame_gap):
    # Extract the first frame_number frames of the video at intervals of frame_gap frames.
    video_reader = cv2.VideoCapture(video_path)
    frame_indexs = [i * frame_gap for i in range(frame_number)]
    frames = []
    for frame_index in frame_indexs:
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = video_reader.read()
        if ret:
            frames.append(frame)
    return frames