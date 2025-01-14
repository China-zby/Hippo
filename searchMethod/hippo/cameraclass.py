import os
import cv2
import math
import json
import pickle
import subprocess
import numpy as np
import pygraphviz as pgv
from copy import deepcopy

from tqdm import tqdm
from config_ import Config
from cluster import Trajectory
from alive_progress import alive_bar
from utils import add_node, ACCURACY_MASK_VALUE, iterate_with_bar, read_json, \
    search_file, generate_action_from_config, mean, \
    SEARCH_SPACE, SEARCH_SPACE_NAMES

class Camera(object):
    def __init__(self, data, info, config, id, objects, metrics, 
                 reward_func_info=None, cache_store_path=None,
                 search_space=None, search_space_name=None):
        self.data = data
        self.width, self.height, self.fps, self.frames, self.chosen_frames = info
        self.attributes = ["area", "speed", "acc", "turn", "turn_speed", "dist"]
        self.config = config
        self.id = id
        self.objects = list(map(lambda x: x.capitalize(), objects))
        self.metrics = metrics

        self.temp = 10.0
        if reward_func_info is not None:
            self.temp, = reward_func_info

        if search_space is not None:
            self.search_space = search_space
        else:
            self.search_space = SEARCH_SPACE
            
        if search_space_name is not None:
            self.search_space_name = search_space_name
        else:
            self.search_space_name = SEARCH_SPACE_NAMES

        self.cache_store_path = cache_store_path
        self.record = None
        self.latency = None
        self.overallPerformance = None
        self.representation = None

    def reset_with_videoid(self, videoid):
        self.videoid = videoid

    def set_suboptimal_solution(self, action):
        self.sosolution = action

    @property
    def suboptimal_solution(self):
        return self.sosolution

    def set_chosen_frames(self, chosen_frames):
        self.chosen_frames = chosen_frames

    def extractTrack(self):
        trackdata = {}
        for dataline in self.data:
            if dataline is None or len(dataline) == 0:
                continue
            for trackdataline in dataline:
                trackid = int(trackdataline["track_id"])
                if trackid not in trackdata:
                    trackdata[trackid] = []
                if abs(trackdataline['left']) < 10 or abs(trackdataline['top']) < 10 or \
                   abs(trackdataline['right'] - self.width) < 10 or abs(trackdataline['bottom'] - self.height) < 10:
                    continue
                trackdata[trackid].append([trackdataline['left'], trackdataline['top'],
                                           trackdataline['right'], trackdataline['bottom']])
        tracks = []
        for trackid in trackdata:
            if len(trackdata[trackid]) < 3:
                continue
            tracks.append(Trajectory(trackdata[trackid]))
        return tracks

    @property
    def vector(self):
        config_data = self.loadConfig()
        config_vector = generate_action_from_config(config_data, 
                                                    search_space_name=self.search_space_name)
        config_vector = "_".join([str(x) for x in config_vector])
        data_type = config_data['database']['datatype']
        res_id = f"{data_type}_{self.id}_{config_vector}.pkl"
        return res_id

    def getRepresentation(self):
        # if self.tracks is not defined, extract tracks from data
        if not hasattr(self, 'tracks'):
            self.tracks = self.extractTrack()
        vector = [0] * 12
        for track in self.tracks:
            for i, attr in enumerate(self.attributes):
                if i < len(self.attributes) - 1:  # For all attributes except 'dist'
                    if "min_" + attr in track.info and (vector[2*i] < track.info["min_" + attr] or vector[2*i] == 0):
                        vector[2*i] = track.info["min_" + attr]
                    if "max_" + attr in track.info and (vector[2*i + 1] > track.info["max_" + attr] or vector[2*i + 1] == 0):
                        vector[2*i + 1] = track.info["max_" + attr]
                else:  # For 'dist' attribute
                    if vector[10] > track.info[attr] or vector[10] == 0:
                        vector[10] = track.info[attr]
        vector[11] = len(self.tracks)
        return vector

    # @property
    # def representation(self):
    #     return self.getRepresentation()

    def updateConfig(self, config):
        self.config.update_cache(config)

    def loadConfig(self):
        return self.config.load_cache()

    @property
    def performance(self):
        save_reward_path = os.path.join(
            self.config.cache_dir, f'reward_{self.id}.json')
        # error_log_path = os.path.join(self.config.cache_dir, f'error_log_{self.id}.txt')
        configData = self.config.load_cache()
        json_data = json.dumps(configData)

        # with open(error_log_path, 'w') as error_file:
        with subprocess.Popen(['go', 'run', './searchMethod/ardent/step.go', save_reward_path, f"{self.id}"],
                              stdin=subprocess.PIPE,
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, text=True) as proc:
            _, _ = proc.communicate(input=json_data)

        with open(save_reward_path) as file:
            record = json.load(file)

        overallPerformance, latencyScore = self.get_overallPerformance(
            record, iflatency=False, classes=self.objects, query_number=len(self.metrics))
        self.overallPerformance = overallPerformance
        self.record = record
        self.latencyScore = latencyScore

        return overallPerformance, latencyScore

    @property
    def reward(self):
        save_reward_path = os.path.join(
            self.config.cache_dir, f'reward_{self.id}.json')
        # error_log_path = os.path.join(self.config.cache_dir, f'error_log_{self.id}.txt')
        configData = self.config.load_cache()
        json_data = json.dumps(configData)

        # with open(error_log_path, 'w') as error_file:
        with subprocess.Popen(['go', 'run', './searchMethod/ardent/step.go', save_reward_path, f"{self.id}"],
                              stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True) as proc:
            _, _ = proc.communicate(input=json_data)

        with open(save_reward_path) as file:
            record = json.load(file)
        overallPerformance = self.get_lowlantencybutperformance(
            record, classes=self.objects, query_number=len(self.metrics))
        self.overallPerformance = overallPerformance
        self.latency = record['Latency']
        self.record = record

        cmetric, cmetricname = [], []
        for objectname in self.objects:
            gtcount = record[objectname.capitalize()+"GtCount"]
            predcount = record[objectname.capitalize()+"PredCount"]
            record[objectname.capitalize()+"CountRate"] = 1 - abs(gtcount -
                                                                  predcount) / gtcount if gtcount != 0 else float(predcount == 0)
            record[objectname.capitalize()+"Sel"] = record[objectname.capitalize() +
                                                           "Sel"] if gtcount != 0 else float(predcount == 0)
            record[objectname.capitalize()+"Agg"] = record[objectname.capitalize() +
                                                           "Agg"] if gtcount != 0 else float(predcount == 0)
        save_record = deepcopy(record)
        for objectname in self.objects:
            del record[objectname.capitalize()+"GtCount"]
            del record[objectname.capitalize()+"PredCount"]
        objects_lower = [x.lower() for x in self.objects]
        metrics_lower = [x.lower() for x in self.metrics]
        for qn in record:
            qn_lower = qn.lower()
            if any(classname in qn_lower for classname in objects_lower):
                if any(metricname in qn_lower for metricname in metrics_lower):
                    cmetric.append(record[qn])
                    cmetricname.append(qn)
        cmetric.append(math.exp(-record['Latency'] / self.temp))
        os.remove(save_reward_path)
        return overallPerformance, save_record, \
            cmetric, cmetricname

    @property
    def ingestion_without_cache(self):
        save_reward_path = os.path.join(
            self.config.cache_dir, f'reward_{self.id}.json')
        configData = self.config.load_cache()
        json_data = json.dumps(configData)

        with subprocess.Popen(['go', 'run', './searchMethod/hippo/step.go', save_reward_path, f"{self.id}"],
                              stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True) as proc:
            _, _ = proc.communicate(input=json_data)
            # print(out, err)

        with open(save_reward_path) as file:
            record = json.load(file)
            # filter out the records with no class
            record = {k: v for k, v in record.items() if any(
                classname.lower() in k.lower() for classname in self.objects + ['latency'])}

        motas, idf1s, hotas, counts = [], [], [], []
        for objectname in self.objects:
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

        self.latency = record['Latency']
        self.record = record

        cmetric, cmetricname = [], []
        for objectname in self.objects:
            gtcount = record[objectname.capitalize()+"GtCount"]
            predcount = record[objectname.capitalize()+"PredCount"]
            record[objectname.capitalize()+"CountRate"] = 1 - abs(gtcount -
                                                                  predcount) / gtcount if gtcount != 0 else float(predcount == 0)
            record[objectname.capitalize()+"Sel"] = record[objectname.capitalize() +
                                                           "Sel"] if gtcount != 0 else float(predcount == 0)
            record[objectname.capitalize()+"Agg"] = record[objectname.capitalize() +
                                                           "Agg"] if gtcount != 0 else float(predcount == 0)
        save_record = deepcopy(record)
        # for objectname in self.objects:
        #     del record[objectname.capitalize()+"GtCount"]
        #     del record[objectname.capitalize()+"PredCount"]
        objects_lower = [x.lower() for x in self.objects]
        metrics_lower = [x.lower() for x in self.metrics]
        for qn in record:
            qn_lower = qn.lower()
            if any(classname in qn_lower for classname in objects_lower):
                if any(metricname in qn_lower for metricname in metrics_lower):
                    cmetric.append(record[qn])
                    cmetricname.append(qn.lower())
        cmetric.append(math.exp(-record['Latency'] / self.temp))
        cmetricname.append('Latency')
        os.remove(save_reward_path)

        othervalues = []
        for metricname in self.metrics:
            fuse_metric = []
            for oi, objectname in enumerate(self.objects):
                ci = cmetricname.index(
                    f"{objectname.lower()}{metricname.lower()}")
                if counts[oi] > 0:
                    fuse_metric.append(cmetric[ci])
            if len(fuse_metric) > 0:
                othervalues.append(mean(fuse_metric))
            else:
                othervalues.append(0.0)
        othervalue = mean(othervalues)
        
        ingestionvalue = (motvalue + othervalue) * 0.5

        overallPerformance = self.incentive_function(
            motvalue, record['Latency'])
        self.overallPerformance = overallPerformance
        
        config_vector = self.vector
        cache_path_name = f"{config_vector}"
        cache_path = os.path.join(
            self.cache_store_path, cache_path_name)
        with open(cache_path, "wb") as f:
            pickle.dump([overallPerformance, ingestionvalue, record['Latency'],
                        hotas + motas + idf1s, save_record, cmetric, cmetricname], f)

        return overallPerformance, ingestionvalue, record['Latency'], hotas + motas + idf1s, save_record, \
            cmetric, cmetricname

    @property
    def ingestion(self):
        save_reward_path = os.path.join(
            self.config.cache_dir, f'reward_{self.id}.json')
        configData = self.config.load_cache()
        json_data = json.dumps(configData)

        config_vector = self.vector
        cache_path_name = f"{config_vector}"
        cache_path = os.path.join(
            self.cache_store_path, cache_path_name)
        has_result_cache = search_file(
            self.cache_store_path, cache_path_name)
        if has_result_cache:
            # print("load from cache")
            with open(cache_path, "rb") as f:
                overallPerformance, ingestionvalue, record_latency, mot_metrics, save_record, \
                    cmetric, cmetricname = pickle.load(f)
            return overallPerformance, ingestionvalue, record_latency, mot_metrics, save_record, \
                cmetric, cmetricname

        with subprocess.Popen(['go', 'run', './searchMethod/hippo/step.go', save_reward_path, f"{self.id}"],
                              stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True) as proc:
            _, _ = proc.communicate(input=json_data)

        with open(save_reward_path) as file:
            record = json.load(file)
            record = {k: v for k, v in record.items() if any(
                classname.lower() in k.lower() for classname in self.objects + ['latency'])}

        motas, idf1s, hotas, counts = [], [], [], []
        for objectname in self.objects:
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

        self.latency = record['Latency']
        self.record = record

        cmetric, cmetricname = [], []
        for objectname in self.objects:
            gtcount = record[objectname.capitalize()+"GtCount"]
            predcount = record[objectname.capitalize()+"PredCount"]
            record[objectname.capitalize()+"CountRate"] = 1 - abs(gtcount -
                                    predcount) / gtcount if gtcount != 0 else float(predcount == 0)
            record[objectname.capitalize()+"Sel"] = record[objectname.capitalize() +
                                            "Sel"] if gtcount != 0 else float(predcount == 0)
            record[objectname.capitalize()+"Agg"] = record[objectname.capitalize() +
                                                "Agg"] if gtcount != 0 else float(predcount == 0)
        save_record = deepcopy(record)
        for objectname in self.objects:
            del record[objectname.capitalize()+"GtCount"]
            del record[objectname.capitalize()+"PredCount"]
        objects_lower = [x.lower() for x in self.objects]
        metrics_lower = [x.lower() for x in self.metrics]
        for qn in record:
            qn_lower = qn.lower()
            if any(classname in qn_lower for classname in objects_lower):
                if any(metricname in qn_lower for metricname in metrics_lower):
                    cmetric.append(record[qn])
                    cmetricname.append(qn.lower())
        cmetric.append(math.exp(-record['Latency'] / self.temp))
        cmetricname.append('Latency')
        os.remove(save_reward_path)

        othervalues = []
        for metricname in self.metrics:
            fuse_metric = []
            for oi, objectname in enumerate(self.objects):
                ci = cmetricname.index(
                    f"{objectname.lower()}{metricname.lower()}")
                if counts[oi] > 0:
                    fuse_metric.append(cmetric[ci])
            if len(fuse_metric) > 0:
                othervalues.append(mean(fuse_metric))
            else:
                othervalues.append(0.0)
        othervalue = mean(othervalues)

        ingestionvalue = (motvalue + othervalue) * 0.5

        overallPerformance = self.incentive_function(
            ingestionvalue, record['Latency'])
        self.overallPerformance = overallPerformance

        with open(cache_path, "wb") as f:
            pickle.dump([overallPerformance, ingestionvalue, record['Latency'],
                        hotas + motas + idf1s, save_record, cmetric, cmetricname], f)

        return overallPerformance, ingestionvalue, record['Latency'], hotas + motas + idf1s, save_record, \
            cmetric, cmetricname

    @property
    def metric(self):
        save_reward_path = os.path.join(
            self.config.cache_dir, f'{self.id}.json')

        configData = self.config.load_cache()
        json_data = json.dumps(configData)

        with subprocess.Popen(['go', 'run', './searchMethod/ardent/step.go', save_reward_path, f"{self.id}"],
                              stdin=subprocess.PIPE,
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, text=True) as proc:
            _, _ = proc.communicate(input=json_data)

        with open(save_reward_path) as file:
            record = json.load(file)

        overallPerformance, _ = self.get_overallPerformance(
            record, classes=self.objects, query_number=len(self.metrics))
        self.overallPerformance = overallPerformance
        self.record = record

        return overallPerformance

    def get_overallPerformance(self, step_records, iflatency=True, classes=['Car', 'Bus', 'Truck'], latency_weight=0.5, query_number=3):
        """
        Calculate weighted reward based on various metrics like latency,
        object selection score, object aggregation score, and count matching score.

        Parameters:
            step_records: Dict containing the records for the current step.
            classes: List of object classes to consider.
            latency_weight: Weight for latency metric.
            query_number: Number of queries for each class.
            temp: Temperature parameter for latency calculation.

        Returns:
            float: Weighted score.
        """

        # Calculate latency score
        latency_score = math.exp(-step_records['Latency'] / self.temp)

        # Generate scores and their corresponding weights
        if iflatency:
            # Initialize score and weight lists
            score_weight = (1 - latency_weight) / (query_number * len(classes))
            scores, score_weights = [latency_score], [latency_weight]
        else:
            score_weight = 1.0 / (query_number * len(classes))
            scores, score_weights = [], []

        if 'sel' in self.metrics or 'agg' in self.metrics:
            suffixs = [metric.capitalize()
                       for metric in self.metrics if metric in ['sel', 'agg']]
            scores += [
                step_records[f"{class_name}{suffix}"]
                if step_records[f"{class_name}GtCount"] != 0 else float(step_records[f"{class_name}PredCount"] == 0)
                for class_name in self.objects
                for suffix in suffixs
            ]
            score_weights += [score_weight] * \
                (len(suffixs) * len(self.objects))

        if 'hota' in self.metrics:
            pass

        if 'count' in self.metrics:
            scores += [
                1.0 - abs(step_records[f"{class_name}GtCount"] -
                          step_records[f"{class_name}PredCount"]) / step_records[f"{class_name}GtCount"]
                if step_records[f"{class_name}GtCount"] != 0 else float(step_records[f"{class_name}PredCount"] == 0)
                for class_name in self.objects
            ]
            score_weights += [score_weight] * len(self.objects)

        # Calculate the final weighted score
        weight_score = sum(score * weight for score,
                           weight in zip(scores, score_weights))

        return weight_score, latency_score

    # def incentive_function(self, acc, delay):
    #     accuracy_value = math.exp(acc) / math.exp(1)
    #     latency_value = math.exp(1.0 / delay) / math.exp(1.0/self.mind)
    #     value = accuracy_value * latency_value
    #     min_value = math.exp(0) / math.exp(1) * \
    #         math.exp(1.0 / self.maxd) / math.exp(1.0/self.mind)
    #     max_value = math.exp(1) / math.exp(1) * \
    #         math.exp(1.0 / self.mind) / math.exp(1.0/self.mind)
    #     reward = (value - min_value) / (max_value - min_value)
    #     return reward

    # def incentive_function(self, acc, delay):
    #     return (math.exp(acc) / math.exp(1) * math.exp(1.5 / delay) / math.exp(1))**0.2

    # def incentive_function(self, acc, delay):
    #     """Calculate incentive based on accuracy and delay."""
    #     delay = math.exp(-delay/25)
    #     return acc + delay

    def map_func(self, x, y):
        y = math.exp(-y/25)
        return x + y

    def incentive_function(self, acc, delay):
        """Calculate incentive based on accuracy and delay."""
        min_z = self.map_func(0, 60)
        max_z = self.map_func(1, 2)

        return (self.map_func(acc, delay) - min_z) / (max_z - min_z)

    # def incentive_function(self, a_i, t_i, alpha=1.0, beta=0.1, gamma=1.0, tau=10.0):
    #     """
    #     Calculate the reward based on running accuracy and processing time.

    #     Parameters:
    #     - a_i: Running accuracy for configuration c_i.
    #     - t_i: Processing time for configuration c_i.
    #     - alpha, beta, gamma, tau: Constants to adjust the trade-off between accuracy and time.

    #     Returns:
    #     - reward: Calculated reward for configuration c_i.
    #     """

    #     reward = alpha * a_i - beta * t_i + \
    #         gamma * np.exp(a_i) * np.exp(-t_i / tau)

    #     return reward

    def get_ingestion_perfandlate(self, step_records, classes=['Car', 'Bus', 'Truck'], query_number=3):
        """
        Calculate weighted reward based on various metrics like latency,
        object selection score, object aggregation score, and count matching score.

        Parameters:
            step_records: Dict containing the records for the current step.
            classes: List of object classes to consider.
            latency_weight: Weight for latency metric.
            query_number: Number of queries for each class.
            temp: Temperature parameter for latency calculation.

        Returns:
            float: Weighted score.
        """

        # Initialize score and weight lists
        score_weight = 1.0 / (query_number * len(classes))

        scores, score_weights = [], []
        # Generate scores and their corresponding weights
        if 'sel' in self.metrics or 'agg' in self.metrics:
            suffixs = [metric.capitalize()
                       for metric in self.metrics if metric in ['sel', 'agg']]
            scores += [
                step_records[f"{class_name}{suffix}"]
                if step_records[f"{class_name}GtCount"] != 0 else float(step_records[f"{class_name}PredCount"] == 0)
                for class_name in self.objects
                for suffix in suffixs
            ]
            score_weights += [score_weight] * \
                (len(suffixs) * len(self.objects))

        if 'count' in self.metrics:
            scores += [
                1.0 - abs(step_records[f"{class_name}GtCount"] -
                          step_records[f"{class_name}PredCount"]) / step_records[f"{class_name}GtCount"]
                if step_records[f"{class_name}GtCount"] != 0 else float(step_records[f"{class_name}PredCount"] == 0)
                for class_name in self.objects
            ]
            score_weights += [score_weight] * len(self.objects)

        accuracy = sum(score * weight for score,
                       weight in zip(scores, score_weights))

        latency = step_records['Latency']

        return self.incentive_function(accuracy, latency)

    def get_lowlantencybutperformance(self, step_records, classes=['Car', 'Bus', 'Truck'], query_number=3):
        """
        Calculate weighted reward based on various metrics like latency,
        object selection score, object aggregation score, and count matching score.

        Parameters:
            step_records: Dict containing the records for the current step.
            classes: List of object classes to consider.
            latency_weight: Weight for latency metric.
            query_number: Number of queries for each class.
            temp: Temperature parameter for latency calculation.

        Returns:
            float: Weighted score.
        """

        # Initialize score and weight lists
        score_weight = 1.0 / (query_number * len(classes))

        scores, score_weights = [], []
        # Generate scores and their corresponding weights
        if 'sel' in self.metrics or 'agg' in self.metrics:
            suffixs = [metric.capitalize()
                       for metric in self.metrics if metric in ['sel', 'agg']]
            scores += [
                step_records[f"{class_name}{suffix}"]
                if step_records[f"{class_name}GtCount"] != 0 else float(step_records[f"{class_name}PredCount"] == 0)
                for class_name in self.objects
                for suffix in suffixs
            ]
            score_weights += [score_weight] * \
                (len(suffixs) * len(self.objects))

        if 'count' in self.metrics:
            scores += [
                1.0 - abs(step_records[f"{class_name}GtCount"] -
                          step_records[f"{class_name}PredCount"]) / step_records[f"{class_name}GtCount"]
                if step_records[f"{class_name}GtCount"] != 0 else float(step_records[f"{class_name}PredCount"] == 0)
                for class_name in self.objects
            ]
            score_weights += [score_weight] * len(self.objects)

        accuracy = sum(score * weight for score,
                       weight in zip(scores, score_weights))

        latency = step_records['Latency']

        return self.incentive_function(accuracy, latency)

    @property
    def eval_unitune(self):
        save_reward_path = os.path.join(
            os.path.dirname(self.config.cache_dir), "records", f'{self.id}.json')
        configData = self.config.load_cache()
        json_data = json.dumps(configData)
        video_path = f"{configData['database']['dataroot']}/dataset/{configData['database']['dataname']}/{configData['database']['datatype']}/video/{self.id}.mp4"
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with subprocess.Popen(['go', 'run', f'{os.path.dirname(__file__)}/step.go', save_reward_path, f"{self.id}"],
                              stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True) as proc:
            _, _ = proc.communicate(input=json_data)

        with open(save_reward_path) as file:
            record = json.load(file)
            record = {k: v for k, v in record.items() if any(
                classname.lower() in k.lower() for classname in self.objects + ['latency'])}

        acc = {}
        hotas, motas, idf1s, counts = [], [], [], []
        for objectname in self.objects:
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
            acc[objectname] = {'hota': hotas[-1], 'mota': motas[-1],
                               'idf1': idf1s[-1], 'count': counts[-1]}

        return acc, frame_count / record['Latency']

    @property
    def graph(self):
        graph = pgv.AGraph(directed=True,
                           fontname='Helvetica', arrowtype='open')

        add_node(graph, -2, "Rewrad: %.2f" %
                 self.overallPerformance, color='red')
        add_node(graph, -1, "Video")

        action_config = self.config.load_cache()

        edges = [(-1, 0), (0, 1)]

        add_node(
            graph, 0, f"SkipNumber: {action_config['videobase']['skipnumber']}")
        add_node(
            graph, 1, f"ScaleDown: {action_config['videobase']['scaledownresolution']}")

        if action_config['filterbase']['flag'] and action_config['roibase']['flag']:
            add_node(
                graph, 2, f"Filter: {action_config['filterbase']['resolution']}")
            add_node(
                graph, 3, f"ROI: {action_config['roibase']['resolution']}")
            edges.append((1, 2))
            edges.append((2, 3))
            next_node = 3
        elif action_config['filterbase']['flag'] and not action_config['roibase']['flag']:
            add_node(
                graph, 2, f"Filter: {action_config['filterbase']['resolution']}")
            edges.append((1, 2))
            next_node = 2
        elif not action_config['filterbase']['flag'] and action_config['roibase']['flag']:
            add_node(
                graph, 2, f"ROI: {action_config['roibase']['resolution']}")
            edges.append((1, 2))
            next_node = 2
        else:
            next_node = 1
            pass

        enhancetools = []
        if action_config['roibase']['denoisingflag']:
            enhancetools.append("Denoising")
        if action_config['roibase']['equalizationflag']:
            enhancetools.append("Equalization")
        if action_config['roibase']['sharpeningflag']:
            enhancetools.append("Sharpening")
        if action_config['roibase']['saturationflag']:
            enhancetools.append("Saturation")
        enhance_node_ids = []
        for enhance_i, enhancetool in enumerate(enhancetools):
            add_node(graph, next_node + enhance_i + 1, f"{enhancetool}")
            edges.append((next_node, next_node + enhance_i + 1))
            enhance_node_ids.append(next_node + enhance_i + 1)

        if len(enhance_node_ids) > 0:
            detect_node_id = max(enhance_node_ids) + 1
            add_node(graph, detect_node_id,
                     f"Detect: {action_config['detectbase']['modeltype']}-{action_config['detectbase']['modelsize']}")
            for enhance_node_id in enhance_node_ids:
                edges.append((enhance_node_id, detect_node_id))
        else:
            detect_node_id = next_node + 1
            add_node(graph, detect_node_id,
                     f"Detect: {action_config['detectbase']['modeltype']}-{action_config['detectbase']['modelsize']}")
            edges.append((next_node, detect_node_id))

        add_node(graph, detect_node_id + 1,
                 f"Track: {action_config['trackbase']['modeltype']}")
        edges.append((detect_node_id, detect_node_id + 1))

        if action_config['postprocessbase']['postprocesstype'] != 'None' and action_config['noisefilterbase']['flag']:
            add_node(graph, detect_node_id + 2, f"PostProcess")
            add_node(graph, detect_node_id + 3, f"NoiseFilter")
            edges.append((detect_node_id + 1, detect_node_id + 2))
            edges.append((detect_node_id + 2, detect_node_id + 3))
            final_node_id = detect_node_id + 3
        elif action_config['postprocessbase']['postprocesstype'] != 'None' and not action_config['noisefilterbase']['flag']:
            add_node(graph, detect_node_id + 2, f"PostProcess")
            edges.append((detect_node_id + 1, detect_node_id + 2))
            final_node_id = detect_node_id + 2
        elif not action_config['postprocessbase']['postprocesstype'] != 'None' and action_config['noisefilterbase']['flag']:
            add_node(graph, detect_node_id + 2, f"NoiseFilter")
            edges.append((detect_node_id + 1, detect_node_id + 2))
            final_node_id = detect_node_id + 2
        else:
            final_node_id = detect_node_id + 1

        add_node(graph, final_node_id + 1, f"Build Index")
        edges.append((final_node_id, final_node_id + 1))

        for edge in edges:
            graph.add_edge(edge[0], edge[1])

        graph.layout(prog='dot')
        return graph

    def set_context(self, sel, agg, count):
        self.sel = sel
        self.agg = agg
        self.count = count

    def add_attribute(self, attr_name, attr_value):
        self.__setattr__(attr_name, attr_value)


def build_cameras(channel, configpath,
                  deviceid, data_name, data_type, method_name, scene_name,
                  objects, metrics,
                  alldir,
                                 cache_store_path="./cache/global/one_camera_with_config"):
    cameras = []
    with alive_bar(channel, bar='brackets', title='Load Cameras: ', spinner="squares") as loadcamerainfobar:
        for cameraid in iterate_with_bar(channel, loadcamerainfobar):
            videopath = os.path.join(alldir.videodir, f"{cameraid}.mp4")
            cap = cv2.VideoCapture(videopath)
            width, height, fps, totalframenumber = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            trackpath = os.path.join(alldir.trackdir, f"{cameraid}.json")
            trackdata = read_json(trackpath)
            cameraConfig = Config(
                configpath, f'{alldir.cacheconfigdir}/{data_type}_cache_config_{cameraid}.yaml',
                deviceid, data_name, data_type, objects,
                f'{method_name}_{data_type}_{cameraid}', scene_name)
            camera = Camera(trackdata, [width, height, fps,
                                        totalframenumber, []],
                            cameraConfig, cameraid, objects, metrics,
                            cache_store_path=cache_store_path)
            cameras.append(camera)
    return cameras


def build_a_camera_with_config(alldir, cameraConfig,
                               objects, metrics,
                               cache_store_path="./cache/global/one_camera_with_config"):
    videopath = os.path.join(alldir.videodir, f"0.mp4")
    cap = cv2.VideoCapture(videopath)
    width, height, fps, totalframenumber = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    camera = Camera([], [width, height, fps,
                         totalframenumber, []],
                    cameraConfig, 0, objects, metrics,
                    cache_store_path=cache_store_path)
    return camera


def build_cameras_with_info(alldir, channel,
                            objects, metrics,
                            configpath, deviceid,
                            method_name, data_name, data_type, scene_name,
                            temp):
    cameras, videopaths, trackdatas = [], [], []
    with alive_bar(channel, bar='brackets', title='Load Cameras: ') as loadcamerainfobar:
        for cameraid in iterate_with_bar(channel, loadcamerainfobar):
            videopath = os.path.join(alldir.videodir, f"{cameraid}.mp4")
            cap = cv2.VideoCapture(videopath)
            width, height, fps, totalframenumber = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            trackpath = os.path.join(alldir.trackdir, f"{cameraid}.json")
            trackdata = read_json(trackpath)
            cameraConfig = Config(
                configpath, f'{alldir.cacheconfigdir}/{data_type}_cache_config_{cameraid}.yaml',
                abs(1 - int(deviceid)), data_name, data_type, objects,
                f'{method_name}_{data_type}_{cameraid}', scene_name)
            camera = Camera(trackdata, [width, height, fps, totalframenumber, []],
                            cameraConfig, cameraid, objects, metrics, reward_func_info=[temp])
            cameras.append(camera)
            videopaths.append(videopath)
            trackdatas.append(trackdata)
    return cameras, videopaths, trackdatas


def build_cameras_with_info_list(alldir, channel_list,
                                 objects, metrics,
                                 configpath, deviceid,
                                 method_name, data_name, data_type, scene_name,
                                 temp, 
                                 search_space=None, search_space_name=None,
                                 cache_store_path="./cache/global/one_camera_with_config"):
    cameras, videopaths, trackdatas = [], [], []
    for cameraid in tqdm(channel_list):
        videopath = os.path.join(alldir.videodir, f"{cameraid}.mp4")
        cap = cv2.VideoCapture(videopath)
        width, height, fps, totalframenumber = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        trackpath = os.path.join(alldir.trackdir, f"{cameraid}.json")
        trackdata = read_json(trackpath)

        cameraConfig = Config(
            configpath, f'{alldir.cacheconfigdir}/{data_type}_cache_config_{cameraid}.yaml',
            int(deviceid), data_name, data_type, objects,
            f'{method_name}_{data_type}_{cameraid}', scene_name)
        camera = Camera(trackdata, [width, height, fps, totalframenumber, []],
                        cameraConfig, cameraid, objects, metrics,
                        reward_func_info=[temp],
                        cache_store_path=cache_store_path,
                        search_space=search_space, search_space_name=search_space_name)

        cameras.append(camera)
        videopaths.append(videopath)
        trackdatas.append(trackdata)
    return cameras, videopaths, trackdatas
