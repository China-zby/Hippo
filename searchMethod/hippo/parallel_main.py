# from stable_baselines3 import REINFORCE
import time
import gymnasium as gym
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

import os
import sys
import json
import utils
import pickle
import random
import argparse
import rec_configs
import numpy as np
import pickle as pkl
import torch.nn.functional as F
import torchvision.transforms as T
from dataloader import VideoDataset2
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from config_ import generate_random_config
from transformers import VideoMAEImageProcessor
from typing import Dict, Optional, Tuple, Union
from stable_baselines3.common.policies import ActorCriticPolicy
from parallel_test_unitune import UNITUNE, UnituneGPModel, UnituneEnv
from cameraclass import build_cameras_with_info, build_a_camera_with_config, build_cameras_with_info_list
from utils import SEARCH_SPACE, SEARCH_SPACE_NAMES, GOLDEN_CONFIG_VECTOR, EFFE_GPU_SPACE_NAMES, \
    run_parral_commands

CANDIDATE_FUNCS = {
    "hippo": rec_configs.hippo_func, 
    "unitune": rec_configs.unitune_func, 
    "skyscraper": rec_configs.skyscraper_func, 
    "otif": rec_configs.otif_func, 
    "hippo_with_skyscraper_cluster": rec_configs.hippo_cluster_skyscraper_func, 
    "hippo_without_pareto_reinforcement_learning": rec_configs.hippo_without_pareto_reinforcement_learning_func, 
    "hippo_without_imitation_learning": rec_configs.hippo_without_imitation_learning_func 
}

class ProbabilisticDistribution:
    def __init__(self, preds):
        self.preds = [torch.softmax(pred, dim=-1) for pred in preds]
        self.bs = preds[0].size(0)
        self.distributions = [torch.distributions.Categorical(
            probs=pred) for pred in self.preds]

    def get_actions(self, deterministic=False):
        if deterministic:
            actions = []
            for bi in range(self.bs):
                action = [torch.argmax(pred[bi]) for pred in self.preds]
                actions.append(action)
                actions = torch.tensor(actions, dtype=torch.long)
        else:
            actions = [dist.sample() for dist in self.distributions]
            actions = torch.stack(actions).t()
        return actions

    def log_prob(self, actions):
        log_probs = []
        for bi, a in enumerate(actions):
            log_prob = sum(dist.log_prob(a[ai])
                           for ai, dist in enumerate(self.distributions))
            log_probs.append(log_prob)
        return torch.stack(log_probs)

    def entropy(self):
        entropies = [dist.entropy().mean() for dist in self.distributions]
        return torch.stack(entropies)  

    def cross_entropy(self, actions):
        losses = []
        for bi in range(self.bs):
            for si in range(len(self.distributions)):
                pred = self.preds[si][bi]
                action = actions[bi][si]
                cross_entropy_loss = -torch.log(pred[action])
                losses.append(cross_entropy_loss)
        return torch.stack(losses).sum()


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.output(x)
        return value


class GCN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_node_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x


class DUPIN(ActorCriticPolicy):
    def __init__(self, hidden_size=64, video_context_size=16, observation_space=None,
                 action_space=None, lr_schedule=None, context_obs=True, use_graph=True,
                 use_rnn=False, iter_generate=False):
        super(DUPIN, self).__init__(observation_space=observation_space,
                                    action_space=action_space, lr_schedule=lr_schedule)
        self.search_space = SEARCH_SPACE
        self.search_space_name = SEARCH_SPACE_NAMES
        self.hidden_size = hidden_size
        # prediction layer
        self.embedding_heads = nn.ModuleList()
        self.predict_heads = nn.ModuleList()
        for i, (name, size) in enumerate(zip(self.search_space_name,
                                             self.search_space)):
            self.predict_heads.append(nn.Linear(hidden_size, size))
            # set embedding with same weight as the linear layer
            embedding = nn.Embedding(size, hidden_size)
            embedding.weight = self.predict_heads[i].weight
            self.embedding_heads.append(embedding)

        self.metric_feature_extractor = nn.Linear(2, hidden_size)

        self.use_rnn = use_rnn
        if use_rnn:
            # print("\033[32mUsing RNN ... ...\033[0m")
            # self.embedding_rnn = nn.LSTM(
            #     hidden_size, hidden_size // 2, 1, batch_first=True, bidirectional=True)
            self.config_rnn = nn.LSTM(
                hidden_size * 2, hidden_size, 1, batch_first=True, bidirectional=True)

        self.config_feature_extractor = nn.Sequential(
            nn.Linear(hidden_size * len(self.search_space), hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2, hidden_size))
        self.aggregator = nn.Linear(hidden_size*2, hidden_size)

        self.use_graph = use_graph
        if use_graph:
            # print("Using Graph ... ...")
            self.type_graph = pickle.load(
                open("./cache/graph/type_graph.pkl", "rb"))
            self.type_gcn = GCN(hidden_size)
            self.module_graph = pickle.load(
                open("./cache/graph/module_graph.pkl", "rb"))
            self.module_gcn = GCN(hidden_size)
            self.value_graph = pickle.load(
                open("./cache/graph/value_graph.pkl", "rb"))
            self.value_gcn = GCN(hidden_size)
            self.value_graph_weight = [[]
                                       for _ in range(len(self.value_graph))]
            self.max_weight_length = 10

            self.aggregator_graph_embedding = nn.Linear(
                hidden_size*3, hidden_size)
            self.fused_embedding = nn.Linear(hidden_size*2, hidden_size)

        if context_obs:
            # print("\033[35mUsing Context observation ... ...\033[0m")
            self.video_context_embedding = nn.Linear(
                video_context_size, hidden_size)
            self.context_aggregator = nn.Sequential(
                nn.Linear(hidden_size*2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size))

        self.init_context = nn.Parameter(
            torch.zeros(hidden_size), requires_grad=False)
        self.iter_generate = iter_generate
        if self.iter_generate:
            self.iter_generator = nn.LSTMCell(hidden_size, hidden_size,
                                              bias=False)
            self.h, self.c = torch.zeros(
                1, hidden_size).to(self.device), \
                torch.zeros(1, hidden_size)

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        context_observation: Union[np.ndarray, Dict[str, np.ndarray]] = None,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        observation = self.obs_to_tensor(observation)
        if context_observation is not None:
            context_observation = np.array(context_observation)
            context_observation = self.obs_to_tensor(context_observation)
        with torch.no_grad():
            actions, _ = self._predict(
                observation, context_observation, deterministic=deterministic)
        return actions.cpu().numpy(), state

    def _predict(self,
                 observation: Union[Tensor, Dict[str, Tensor]],
                 context_observation: Union[Tensor, Dict[str, Tensor]] = None,
                 deterministic: bool = False) -> torch.Tensor:
        if len(observation.shape) == 1:
            observation = observation.reshape(1, -1)
            observation = observation.unsqueeze(0)
        elif len(observation.shape) == 2:
            observation = observation.unsqueeze(0)
        if context_observation is not None:
            if len(context_observation.shape) == 1:
                context_observation = context_observation.unsqueeze(0)
        batch_context = self.extract_batch_context(
            observation, context_observation)
        action, log_probs, _, _ = self.extract_action_from_context_based_on_mlp(
            batch_context)

        return action, log_probs

    def obs_to_tensor(self, observation):
        observation = torch.from_numpy(observation).float().to(self.device)
        return observation

    def extract_graph_embedding(self):
        embeddings = []
        for i, name in enumerate(self.search_space_name):
            param_ids = torch.LongTensor(
                list(range(self.search_space[i]))).to(self.device)
            embedding = self.embedding_heads[i](param_ids)
            embeddings.append(embedding)
        embeddings = torch.cat(embeddings)

        type_edge_index = torch.tensor(self.type_graph).t().contiguous()
        module_edge_index = torch.tensor(self.module_graph).t().contiguous()
        value_edge_index = torch.tensor(self.value_graph).t().contiguous()
        value_edge_weight = torch.tensor(
            [sum(v_list) / (len(v_list) + 1e-9) for v_list in self.value_graph_weight]).to(self.device)

        type_edge_data = Data(x=embeddings, edge_index=type_edge_index)
        module_edge_data = Data(x=embeddings, edge_index=module_edge_index)
        value_edge_data = Data(
            x=embeddings, edge_index=value_edge_index, edge_weight=value_edge_weight)

        type_edge_data = type_edge_data.to(self.device)
        module_edge_data = module_edge_data.to(self.device)
        value_edge_data = value_edge_data.to(self.device)

        type_embedding = self.type_gcn(type_edge_data)
        module_embedding = self.module_gcn(module_edge_data)
        value_embedding = self.value_gcn(value_edge_data)

        fused_embedding = torch.cat(
            [type_embedding, module_embedding, value_embedding], dim=-1)
        embeddings = self.aggregator_graph_embedding(fused_embedding)

        graph_embedding_dict = {}
        for i, name in enumerate(self.search_space_name):
            start_idx, end_idx = sum(self.search_space[:i]), sum(
                self.search_space[:i+1])
            graph_embedding_dict[name] = embeddings[start_idx:end_idx]

        return graph_embedding_dict

    def extract_batch_context(self, observations, context_obs=None):
        graph_embedding_dict = None
        if self.use_graph:
            graph_embedding_dict = self.extract_graph_embedding()

        batch_context = []
        for bi, ob in enumerate(observations):
            if self.use_rnn:
                context = self.extract_context_based_on_rnn(
                    ob, context_observation=context_obs[bi] if context_obs is not None else None,
                    graph_embeddings=graph_embedding_dict)
            else:
                context = self.extract_context_based_on_mlp(
                    ob, context_observation=context_obs[bi] if context_obs is not None else None,
                    graph_embeddings=graph_embedding_dict)
            batch_context.append(context)
        batch_context = torch.cat(batch_context, dim=0)
        # normalize the context
        # batch_context = (batch_context - self.mean) / self.std
        # self.batch_context_vector_pool.append(batch_context.detach())
        return batch_context

    def update_mean_std(self, alpha=0.9):
        context_vectors = torch.cat(self.batch_context_vector_pool, dim=0)
        mean = context_vectors.mean(dim=0)
        std = context_vectors.std(dim=0)
        self.mean.copy_(alpha * self.mean.data + (1 - alpha) * mean)
        self.std.copy_(alpha * self.std.data + (1 - alpha) * std)
        self.batch_context_vector_pool = []

    def extract_context_based_on_mlp(self, observation,
                                     context_observation=None,
                                     graph_embeddings=None):
        '''
        observation: (maxsize, 2 + len(search_space))
        '''
        # If context_observation is not None, then the context_observation is used as the context
        if context_observation is not None:
            video_context = self.video_context_embedding(
                context_observation).unsqueeze(0)
        # Filter every row that is not -1. The size of obs is (B, maxsize, 2 + len(search_space))
        ob = observation[observation[:, 0] != -1]
        metric_observation, config_observation = ob[:, :2], ob[:, 2:]
        config_observation = config_observation.long()
        embeddings = []
        for i, name in enumerate(self.search_space_name):
            embedding = self.embedding_heads[i](config_observation[:, i])
            if graph_embeddings is not None:
                embedding = torch.cat(
                    [embedding, graph_embeddings[name][config_observation[:, i]]], dim=-1)
                embedding = self.fused_embedding(embedding)
            embeddings.append(embedding)
        # (1, len(search_space), hidden_size)
        embeddings = torch.stack(embeddings, dim=1)
        B, S, H = embeddings.size()
        embeddings = embeddings.view(B, S*H)

        config_feature_set = self.config_feature_extractor(embeddings)
        metric_feature_set = self.metric_feature_extractor(metric_observation)
        feature_set = torch.cat(
            [config_feature_set, metric_feature_set], dim=-1)
        observation_context = self.aggregator(
            torch.max(feature_set, dim=0)[0].unsqueeze(0))

        if context_observation is not None:
            context = self.context_aggregator(
                torch.cat([observation_context, video_context], dim=-1))

        return context

    def extract_context_based_on_rnn(self, observation,
                                     context_observation=None,
                                     graph_embeddings=None):
        '''
        observation: (maxsize, 2 + len(search_space))
        '''
        # If context_observation is not None, then the context_observation is used as the context
        if context_observation is not None:
            video_context = self.video_context_embedding(
                context_observation).unsqueeze(0)
        # Filter every row that is not -1. The size of obs is (B, maxsize, 2 + len(search_space))
        ob = observation[observation[:, 0] != -1]

        if len(ob) != 0:
            metric_observation, config_observation = ob[:, :2], ob[:, 2:]
            config_observation = config_observation.long()
            embeddings = []
            for i, name in enumerate(self.search_space_name):
                embedding = self.embedding_heads[i](config_observation[:, i])
                if graph_embeddings is not None:
                    embedding = torch.cat(
                        [embedding, graph_embeddings[name][config_observation[:, i]]], dim=-1)
                    embedding = self.fused_embedding(embedding)
                embeddings.append(embedding)
            # (1, len(search_space), hidden_size)
            embeddings = torch.stack(embeddings, dim=1)
            B, S, H = embeddings.size()
            # if self.use_rnn:
            #     embeddings, _ = self.embedding_rnn(embeddings)
            embeddings = embeddings.reshape(B, S*H)
            config_feature_set = self.config_feature_extractor(embeddings)
            metric_feature_set = self.metric_feature_extractor(
                metric_observation)
            feature_set = torch.cat(
                [config_feature_set, metric_feature_set], dim=-1)
            if self.use_rnn:
                _, last_feature = self.config_rnn(feature_set)
                last_hidden = torch.cat(
                    [last_feature[0][0], last_feature[0][1]], dim=-1).unsqueeze(0)
            else:
                last_hidden = torch.max(feature_set, dim=0)[0].unsqueeze(0)
            observation_context = self.aggregator(last_hidden)
        else:
            observation_context = self.init_context.unsqueeze(0)

        if context_observation is not None:
            context = self.context_aggregator(
                torch.cat([observation_context, video_context], dim=-1))

        return context

    def extract_action_from_context_based_on_mlp(self, batch_context,
                                                 actions=None,
                                                 deterministic=False,
                                                 return_logits=False):
        """_summary_

        Args:
            batch_context (_type_): [B, H]
            actions (_type_, optional): _description_. Defaults to None.
            deterministic (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        # for i, name in enumerate(self.search_space_name):
        #     # pred = getattr(self, name)(batch_context) false code here because the inplace operation
        #     pred = getattr(self, name)(batch_context)
        #     pred_fallten.append(pred)
        preds = []
        for i, name in enumerate(self.search_space_name):
            pred = self.predict_heads[i](batch_context)
            preds.append(pred)

        preds_distribution = ProbabilisticDistribution(preds)
        actions = preds_distribution.get_actions(
            deterministic=deterministic)

        log_probs = preds_distribution.log_prob(actions)
        entropy = preds_distribution.entropy()

        if return_logits:
            return actions, log_probs, entropy, preds_distribution

        return actions, log_probs, entropy, None

    def reset_hidden_state(self):
        self.h = torch.zeros(1, self.hidden_size)
        self.c = torch.zeros(1, self.hidden_size)

    def extract_action_from_context_based_on_rnn(self, batch_context,
                                                 actions=None,
                                                 deterministic=False,
                                                 return_logits=False):
        """_summary_

        Args:
            batch_context (_type_): [B, H]
            actions (_type_, optional): _description_. Defaults to None.
            deterministic (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """

        if self.iter_generate:
            hidden_state, cell_state = self.iter_generator(
                batch_context, (self.h.to(batch_context.device),
                                self.c.to(batch_context.device)))
            self.h, self.c = hidden_state, cell_state
        else:
            hidden_state = batch_context

        preds = []
        for i, name in enumerate(self.search_space_name):
            pred = self.predict_heads[i](hidden_state)
            preds.append(pred)

        preds_distribution = ProbabilisticDistribution(preds)
        actions = preds_distribution.get_actions(
            deterministic=deterministic)

        log_probs = preds_distribution.log_prob(actions)
        entropy = preds_distribution.entropy()

        if return_logits:
            return actions, log_probs, entropy, preds_distribution

        return actions, log_probs, entropy, None

    def forward(self, obs: Tensor, context_obs: Tensor = None, deterministic: bool = False,
                return_logits=False) -> Tuple[Tensor]:
        batch_context = self.extract_batch_context(obs, context_obs)
        if self.iter_generate:
            action, log_probs, entropy, logits = self.extract_action_from_context_based_on_rnn(
                batch_context, deterministic=deterministic, return_logits=return_logits)
        else:
            action, log_probs, entropy, logits = self.extract_action_from_context_based_on_mlp(
                batch_context, deterministic=deterministic, return_logits=return_logits)

        if return_logits:
            return action, log_probs, entropy, logits

        return action, log_probs, entropy

    def evaluate_actions(self, obs: Union[Tensor, Dict[str, Tensor]], actions: Tensor) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        batch_context = self.extract_batch_context(obs)
        _, log_probs, entropy = self.extract_action_from_context_based_on_mlp(
            batch_context)
        batch_value = self.value_net(batch_context)

        return batch_value, log_probs, entropy

    def predict_values(
            self,
            obs: Union[np.ndarray, Dict[str, np.ndarray]]):
        if len(obs.shape) == 1:
            obs = obs.reshape(1, -1)
            obs = obs.unsqueeze(0)
        elif len(obs.shape) == 2:
            obs = obs.unsqueeze(0)

        batch_context = self.extract_batch_context(obs)
        batch_value = self.value_net(batch_context)

        return batch_value

    def update_graph_weights(self, values):
        for action, weight in values:
            for i in range(len(self.search_space)):
                for j in range(len(self.search_space)):
                    embed_i = sum(self.search_space[:i]) + action[i]
                    embed_j = sum(self.search_space[:j]) + action[j]
                    if [embed_i, embed_j] in self.value_graph:
                        edge_idx = self.value_graph.index(
                            [embed_i, embed_j])
                        if len(self.value_graph_weight[edge_idx]) < self.max_weight_length:
                            self.value_graph_weight[edge_idx].append(weight)
                        else:
                            self.value_graph_weight[edge_idx].pop(0)
                            self.value_graph_weight[edge_idx].append(weight)


class HippoEnv(gym.Env):
    def __init__(self, dataset, feature_extractor_model, camera_config,
                 max_iter, max_pareto_set_size,
                 debug=False, use_init_solution=True, use_cache=False):
        super(HippoEnv, self).__init__()
        self.dataset = dataset
        self.feature_extractor = feature_extractor_model
        self.pareto_set = []
        self.video_ids = dataset.video_ids
        self.video_ids_iter = dataset.video_ids.copy()
        self.video_camera_config = camera_config
        self.max_iter = max_iter
        self.max_pareto_set_size = max_pareto_set_size
        self.iter_num = 0
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.max_pareto_set_size, len(SEARCH_SPACE) + 2), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete(SEARCH_SPACE)
        self.debug = debug
        self.batch = 0
        # if self.debug:
        #     print("\033[91mDebug mode is on ... ...\033[0m")

        self.use_init_solution = use_init_solution
        if use_init_solution:
            self.init_solution = [[5, 5, 1, 4, 4, 1, 1, 2, 4],
                                  [0, 5, 0, 1, 2, 1, 4, 4, 1],
                                  [0, 1, 0, 1, 0, 1, 1, 4, 3],
                                  [0, 1, 7, 1, 1, 1, 4, 1, 3],
                                  [1, 0, 0, 1, 1, 1, 1, 3, 1]]

        self.use_cache = use_cache

    @property
    def skyscraper_configs(self):
        context_config_number = 2 # self.max_pareto_set_size
        config_vectors = []
        for _ in range(context_config_number):
            config_vectors.append(utils.generate_random_config_vector())
        return config_vectors

    @staticmethod
    def roc_func(pareto_set):
        pareto_set = sorted(pareto_set, key=lambda x: x[0])
        area = pareto_set[0][0] * pareto_set[0][1]
        for i in range(1, len(pareto_set)):
            area += (pareto_set[i][1] + pareto_set[i-1][1]) \
                * (pareto_set[i][0] - pareto_set[i-1][0]) / 2
        return area

    @staticmethod
    def soc_func(pareto_set):
        if len(pareto_set) == 0:
            return 0.0
        pareto_set = sorted(pareto_set, key=lambda x: x[0])
        area = pareto_set[0][0] * pareto_set[0][1]
        for i in range(1, len(pareto_set)):
            area += pareto_set[i][1] * (pareto_set[i][0] - pareto_set[i-1][0])
        return area

    @staticmethod
    def seq_reward_func(v0, vt, vt_1):
        delta_t_0 = (vt - v0) / v0
        delta_t_t_1 = (vt_1 - vt) / vt

        if delta_t_0 > 0:
            r = ((1 + delta_t_0)**2 - 1)*abs(1 + delta_t_t_1)
        else:
            r = - ((1 - delta_t_0)**2 - 1)*abs(1 - delta_t_t_1)
        return r

    def step_reward(self, direction=True, step_reward_scale=1.0):
        if not direction:
            return 0.0
        new_pareto_roc = self.soc_func(self.pareto_set)
        reward = max(new_pareto_roc - self.pareto_roc, 0) * step_reward_scale

        return reward

    def win_reward(self, win_reward_scale=10.0):
        roc = self.roc_func(self.pareto_set)
        mean_acc, mean_lat = [], []
        for res in self.pareto_set:
            mean_acc.append(res[0])
            mean_lat.append(res[1])
        mean_acc = sum(mean_acc) / len(mean_acc)
        mean_lat = sum(mean_lat) / len(mean_lat)
        win_rewards = [roc, mean_acc, mean_lat]
        win_reward = mean(win_rewards) * win_reward_scale
        return win_reward

    def prepare_observation(self, pareto_set):
        processed_pareto_set = np.ones(
            (self.max_pareto_set_size, len(SEARCH_SPACE) + 2)) * -1
        for i, point in enumerate(pareto_set):
            processed_pareto_set[i, :2] = point[:2]
            processed_pareto_set[i, 2:] = point[2]
        return processed_pareto_set

    def prepare_context_observation(self):
        return self._prepare_context_observation(self.context)

    @staticmethod
    def _prepare_context_observation(context):
        cameraid = context['Cameraid']
        timeid = context['Time']
        quality = context['Quailty']
        traj_context = [context['mean_speed'], context['std_speed'], context['mean_acceleration'], context['std_acceleration'],
                        context['mean_traj_linearity'], context['mean_traj_length'], context['traj_count'],
                        context['proportion_of_stopped_vehicles'],
                        context['mean_traj_densitys'], context['std_traj_densitys'],
                        context['car_rate'], context['bus_rate'], context['truck_rate']]
        return [cameraid, timeid, quality] + traj_context

    def run_config(self, action):
        self.video_camera_config.updateConfig(utils.generate_config(action,
                                                                    self.video_camera_config.loadConfig()))
        if self.use_cache:
            _, motvalue, _, _, _, \
                cmetric, _ = self.video_camera_config.ingestion
        else:
            _, motvalue, _, _, _, \
                cmetric, _ = self.video_camera_config.ingestion_without_cache
        accuracy, latency = motvalue, cmetric[-1]
        return accuracy, latency

    def step(self, action, env=None):
        if len(action.shape) == 2:
            action = action[0]
        self.video_camera_config.id = self.current_video
        if self.debug:
            accuracy, latency = random.random(), random.random()
            hotas, motas, idf1s = [0.0], [0.0], [0.0]
        else:
            self.video_camera_config.updateConfig(utils.generate_config(action,
                                                                        self.video_camera_config.loadConfig()))
            if self.use_cache:
                _, motvalue, _, metrics_list, _, \
                    cmetric, _ = self.video_camera_config.ingestion
            else:
                _, motvalue, _, metrics_list, _, \
                    cmetric, _ = self.video_camera_config.ingestion_without_cache
            accuracy, latency = motvalue, cmetric[-1]
            print(accuracy)
            num_objects = len(metrics_list)#de
            hotas = metrics_list[:num_objects]#de
            motas = metrics_list[num_objects:2*num_objects]#de
            idf1s = metrics_list[2*num_objects:3*num_objects]#de
            print(hotas)
        # not self.pareto_set or accuracy < min(pareto[0] for pareto in self.pareto_set)
        direction = True

        if direction:
            self.pareto_set.append((accuracy, latency, action,hotas,motas,idf1s))#de
            self.pareto_set = self.identify_pareto(self.pareto_set)
            self.pareto_set = sorted(self.pareto_set, key=lambda x: x[0])

        observation = self.prepare_observation(self.pareto_set)
        context_observation = self.prepare_context_observation()

        if env is not None:
            distances = []
            for video_id in env.video_ids:
                env_context = env.dataset.getitem_context(video_id)
                env_context = self._prepare_context_observation(env_context)
                distances.append([np.linalg.norm(np.array([co - ec for co, ec in zip(context_observation, env_context)])),
                                  env_context])
            context_observation = min(distances, key=lambda x: x[0])[1]

        reward = self.step_reward(direction=direction)
        done, truncated = False, False
        info = {"value": [accuracy, latency, action]}
        self.pareto_num = len(self.pareto_set)
        self.pareto_roc = self.soc_func(self.pareto_set)

        if len(self.pareto_set) == self.max_pareto_set_size:
            reward += self.win_reward()
            done = True

        if self.iter_num >= self.max_iter - 1:
            truncated = True

        self.iter_num += 1
        return [observation, context_observation], reward, done, truncated, info

    def reset(self, video_id=None, env=None):
        if video_id is None:
            if len(self.video_ids_iter) == 0:
                self.video_ids_iter = self.video_ids.copy()
            random.shuffle(self.video_ids_iter)
            video_id = self.video_ids_iter.pop()

        self.current_video = video_id
        self.context = self.dataset.getitem_context(self.current_video)
        self.video_camera_config.id = video_id

        if not self.use_init_solution:
            action = None
            self.pareto_set = []
            accuracy, latency = -1.0, -1.0
        else:
            self.pareto_set = []
            if len(self.init_solution) == 0:
                self.init_solution = [GOLDEN_CONFIG_VECTOR]
            for action in self.init_solution:
                self.video_camera_config.updateConfig(utils.generate_config(action,
                                                                            self.video_camera_config.loadConfig()))
                if self.debug:
                    accuracy, latency = random.random(), random.random()
                else:
                    if self.use_cache:
                        _, motvalue, _, _, _, \
                            cmetric, _ = self.video_camera_config.ingestion
                    else:
                        _, motvalue, _, _, _, \
                            cmetric, _ = self.video_camera_config.ingestion_without_cache
                    accuracy, latency = motvalue, cmetric[-1]
                    
                self.pareto_set.append([accuracy, latency, action])
            self.pareto_set = self.identify_pareto(self.pareto_set)
        self.pareto_set = sorted(self.pareto_set, key=lambda x: x[0])

        self.iter_num = 0
        observation = self.prepare_observation(self.pareto_set)
        context_observation = self.prepare_context_observation()

        if env is not None:
            distances = []
            for video_id in env.video_ids:
                env_context = env.dataset.getitem_context(video_id)
                env_context = self._prepare_context_observation(env_context)
                distances.append([np.linalg.norm(np.array([co - ec for co, ec in zip(context_observation, env_context)])),
                                  env_context])
            context_observation = min(distances, key=lambda x: x[0])[1]

        self.pareto_num = len(self.pareto_set)
        self.pareto_roc = self.soc_func(self.pareto_set)

        return [observation, context_observation], {"value": [accuracy, latency, action]}

    @staticmethod
    def is_dominated(x, y):
        return all(x[i] <= y[i] for i in range(2)) and any(x[i] < y[i] for i in range(2))

    def identify_pareto(self, solutions):
        pareto_front = []
        for solution in solutions:
            if not any(self.is_dominated(solution, other) for other in solutions):
                if ",".join([str(int(res)) for res in solution[-1]]) not in [",".join([str(int(res)) for res in pareto[-1]]) for pareto in pareto_front]:
                    pareto_front.append(solution)
        return pareto_front

    def extract_features(self, frames):
        return self.feature_extractor.extract(frames)

    def render(self, mode="human"):
        pass

    def collocate_pareto_set(self, video_id, camera_type="train", cache_dir="./cache/global/one_camera_with_config"):
        ingestion_result_path_name_list = os.listdir(cache_dir)
        filtered_ingestion_result_path_name_list = [
            path_name for path_name in ingestion_result_path_name_list if f"{camera_type}_{video_id}_" in path_name]
        result_sets = []
        for ingestion_result_path_name in filtered_ingestion_result_path_name_list:
            ingestion_result_path = os.path.join(
                cache_dir, ingestion_result_path_name)
            with open(ingestion_result_path, "rb") as f:
                ingestion_result = pickle.load(f)
                cache_info = ingestion_result_path_name.rstrip(".pkl").split("_")[
                    1:]
                video_id, config_vector = int(cache_info[0]), cache_info[1:]
                config_vector = [int(res) for res in config_vector]
                _, ingestionvalue, _, _, _, cmetric, _ = ingestion_result
                result_sets.append(
                    [ingestionvalue, cmetric[-1], config_vector])
        # print("result_sets: ", result_sets)
        pareto_set = self.identify_pareto(result_sets)
        pareto_set = sorted(pareto_set, key=lambda x: x[0])
        return pareto_set

    def collocate_no_pareto_set(self, video_id, camera_type="train", cache_dir="./cache/global/one_camera_with_config"):
        return_pareto_set = {eff_bound:None for eff_bound in range(1, 10)}
        ingestion_result_path_name_list = os.listdir(cache_dir)
        filtered_ingestion_result_path_name_list = [
            path_name for path_name in ingestion_result_path_name_list if f"{camera_type}_{video_id}" in path_name]
        
        for ingestion_result_path_name in filtered_ingestion_result_path_name_list:
            ingestion_result_path = os.path.join(
                cache_dir, ingestion_result_path_name)
            with open(ingestion_result_path, "rb") as f:
                ingestion_result = pickle.load(f)
                cache_info = ingestion_result_path_name.rstrip(".pkl").split("_")[
                    1:]
                video_id, config_vector = int(cache_info[0]), cache_info[1:]
                config_vector = [int(res) for res in config_vector]
                _, ingestionvalue, _, _, _, cmetric, _ = ingestion_result
                
                eff_bound = int(cmetric[-1] / 0.1)
                if eff_bound in return_pareto_set:
                    if return_pareto_set[eff_bound] is None:
                        return_pareto_set[eff_bound] = [ingestionvalue, cmetric[-1], config_vector]
                    else:
                        if ingestionvalue > return_pareto_set[eff_bound][0]:
                            return_pareto_set[eff_bound] = [ingestionvalue, cmetric[-1], config_vector]
        
        result_sets = []
        for eff_bound in range(1, 10):
            if return_pareto_set[eff_bound] is not None:
                result_sets.append(return_pareto_set[eff_bound])
        
        pareto_set = result_sets # self.identify_pareto(result_sets)
        pareto_set = sorted(pareto_set, key=lambda x: x[0])
        return pareto_set
    def set_batch(self, batch):
        self.batch = batch

    @staticmethod
    def evenly_sample_indices(data, num_samples=10):
 
        quantiles = np.linspace(0, 1, num_samples)  
        quantile_values = np.quantile(data, quantiles)
        sorted_indices = np.argsort(data)
        sorted_data = data[sorted_indices]
        sample_indices = []
        for value in quantile_values:
            idx = np.abs(sorted_data - value).argmin()
            sample_indices.append(sorted_indices[idx])

        return np.unique(sample_indices)

    def step_without_ingestion(self, accuracy, latency, action):
        if len(action.shape) == 2:
            action = action[0]
        self.video_camera_config.id = self.current_video
        self.video_camera_config.updateConfig(utils.generate_config(action,
                                                                    self.video_camera_config.loadConfig()))

        self.pareto_set.append((accuracy, latency, action))
        self.pareto_set = self.identify_pareto(self.pareto_set)
        self.pareto_set = sorted(self.pareto_set, key=lambda x: x[0])

        observation = self.prepare_observation(self.pareto_set)
        context_observation = self.prepare_context_observation()

        done, truncated = False, False
        # info = {"value": [accuracy, latency, action]}
        self.pareto_num = len(self.pareto_set)
        self.pareto_roc = self.soc_func(self.pareto_set)

        if len(self.pareto_set) == self.max_pareto_set_size:
            done = True

        if self.iter_num >= self.max_iter - 1:
            truncated = True

        self.iter_num += 1
        return [observation, context_observation], done, truncated

    def eval_config(self, action, video_id):
        if isinstance(action, np.ndarray):
            if len(action.shape) == 2:
                action = action[0]
        self.video_camera_config.id = video_id
        self.video_camera_config.updateConfig(utils.generate_config(action,
                                                                    self.video_camera_config.loadConfig()))
        if self.debug:
            accuracy, latency = random.random(), random.random()
        else:
            if self.use_cache:
                _, motvalue, _, _, _, \
                    cmetric, _ = self.video_camera_config.ingestion
            else:
                _, motvalue, _, _, _, \
                    cmetric, _ = self.video_camera_config.ingestion_without_cache
            accuracy, latency = motvalue, cmetric[-1]
        return accuracy, latency

    def filter_duplicate(self, solutions):
        # solutions : [acc, lat, config] * N
        no_duplicate_solutions = []
        for solution in solutions:
            new_flag = True
            for no_duplicate_solution in no_duplicate_solutions:
                no_duplicate_acc, no_duplicate_lat, no_duplicate_config = no_duplicate_solution
                acc, lat, config = solution
                if (no_duplicate_acc - acc)**2 + (no_duplicate_lat - lat)**2 < 1e-6:
                    new_flag = False
                    break
            if new_flag:
                no_duplicate_solutions.append(solution)
        return no_duplicate_solutions


def prepare_context_observation(context):
    cameraid = context['Cameraid']
    timeid = context['Time']
    quality = context['Quailty']
    traj_context = [context['mean_speed'], context['std_speed'], context['mean_acceleration'], context['std_acceleration'],
                    context['mean_traj_linearity'], context['mean_traj_length'], context['traj_count'],
                    context['proportion_of_stopped_vehicles'],
                    context['mean_traj_densitys'], context['std_traj_densitys'],
                    context['car_rate'], context['bus_rate'], context['truck_rate']]
    return [cameraid, timeid, quality] + traj_context


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="hippo")
    parser.add_argument('--seed', type=int, default=581, help='random seed')
    parser.add_argument('--test_video_num_list', type=int, nargs='+', help='test video streams') 
    parser.add_argument('--method_list', type=str, nargs='+', help='method list', default=['hippo','skyscraper','otif','unitune'
                                                                                           ])
    args = parser.parse_args()
    
seed = args.seed
test_video_num_list = args.test_video_num_list
method_list = args.method_list
func_list = [CANDIDATE_FUNCS[method] for method in method_list]

print("seed: ", seed)
print("test_video_num_list: ", test_video_num_list)
print("method_list: ", method_list)

random.seed(seed)  # Set random seed
np.random.seed(seed)  # Set random seed
torch.manual_seed(seed)  # Set random seed
torch.cuda.manual_seed(seed)  # Set random seed
torch.cuda.manual_seed_all(seed)  # Set random seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
channel = 400  # Total number of video stream channels for testing. This doesn't mean all video streams can be processed simultaneously, but refers to the number of streams for basic information retrieval.
method_name = "parallel_test"  # The name of the test method
data_name = "hippo"  # The name of the dataset
data_type = "test"  # The type of the dataset
scene_name = "hippo"  # The name of the scene
device_id = 1  # GPU ID
lr = 1e-2  # Learning rate
temp = 10.0  # Softmax temperature. Since we are measuring video processing efficiency, the temperature is a parameter that normalizes processing time.
max_iter = 10  # Maximum number of iterations. This refers to the maximum number of iterations for Hippo and many comparison algorithms to find configurations for each video stream cluster.
use_rnn = True  # Whether the Hippo agent uses RNN
batch_size = 32  # Batch size for training the Hippo agent
update_step = 64  # Update step for training the Hippo agent
hidden_size = 32  # Hidden layer size for the Hippo agent
lr_decay_flag = True  # Whether to use learning rate decay for training the Hippo agent
max_pareto_set_size = 10  # The maximum size of the Pareto set that the Hippo agent finds for video streams
video_context_size = 16  # Video stream context size for the Hippo agent
use_init_solution = False  # Whether the Hippo agent uses an initial solution
context_path_name = "context_01.csv"  # The context file name. Since both the training and test video streams lack a training module for context, all contexts are manually labeled and stored here.
framenumber, framegap = 16, 16  # The number of frames and frame gap for reading video streams
efficiency_targets = [0.50, 0.60, 0.70, 0.80, 0.90]  # Processing efficiency targets
train_video_list = list(range(800))  # List of all video stream IDs in the training set
filter_video_ids = [34, 60, 103, 133, 137, 142, 145, 149, 151, 156, 160, 162, 163, 168, 176, 177, 181, 183, 196, 300, 302, 367, 374, 376, 410, 469, 470, 476, 483, 495, 521, 534, 653, 673, 683, 692, 705, 730, 786, 799]  # Filtered video stream IDs
for filter_video_id in filter_video_ids:  # Remove certain video streams, as some video streams in the training set don't have meaningful targets, so any configuration is irrelevant.
    train_video_list.remove(filter_video_id)
debug = False  # Whether to enable debug mode. In debug mode, video streams are not parsed, and random data is generated for quick code testing.
scene_num = 8  # The number of scenes. This refers to the total number of scenes involving video streams.
object_nan = 100  # Invalid value. Some values in our dataset are invalid, and this is used to represent those invalid values.
use_cache = True  # Whether to use caching. If caching is enabled, parsing some video streams can use cached data, speeding up the process.
train_fit = True  # Whether to map the test set's context to the training set. The details are explained in the `hippo_func`.
max_pool_length = 6  # This is a parameter used to find the optimal configuration.
latency_bound = 60  # The processing time limit. Since the video data is one minute long, processing times less than 60 seconds are considered real-time.
memory_bound = 23000  # Memory limit
train_gap = 100  # Interval between video streams in the training set
test_gap = 50  # Interval between video streams in the test set
train_scene_dict, test_scene_dict = {}, {}  # Dictionaries for the training and testing scenes. They group video stream IDs that belong to the same scene.
for scene_id in range(scene_num):  # Generate the scene dictionaries for training and testing
    for i in range(train_gap):  # Training set scene dictionary
        train_scene_dict[scene_id * train_gap + i] = scene_id
    for j in range(test_gap):  # Testing set scene dictionary
        test_scene_dict[scene_id * test_gap + j] = scene_id

    define_video_list = {100: [4, 8, 11, 14, 16, 17, 18, 30, 32, 33, 44, 45, 46, 50, 54, 58, 65, 67, 73, 74, 75, 85, 86, 88, 91, 92, 117, 118, 120, 122, 123, 127, 136, 139, 140, 142, 146, 149, 150, 151, 153, 156, 157, 158, 167, 176, 178, 187, 190, 196, 201, 208, 211, 219, 222, 223, 225, 233, 234, 238, 244, 246, 251, 255, 267, 273, 276, 282, 283, 285, 286, 287, 292, 295, 298, 308, 317, 319, 320, 323, 324, 328, 329, 331, 332, 344, 345, 348, 352, 355, 363, 367, 386, 389, 390, 393, 394, 395, 396, 399],
                         109: [3, 4, 8, 11, 14, 16, 17, 18, 30, 32, 33, 44, 45, 46, 50, 54, 58, 62, 65, 67, 73, 74, 75, 85, 86, 88, 91, 92, 104, 117, 118, 120, 122, 123, 127, 136, 139, 140, 142, 146, 149, 150, 151, 153, 156, 157, 158, 167, 170, 176, 178, 187, 189, 190, 196, 201, 208, 211, 219, 222, 223, 225, 233, 234, 237, 238, 244, 246, 251, 255, 267, 273, 276, 282, 283, 285, 286, 287, 292, 295, 298, 299, 308, 317, 319, 320, 323, 324, 328, 329, 331, 332, 344, 345, 348, 349, 352, 355, 363, 367, 375, 386, 389, 390, 393, 394, 395, 396, 399], 118: [3, 4, 8, 11, 14, 16, 17, 18, 30, 32, 33, 38, 44, 45, 46, 50, 54, 58, 62, 65, 67, 73, 74, 75, 85, 86, 88, 90, 91, 92, 104, 117, 118, 120, 122, 123, 126, 127, 136, 139, 140, 142, 146, 149, 150, 151, 153, 156, 157, 158, 161, 167, 170, 176, 178, 187, 189, 190, 196, 201, 206, 208, 211, 219, 222, 223, 225, 233, 234, 237, 238, 244, 246, 251, 255, 259, 267, 273, 276, 279, 282, 283, 285, 286, 287, 292, 295, 298, 299, 301, 308, 317, 319, 320, 323, 324, 328, 329, 331, 332, 344, 345, 348, 349, 352, 355, 363, 367, 372, 375, 386, 389, 390, 393, 394, 395, 396, 399], 125: [3, 4, 8, 11, 14, 16, 17, 18, 30, 32, 33, 36, 38, 44, 45, 46, 50, 54, 55, 58, 62, 65, 67, 73, 74, 75, 85, 86, 88, 90, 91, 92, 104, 117, 118, 120, 122, 123, 126, 127, 136, 139, 140, 142, 146, 147, 149, 150, 151, 153, 156, 157, 158, 161, 167, 170, 176, 178, 182, 187, 189, 190, 196, 201, 206, 208, 211, 219, 222, 223, 225, 227, 233, 234, 237, 238, 244, 246, 251, 255, 259, 267, 273, 276, 277, 279, 282, 283, 285, 286, 287, 292, 295, 298, 299, 301, 303, 308, 317, 319, 320, 323, 324, 328, 329, 331, 332, 344, 345, 348, 349, 352, 355, 363, 367, 372, 375, 386, 389, 390, 393, 394, 395, 396, 399], 127: [3, 4, 8, 11, 14, 16, 17, 18, 30, 32, 33, 36, 38, 44, 45, 46, 50, 54, 55, 58, 61, 62, 65, 67, 73, 74, 75, 85, 86, 88, 90, 91, 92, 104, 117, 118, 120, 122, 123, 126, 127, 136, 139, 140, 142, 146, 147, 149, 150, 151, 153, 156, 157, 158, 161, 167, 170, 176, 178, 182, 187, 189, 190, 196, 201, 206, 208, 211, 213, 219, 222, 223, 225, 227, 233, 234, 237, 238, 244, 246, 251, 255, 259, 267, 273, 276, 277, 279, 282, 283, 285, 286, 287, 292, 295, 298, 299, 301, 303, 308, 317, 319, 320, 323, 324, 328, 329, 331, 332, 344, 345, 348, 349, 352, 355, 363, 367, 372, 375, 386, 389, 390, 393, 394, 395, 396, 399], 136: [3, 4, 8, 11, 14, 16, 17, 18, 27, 30, 32, 33, 36, 38, 44, 45, 46, 50, 54, 55, 58, 61, 62, 65, 67, 73, 74, 75, 85, 86, 87, 88, 90, 91, 92, 104, 106, 117, 118, 120, 122, 123, 126, 127, 136, 139, 140, 142, 146, 147, 149, 150, 151, 153, 156, 157, 158, 161, 167, 170, 176, 178, 182, 184, 187, 189, 190, 196, 201, 206, 208, 211, 213, 219, 222, 223, 225, 226, 227, 233, 234, 237, 238, 244, 246, 251, 253, 255, 259, 267, 273, 276, 277, 279, 282, 283, 285, 286, 287, 292, 293, 295, 298, 299, 301, 303, 308, 317, 319, 320, 323, 324, 328, 329, 331, 332, 336, 344, 345, 348, 349, 352, 355, 356, 363, 367, 372, 375, 386, 389, 390, 393, 394, 395, 396, 399], 145: [3, 4, 8, 11, 14, 16, 17, 18, 27, 30, 32, 33, 36, 38, 41, 42, 44, 45, 46, 50, 53, 54, 55, 58, 61, 62, 65, 67, 73, 74, 75, 85, 86, 87, 88, 90, 91, 92, 103, 104, 106, 117, 118, 120, 122, 123, 126, 127, 136, 139, 140, 142, 146, 147, 149, 150, 151, 153, 156, 157, 158, 161, 167, 170, 176, 178, 182, 184, 187, 189, 190, 196, 199, 201, 204, 206, 208, 211, 213, 219, 222, 223, 225, 226, 227, 233, 234, 237, 238, 244, 246, 251, 253, 255, 258, 259, 267, 273, 276, 277, 279, 282, 283, 285, 286, 287, 292, 293, 295, 298, 299, 301, 303, 308, 317, 319, 320, 323, 324, 328, 329, 331, 332, 333, 336, 344, 345, 348, 349, 352, 355, 356, 363, 367, 372, 375, 376, 386, 389, 390, 393, 394, 395, 396, 399], 150: [3, 4, 8, 11, 14, 16, 17, 18, 22, 27, 30, 32, 33, 36, 38, 41, 42, 44, 45, 46, 50, 53, 54, 55, 58, 61, 62, 65, 67, 73, 74, 75, 85, 86, 87, 88, 90, 91, 92, 103, 104, 105, 106, 117, 118, 120, 122, 123, 126, 127, 136, 139, 140, 142, 146, 147, 149, 150, 151, 153, 156, 157, 158, 161, 167, 170, 176, 178, 182, 184, 187, 189, 190, 196, 199, 201, 204, 206, 208, 211, 213, 219, 222, 223, 225, 226, 227, 233, 234, 237, 238, 244, 246, 247, 251, 253, 255, 258, 259, 267, 273, 276, 277, 279, 282, 283, 285, 286, 287, 292, 293, 295, 298, 299, 301, 303, 308, 310, 317, 319, 320, 323, 324, 328, 329, 331, 332, 333, 336, 344, 345, 348, 349, 352, 355, 356, 363, 367, 372, 375, 376, 386, 389, 390, 391, 393, 394, 395, 396, 399], 159: [3, 4, 8, 11, 14, 16, 17, 18, 22, 27, 30, 31, 32, 33, 36, 38, 41, 42, 44, 45, 46, 50, 53, 54, 55, 58, 61, 62, 65, 67, 73, 74, 75, 81, 85, 86, 87, 88, 90, 91, 92, 103, 104, 105, 106, 117, 118, 120, 122, 123, 126, 127, 129, 136, 139, 140, 142, 146, 147, 149, 150, 151, 153, 155, 156, 157, 158, 161, 167, 170, 176, 178, 182, 184, 187, 189, 190, 196, 199, 201, 204, 206, 208, 210, 211, 213, 219, 222, 223, 225, 226, 227, 233, 234, 237, 238, 244, 246, 247, 251, 252, 253, 255, 258, 259, 267, 273, 276, 277, 279, 282, 283, 285, 286, 287, 292, 293, 295, 298, 299, 301, 303, 308, 310, 317, 319, 320, 323, 324, 328, 329, 331, 332, 333, 336, 342, 344, 345, 348, 349, 352, 355, 356, 361, 363, 367, 370, 372, 375, 376, 386, 389, 390, 391, 393, 394, 395, 396, 399], 168: [1, 3, 4, 8, 11, 14, 16, 17, 18, 22, 27, 30, 31, 32, 33, 36, 38, 41, 42, 44, 45, 46, 50, 53, 54, 55, 58, 61, 62, 65, 67, 68, 73, 74, 75, 81, 85, 86, 87, 88, 90, 91, 92, 103, 104, 105, 106, 117, 118, 120, 122, 123, 126, 127, 129, 136, 139, 140, 142, 145, 146, 147, 149, 150, 151, 153, 155, 156, 157, 158, 161, 167, 169, 170, 176, 178, 182, 184, 187, 189, 190, 196, 199, 201, 204, 206, 208, 210, 211, 213, 219, 222, 223, 225, 226, 227, 233, 234, 237, 238, 242, 244, 246, 247, 249, 251, 252, 253, 255, 258, 259, 267, 273, 276, 277, 279, 282, 283, 285, 286, 287, 288, 292, 293, 295, 298, 299, 301, 303, 308, 310, 317, 319, 320, 323, 324, 328, 329, 331, 332, 333, 336, 337, 342, 344, 345, 348, 349, 352, 355, 356, 361, 363, 367, 370, 372, 375, 376, 386, 389, 390, 391, 392, 393, 394, 395, 396, 399], 175: [1, 3, 4, 8, 11, 14, 16, 17, 18, 19, 22, 27, 30, 31, 32, 33, 36, 38, 41, 42, 44, 45, 46, 50, 53, 54, 55, 58, 61, 62, 65, 67, 68, 73, 74, 75, 77, 81, 85, 86, 87, 88, 90, 91, 92, 103, 104, 105, 106, 109, 117, 118, 120, 122, 123, 126, 127, 129, 136, 139, 140, 142, 145, 146, 147, 149, 150, 151, 153, 155, 156, 157, 158, 161, 167, 169, 170, 175, 176, 178, 182, 184, 187, 189, 190, 196, 199, 201, 204, 206, 208, 210, 211, 213, 219, 222, 223, 225, 226, 227, 232, 233, 234, 237, 238, 242, 244, 246, 247, 249, 251, 252, 253, 255, 258, 259, 267, 273, 276, 277, 279, 282, 283, 285, 286, 287, 288, 292, 293, 294, 295, 298, 299, 301, 303, 308, 310, 317, 319, 320, 323, 324, 328, 329, 331, 332, 333, 336, 337, 342, 344, 345, 346, 348, 349, 352, 355, 356, 361, 363, 367, 370, 372, 375, 376, 386, 389, 390, 391, 392, 393, 394, 395, 396, 399], 177: [1, 3, 4, 8, 11, 14, 16, 17, 18, 19, 20, 22, 27, 30, 31, 32, 33, 36, 38, 41, 42, 44, 45, 46, 50, 53, 54, 55, 58, 61, 62, 65, 67, 68, 73, 74, 75, 77, 81, 85, 86, 87, 88, 90, 91, 92, 103, 104, 105, 106, 109, 117, 118, 120, 122, 123, 126, 127, 129, 136, 139, 140, 142, 145, 146, 147, 149, 150, 151, 153, 155, 156, 157, 158, 161, 167, 169, 170, 175, 176, 178, 182, 184, 187, 189, 190, 196, 199, 201, 204, 206, 208, 210, 211, 213, 219, 222, 223, 225, 226, 227, 232, 233, 234, 237, 238, 242, 244, 246, 247, 249, 251, 252, 253, 255, 258, 259, 266, 267, 273, 276, 277, 279, 282, 283, 285, 286, 287, 288, 292, 293, 294, 295, 298, 299, 301, 303, 308, 310, 317, 319, 320, 323, 324, 328, 329, 331, 332, 333, 336, 337, 342, 344, 345, 346, 348, 349, 352, 355, 356, 361, 363, 367, 370, 372, 375, 376, 386, 389, 390, 391, 392, 393, 394, 395, 396, 399], 186: [1, 3, 4, 8, 11, 14, 16, 17, 18, 19, 20, 22, 27, 29, 30, 31, 32, 33, 36, 38, 41, 42, 44, 45, 46, 50, 53, 54, 55, 58, 61, 62, 65, 67, 68, 73, 74, 75, 76, 77, 81, 85, 86, 87, 88, 90, 91, 92, 103, 104, 105, 106, 109, 117, 118, 120, 122, 123, 126, 127, 129, 136, 139, 140, 142, 145, 146, 147, 148, 149, 150, 151, 152, 153, 155, 156, 157, 158, 161, 167, 169, 170, 175, 176, 178, 182, 184, 187, 189, 190, 196, 199, 201, 204, 206, 208, 210, 211, 213, 219, 222, 223, 225, 226, 227, 232, 233, 234, 237, 238, 241, 242, 244, 246, 247, 249, 250, 251, 252, 253, 255, 258, 259, 263, 266, 267, 273, 276, 277, 279, 282, 283, 285, 286, 287, 288, 292, 293, 294, 295, 298, 299, 301, 303, 305, 308, 310, 317, 319, 320, 323, 324, 328, 329, 331, 332, 333, 336, 337, 342, 344, 345, 346, 348, 349, 352, 355, 356, 361, 363, 365, 367, 370, 372, 375, 376, 386, 389, 390, 391, 392, 393, 394, 395, 396, 399], 195: [1, 3, 4, 8, 11, 14, 16, 17, 18, 19, 20, 21, 22, 27, 29, 30, 31, 32, 33, 36, 38, 41, 42, 44, 45, 46, 50, 53, 54, 55, 58, 61, 62, 65, 67, 68, 73, 74, 75, 76, 77, 81, 85, 86, 87, 88, 90, 91, 92, 97, 102, 103, 104, 105, 106, 109, 117, 118, 120, 122, 123, 126, 127, 129, 136, 139, 140, 142, 145, 146, 147, 148, 149, 150, 151, 152, 153, 155, 156, 157, 158, 161, 167, 169, 170, 175, 176, 178, 182, 183, 184, 187, 189, 190, 196, 199, 201, 204, 206, 208, 210, 211, 213, 215, 219, 221, 222, 223, 225, 226, 227, 232, 233, 234, 237, 238, 241, 242, 244, 246, 247, 249, 250, 251, 252, 253, 255, 258, 259, 263, 266, 267, 268, 273, 276, 277, 279, 282, 283, 285, 286, 287, 288, 292, 293, 294, 295, 298, 299, 300, 301, 303, 305, 308, 310, 317, 319, 320, 323, 324, 328, 329, 331, 332, 333, 336, 337, 342, 344, 345, 346, 348, 349, 352, 355, 356, 361, 363, 365, 367, 370, 372, 375, 376, 386, 388, 389, 390, 391, 392, 393, 394, 395, 396, 399], 200: [1, 2, 3, 4, 8, 11, 14, 16, 17, 18, 19, 20, 21, 22, 27, 29, 30, 31, 32, 33, 36, 38, 41, 42, 44, 45, 46, 50, 53, 54, 55, 56, 58, 61, 62, 65, 67, 68, 73, 74, 75, 76, 77, 81, 85, 86, 87, 88, 90, 91, 92, 97, 102, 103, 104, 105, 106, 109, 117, 118, 120, 122, 123, 126, 127, 129, 133, 136, 139, 140, 142, 145, 146, 147, 148, 149, 150, 151, 152, 153, 155, 156, 157, 158, 161, 167, 169, 170, 175, 176, 178, 182, 183, 184, 187, 189, 190, 196, 199, 201, 204, 206, 208, 210, 211, 213, 215, 219, 221, 222, 223, 225, 226, 227, 232, 233, 234, 237, 238, 241, 242, 243, 244, 246, 247, 249, 250, 251, 252, 253, 255, 258, 259, 263, 266, 267, 268, 273, 276, 277, 279, 282, 283, 285, 286, 287, 288, 292, 293, 294, 295, 298, 299, 300, 301, 303, 305, 308, 310, 317, 319, 320, 323, 324, 328, 329, 331, 332, 333, 336, 337, 342, 344, 345, 346, 348, 349, 352, 355, 356, 361, 363, 365, 367, 370, 372, 375, 376, 378, 386, 388, 389, 390, 391, 392, 393, 394, 395, 396, 399], 209: [1, 2, 3, 4, 8, 11, 14, 16, 17, 18, 19, 20, 21, 22, 27, 29, 30, 31, 32, 33, 35, 36, 38, 41, 42, 44, 45, 46, 50, 53, 54, 55, 56, 58, 61, 62, 65, 67, 68, 73, 74, 75, 76, 77, 78, 81, 85, 86, 87, 88, 90, 91, 92, 97, 102, 103, 104, 105, 106, 108, 109, 117, 118, 120, 122, 123, 126, 127, 129, 133, 136, 139, 140, 142, 145, 146, 147, 148, 149, 150, 151, 152, 153, 155, 156, 157, 158, 161, 163, 167, 169, 170, 175, 176, 178, 182, 183, 184, 187, 189, 190, 196, 199, 201, 202, 204, 206, 208, 210, 211, 213, 215, 219, 221, 222, 223, 225, 226, 227, 232, 233, 234, 237, 238, 241, 242, 243, 244, 246, 247, 249, 250, 251, 252, 253, 255, 256, 258, 259, 263, 266, 267, 268, 273, 276, 277, 279, 282, 283, 285, 286, 287, 288, 291, 292, 293, 294, 295, 298, 299, 300, 301, 303, 305, 306, 308, 310, 317, 319, 320, 323, 324, 328, 329, 331, 332, 333, 336, 337, 342, 344, 345, 346, 348, 349, 352, 355, 356, 361, 363, 365, 367, 368, 370, 372, 375, 376, 378, 386, 388, 389, 390, 391, 392, 393, 394, 395, 396, 399], 218: [1, 2, 3, 4, 8, 11, 14, 16, 17, 18, 19, 20, 21, 22, 27, 29, 30, 31, 32, 33, 35, 36, 38, 39, 41, 42, 44, 45, 46, 50, 53, 54, 55, 56, 58, 61, 62, 65, 67, 68, 71, 73, 74, 75, 76, 77, 78, 81, 85, 86, 87, 88, 90, 91, 92, 97, 100, 102, 103, 104, 105, 106, 108, 109, 117, 118, 120, 122, 123, 126, 127, 128, 129, 133, 136, 139, 140, 142, 145, 146, 147, 148, 149, 150, 151, 152, 153, 155, 156, 157, 158, 161, 163, 167, 169, 170, 175, 176, 178, 182, 183, 184, 187, 189, 190, 192, 196, 199, 201, 202, 204, 206, 208, 210, 211, 213, 215, 219, 221, 222, 223, 225, 226, 227, 232, 233, 234, 237, 238, 241, 242, 243, 244, 246, 247, 248, 249, 250, 251, 252, 253, 255, 256, 258, 259, 263, 266, 267, 268, 273, 276, 277, 279, 281, 282, 283, 285, 286, 287, 288, 291, 292, 293, 294, 295, 298, 299, 300, 301, 303, 305, 306, 308, 310, 317, 319, 320, 323, 324, 326, 328, 329, 331, 332, 333, 336, 337, 342, 344, 345, 346, 348, 349, 352, 355, 356, 361, 363, 365, 367, 368, 370, 372, 375, 376, 378, 380, 386, 388, 389, 390, 391, 392, 393, 394, 395, 396, 399], 225: [1, 2, 3, 4, 8, 11, 14, 16, 17, 18, 19, 20, 21, 22, 27, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 41, 42, 44, 45, 46, 50, 53, 54, 55, 56, 58, 61, 62, 65, 67, 68, 71, 73, 74, 75, 76, 77, 78, 81, 85, 86, 87, 88, 90, 91, 92, 97, 100, 102, 103, 104, 105, 106, 108, 109, 117, 118, 120, 122, 123, 125, 126, 127, 128, 129, 133, 136, 139, 140, 142, 145, 146, 147, 148, 149, 150, 151, 152, 153, 155, 156, 157, 158, 161, 163, 167, 169, 170, 175, 176, 178, 182, 183, 184, 185, 187, 189, 190, 192, 196, 199, 201, 202, 204, 206, 208, 210, 211, 213, 215, 219, 221, 222, 223, 225, 226, 227, 232, 233, 234, 237, 238, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 255, 256, 258, 259, 263, 266, 267, 268, 273, 274, 276, 277, 279, 281, 282, 283, 285, 286, 287, 288, 291, 292, 293, 294, 295, 298, 299, 300, 301, 303, 305, 306, 308, 310, 317, 319, 320, 323, 324, 326, 327, 328, 329, 331, 332, 333, 336, 337, 342, 344, 345, 346, 348, 349, 352, 355, 356, 361, 363, 365, 367, 368, 370, 371, 372, 375, 376, 378, 380, 386, 388, 389, 390, 391, 392, 393, 394, 395, 396, 399], 227: [1, 2, 3, 4, 8, 11, 14, 16, 17, 18, 19, 20, 21, 22, 27, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 41, 42, 44, 45, 46, 50, 53, 54, 55, 56, 58, 61, 62, 65, 67, 68, 71, 73, 74, 75, 76, 77, 78, 81, 83, 85, 86, 87, 88, 90, 91, 92, 97, 100, 102, 103, 104, 105, 106, 108, 109, 117, 118, 120, 122, 123, 125, 126, 127, 128, 129, 133, 136, 139, 140, 142, 145, 146, 147, 148, 149, 150, 151, 152, 153, 155, 156, 157, 158, 161, 163, 167, 169, 170, 175, 176, 178, 182, 183, 184, 185, 187, 189, 190, 192, 195, 196, 199, 201, 202, 204, 206, 208, 210, 211, 213, 215, 219, 221, 222, 223, 225, 226, 227, 232, 233, 234, 237, 238, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 255, 256, 258, 259, 263, 266, 267, 268, 273, 274, 276, 277, 279, 281, 282, 283, 285, 286, 287, 288, 291, 292, 293, 294, 295, 298, 299, 300, 301, 303, 305, 306, 308, 310, 317, 319, 320, 323, 324, 326, 327, 328, 329, 331, 332, 333, 336, 337, 342, 344, 345, 346, 348, 349, 352, 355, 356, 361, 363, 365, 367, 368, 370, 371, 372, 375, 376, 378, 380, 386, 388, 389, 390, 391, 392, 393, 394, 395, 396, 399], 236: [0, 1, 2, 3, 4, 8, 11, 14, 16, 17, 18, 19, 20, 21, 22, 27, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 41, 42, 44, 45, 46, 50, 53, 54, 55, 56, 58, 61, 62, 65, 67, 68, 71, 73, 74, 75, 76, 77, 78, 81, 83, 85, 86, 87, 88, 90, 91, 92, 95, 97, 100, 102, 103, 104, 105, 106, 108, 109, 117, 118, 120, 122, 123, 125, 126, 127, 128, 129, 133, 135, 136, 139, 140, 142, 145, 146, 147, 148, 149, 150, 151, 152, 153, 155, 156, 157, 158, 161, 163, 167, 169, 170, 175, 176, 178, 182, 183, 184, 185, 187, 188, 189, 190, 192, 195, 196, 199, 201, 202, 204, 206, 208, 210, 211, 213, 214, 215, 219, 221, 222, 223, 225, 226, 227, 232, 233, 234, 237, 238, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 255, 256, 258, 259, 263, 266, 267, 268, 273, 274, 276, 277, 279, 280, 281, 282, 283, 285, 286, 287, 288, 291, 292, 293, 294, 295, 298, 299, 300, 301, 303, 304, 305, 306, 308, 310, 317, 319, 320, 323, 324, 326, 327, 328, 329, 331, 332, 333, 336, 337, 342, 344, 345, 346, 348, 349, 352, 353, 355, 356, 361, 363, 365, 367, 368, 370, 371, 372, 375, 376, 377, 378, 380, 386, 388, 389, 390, 391, 392, 393, 394, 395, 396, 399], 245: [0, 1, 2, 3, 4, 8, 11, 14, 16, 17, 18, 19, 20, 21, 22, 23, 27, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 41, 42, 44, 45, 46, 50, 53, 54, 55, 56, 58, 61, 62, 65, 67, 68, 71, 73, 74, 75, 76, 77, 78, 81, 83, 85, 86, 87, 88, 90, 91, 92, 95, 97, 98, 100, 102, 103, 104, 105, 106, 108, 109, 110, 117, 118, 120, 122, 123, 125, 126, 127, 128, 129, 133, 135, 136, 139, 140, 142, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 161, 163, 167, 169, 170, 175, 176, 178, 182, 183, 184, 185, 187, 188, 189, 190, 192, 195, 196, 199, 201, 202, 204, 206, 208, 209, 210, 211, 213, 214, 215, 219, 221, 222, 223, 225, 226, 227, 232, 233, 234, 237, 238, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 255, 256, 258, 259, 262, 263, 266, 267, 268, 273, 274, 276, 277, 279, 280, 281, 282, 283, 285, 286, 287, 288, 291, 292, 293, 294, 295, 298, 299, 300, 301, 303, 304, 305, 306, 308, 309, 310, 317, 319, 320, 323, 324, 326, 327, 328, 329, 331, 332, 333, 336, 337, 342, 344, 345, 346, 348, 349, 352, 353, 355, 356, 361, 363, 365, 367, 368, 370, 371, 372, 375, 376, 377, 378, 380, 381, 386, 388, 389, 390, 391, 392, 393, 394, 395, 396, 399], 250: [0, 1, 2, 3, 4, 8, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 27, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 41, 42, 44, 45, 46, 50, 53, 54, 55, 56, 58, 61, 62, 65, 67, 68, 71, 73, 74, 75, 76, 77, 78, 81, 83, 85, 86, 87, 88, 90, 91, 92, 95, 97, 98, 100, 102, 103, 104, 105, 106, 108, 109, 110, 117, 118, 120, 122, 123, 125, 126, 127, 128, 129, 133, 135, 136, 139, 140, 142, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 161, 163, 167, 169, 170, 175, 176, 178, 182, 183, 184, 185, 187, 188, 189, 190, 192, 195, 196, 198, 199, 201, 202, 204, 206, 208, 209, 210, 211, 213, 214, 215, 219, 221, 222, 223, 225, 226, 227, 232, 233, 234, 237, 238, 239, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 255, 256, 258, 259, 262, 263, 266, 267, 268, 273, 274, 276, 277, 279, 280, 281, 282, 283, 285, 286, 287, 288, 291, 292, 293, 294, 295, 298, 299, 300, 301, 303, 304, 305, 306, 308, 309, 310, 317, 319, 320, 322, 323, 324, 326, 327, 328, 329, 331, 332, 333, 336, 337, 342, 344, 345, 346, 348, 349, 352, 353, 355, 356, 358, 361, 363, 365, 367, 368, 370, 371, 372, 375, 376, 377, 378, 380, 381, 386, 388, 389, 390, 391, 392, 393, 394, 395, 396, 399], 259: [0, 1, 2, 3, 4, 8, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 27, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 41, 42, 43, 44, 45, 46, 50, 53, 54, 55, 56, 58, 61, 62, 65, 67, 68, 71, 73, 74, 75, 76, 77, 78, 81, 83, 85, 86, 87, 88, 89, 90, 91, 92, 95, 96, 97, 98, 100, 102, 103, 104, 105, 106, 108, 109, 110, 117, 118, 120, 122, 123, 125, 126, 127, 128, 129, 131, 133, 135, 136, 139, 140, 142, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 161, 163, 167, 169, 170, 172, 175, 176, 178, 182, 183, 184, 185, 187, 188, 189, 190, 192, 195, 196, 198, 199, 201, 202, 203, 204, 206, 208, 209, 210, 211, 213, 214, 215, 219, 221, 222, 223, 225, 226, 227, 232, 233, 234, 237, 238, 239, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 255, 256, 257, 258, 259, 262, 263, 266, 267, 268, 273, 274, 276, 277, 279, 280, 281, 282, 283, 285, 286, 287, 288, 291, 292, 293, 294, 295, 298, 299, 300, 301, 303, 304, 305, 306, 308, 309, 310, 317, 319, 320, 322, 323, 324, 326, 327, 328, 329, 331, 332, 333, 335, 336, 337, 342, 344, 345, 346, 348, 349, 352, 353, 355, 356, 358, 361, 363, 364, 365, 367, 368, 370, 371, 372, 375, 376, 377, 378, 380, 381, 386, 388, 389, 390, 391, 392, 393, 394, 395, 396, 399], 268: [0, 1, 2, 3, 4, 8, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 41, 42, 43, 44, 45, 46, 50, 53, 54, 55, 56, 58, 61, 62, 65, 67, 68, 71, 72, 73, 74, 75, 76, 77, 78, 81, 83, 85, 86, 87, 88, 89, 90, 91, 92, 95, 96, 97, 98, 100, 102, 103, 104, 105, 106, 108, 109, 110, 117, 118, 120, 122, 123, 124, 125, 126, 127, 128, 129, 131, 133, 135, 136, 139, 140, 142, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 161, 163, 167, 169, 170, 172, 175, 176, 177, 178, 180, 182, 183, 184, 185, 187, 188, 189, 190, 192, 195, 196, 198, 199, 201, 202, 203, 204, 206, 208, 209, 210, 211, 213, 214, 215, 219, 221, 222, 223, 225, 226, 227, 230, 232, 233, 234, 237, 238, 239, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 255, 256, 257, 258, 259, 262, 263, 266, 267, 268, 273, 274, 276, 277, 279, 280, 281, 282, 283, 285, 286, 287, 288, 290, 291, 292, 293, 294, 295, 298, 299, 300, 301, 303, 304, 305, 306, 308, 309, 310, 316, 317, 319, 320, 322, 323, 324, 326, 327, 328, 329, 331, 332, 333, 335, 336, 337, 342, 344, 345, 346, 348, 349, 352, 353, 355, 356, 358, 359, 361, 363, 364, 365, 367, 368, 370, 371, 372, 375, 376, 377, 378, 380, 381, 386, 388, 389, 390, 391, 392, 393, 394, 395, 396, 399], 275: [0, 1, 2, 3, 4, 8, 10, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 41, 42, 43, 44, 45, 46, 50, 53, 54, 55, 56, 58, 61, 62, 64, 65, 67, 68, 71, 72, 73, 74, 75, 76, 77, 78, 81, 83, 85, 86, 87, 88, 89, 90, 91, 92, 95, 96, 97, 98, 100, 102, 103, 104, 105, 106, 108, 109, 110, 114, 117, 118, 120, 122, 123, 124, 125, 126, 127, 128, 129, 131, 133, 135, 136, 139, 140, 142, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 161, 163, 167, 169, 170, 172, 175, 176, 177, 178, 179, 180, 182, 183, 184, 185, 187, 188, 189, 190, 192, 195, 196, 198, 199, 201, 202, 203, 204, 206, 208, 209, 210, 211, 213, 214, 215, 219, 221, 222, 223, 225, 226, 227, 230, 232, 233, 234, 237, 238, 239, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 255, 256, 257, 258, 259, 262, 263, 266, 267, 268, 273, 274, 276, 277, 278, 279, 280, 281, 282, 283, 285, 286, 287, 288, 290, 291, 292, 293, 294, 295, 298, 299, 300, 301, 303, 304, 305, 306, 308, 309, 310, 316, 317, 318, 319, 320, 322, 323, 324, 326, 327, 328, 329, 331, 332, 333, 335, 336, 337, 342, 344, 345, 346, 348, 349, 352, 353, 355, 356, 358, 359, 361, 363, 364, 365, 366, 367, 368, 370, 371, 372, 375, 376, 377, 378, 380, 381, 386, 388, 389, 390, 391, 392, 393, 394, 395, 396, 399], 277: [0, 1, 2, 3, 4, 8, 10, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 41, 42, 43, 44, 45, 46, 50, 53, 54, 55, 56, 58, 61, 62, 64, 65, 67, 68, 71, 72, 73, 74, 75, 76, 77, 78, 81, 83, 85, 86, 87, 88, 89, 90, 91, 92, 95, 96, 97, 98, 100, 102, 103, 104, 105, 106, 108, 109, 110, 114, 117, 118, 120, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 133, 135, 136, 139, 140, 142, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 161, 163, 167, 169, 170, 172, 175, 176, 177, 178, 179, 180, 182, 183, 184, 185, 187, 188, 189, 190, 192, 195, 196, 198, 199, 201, 202, 203, 204, 206, 208, 209, 210, 211, 213, 214, 215, 219, 221, 222, 223, 225, 226, 227, 230, 232, 233, 234, 237, 238, 239, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 255, 256, 257, 258, 259, 262, 263, 266, 267, 268, 273, 274, 276, 277, 278, 279, 280, 281, 282, 283, 285, 286, 287, 288, 290, 291, 292, 293, 294, 295, 298, 299, 300, 301, 303, 304, 305, 306, 308, 309, 310, 316, 317, 318, 319, 320, 322, 323, 324, 326, 327, 328, 329, 331, 332, 333, 335, 336, 337, 338, 342, 344, 345, 346, 348, 349, 352, 353, 355, 356, 358, 359, 361, 363, 364, 365, 366, 367, 368, 370, 371, 372, 375, 376, 377, 378, 380, 381, 386, 388, 389, 390, 391, 392, 393, 394, 395, 396, 399], 286: [0, 1, 2, 3, 4, 8, 10, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 41, 42, 43, 44, 45, 46, 47, 50, 53, 54, 55, 56, 58, 61, 62, 64, 65, 67, 68, 71, 72, 73, 74, 75, 76, 77, 78, 81, 83, 85, 86, 87, 88, 89, 90, 91, 92, 94, 95, 96, 97, 98, 100, 102, 103, 104, 105, 106, 108, 109, 110, 114, 117, 118, 120, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 133, 134, 135, 136, 139, 140, 142, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 161, 163, 167, 169, 170, 172, 175, 176, 177, 178, 179, 180, 182, 183, 184, 185, 187, 188, 189, 190, 191, 192, 195, 196, 198, 199, 201, 202, 203, 204, 206, 208, 209, 210, 211, 213, 214, 215, 219, 221, 222, 223, 225, 226, 227, 230, 232, 233, 234, 235, 237, 238, 239, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 255, 256, 257, 258, 259, 262, 263, 266, 267, 268, 272, 273, 274, 276, 277, 278, 279, 280, 281, 282, 283, 285, 286, 287, 288, 290, 291, 292, 293, 294, 295, 298, 299, 300, 301, 303, 304, 305, 306, 307, 308, 309, 310, 311, 316, 317, 318, 319, 320, 322, 323, 324, 326, 327, 328, 329, 331, 332, 333, 335, 336, 337, 338, 342, 344, 345, 346, 348, 349, 352, 353, 355, 356, 358, 359, 361, 363, 364, 365, 366, 367, 368, 370, 371, 372, 375, 376, 377, 378, 380, 381, 386, 388, 389, 390, 391, 392, 393, 394, 395, 396, 398, 399], 295: [0, 1, 2, 3, 4, 8, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 41, 42, 43, 44, 45, 46, 47, 50, 53, 54, 55, 56, 58, 61, 62, 64, 65, 67, 68, 71, 72, 73, 74, 75, 76, 77, 78, 81, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 94, 95, 96, 97, 98, 100, 101, 102, 103, 104, 105, 106, 108, 109, 110, 114, 117, 118, 120, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 133, 134, 135, 136, 139, 140, 142, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 161, 163, 167, 169, 170, 172, 174, 175, 176, 177, 178, 179, 180, 182, 183, 184, 185, 187, 188, 189, 190, 191, 192, 195, 196, 198, 199, 201, 202, 203, 204, 206, 208, 209, 210, 211, 213, 214, 215, 219, 221, 222, 223, 225, 226, 227, 230, 231, 232, 233, 234, 235, 237, 238, 239, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 255, 256, 257, 258, 259, 262, 263, 266, 267, 268, 272, 273, 274, 276, 277, 278, 279, 280, 281, 282, 283, 285, 286, 287, 288, 290, 291, 292, 293, 294, 295, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 316, 317, 318, 319, 320, 322, 323, 324, 326, 327, 328, 329, 331, 332, 333, 335, 336, 337, 338, 342, 344, 345, 346, 348, 349, 352, 353, 355, 356, 358, 359, 361, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 375, 376, 377, 378, 380, 381, 386, 388, 389, 390, 391, 392, 393, 394, 395, 396, 398, 399], 300: [0, 1, 2, 3, 4, 8, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 41, 42, 43, 44, 45, 46, 47, 50, 53, 54, 55, 56, 58, 61, 62, 64, 65, 67, 68, 71, 72, 73, 74, 75, 76, 77, 78, 81, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 94, 95, 96, 97, 98, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 114, 117, 118, 120, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 133, 134, 135, 136, 139, 140, 142, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 161, 163, 167, 169, 170, 172, 174, 175, 176, 177, 178, 179, 180, 182, 183, 184, 185, 187, 188, 189, 190, 191, 192, 195, 196, 198, 199, 201, 202, 203, 204, 205, 206, 208, 209, 210, 211, 213, 214, 215, 219, 221, 222, 223, 225, 226, 227, 230, 231, 232, 233, 234, 235, 237, 238, 239, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 255, 256, 257, 258, 259, 262, 263, 266, 267, 268, 272, 273, 274, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 290, 291, 292, 293, 294, 295, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 316, 317, 318, 319, 320, 322, 323, 324, 326, 327, 328, 329, 331, 332, 333, 335, 336, 337, 338, 339, 342, 344, 345, 346, 348, 349, 352, 353, 355, 356, 358, 359, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 375, 376, 377, 378, 380, 381, 386, 388, 389, 390, 391, 392, 393, 394, 395, 396, 398, 399], 309: [0, 1, 2, 3, 4, 7, 8, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 50, 52, 53, 54, 55, 56, 58, 61, 62, 64, 65, 67, 68, 71, 72, 73, 74, 75, 76, 77, 78, 81, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 94, 95, 96, 97, 98, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 114, 117, 118, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 133, 134, 135, 136, 139, 140, 142, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 160, 161, 163, 167, 169, 170, 172, 174, 175, 176, 177, 178, 179, 180, 182, 183, 184, 185, 187, 188, 189, 190, 191, 192, 195, 196, 198, 199, 201, 202, 203, 204, 205, 206, 208, 209, 210, 211, 213, 214, 215, 219, 221, 222, 223, 225, 226, 227, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 255, 256, 257, 258, 259, 262, 263, 266, 267, 268, 272, 273, 274, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 313, 316, 317, 318, 319, 320, 322, 323, 324, 326, 327, 328, 329, 331, 332, 333, 335, 336, 337, 338, 339, 342, 344, 345, 346, 348, 349, 352, 353, 355, 356, 357, 358, 359, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 375, 376, 377, 378, 380, 381, 386, 388, 389, 390, 391, 392, 393, 394, 395, 396, 398, 399], 318: [0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 50, 52, 53, 54, 55, 56, 58, 61, 62, 64, 65, 67, 68, 71, 72, 73, 74, 75, 76, 77, 78, 81, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 114, 117, 118, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 133, 134, 135, 136, 137, 139, 140, 142, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 160, 161, 163, 165, 167, 169, 170, 172, 174, 175, 176, 177, 178, 179, 180, 182, 183, 184, 185, 187, 188, 189, 190, 191, 192, 195, 196, 198, 199, 201, 202, 203, 204, 205, 206, 208, 209, 210, 211, 213, 214, 215, 218, 219, 221, 222, 223, 225, 226, 227, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 262, 263, 266, 267, 268, 272, 273, 274, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 316, 317, 318, 319, 320, 322, 323, 324, 326, 327, 328, 329, 331, 332, 333, 335, 336, 337, 338, 339, 342, 344, 345, 346, 348, 349, 352, 353, 354, 355, 356, 357, 358, 359, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 375, 376, 377, 378, 380, 381, 382, 386, 388, 389, 390, 391, 392, 393, 394, 395, 396, 398, 399], 325: [0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 50, 52, 53, 54, 55, 56, 58, 61, 62, 64, 65, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 78, 81, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 114, 117, 118, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 133, 134, 135, 136, 137, 139, 140, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 160, 161, 163, 165, 167, 168, 169, 170, 172, 174, 175, 176, 177, 178, 179, 180, 182, 183, 184, 185, 187, 188, 189, 190, 191, 192, 195, 196, 198, 199, 201, 202, 203, 204, 205, 206, 208, 209, 210, 211, 213, 214, 215, 216, 218, 219, 221, 222, 223, 225, 226, 227, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 261, 262, 263, 266, 267, 268, 272, 273, 274, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 316, 317, 318, 319, 320, 322, 323, 324, 325, 326, 327, 328, 329, 331, 332, 333, 335, 336, 337, 338, 339, 342, 344, 345, 346, 348, 349, 351, 352, 353, 354, 355, 356, 357, 358, 359, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 375, 376, 377, 378, 380, 381, 382, 386, 388, 389, 390, 391, 392, 393, 394, 395, 396, 398, 399], 327: [0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 50, 52, 53, 54, 55, 56, 58, 61, 62, 64, 65, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 78, 81, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 114, 117, 118, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 133, 134, 135, 136, 137, 139, 140, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 160, 161, 163, 165, 167, 168, 169, 170, 172, 174, 175, 176, 177, 178, 179, 180, 182, 183, 184, 185, 187, 188, 189, 190, 191, 192, 195, 196, 198, 199, 200, 201, 202, 203, 204, 205, 206, 208, 209, 210, 211, 213, 214, 215, 216, 218, 219, 221, 222, 223, 225, 226, 227, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 261, 262, 263, 266, 267, 268, 272, 273, 274, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 316, 317, 318, 319, 320, 322, 323, 324, 325, 326, 327, 328, 329, 331, 332, 333, 335, 336, 337, 338, 339, 342, 344, 345, 346, 347, 348, 349, 351, 352, 353, 354, 355, 356, 357, 358, 359, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 375, 376, 377, 378, 380, 381, 382, 386, 388, 389, 390, 391, 392, 393, 394, 395, 396, 398, 399], 336: [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 50, 52, 53, 54, 55, 56, 58, 61, 62, 64, 65, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 78, 80, 81, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 113, 114, 117, 118, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 133, 134, 135, 136, 137, 139, 140, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 160, 161, 163, 165, 167, 168, 169, 170, 172, 174, 175, 176, 177, 178, 179, 180, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 195, 196, 198, 199, 200, 201, 202, 203, 204, 205, 206, 208, 209, 210, 211, 213, 214, 215, 216, 218, 219, 220, 221, 222, 223, 225, 226, 227, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 261, 262, 263, 266, 267, 268, 272, 273, 274, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 316, 317, 318, 319, 320, 322, 323, 324, 325, 326, 327, 328, 329, 331, 332, 333, 335, 336, 337, 338, 339, 342, 343, 344, 345, 346, 347, 348, 349, 351, 352, 353, 354, 355, 356, 357, 358, 359, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 375, 376, 377, 378, 380, 381, 382, 383, 386, 388, 389, 390, 391, 392, 393, 394, 395, 396, 398, 399], 345: [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 50, 52, 53, 54, 55, 56, 58, 61, 62, 64, 65, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 113, 114, 117, 118, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 160, 161, 163, 165, 167, 168, 169, 170, 172, 174, 175, 176, 177, 178, 179, 180, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 195, 196, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 213, 214, 215, 216, 218, 219, 220, 221, 222, 223, 225, 226, 227, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 261, 262, 263, 266, 267, 268, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 316, 317, 318, 319, 320, 322, 323, 324, 325, 326, 327, 328, 329, 331, 332, 333, 335, 336, 337, 338, 339, 340, 342, 343, 344, 345, 346, 347, 348, 349, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 375, 376, 377, 378, 380, 381, 382, 383, 386, 388, 389, 390, 391, 392, 393, 394, 395, 396, 398, 399]}
coarseness = 0.8  # Parameter for OTIF. Since OTIF uses a greedy strategy, a larger coarseness reduces the search space, similar to a step size. OTIF's parameters are monotonic, so saving more resources generally leads to greater efficiency, but this parameter indicates how much efficiency improvement can be gained by saving 80% of resources, with a corresponding reduction in accuracy.
min_cluster_num, max_cluster_num = 2, 5  # Minimum and maximum number of clusters. In general, the more clusters there are, the more video stream clusters can run in parallel, but it also consumes more computational resources, limiting many possible solutions.
use_experience = True  # Whether Hippo uses experience. Experience refers to the configuration set results sampled from previous training datasets.
use_higher_context = False  # Whether to use higher-level context, such as video stream context. This context refers to features of the video stream, such as frame rate, resolution, etc.
configpath = "./cache/base.yaml"  # Path to the configuration file. This configuration file is for Hippo and is essentially a default configuration with minimal impact.
latency_constrain = "LinearSum"  # Latency constraint. This refers to the maximum latency, which is typically linear, meaning that the greater the latency, the more resources consumed.
save_dir = "./sklearn_models"  # Path to save the parameters for the UniTune algorithm, specifically for saving sklearn models.
log_name = "test_reinforce_tensorboard_all_context"  # Name of the log, this refers to the tensorboard log.
log_dir = f"./{log_name}/"  # Path to the log.
writer = SummaryWriter(log_dir=log_dir)  # Tensorboard writer.
objects, metrics = ["car", "bus", "truck"], ["sel", "agg", "cq3"]  # Objects and metrics. Objects refer to the entities in the video stream, and metrics refer to the measurement indices, such as sel (selection query: whether there is a target), agg (aggregation query: how many targets), cq3 (event query: whether a left turn occurred).
alldir = utils.DataPaths('/home/lzp/otif-dataset/dataset',
                         data_name, data_type, method_name)  # Path to the dataset.
train_alldir = utils.DataPaths('/home/lzp/otif-dataset/dataset',
                               data_name, "train", method_name)  # Path to the training dataset.

utils.makedirs([alldir.visdir,
                alldir.cachedir,
                alldir.graphdir,
                alldir.recorddir,
                alldir.framedir,
                alldir.storedir,
                alldir.cacheconfigdir,
                alldir.cacheresultdir])

# Read video stream information from the dataset, including various details for training videos.
train_cameras, train_videopaths, train_trackdatas = build_cameras_with_info_list(train_alldir, train_video_list,
                                                                                 objects, metrics,
                                                                                 configpath, device_id,
                                                                                 method_name, data_name, "train",
                                                                                 scene_name, temp)
# Read video stream information from the dataset, including various details for test videos.
test_cameras, test_videopaths, test_trackdatas = build_cameras_with_info(alldir, channel,
                                                                         objects, metrics,
                                                                         configpath, device_id,
                                                                         method_name, data_name, data_type,
                                                                         scene_name, temp)

model_ckpt = "/mnt/data_ssd1/lzp/MCG_NJUvideomae_base"
image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)  # These parameters are for another context extraction method. This method extracts features from video streams by processing several frames. However, it is not currently used, so it can be ignored.
mean = image_processor.image_mean
std = image_processor.image_std

if "shortest_edge" in image_processor.size:
    height = width = image_processor.size["shortest_edge"]
else:
    height = image_processor.size["height"]
    width = image_processor.size["width"]
resize_to = (height, width)

train_transform = T.Compose([T.ToPILImage(),
                             T.Resize(resize_to),
                             T.ToTensor(),
                             T.Normalize(mean,
                                         std)])

train_dataset, test_dataset = VideoDataset2(
    train_videopaths, train_trackdatas, framenumber,
    framegap, objects=objects, transform=train_transform,
    flag="train", context_path_name=context_path_name), VideoDataset2(
    test_videopaths, test_trackdatas, framenumber,
    framegap, objects=objects, transform=train_transform,
    flag=data_type, context_path_name=context_path_name)

train_envconfigpath = "./cache/train_base.yaml"
train_envcachepath = "./cache/train_env.yaml"
train_env_camera_config = generate_random_config(
    train_envconfigpath, train_envcachepath, device_id, data_name, "train", objects, "parallel_train", scene_name)
train_env_camera = build_a_camera_with_config(
    alldir, train_env_camera_config, objects, metrics)

envconfigpath = "./cache/base.yaml"
envcachepath = "./cache/env.yaml"
env_camera_config = generate_random_config(
    envconfigpath, envcachepath, device_id, data_name, data_type, objects, method_name, scene_name)
env_camera = build_a_camera_with_config(
    alldir, env_camera_config, objects, metrics)

observation_space = gym.spaces.Box(
    low=-np.inf, high=np.inf, shape=(max_pareto_set_size, len(SEARCH_SPACE) + 2), dtype=np.float32)
action_space = gym.spaces.MultiDiscrete(SEARCH_SPACE)  # This refers to the search space, which defines the configuration space, mainly for video parsing.
policy = DUPIN(hidden_size=hidden_size, video_context_size=video_context_size,  # This is the DUPIN policy network. DUPIN is a reinforcement learning method where an RNN uses video stream context information and iteratively recommends configurations that satisfy Pareto constraints.
               observation_space=observation_space,
               action_space=action_space, use_rnn=use_rnn)
policy_weight_path = ""
if policy_weight_path:
    policy_weights = torch.load(policy_weight_path)
    policy.load_state_dict(policy_weights)

policy_unitune = UNITUNE(hidden_size=64, observation_space=observation_space,
                         action_space=action_space)  # This is the UniTune algorithm's agent. The agent combines reinforcement learning and Bayesian optimization for configuration search.
policy_unitune_weight_path = "./unitune_tensorboard/eval_result/last_model_1024"
policy_unitune_weights = torch.load(policy_unitune_weight_path)
policy_unitune.load_state_dict(policy_unitune_weights)
bo_model_unitune = UnituneGPModel(action_size=len(SEARCH_SPACE),
                                  context_size=len(SEARCH_SPACE)+3)
bo_model_unitune.load_gpmodel(
    "./unitune_tensorboard/eval_result/gpmodel_1024")

policy.eval()
env = HippoEnv(train_dataset, None, train_env_camera,
               max_iter, max_pareto_set_size,
               debug=debug, use_init_solution=use_init_solution, use_cache=use_cache)

test_env = HippoEnv(test_dataset, None, env_camera,
                    max_iter, max_pareto_set_size,
                    debug=debug, use_init_solution=use_init_solution, use_cache=use_cache)

unitune_env = UnituneEnv(test_dataset, None, env_camera,
                         max_pareto_set_size, effiency_targets)

if not os.path.exists("./clusters"):
    os.makedirs("./clusters")
if not os.path.exists("./paretos/figures"):
    os.makedirs("./paretos/figures")
if not os.path.exists("./paretos/values"):
    os.makedirs("./paretos/values")

# # Efficiency estimation model
gpu_resource_info_path = "./GPUInfo.json"  # Path to GPU resource information. This is used to read the GPU resource consumption for different configurations.
with open(gpu_resource_info_path, "r") as f:
    gpu_resource_info_raw = json.load(f)
    gpu_resource_info = {}
    indexs = [SEARCH_SPACE_NAMES.index(space_name)
              for space_name in EFFE_GPU_SPACE_NAMES]
    for config_action_str in gpu_resource_info_raw:
        gpu_resource = gpu_resource_info_raw[config_action_str]
        config_action_list = config_action_str.split("_")
        config_action_str = "_".join(
            [config_action_list[index] for index in indexs])
        gpu_resource_info[config_action_str] = gpu_resource

all_repre_train_video_ids = []
start = time.time()
for video_num in test_video_num_list:
    if video_num in define_video_list:
        video_ids = define_video_list[video_num]

    for function_name, recommend_function in zip(method_list,
                                                 func_list):
        method_config_path = f"./cache/{function_name}.yaml"
        method_cache_path = f"./cache/{function_name}_cache.yaml"

        method_camera_config = generate_random_config(
            method_config_path, method_cache_path, device_id, data_name, data_type, objects, method_name, scene_name)
        method_camera = build_a_camera_with_config(
            alldir, method_camera_config, objects, metrics)

        inputs = {
            "seed": seed, "video_num": video_num, "video_ids": video_ids,
            "policy": policy, "env": env, "test_env": test_env,
            "train_scene_dict": train_scene_dict, "test_scene_dict": test_scene_dict,
            "cluster_nums": [min_cluster_num, max_cluster_num],
            "gpu_resource_info": gpu_resource_info, "indexs": indexs,
            "configpath": method_config_path, "deviceid": device_id,
            "data_name": data_name, "data_type": data_type, "scene_name": scene_name,
            "objects": objects, "temp": temp, "max_pool_lenth": max_pool_lenth,
            "latency_bound": latency_bound, "memory_bound": memory_bound,
            "max_pareto_set_size": max_pareto_set_size, "object_nan": object_nan,
            "train_fit": train_fit, "use_experience": use_experience,
            "policy_unitune": policy_unitune, "bo_model_unitune": bo_model_unitune,
            "random_camera": method_camera, "effiency_targets": effiency_targets,
            "coarseness": coarseness,
            "use_cache": use_cache, "use_init_solution": use_init_solution,
            "unitune_test_env": unitune_env}

        # recommend_function(**inputs)
        find_solution, run_cmds, \
            ingestion_result_paths, \
            ClusterNum, optimal_config_indices = recommend_function(
                **inputs)  # Obtain configuration run commands for the video stream.

        if not os.path.exists("./paretos/results"):
            os.makedirs("./paretos/results")
        with open(f"./paretos/results/{function_name}_{video_num}_optimal_config_indices.json", "w") as f:
            json.dump(optimal_config_indices, f)

        if find_solution:  # If a solution was found
            print("run_cmds: ", run_cmds)
            run_parral_commands(run_cmds)  # Run the command
            for ingestion_result_path in ingestion_result_paths:
                ingestion_result = utils.read_json(ingestion_result_path)  # Read the result
            final_result = utils.combine_parallel_result(
                ingestion_result_paths, objects, ClusterNum)  # Combine the results of multiple clusters
            pretty_final_result = json.dumps(
                final_result, indent=4, sort_keys=True)  # Save the result in JSON format

            save_result_path = f"./ingestion_results/{video_num}"  # Folder to save the result
            if not os.path.exists(save_result_path):
                os.makedirs(save_result_path)
            with open(f"{save_result_path}/{function_name}.json", "w") as f:
                f.write(pretty_final_result)
print("time: ", time.time() - start)
print("all_repre_train_video_ids: ", all_repre_train_video_ids)
