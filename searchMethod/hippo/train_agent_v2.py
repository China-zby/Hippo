# from stable_baselines3 import REINFORCE
import torch
import seaborn as sns
import torch.nn as nn
import gymnasium as gym
from torch import Tensor
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import os
import math
import utils
import random
import pickle
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from config_ import generate_random_config
from typing import Dict, Optional, Tuple, Union
from dataloader import VideoDataset2, build_transform
from stable_baselines3.common.policies import ActorCriticPolicy
from utils import SEARCH_SPACE, SEARCH_SPACE_NAMES, GOLDEN_CONFIG_VECTOR, mean
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from cameraclass import build_cameras_with_info_list


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
        return torch.stack(entropies)  # 返回所有分布熵的总和

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
            print("\033[32mUsing RNN ... ...\033[0m")
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
            print("Using Graph ... ...")
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
            print("\033[35mUsing Context observation ... ...\033[0m")
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
                 max_iter, max_pareto_set_size, effiency_targets, debug=False, use_init_solution=True):
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
        self.effiency_targets = effiency_targets
        self.debug = debug
        self.batch = 0
        if self.debug:
            print("\033[91mDebug mode is on ... ...\033[0m")

        self.use_init_solution = use_init_solution

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

    def step(self, action, env=None):
        if len(action.shape) == 2:
            action = action[0]
        self.video_camera_config.id = self.current_video
        if self.debug:
            accuracy, latency = random.random(), random.random()
        else:
            self.video_camera_config.updateConfig(utils.generate_config(action,
                                                                        self.video_camera_config.loadConfig()))
            _, ingvalue, _, _, _, \
                cmetric, _ = self.video_camera_config.ingestion
            accuracy, latency = ingvalue, cmetric[-1]

        # not self.pareto_set or accuracy < min(pareto[0] for pareto in self.pareto_set)
        direction = True

        if direction:
            self.pareto_set.append((accuracy, latency, action))
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
            # if random.random() < 0.25:
            #     train_video_ids = [84, 200, 206, 237, 267, 268, 304, 358, 364, 464, 471, 479, 489, 501, 543, 560, 601, 716, 769]
            #     self.current_video = random.choice(train_video_ids)
            # else:
            if len(self.video_ids_iter) == 0:
                self.video_ids_iter = self.video_ids.copy()
            random.shuffle(self.video_ids_iter)
            self.current_video = self.video_ids_iter.pop()
        else:
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
                    _, ingvalue, _, _, _, \
                        cmetric, _ = self.video_camera_config.ingestion
                    accuracy, latency = ingvalue, cmetric[-1]
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
            path_name for path_name in ingestion_result_path_name_list if f"{camera_type}_{video_id}" in path_name]
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
        pareto_set = self.identify_pareto(result_sets)
        pareto_set = sorted(pareto_set, key=lambda x: x[0])
        return pareto_set

    def set_batch(self, batch):
        self.batch = batch

    @staticmethod
    def evenly_sample_indices(data, num_samples=10):
        # 确定每个采样点所在的分位数区间
        quantiles = np.linspace(0, 1, num_samples)  # 避免0和1，这样不会取到最小和最大值之外的数
        # 计算每个分位数对应的数据值
        quantile_values = np.quantile(data, quantiles)

        # 对原数据进行排序，获取排序后的索引
        sorted_indices = np.argsort(data)
        sorted_data = data[sorted_indices]

        # 为每个分位数找到最接近的数据点索引
        sample_indices = []
        for value in quantile_values:
            # 找到最接近分位数值的数据点索引
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


# REINFORCE训练过程


def reinforce(env, test_env,
              policy, optimizer, deviceid,
              batch_size, update_step,
              writer, save_path,
              episodes=1000, baseline_window_size=10, strong_update_start_step=1000,
              gamma=0.99, lr=None, entropy_coef=0.005, l1_lambda=0.0001, pareto_coef=5.0,
              lr_decay=False, video_list=[0], writer_window=10,
              save_train=False, strong_update_step=10):
    bast_metric, step_i = 0, 0
    policy.to(deviceid)

    reward_window = deque(maxlen=baseline_window_size)
    baseline_window = {vid: [deque(maxlen=baseline_window_size) for _ in range(
        env.max_iter)] for vid in env.video_ids}

    custom_columns = [
        "[progress.description]{task.description}",
        BarColumn(bar_width=30),  # 使用默认宽度的进度条
        "[progress.percentage]{task.percentage:>3.0f}%",  # 显示百分比
        # 显示当前步数/总步数
        "(", TextColumn(
            "[progress.completed]{task.completed}"), "/", TextColumn("{task.total}"), ")",
        TimeRemainingColumn(),  # 显示预计剩余时间
    ]
    with Progress(*custom_columns) as progress:
        task = progress.add_task("[cyan]Processing...", total=episodes)
        for batch in range(0, episodes, batch_size):
            batch_idxs, batch_vids, batch_rewards, \
                batch_saved_entropys, batch_saved_log_probs = [], [], [], [], []

            action_values = []

            if batch % 500 == 0 and batch >= strong_update_start_step:
                strong_update_step += 1

            policy.train()
            env.set_batch(batch)

            if batch >= strong_update_start_step:
                for _ in range(strong_update_step):
                    pareto_classify_losses = []
                    for train_video_id in env.video_ids:
                        observation_, _ = env.reset(train_video_id)
                        pareto_set_gt = env.collocate_pareto_set(
                            env.current_video)
                        if len(pareto_set_gt) > 10:
                            pareto_set_gt_idxs = env.evenly_sample_indices(
                                np.array([res[0] for res in pareto_set_gt]), num_samples=10)
                            pareto_set_gt = [pareto_set_gt[idx]
                                             for idx in pareto_set_gt_idxs]
                            pareto_set_gt = sorted(
                                pareto_set_gt, key=lambda x: x[0], reverse=True)
                        pareto_set_gt_queue = pareto_set_gt.copy()

                        policy.reset_hidden_state()

                        done, truncated = False, False
                        while not done and not truncated:
                            pareto_observation = torch.from_numpy(
                                observation_[0]).float().unsqueeze(0).to(deviceid)
                            context_observation = torch.tensor(
                                observation_[1], dtype=torch.float).unsqueeze(0).to(deviceid)
                            action, log_probs, \
                                entropy, action_distribution = policy(
                                    pareto_observation,
                                    context_observation,
                                    return_logits=True)
                            if len(pareto_set_gt_queue) == 0:
                                done = True
                                continue
                            pareto_acc, pareto_lat, pareto_action = pareto_set_gt_queue.pop(
                                0)
                            observation_, done, truncated = env.step_without_ingestion(pareto_acc, pareto_lat,
                                                                                       torch.tensor([pareto_action]))
                            pareto_classify_losses.append(
                                action_distribution.cross_entropy(torch.tensor([pareto_action])))

                    pareto_classify_loss = torch.mean(
                        torch.stack(pareto_classify_losses))
                    optimizer.zero_grad()
                    pareto_classify_loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.1)
                    optimizer.step()

            for episode in range(batch_size):
                episode_log_probs, episode_entropys, episode_rewards = [], [], []

                observation_, _ = env.reset()
                pareto_set_gt = env.collocate_pareto_set(env.current_video)
                done, truncated = False, False

                policy.reset_hidden_state()

                plt.figure(figsize=(12, 6))

                plt.subplot(1, 2, 2)
                plt.scatter([res[0] for res in pareto_set_gt],
                            [res[1] for res in pareto_set_gt], c='r', marker='o')
                while not done and not truncated:
                    observation = torch.from_numpy(
                        observation_[0]).float().unsqueeze(0).to(deviceid)
                    context_observation = torch.tensor(
                        observation_[1], dtype=torch.float).unsqueeze(0).to(deviceid)
                    action, log_probs, entropy, action_distribution = policy(
                        observation, context_observation, return_logits=True)
                    observation_, reward, done, truncated, info = env.step(
                        action.cpu())

                    if policy.use_graph:
                        action_values.append([action.cpu().numpy()[0].tolist(),
                                              (info["value"][0] + info["value"][1])/2.0])

                    # plot the pareto set and the new action
                    new_acc, new_lat = info["value"][:2]
                    plt.subplot(1, 2, 1)
                    plt.scatter([new_acc], [new_lat], c='b', marker='x')
                    iter_num = env.iter_num
                    plt.text(new_acc, new_lat, f"{iter_num}")
                    plt.text(0.7, 0.9 - (iter_num - 1) * 0.05,
                             f"{iter_num}: {reward:.6f}")
                    plt.xlim(0, 1)
                    plt.ylim(0, 1)
                    plt.xlabel("Accuracy")
                    plt.ylabel("Latency")

                    episode_log_probs.append(log_probs)
                    episode_entropys.append(entropy)
                    episode_rewards.append(reward)

                    if len(pareto_set_gt) == 0:
                        continue

                    pred_acc, pred_lat = info["value"][0], info["value"][1]
                    dist_with_pareto_set = [math.sqrt(
                        (pred_acc - res[0])**2 + (pred_lat - res[1])**2) for res in pareto_set_gt]
                    mini_action_idx = np.argmin(dist_with_pareto_set)

                    plt.subplot(1, 2, 2)
                    plt.scatter([pred_acc], [pred_lat], c='b', marker='x')
                    plt.plot([pred_acc, pareto_set_gt[mini_action_idx][0]], [
                        pred_lat, pareto_set_gt[mini_action_idx][1]], 'r--')
                    plt.xlim(0, 1)
                    plt.ylim(0, 1)
                    plt.xlabel("Accuracy")
                    plt.ylabel("Latency")
                    plt.savefig(os.path.join("./test_new_idea",
                                f"step_{batch}_{episode}.png"))

                plt.close()
                plt.cla()
                plt.clf()

                if save_train:
                    save_train_path = os.path.join(
                        os.path.dirname(save_path), "train_result")
                    save_train_result_path = os.path.join(
                        save_train_path, f"step_{batch}")
                    if not os.path.exists(save_train_result_path):
                        os.makedirs(save_train_result_path)
                    plt.scatter([res[0] for res in env.pareto_set],
                                [res[1] for res in env.pareto_set], c='r',
                                marker='x', label=f'pareto set {env.current_video}')
                    plt.savefig(os.path.join(save_train_result_path,
                                f"pareto_set_{env.current_video}.png"))
                    plt.xlim(0, 1)
                    plt.ylim(0, 1)
                    plt.close()
                    plt.cla()
                    plt.clf()
                    with open(os.path.join(save_train_result_path, f"pareto_set_{env.current_video}.txt"), "w") as f:
                        mean_acc, mean_lat = [], []
                        for res in env.pareto_set:
                            f.write(f"{res[0]} {res[1]} {res[2]}\n")
                            mean_acc.append(res[0])
                            mean_lat.append(res[1])
                        mean_acc = sum(mean_acc) / len(mean_acc)
                        mean_lat = sum(mean_lat) / len(mean_lat)
                        f.write(f"{mean_acc} {mean_lat} {00000000}\n")
                        writer.add_scalar(
                            f'train/mean_acc_{env.current_video}', mean_acc, batch)
                        writer.add_scalar(
                            f'train/mean_lat_{env.current_video}', mean_lat, batch)
                        writer.add_scalar(
                            f'train/pset_num_{env.current_video}', len(env.pareto_set), batch)

                for reward in episode_rewards:
                    step_i += 1
                    writer.add_scalar("Reward/step_reward",
                                      reward, step_i)

                reward_window.append(sum(episode_rewards))

                # Calculate return episode rewards
                R = 0
                episode_returns, episode_idxs, episode_vids = [], [], []
                for i, r in enumerate(episode_rewards[::-1]):
                    R = r + gamma * R
                    interactive_round = len(episode_rewards) - i - 1
                    baseline_window[env.current_video][interactive_round].append(
                        R)
                    episode_returns.insert(0, R)
                    episode_idxs.insert(0, interactive_round)
                    episode_vids.insert(0, env.current_video)

                batch_vids.extend(episode_vids)
                batch_idxs.extend(episode_idxs)
                batch_rewards.extend(episode_returns)
                batch_saved_log_probs.extend(episode_log_probs)
                batch_saved_entropys.extend(episode_entropys)

                progress.update(task, advance=1)
                writer.add_scalar("Reward/episode_sum_reward",
                                  np.sum(episode_rewards), batch + episode)
                if batch + episode > writer_window:
                    writer.add_scalar("Reward/window_mean_reward",
                                      np.mean(reward_window), batch + episode)

            if policy.use_graph:
                policy.update_graph_weights(action_values)

            # Calculate the baseline
            baseline_batch_rewards = [
                float(np.mean(baseline_window[vid][i])) for vid, i in zip(batch_vids, batch_idxs)]

            batch_rewards = [r - b for r,
                             b in zip(batch_rewards, baseline_batch_rewards)]

            batch_saved_log_probs = torch.cat(batch_saved_log_probs)
            batch_saved_entropys = torch.cat(batch_saved_entropys)

            # Policy gradient update
            policy_losses, entropy_losses = [], []
            for entropy, log_prob, R in zip(batch_saved_entropys,
                                            batch_saved_log_probs,
                                            batch_rewards):
                policy_losses.append(-log_prob * R)
                entropy_losses.append(-entropy)
            policy_loss = torch.stack(policy_losses).sum()
            entropy_loss = torch.stack(entropy_losses).sum()

            l1_loss = sum(p.abs().sum() for p in policy.parameters())

            optimizer.zero_grad()
            total_loss = policy_loss + entropy_coef * entropy_loss + \
                l1_lambda * l1_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()

            if lr_decay:
                lr_now = lr * (1 - batch / episodes)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_now

            writer.add_scalar("Loss/policy_loss", policy_loss, batch)

            if False: # batch % update_step == 0:
                policy.eval()
                # policy.update_mean_std()
                save_result_path = os.path.join(save_path, f"step_{batch}")
                if not os.path.exists(save_result_path):
                    os.makedirs(save_result_path)
                total_acc, total_lat = 0, 0
                for video_id in test_env.video_ids:
                    obs_, _ = test_env.reset(video_id, env=env)
                    done, truncated = False, False
                    policy.reset_hidden_state()
                    total_reward = 0
                    while not done and not truncated:
                        obs = obs_[0]
                        context_obs = obs_[1]
                        action, _states = policy.predict(
                            obs, context_obs, deterministic=True)
                        obs_, rewards, done, truncated, _info = test_env.step(
                            action, env=env)
                        total_reward += rewards
                    plt.scatter([res[0] for res in test_env.pareto_set],
                                [res[1] for res in test_env.pareto_set], c='r',
                                marker='x', label=f'pareto set {video_id}')
                    plt.savefig(os.path.join(save_result_path,
                                f"pareto_set_{video_id}.png"))
                    plt.xlim(0, 1)
                    plt.ylim(0, 1)
                    plt.close()
                    plt.cla()
                    plt.clf()
                    with open(os.path.join(save_result_path, f"pareto_set_{video_id}.txt"), "w") as f:
                        mean_acc, mean_lat = [], []
                        for res in test_env.pareto_set:
                            f.write(f"{res[0]} {res[1]} {res[2]}\n")
                            mean_acc.append(res[0])
                            mean_lat.append(res[1])
                        mean_acc = sum(mean_acc) / len(mean_acc)
                        mean_lat = sum(mean_lat) / len(mean_lat)
                        f.write(f"{mean_acc} {mean_lat} {00000000}\n")
                        writer.add_scalar(
                            f'eval/mean_acc_{video_id}', mean_acc, batch)
                        writer.add_scalar(
                            f'eval/mean_lat_{video_id}', mean_lat, batch)
                        writer.add_scalar(
                            f'eval/pset_num_{video_id}', len(test_env.pareto_set), batch)
                        total_acc += mean_acc
                        total_lat += mean_lat
                total_acc = total_acc / len(video_list)
                total_lat = total_lat / len(video_list)

                total_metric = total_acc * 0.5 + total_lat * 0.5 + total_reward + 0.5 * \
                    math.exp(-(max_pareto_set_size-len(test_env.pareto_set)) /
                             max_pareto_set_size)  # plus the pareto number
                if total_metric > bast_metric:
                    bast_metric = total_metric
                    torch.save(policy.state_dict(), os.path.join(
                        save_path, f"best_model_{batch}"))
            torch.save(policy.state_dict(), os.path.join(
                save_path, f"last_model_{batch}"))
            if policy.use_graph:
                pickle.dump(policy.value_graph_weight, open(
                    os.path.join(save_path, f"graph_value_{batch}.pkl"), "wb"))


if __name__ == "__main__":
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    channel = 800
    method_name = "hippo"
    data_name = "hippo"
    scene_name = "hippo"
    configpath = "./cache/base.yaml"
    deviceid = 1
    policy_deviceid = torch.device(f"cuda:{1 - deviceid}")
    lr = 1e-2
    temp = 10.0
    debug = False
    max_iter = 10
    use_rnn = True
    hidden_size = 128
    use_graph = True
    save_train = True
    entropy_coef = 0.01
    iter_generate = True
    lr_decay_flag = True
    strong_update_step = 2
    video_context_size = 16
    max_pareto_set_size = 5
    use_init_solution = False
    baseline_window_size = 10
    framenumber, framegap = 16, 16
    device = torch.device("cuda:0")
    context_path_name = "context_01.csv"  # context_norm.csv
    context_path_dir = "/mnt/data_ssd1/lzp/otif-dataset/dataset/hippo"
    effiency_targets = []
    train_video_list = list(range(channel))
    # [0,   1,   4,   5,   10,  100, 105, 128, 173,
    # 188, 206, 210, 256, 277, 294, 325, 337, 363,
    # 382, 397, 412, 432, 451, 468, 491, 507, 512,
    # 516, 517, 591, 605, 619, 635, 679, 698, 700,
    # 715, 758, 781, 789]
    test_video_list = [] # [3,  25,  50,  79, 104, 127, 160, 184]
    batch_size = 32  # len(train_video_list)
    update_step = batch_size * 8
    log_dir = "./logs/hippo_checkpoint/"
    writer = SummaryWriter(log_dir=log_dir)
    objects, metrics = ["car", "bus", "truck"], ["sel", "agg", "cq3"]
    alldir = utils.DataPaths('/home/lzp/otif-dataset/dataset',
                             data_name, "train", method_name)
    save_path = os.path.join(log_dir, "eval_result")
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    utils.makedirs([alldir.visdir,
                    alldir.cachedir,
                    alldir.graphdir,
                    alldir.recorddir,
                    alldir.framedir,
                    alldir.storedir,
                    alldir.cacheconfigdir,
                    alldir.cacheresultdir])

    train_cameras, train_videopaths, train_trackdatas = build_cameras_with_info_list(alldir, train_video_list,
                                                                                     objects, metrics,
                                                                                     configpath, deviceid,
                                                                                     method_name, data_name, "train",
                                                                                     scene_name, temp)

    # test_cameras, test_videopaths, test_trackdatas = build_cameras_with_info_list(alldir, test_video_list,
    #                                                                               objects, metrics,
    #                                                                               configpath, deviceid,
    #                                                                               method_name, data_name, "test",
    #                                                                               scene_name, temp)

    train_transform = build_transform()

    train_dataset = VideoDataset2(
        train_videopaths, train_trackdatas, framenumber,
        framegap, objects=objects, transform=train_transform,
        flag="train", context_path_name=context_path_name)
    # , VideoDataset2(        , test_dataset
    #     test_videopaths, test_trackdatas, framenumber,
    #     framegap, objects=objects, transform=train_transform,
    #     flag="test", context_path_name=context_path_name)

    observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(max_pareto_set_size, len(SEARCH_SPACE) + 2), dtype=np.float32)
    action_space = gym.spaces.MultiDiscrete(SEARCH_SPACE)
    policy = DUPIN(hidden_size=hidden_size, video_context_size=video_context_size,
                   observation_space=observation_space,
                   action_space=action_space, use_graph=use_graph, use_rnn=use_rnn,
                   iter_generate=iter_generate)

    # "./reinforce_tensorboard_all/eval_result/best_model_8896"
    pretrain_weight = "" # "./reinforce_tensorboard_all_context_graph2/eval_result/best_model_512"
    if os.path.exists(pretrain_weight):
        pretrain_weights = torch.load(pretrain_weight)  # ["state_dict"]
        policy.load_state_dict(pretrain_weights, strict=False)
        policy_keys = set(policy.state_dict().keys())
        pretrain_weights_keys = set(pretrain_weights.keys())
        missing_keys = policy_keys - pretrain_weights_keys
        print("\033[93mParameters not loaded from the weights file:\033[0m")
        for key in missing_keys:
            print("\033[93m" + key + "\033[0m")
        if use_graph:
            policy.value_graph_weight = pickle.load(open(
                "./reinforce_tensorboard_all_context_graph/eval_result/graph_value_1600.pkl", "rb"))

    env = HippoEnv(train_dataset, None, train_cameras[0],
                   max_iter, max_pareto_set_size, effiency_targets, debug=debug,
                   use_init_solution=use_init_solution)
    # test_env = HippoEnv(test_dataset, None, test_cameras[0],
    #                     max_iter, max_pareto_set_size, effiency_targets, debug=debug,
    #                     use_init_solution=use_init_solution)

    optimizer = optim.Adam(policy.parameters(), lr=lr)

    reinforce(env, env, policy, optimizer, policy_deviceid,
              batch_size, update_step,
              writer, save_path, lr=lr,
              episodes=100000, baseline_window_size=baseline_window_size,
              gamma=0.99, lr_decay=lr_decay_flag,
              video_list=test_video_list, entropy_coef=entropy_coef,
              save_train=save_train, strong_update_step=strong_update_step)
