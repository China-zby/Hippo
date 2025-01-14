# from stable_baselines3 import REINFORCE
import gymnasium as gym
import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

import os
import GPy
import math
import utils
import random
import numpy as np
from config_ import generate_random_config
from typing import Dict, Optional, Tuple, Union
from dataloader import VideoDataset2, build_transform
from stable_baselines3.common.policies import ActorCriticPolicy
from utils import SEARCH_SPACE, SEARCH_SPACE_NAMES, GOLDEN_CONFIG_VECTOR
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from cameraclass import build_cameras_with_info_list, build_a_camera_with_config


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


class ContextualGPModel():
    def __init__(self, action_size=24, context_size=512):
        action_kernel = GPy.kern.Exponential(
            input_dim=action_size, active_dims=range(action_size))
        context_kernel = GPy.kern.Linear(
            input_dim=context_size, active_dims=range(action_size, action_size + context_size))

        self.kernel = action_kernel + context_kernel
        self.trained = False
        
        self.max_update_size = 4096

    def fit(self, configs, cluster_contexts, pls):
        print('fitting gp model...')
        print('configs shape: ', configs.shape)
        print('cluster_contexts shape: ', cluster_contexts.shape)
        print('pls shape: ', pls.shape)
        X = np.hstack([configs, cluster_contexts])
        y = pls
        self.pmodel = GPy.models.GPRegression(X, y[:, 0:1], self.kernel)
        self.lmodel = GPy.models.GPRegression(X, y[:, 1:2], self.kernel)
        self.pmodel.optimize()
        self.lmodel.optimize()
        self.trained = True
        self.X = X
        self.y = y

    def predict(self, config_vector, context_vector):
        X = np.hstack([config_vector, context_vector])[np.newaxis, :]
        pmean, pvariance = self.pmodel.predict(X)
        lmean, lvariance = self.lmodel.predict(X)
        pmean, lmean = float(pmean[0]), float(lmean[0])
        pvariance, lvariance = float(pvariance[0]), float(lvariance[0])

        return pmean, pvariance, lmean, lvariance

    def update(self, config_vector, context_vector, pl):
        X = np.hstack([config_vector, context_vector])
        y = pl
        self.pmodel.set_XY(np.vstack([self.X, X]),
                           np.vstack([self.y[:, 0:1], y[:, 0:1]]))
        self.lmodel.set_XY(np.vstack([self.X, X]),
                           np.vstack([self.y[:, 1:2], y[:, 1:2]]))
        self.pmodel.optimize()
        self.lmodel.optimize()
        self.trained = True
        self.X = np.vstack([self.X, X])
        self.y = np.vstack([self.y, y])
        
        if len(self.X) > self.max_update_size:
            self.X = self.X[-self.max_update_size:]
            self.y = self.y[-self.max_update_size:]

    def get_config_vector(self, context, ori_config_vector, boundline=None):
        radius = 1
        safe_config_vectors = []
        while len(safe_config_vectors) < 1:
            if radius >= 4:
                break
            config_vectors = utils.generate_random_set_config_vector(
                ori_config_vector, radius)
            for config_vector in config_vectors:
                pm, pa, lm, la = self.predict(config_vector, context)
                if pm - pa * 0.5 > boundline:
                    safe_config_vectors.append(
                        [config_vector, lm + la * 0.25, pm + pa * 0.25])
            radius += 1
        if len(safe_config_vectors) > 0:
            return max(safe_config_vectors, key=lambda x: x[1])
        else:
            return utils.generate_random_config_vector(), 0.0, 0.0

    def save_gpmodel(self, filename):
        self.pmodel.save_model(filename + '_pmodel')
        self.lmodel.save_model(filename + '_lmodel')

    def load_gpmodel(self, filename):
        self.pmodel = GPy.core.model.Model.load_model(filename + '_pmodel.zip')
        self.lmodel = GPy.core.model.Model.load_model(filename + '_lmodel.zip')
        self.trained = True


class DUPIN(ActorCriticPolicy):  # ActorCriticPolicy
    def __init__(self, hidden_size=128, video_context_size=16,
                 observation_space=None, action_space=None,
                 lr_schedule=None, context_obs=True):
        super(DUPIN, self).__init__(observation_space=observation_space,
                                    action_space=action_space, lr_schedule=lr_schedule)
        self.search_space = SEARCH_SPACE
        self.search_space_name = SEARCH_SPACE_NAMES
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

        self.metric_feature_extractor = nn.Linear(3, hidden_size)
        self.config_feature_extractor = nn.Sequential(
            nn.Linear(hidden_size * len(self.search_space), hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2, hidden_size))

        self.aggregator = nn.Linear(hidden_size*2, hidden_size)

        if context_obs:
            self.video_context_embedding = nn.Linear(
                video_context_size, hidden_size)
            self.context_aggregator = nn.Sequential(
                nn.Linear(hidden_size*2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size))

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
        action, log_probs, _ = self.extract_action_from_context_based_on_mlp(
            batch_context)

        return action, log_probs

    def obs_to_tensor(self, observation):
        observation = torch.from_numpy(observation).float().to(self.device)
        return observation

    def extract_batch_context(self, observations, context_obs=None):
        batch_context = []
        for bi, ob in enumerate(observations):
            if context_obs is not None:
                context = self.extract_context_based_on_mlp(
                    ob, context_obs[bi])
            else:
                context = self.extract_context_based_on_mlp(ob)
            batch_context.append(context)
        batch_context = torch.cat(batch_context, dim=0)
        return batch_context

    def extract_context_based_on_mlp(self, observation, context_observation=None):
        '''
        observation: (maxsize, 2 + len(search_space))
        '''
        # If context_observation is not None, then the context_observation is used as the context
        if context_observation is not None:
            video_context = self.video_context_embedding(
                context_observation).unsqueeze(0)
        # Filter every row that is not -1. The size of obs is (B, maxsize, 2 + len(search_space))
        ob = observation[observation[:, 0] != -1]
        metric_observation, config_observation = ob[:, :3], ob[:, 3:]
        config_observation = config_observation.long()
        embeddings = []
        for i, name in enumerate(self.search_space_name):
            embedding = self.embedding_heads[i](config_observation[:, i])
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
        else:
            context = observation_context

        return context

    def extract_context_based_on_rnn(self, observation):
        '''
        observation: (maxsize, 2 + len(search_space))
        '''
        # Filter every row that is not -1. The size of obs is (B, maxsize, 2 + len(search_space))
        ob = observation[observation[:, 0] != -1]
        metric_observation, config_observation = ob[:, :2], ob[:, 2:]
        config_observation = config_observation.long()
        embeddings = []
        for i, name in enumerate(self.search_space_name):
            embedding = getattr(self, name + "_embedding")
            embeddings.append(embedding(config_observation[:, i]))
        # (1, len(search_space), hidden_size)
        embeddings = torch.stack(embeddings, dim=1)

        config_feature_set = self.config_feature_extractor(embeddings)[
            1].squeeze(0)
        metric_feature_set = self.metric_feature_extractor(
            metric_observation)
        feature_set = torch.cat(
            [config_feature_set, metric_feature_set], dim=-1)
        context = torch.max(feature_set, dim=0)[0].unsqueeze(0)

        return context

    def extract_action_from_context_based_on_mlp(self, batch_context,
                                                 actions=None,
                                                 deterministic=False):
        """_summary_

        Args:
            batch_context (_type_): [B, H]
            actions (_type_, optional): _description_. Defaults to None.
            deterministic (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        pred_fallten = []
        # for i, name in enumerate(self.search_space_name):
        #     # pred = getattr(self, name)(batch_context) false code here because the inplace operation
        #     pred = getattr(self, name)(batch_context)
        #     pred_fallten.append(pred)
        for i, name in enumerate(self.search_space_name):
            pred = self.predict_heads[i](batch_context)
            pred_fallten.append(pred)

        pred_fallten = torch.cat(pred_fallten, dim=-1)
        pred_fallten_distribution = self.action_dist.proba_distribution(
            action_logits=pred_fallten)
        actions = pred_fallten_distribution.get_actions(
            deterministic=deterministic)
        log_probs = pred_fallten_distribution.log_prob(actions)
        entropy = pred_fallten_distribution.entropy()

        return actions, log_probs, entropy

    def extract_action_from_context_based_on_rnn(self, batch_context, actions=None, deterministic=False):
        B, H = batch_context.size()
        input_data = torch.zeros(B, 1, H).cuda()

        pred_fallten = []
        pred_actions, log_probs, entropy = [], [], []
        for i, name in enumerate(self.search_space_name):
            output, batch_context = self.decoder(
                input_data, batch_context.unsqueeze(0))
            batch_context = batch_context.squeeze(0)

            pred = getattr(self, name)(output[:, 0, :])
            pred_dist = torch.distributions.Categorical(logits=pred)
            pred_params = pred_dist.sample()

            pred_fallten.append(pred)

            input_data = getattr(
                self, name + "_embedding")(pred_params).unsqueeze(1)

            pred_actions.append(pred_params)
            if actions is not None:
                log_prob = pred_dist.log_prob(actions[:, i])
            else:
                log_prob = pred_dist.log_prob(pred_params)
            log_probs.append(log_prob)
            entropy.append(pred_dist.entropy())

        pred_fallten = torch.cat(pred_fallten, dim=-1)
        pred_fallten_distribution = self.action_dist.proba_distribution(
            action_logits=pred_fallten)
        actions = pred_fallten_distribution.get_actions(
            deterministic=deterministic)
        log_prob = pred_fallten_distribution.log_prob(actions)
        entropy = pred_fallten_distribution.entropy()

        pred_actions = torch.stack(pred_actions, dim=-1)
        entropy = torch.stack(entropy, dim=-1).sum(dim=-1)
        log_probs = torch.stack(log_probs, dim=-1).sum(dim=-1)

        return pred_actions, log_probs, entropy

    def forward(self, obs: Tensor, context_obs: Tensor = None, deterministic: bool = False) -> Tuple[Tensor]:
        batch_context = self.extract_batch_context(obs, context_obs)
        action, log_probs, _ = self.extract_action_from_context_based_on_mlp(
            batch_context, deterministic=deterministic)
        # batch_value = self.value_net(batch_context)

        return action, log_probs

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


class UnituneEnv(gym.Env):
    def __init__(self, dataset, feature_extractor_model, camera_config,
                 max_iter, max_pareto_set_size, effiency_targets, debug=False, use_init_solution=True):
        super(UnituneEnv, self).__init__()
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
        self.effiency_target = self.effiency_targets[0]
        self.debug = debug
        if self.debug:
            print("\033[91mDebug mode is on ... ...\033[0m")

        self.use_init_solution = use_init_solution
        if use_init_solution:
            self.init_solution = [[5, 5, 1, 4, 4, 1, 1, 2, 4],
                                  [0, 5, 0, 1, 2, 1, 4, 4, 1],
                                  [0, 1, 0, 1, 0, 1, 1, 4, 3],
                                  [0, 1, 7, 1, 1, 1, 4, 1, 3],
                                  [1, 0, 0, 1, 1, 1, 1, 3, 1]]

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
        pareto_set = sorted(pareto_set, key=lambda x: x[0])
        area = pareto_set[0][0] * pareto_set[0][1]
        for i in range(1, len(pareto_set)):
            area += pareto_set[i][1] * (pareto_set[i][0] - pareto_set[i-1][0])
        return area

    def step_reward(self, at, lt):
        at_1, lt_1 = self.solution[0], self.solution[1]
        if lt > self.effiency_target and lt_1 > self.effiency_target:
            reward = 1 + (at - at_1) + (at - self.a0)
        elif lt > self.effiency_target and lt_1 <= self.effiency_target:
            reward = 1 + (at - at_1) + (at - self.a0) + \
                (lt - lt_1) + (lt - self.l0)
        elif lt <= self.effiency_target and lt_1 > self.effiency_target:
            reward = (lt - lt_1) + (lt - self.l0)
        elif lt <= self.effiency_target and lt_1 <= self.effiency_target:
            reward = (lt - lt_1) + (lt - self.l0)

        return reward

    def win_reward(self):
        pass

    def prepare_observation(self, pareto_set):
        processed_pareto_set = np.ones(
            (self.max_pareto_set_size, len(SEARCH_SPACE) + 3)) * -1
        for i, point in enumerate(pareto_set):
            processed_pareto_set[i, :2] = point[:2]
            processed_pareto_set[i, 2:3] = self.effiency_target
            processed_pareto_set[i, 3:] = point[2]
        return processed_pareto_set

    def prepare_context_observation(self):
        cameraid = self.context['Cameraid']
        timeid = self.context['Time']
        quality = self.context['Quailty']
        traj_context = [self.context['mean_speed'], self.context['std_speed'], self.context['mean_acceleration'], self.context['std_acceleration'],
                        self.context['mean_traj_linearity'], self.context['mean_traj_length'], self.context['traj_count'],
                        self.context['proportion_of_stopped_vehicles'],
                        self.context['mean_traj_densitys'], self.context['std_traj_densitys'],
                        self.context['car_rate'], self.context['bus_rate'], self.context['truck_rate']]
        return [cameraid, timeid, quality] + traj_context

    def step(self, action):
        if len(action.shape) == 2:
            action = action[0]
        self.video_camera_config.id = self.current_video
        if self.debug:
            accuracy, latency = random.random(), random.random()
        else:
            self.video_camera_config.updateConfig(utils.generate_config(action,
                                                                        self.video_camera_config.loadConfig()))
            _, motvalue, _, _, _, \
                cmetric, _ = self.video_camera_config.ingestion
            accuracy, latency = motvalue, cmetric[-1]

        self.a, self.l = accuracy, latency

        reward = self.step_reward(accuracy, latency)

        if latency > self.effiency_target:
            if accuracy > self.solution[0]:
                self.solution = [accuracy, latency, action]

        observation, done, truncated, info = self.prepare_observation(
            [self.solution]), False, False, {}

        if self.iter_num >= self.max_iter - 1:
            truncated = True

        self.iter_num += 1
        return observation, reward, done, truncated, info

    def reset(self):
        if len(self.video_ids_iter) == 0:
            self.video_ids_iter = self.video_ids.copy()
        random.shuffle(self.video_ids_iter)
        self.current_video = self.video_ids_iter.pop()
        self.video_camera_config.id = self.current_video

        self.video_camera_config.updateConfig(utils.generate_config(GOLDEN_CONFIG_VECTOR,
                                                                    self.video_camera_config.loadConfig()))
        if self.debug:
            accuracy, latency = random.random(), random.random()
        else:
            _, motvalue, _, _, _, \
                cmetric, _ = self.video_camera_config.ingestion
            accuracy, latency = motvalue, cmetric[-1]
        self.a0, self.l0 = accuracy, latency
        self.solution = [accuracy, latency, GOLDEN_CONFIG_VECTOR]

        observation = self.prepare_observation([self.solution])
        self.iter_num = 0
        return observation, {}

    def reset_with_videoid(self, video_id):
        self.current_video = video_id
        self.video_camera_config.id = video_id
        self.video_camera_config.updateConfig(utils.generate_config(GOLDEN_CONFIG_VECTOR,
                                                                    self.video_camera_config.loadConfig()))
        if self.debug:
            accuracy, latency = random.random(), random.random()
        else:
            _, motvalue, _, _, _, \
                cmetric, _ = self.video_camera_config.ingestion
            accuracy, latency = motvalue, cmetric[-1]
        self.solution = [accuracy, latency, GOLDEN_CONFIG_VECTOR]
        self.a0, self.l0 = accuracy, latency

        observation = self.prepare_observation([self.solution])
        self.iter_num = 0
        return observation, {}

    @staticmethod
    def is_dominated(x, y):
        return all(x[i] <= y[i] for i in range(2)) and any(x[i] < y[i] for i in range(2))

    def identify_pareto(self, solutions):
        pareto_front = []
        for solution in solutions:
            if not any(self.is_dominated(solution, other) for other in solutions):
                pareto_front.append(solution)
        if len(pareto_front) > self.max_pareto_set_size:
            pareto_front = sorted(pareto_front, key=lambda x: x[0])
            pareto_front = pareto_front[:self.max_pareto_set_size]
        return pareto_front

    def extract_features(self, frames):
        return self.feature_extractor.extract(frames)

    def render(self, mode="human"):
        pass


# REINFORCE训练过程
def reinforce(env, test_env,
              policy, bo_model, optimizer,
              batch_size, writer, save_path,
              episodes=1000,
              gamma=0.99, lr=None,
              lr_decay=False, video_list=[0], writer_window=10):
    bast_metric, step_i = 0, 0

    reward_window = deque(maxlen=10)
    baseline_window = {vid: [deque(maxlen=100) for _ in range(
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
            batch_saved_log_probs = []
            batch_rewards = []
            batch_idxs = []
            batch_vids = []

            bo_config_vectors, bo_context_vectors, bo_pls = [], [], []
            for episode in range(batch_size):
                random_effiency_target = random.choice(env.effiency_targets)
                env.effiency_target = random_effiency_target
                saved_log_probs = []
                rewards = []
                observation, _ = env.reset()
                done = False
                interactive_round = 0

                # Generate an episode
                while not done:
                    observation = torch.from_numpy(
                        observation).float().unsqueeze(0)
                    action, log_probs = policy(observation)

                    bo_config_vectors.append(action.tolist()[0])
                    bo_context_vectors.append(
                        GOLDEN_CONFIG_VECTOR + [env.a0, env.l0] + [random_effiency_target])

                    observation, reward, done, truncated, _ = env.step(action)

                    bo_pls.append([env.a, env.l])

                    saved_log_probs.append(log_probs)
                    rewards.append(reward)
                    interactive_round += 1
                    step_i += 1
                    writer.add_scalar("Reward/step_reward",
                                      reward, step_i)

                    if truncated:
                        break

                reward_window.append(sum(rewards))

                # Calculate return rewards
                R = 0
                vids = []
                idxs = []
                returns = []
                for i, r in enumerate(rewards[::-1]):
                    R = r + gamma * R
                    interactive_round = len(rewards) - i - 1
                    baseline_window[env.current_video][interactive_round].append(
                        R)
                    returns.insert(0, R)
                    idxs.insert(0, interactive_round)
                    vids.insert(0, env.current_video)

                batch_vids.extend(vids)
                batch_idxs.extend(idxs)
                batch_rewards.extend(returns)
                batch_saved_log_probs.extend(saved_log_probs)

                progress.update(task, advance=1)
                writer.add_scalar("Reward/episode_sum_reward",
                                  np.sum(rewards), batch + episode)
                writer.add_scalar("Reward/episode_mean_reward",
                                  np.mean(rewards), batch + episode)
                if batch + episode > writer_window:
                    writer.add_scalar("Reward/window_mean_reward",
                                      np.mean(reward_window), batch + episode)

            # 计算baseline
            baseline_batch_rewards = [
                float(np.mean(baseline_window[vid][i])) for vid, i in zip(batch_vids, batch_idxs)]

            batch_rewards = [r - b for r,
                             b in zip(batch_rewards, baseline_batch_rewards)]
            batch_saved_log_probs = torch.cat(batch_saved_log_probs)

            # 策略梯度更新
            policy_loss = []
            for log_prob, R in zip(batch_saved_log_probs, batch_rewards):
                policy_loss.append(-log_prob * R)

            optimizer.zero_grad()
            policy_loss = torch.stack(policy_loss).sum()
            policy_loss.backward()
            optimizer.step()

            if lr_decay:
                lr_now = lr * (1 - batch / episodes)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_now

            writer.add_scalar("Loss/policy_loss", policy_loss, batch)

            bo_config_vectors = np.array(bo_config_vectors)
            bo_context_vectors = np.array(bo_context_vectors)
            bo_pls = np.array(bo_pls)

            if bo_model.trained:
                bo_model.update(bo_config_vectors, bo_context_vectors, bo_pls)
            else:
                bo_model.fit(bo_config_vectors, bo_context_vectors, bo_pls)

            save_result_path = os.path.join(save_path, f"step_{batch}")
            if not os.path.exists(save_result_path):
                os.makedirs(save_result_path)

            if batch % (batch_size * 4) == 0:
                total_acc, total_lat = 0, 0
                for video_id in test_env.video_ids:
                    pareto_set = []
                    for effiency_target in test_env.effiency_targets:
                        test_env.effiency_target = effiency_target
                        obs, _ = test_env.reset_with_videoid(video_id)
                        done = False
                        truncated = False
                        total_reward = 0
                        while not done and not truncated:
                            action, _states = policy.predict(
                                obs, deterministic=True)
                            obs, rewards, done, truncated, _info = test_env.step(
                                action)
                            total_reward += rewards
                        pareto_set.append([test_env.solution[0],
                                           test_env.solution[1],
                                           test_env.solution[2]])
                        
                    print(f"\nvideo_id - 1: {video_id}, pareto_set: {len(pareto_set)}")
                    pareto_set = env.identify_pareto(pareto_set)
                    print(f"video_id - 2: {video_id}, pareto_set: {len(pareto_set)}\n")

                    plt.scatter([res[0] for res in pareto_set],
                                [res[1] for res in pareto_set], c='r',
                                marker='x', label=f'pareto set {video_id}')
                    plt.savefig(os.path.join(save_result_path,
                                f"pareto_set_{video_id}.png"))
                    plt.close()
                    plt.cla()
                    plt.clf()
                    with open(os.path.join(save_result_path, f"pareto_set_{video_id}.txt"), "w") as f:
                        mean_acc, mean_lat = [], []
                        for res in pareto_set:
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
                            f'eval/pset_num_{video_id}', len(pareto_set), batch)
                        total_acc += mean_acc
                        total_lat += mean_lat
                total_acc = total_acc / len(video_list)
                total_lat = total_lat / len(video_list)

                total_metric = total_acc * 0.5 + total_lat * 0.5 + total_reward + 0.5 * \
                    math.exp(-(max_pareto_set_size-len(pareto_set)) /
                             max_pareto_set_size)  # plus the pareto number
                if total_metric > bast_metric:
                    bast_metric = total_metric
                    torch.save(policy.state_dict(), os.path.join(
                        save_path, f"best_model_{batch}"))
            torch.save(policy.state_dict(), os.path.join(
                save_path, f"last_model_{batch}"))
            bo_model.save_gpmodel(os.path.join(save_path, f"gpmodel_{batch}"))


if __name__ == "__main__":
    seed = 2024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    channel = 800
    method_name = "unitune"
    data_name = "hippo"
    scene_name = "hippo"
    configpath = "./cache/base.yaml"
    deviceid = 1
    lr = 1e-1
    temp = 10.0
    max_iter = 10
    batch_size = 32
    update_step = 64
    lr_decay_flag = True
    max_pareto_set_size = 8
    framenumber, framegap = 16, 16
    effiency_targets = [0.4, 0.5, 0.6, 0.7, 0.8]
    train_video_list = [0,   1,   4,   5,  10, 100, 105, 128, 173, 188, 206, 210, 256, 277, 294, 325, 337, 363, 382, 397,
                        412, 432, 451, 468, 491, 507, 512, 516, 517, 591, 605, 619, 635, 679, 698, 700, 715, 758, 781, 789]
    test_video_list = [3,  25,  50,  79, 104, 127, 160, 184]
    log_dir = "./unitune_tensorboard/"
    context_path_name = "context_01.csv"
    writer = SummaryWriter(log_dir=log_dir)
    objects, metrics = ["car", "bus", "truck"], ["sel", "agg", "topk"]
    alldir = utils.DataPaths('/home/lzp/otif-dataset/dataset',
                             data_name, "train", method_name)
    save_path = os.path.join(log_dir, "eval_result")

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

    test_cameras, test_videopaths, test_trackdatas = build_cameras_with_info_list(alldir, test_video_list,
                                                                                  objects, metrics,
                                                                                  configpath, deviceid,
                                                                                  method_name, data_name, "test",
                                                                                  scene_name, temp)

    train_transform = build_transform()

    train_dataset, test_dataset = VideoDataset2(
        train_videopaths, train_trackdatas, framenumber,
        framegap, objects=objects, transform=train_transform,
        flag="train", context_path_name=context_path_name), VideoDataset2(
        test_videopaths, test_trackdatas, framenumber,
        framegap, objects=objects, transform=train_transform,
        flag="test", context_path_name=context_path_name)

    observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(max_pareto_set_size, len(SEARCH_SPACE) + 2), dtype=np.float32)
    action_space = gym.spaces.MultiDiscrete(SEARCH_SPACE)
    bo_model = ContextualGPModel(action_size=len(SEARCH_SPACE),
                                 context_size=len(SEARCH_SPACE)+3)
    policy = DUPIN(hidden_size=64, video_context_size=16,
                   observation_space=observation_space,
                   action_space=action_space)
    # pretrain_weight = "./reinforce_tensorboard_all/eval_result/best_model_8896"
    # pretrain_weights = torch.load(pretrain_weight)["state_dict"]
    # policy.load_state_dict(pretrain_weights, strict=False)
    # policy_keys = set(policy.state_dict().keys())
    # pretrain_weights_keys = set(pretrain_weights.keys())
    # missing_keys = policy_keys - pretrain_weights_keys
    # print("\033[93mParameters not loaded from the weights file:\033[0m")
    # for key in missing_keys:
    #     print("\033[93m" + key + "\033[0m")

    env = UnituneEnv(train_dataset, None, train_cameras[0],
                     max_iter, max_pareto_set_size, effiency_targets)
    test_env = UnituneEnv(test_dataset, None, test_cameras[0],
                          max_iter, max_pareto_set_size, effiency_targets)

    optimizer = optim.Adam(policy.parameters(), lr=lr)

    reinforce(env, test_env, policy, bo_model, optimizer,
              batch_size, writer, save_path, lr=lr,
              episodes=100000, gamma=0.99, lr_decay=lr_decay_flag,
              video_list=test_video_list)
