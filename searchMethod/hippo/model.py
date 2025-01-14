import os
import math
import random

import torch
import numpy as np
import pandas as pd
from torch import nn
import seaborn as sns
import pygraphviz as pgv
from torchvision import models
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

from transformers import VideoMAEForVideoClassification

from utils import add_node, generate_config, mean, \
    distance_matrix, replace_invalid_values, \
    SEARCH_SPACE, KNOB_TYPES, POS_TYPES


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[x, 0, :]  # .repeat(x.size(0), 1)


class GraphConvNet(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GraphConvNet, self).__init__()

        self.conv1 = GCNConv(num_node_features, 128)
        self.conv2 = GCNConv(128, num_classes)
        self.act = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index, edge_weight)
        x = self.act(x)
        x = self.conv2(x, edge_index, edge_weight)

        return x


class KnobEmbedding(nn.Module):
    def __init__(self, num_total_tokens, hidden_size, num_knob=5, scl=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_total_tokens, hidden_size)
        self.knob_embedding = nn.Embedding(num_knob + 1, hidden_size)
        self.positional_embedding = PositionalEncoding(
            hidden_size, max_len=100)

    def forward(self, x, knob=None, pos=None, graph_emb=None):
        if x.shape[-1] != self.hidden_size:
            xe = self.embedding(x) + \
                self.knob_embedding(knob) + \
                self.positional_embedding(pos)
        else:
            xe = x + \
                self.knob_embedding(knob) + \
                self.positional_embedding(pos)
        if graph_emb is not None:
            xe = xe + graph_emb
        return xe


class VideoExtractor(nn.Module):
    def __init__(self, model_ckpt, objects=[], framenumber=8,
                 context_size=768, process_quailty_number=3):
        super().__init__()

        # define the pretrained video extractor based on torchvision
        self.video_extractor = VideoMAEForVideoClassification.from_pretrained(
            model_ckpt,
            # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
            ignore_mismatched_sizes=True,
        )
        self.fc_layer = nn.Sequential(nn.Linear(context_size, context_size),
                                      nn.ReLU(),
                                      nn.Linear(context_size, process_quailty_number))

        self.objects = objects
        self.framenumber = framenumber
        layer_names = ['patch_embeddings'] + \
            [f'encoder.layer.{i}.' for i in [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
        self.frozen_layers(layer_names)

    def frozen_layers(self, frozen_layer_list):
        for name, param in self.video_extractor.named_parameters():
            if any(layer_name in name for layer_name in frozen_layer_list):
                param.requires_grad = False
                print(f'Froze {name}')

    def forward(self, images, representations=None, compute_loss=True):
        b, c, t, h, w = images.shape  # batch_size, num_frames, num_channels, height, width
        context_vector = self.video_extractor(
            images, output_hidden_states=True)['hidden_states'][-1].mean(1).squeeze()
        pred_process_quailty = self.fc_layer(context_vector)

        if not self.training:
            return context_vector, representations, pred_process_quailty

        if compute_loss:
            target_distances = distance_matrix(representations)  # b, b
            pred_distances = distance_matrix(context_vector)  # b, b
            dist_p, dist_n = [], []
            for bi in range(b):
                vi, vj = random.sample([bj for bj in range(b) if bj != bi], 2)
                if target_distances[bi][vi] < target_distances[bi][vj]:
                    dist_p.append(pred_distances[bi][vi])
                    dist_n.append(pred_distances[bi][vj])
                else:
                    dist_p.append(pred_distances[bi][vj])
                    dist_n.append(pred_distances[bi][vi])
            dist_p = torch.stack(dist_p)
            dist_n = torch.stack(dist_n)
            loss = torch.clamp(dist_p - dist_n + 1.0, min=0.0)

            return loss
        else:
            return pred_process_quailty

    @torch.no_grad()
    def extract(self, images):
        b, c, t, h, w = images.shape
        context_vector = self.video_extractor(
            images, output_hidden_states=True)['hidden_states'][-1].mean(1).squeeze()
        return context_vector


class HighWayNetwork(nn.Module):
    def __init__(self, context_size, hidden_size):
        super().__init__()

        self.highway = nn.Sequential(
            nn.Linear(context_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.gate = nn.Sequential(
            nn.Linear(context_size, hidden_size),
            nn.Sigmoid(),
        )

        # For matching the dimensions.
        self.transform = nn.Linear(context_size, hidden_size)

    def forward(self, context_vector):
        # Transform the input to match the hidden_size
        transformed_context = self.transform(context_vector)
        return self.highway(context_vector) * self.gate(context_vector) + transformed_context * (1 - self.gate(context_vector))


class TransWatcher(nn.Module):
    def __init__(self,
                 hidden_size,
                 metric_size,
                 context_size=512,
                 search_spaces=SEARCH_SPACE,  # noise reduction [26]
                 knob_types=[0,
                             0,
                             0, 1, 1,
                             0, 1, 1,
                             0, 1, 1, 1, 1,
                             0, 1, 1,
                             0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             0, 1,
                             0, 1],
                 thre_types=[0,
                             0,
                             0, 0, 1,
                             0, 0, 1,
                             0, 0, 0, 0, 0,
                             0, 0, 1,
                             0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             0, 0,
                             0, 0],):
        super().__init__()
        self.search_spaces = search_spaces
        self.knob_types = knob_types
        self.thre_types = thre_types
        self.num_total_tokens = sum(search_spaces)

        # build high-way network for context reader
        self.context_reader = HighWayNetwork(context_size, hidden_size)

        self.embedding = nn.Embedding(self.num_total_tokens, hidden_size)
        self.knob_embedding = nn.Embedding(6, hidden_size)
        self.positional_embedding = PositionalEncoding(hidden_size, max_len=31)

        self.watcher = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=2), num_layers=3)
        self.predictor = nn.Linear(hidden_size, metric_size)

    def forward(self, contexts, actions, knob_vector, pos_vector):
        context_vector = self.context_reader(contexts)
        config_vector = self.embedding(actions)
        states = torch.cat([context_vector.unsqueeze(1),
                            config_vector], dim=1)
        knob_embeddings = self.knob_embedding(
            knob_vector)
        pos_embeddings = self.positional_embedding(pos_vector)

        states = states + knob_embeddings + pos_embeddings
        states = states.transpose(0, 1)
        states = self.watcher(states)
        states = states.transpose(0, 1)

        final_state = states[:, 0, :]
        metric = self.predictor(final_state)
        metric = torch.sigmoid(metric)
        return metric


class TransController(nn.Module):
    def __init__(self,
                 batch_size,
                 hidden_size,
                 device,
                 eval=False,
                 eval_load_path=None,
                 objects=None,
                 history_weight=0.1,
                 dynamic_graph_rate=0.9,
                 context_size=512,
                 nhead=2,
                 num_encoder_layers=2,
                 num_decoder_layers=2,
                 dropout=0.1,
                 dim_feedforward=128):
        super(TransController, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size

        self.history_weight = history_weight
        self.context_fusion_layer = nn.Linear(
            hidden_size + context_size, hidden_size)
        self.predictor = nn.Transformer(d_model=hidden_size,
                                        nhead=nhead,
                                        num_encoder_layers=num_encoder_layers,
                                        num_decoder_layers=num_decoder_layers,
                                        dim_feedforward=dim_feedforward,
                                        dropout=dropout, activation='relu')

        self.search_spaces = SEARCH_SPACE
        self.knob_types = KNOB_TYPES
        num_total_tokens = sum(self.search_spaces)

        self.objects = objects

        self.device = device

        num_total_tokens += 1
        self.convmodel = GraphConvNet(hidden_size, hidden_size)

        self.embedding = KnobEmbedding(num_total_tokens, hidden_size)

        # * share the weight between the knob prediction layer and the embedding layer
        self.knob_layer = nn.Linear(hidden_size, num_total_tokens)

        self.topk_best_cfg_seeds = []

        if not eval:
            self.generate_vdbms_seed(self.hidden_size)
            self.nodes, self.edges, self.node_dict, self.remap_node_dict = self.generate_corr_weight()
            self.knob_layer.weight = self.embedding.embedding.weight
        else:
            self.load_corr_weight(eval_load_path)
            self.load_ema_seed(eval_load_path)

        self.loss_func = nn.CrossEntropyLoss()
        self.dynamic_graph_rate = dynamic_graph_rate

    def save_ema_seed(self, save_path):
        torch.save(self.search_random_state, save_path)

    def load_ema_seed(self, load_path):
        print("\033[92m" + f"Load ema seed from {load_path}" + "\033[0m")
        load_path = os.path.join(load_path, "ema_seed.pt")
        self.search_random_state = torch.load(load_path)
        print(self.search_random_state)

    def save_corr_weight(self, save_path):
        # save path : ... pth
        torch.save((self.nodes, self.edges, self.node_dict,
                   self.remap_node_dict), save_path)

    def load_corr_weight(self, load_path):
        # load path : ... pth print red info
        print("\033[91m" + f"Load corr weight from {load_path}" + "\033[0m")
        load_path = os.path.join(load_path, "corr_weight.pt")
        self.nodes, self.edges, self.node_dict, self.remap_node_dict = torch.load(
            load_path)

    def generate_thre_weight(self):
        thre_weights = [[[1] for _ in range(self.search_spaces[i])] for i in range(
            len(self.search_spaces))]
        return thre_weights

    def update_thre_weight(self, rewards, actions):
        for action, reward in zip(actions, rewards):
            for i in range(len(self.search_spaces)):
                if len(self.thre_weights[i][action[i]]) > self.lds_size:
                    self.thre_weights[i][action[i]].pop(0)
                    self.thre_weights[i][action[i]].append(reward)
                else:
                    self.thre_weights[i][action[i]].append(reward)

    def smooth_thre_weight(self):
        smoothed_thre_weights = []
        for i in range(len(self.search_spaces)):
            if self.thre_types[i] != 1:
                smoothed_thre_weights.append(None)
            else:
                weights = []
                for j in range(self.search_spaces[i]):
                    weights.append(mean(self.thre_weights[i][j]))
                weights = np.clip(weights, 0, 1)
                kde = gaussian_kde(np.linspace(
                    0, 1, len(weights)), weights=weights)
                smoothed_weights = kde(np.linspace(0, 1, len(weights)))
                smoothed_thre_weights.append(smoothed_weights)
        return smoothed_thre_weights

    def generate_lds_weight(self, actions):
        smoothed_thre_weights = self.smooth_thre_weight()
        lds_weight = []
        for i in range(len(actions)):
            action = actions[i]
            for j in range(len(self.search_spaces)):
                if self.thre_types[j] == 1:
                    lds_weight.append(
                        1 / (1.0 - smoothed_thre_weights[j][action[j]]) - 1.0)
                else:
                    lds_weight.append(1)
        return lds_weight

    def generate_scl_weight(self, actions):
        # generate weithts by nodes and edges
        scl_weight = []
        for i in range(len(actions)):
            action = actions[i]
            for j in range(len(self.search_spaces)):
                if self.knob_types[j] == 0:
                    scl_weight.append(1)
                else:
                    scl_weight.append(action[j])
        return scl_weight

    def make_action(self, input_state):
        hidden_cell = self.init_hidden(
            batch_size=input_state.shape[0], device=input_state.device)

        activations = []
        entropies = []
        log_probs = []

        # Generate skipnumber and scaledownresolution
        for step_i in range(len(self.search_spaces)):
            hidden, cell = self.encoder_cell(input_state, hidden_cell)
            hidden_cell = (hidden, cell)
            cat_hidden = torch.cat(hidden_cell, dim=1)

            logits = self._decoders[step_i](cat_hidden)
            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            entropy = -(log_prob * probs).sum(1, keepdim=False)

            action = probs.multinomial(num_samples=1).data
            selected_log_prob = log_prob.gather(1, action)
            input_action = action[:, 0] + sum(self.search_spaces[:step_i])
            input_state = self.embedding(input_action) + self.knob_embedding(
                torch.tensor([self.knob_types[step_i]]).to(input_state.device))

            activations.append(action[:, 0])
            entropies.append(entropy)
            log_probs.append(selected_log_prob)

        activations = torch.stack(activations).transpose(0, 1)
        return activations, torch.cat(log_probs), torch.cat(entropies)

    def init_hidden(self, batch_size=1, device='cpu'):
        return (torch.zeros(batch_size, self.hidden_size).to(device),
                torch.zeros(batch_size, self.hidden_size).to(device))

    def visualize_network(self, config, step_i,
                          activations, rewards, accuracys, save_dir="./searchMethod/streamline/log/"):
        assert os.path.exists(
            save_dir), f"Save directory {save_dir} does not exist."
        if not os.path.exists(os.path.join(save_dir, f"{step_i}/graph")):
            os.makedirs(os.path.join(save_dir, f"{step_i}/graph"))

        if not os.path.exists(os.path.join(save_dir, f"{step_i}/cfgs")):
            os.makedirs(os.path.join(save_dir, f"{step_i}/cfgs"))

        df = pd.DataFrame(accuracys)
        df.to_csv(os.path.join(
            save_dir, f"{step_i}/accuracy.csv"), index=False)
        df['Graph'] = df.index
        df = df.melt(id_vars='Graph', var_name='Configuration',
                     value_name='Value')

        plt.figure(figsize=(15, 6))
        # Change the color palette using `palette` and mark the style with `style`
        sns.lineplot(data=df, x='Graph', y='Value',
                     hue='Configuration', style='Configuration', palette='cool')
        # To add markers, we'll use scatterplot with the same hue and style
        sns.scatterplot(data=df, x='Graph', y='Value',
                        hue='Configuration', style='Configuration', palette='cool')
        plt.legend(bbox_to_anchor=(1.015, 1), loc=2, borderaxespad=0.)
        plt.savefig(os.path.join(save_dir, f"{step_i}/accuracy.png"))

        for action_i, (action, reward) in enumerate(zip(activations, rewards)):
            graph = pgv.AGraph(directed=True, strict=True,
                               fontname='Helvetica', arrowtype='open')

            add_node(graph, -2, "Rewrad: %.2f" % reward, color='red')
            add_node(graph, -1, "Video")

            base_config = config.load_cache()
            action_config = generate_config(action, base_config)
            config.update_cache(action_config)
            config.save(os.path.join(
                save_dir, f'{step_i}/cfgs/{action_i+1}.yaml'))

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

            enhance_node_ids = []
            for enhance_i, enhancetool in enumerate(action_config['roibase']['enhancetools']):
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

            if action_config['postprocessbase']['flag'] and action_config['noisefilterbase']['flag']:
                add_node(graph, detect_node_id + 2, f"PostProcess")
                add_node(graph, detect_node_id + 3, f"NoiseFilter")
                edges.append((detect_node_id + 1, detect_node_id + 2))
                edges.append((detect_node_id + 2, detect_node_id + 3))
                final_node_id = detect_node_id + 3
            elif action_config['postprocessbase']['flag'] and not action_config['noisefilterbase']['flag']:
                add_node(graph, detect_node_id + 2, f"PostProcess")
                edges.append((detect_node_id + 1, detect_node_id + 2))
                final_node_id = detect_node_id + 2
            elif not action_config['postprocessbase']['flag'] and action_config['noisefilterbase']['flag']:
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
            graph.draw(os.path.join(
                save_dir, f"./{step_i}/graph/graph_{action_i+1}.png"))

    def generate_vdbms_seed(self, hidden_size):
        self.search_random_state = nn.Parameter(
            torch.zeros(hidden_size), requires_grad=True)

    def update_topk_seeds(self, rewards, actions, search_states):
        generated_seeds = search_states.clone().detach()
        for reward, action, generated_seed in zip(rewards, actions, generated_seeds):
            if len(self.topk_best_cfg_seeds) < 10:
                self.topk_best_cfg_seeds.append(
                    (reward, action, generated_seed))
            else:
                min_idx = np.argmin([seed[0]
                                    for seed in self.topk_best_cfg_seeds])
                if reward > self.topk_best_cfg_seeds[min_idx][0]:
                    self.topk_best_cfg_seeds[min_idx] = (
                        reward, action, generated_seed)
            # Sort by reward
            self.topk_best_cfg_seeds = sorted(
                self.topk_best_cfg_seeds, key=lambda x: x[0], reverse=True)

    @property
    def ema_seed(self):
        # if len(self.topk_best_cfg_seeds) == 0: return self.search_random_state
        # for seed_i, seed in enumerate(self.topk_best_cfg_seeds):
        #     if seed_i == 0: ema_seed = seed[2]
        #     else: ema_seed = ema_seed + seed[2]
        # ema_seed = ema_seed / len(self.topk_best_cfg_seeds)
        # ema_seed = ema_seed.unsqueeze(0).expand(self.batch_size, -1)
        # return ema_seed * 0.5 + self.search_random_state * 0.5
        return self.search_random_state

    def generate_corr_weight(self):
        node_id = 0
        nodes, edges, node_dict, remap_node_dict = [], {}, {}, {}
        nodes.append(node_id)
        node_id += 1
        for i in range(len(self.search_spaces)):
            for j in range(self.search_spaces[i]):
                node_dict[node_id] = (i, j)
                remap_node_dict[(i, j)] = node_id
                nodes.append(node_id)
                node_id += 1
        for j in range(self.search_spaces[0]):
            edges[(0, j)] = 1.0
        for i in range(1, len(self.search_spaces)):
            for m in range(i):
                for j in range(self.search_spaces[i - (m+1)]):
                    for k in range(self.search_spaces[i]):
                        start_node = remap_node_dict[(i - (m+1), j)]
                        end_node = remap_node_dict[(i, k)]
                        edges[(start_node, end_node)] = 1.0

        return nodes, edges, node_dict, remap_node_dict

    def update_corr_weight(self, rewards, actions):
        for reward, action in zip(rewards, actions):
            for search_i in range(len(self.search_spaces)):
                node_id = self.remap_node_dict[(
                    search_i, int(action[search_i].item()))]
                if search_i == 0:
                    start_node_ids, end_node_ids = [0], [node_id]
                else:
                    start_node_ids, end_node_ids = [self.remap_node_dict[(search_j,
                                                                          int(action[search_j].item()))] for search_j in range(search_i)], \
                        [self.remap_node_dict[(search_i, int(
                            action[search_i].item()))] for _ in range(search_i)]
                for start_node_id, end_node_id in zip(start_node_ids, end_node_ids):
                    if (start_node_id, end_node_id) not in self.edges:
                        self.edges[(start_node_id, end_node_id)] = reward
                    else:
                        self.edges[(start_node_id, end_node_id)] = self.edges[(
                            start_node_id, end_node_id)] * self.dynamic_graph_rate + reward * (1 - self.dynamic_graph_rate)

    def graph_embedding(self):
        nodes = torch.tensor(self.nodes).to(self.device)
        node_features = self.embedding.embedding(nodes)
        edge_index = []
        edge_weights = []
        for node_pair, edge_value in self.edges.items():
            edge_index.append([node_pair[0], node_pair[1]])
            edge_weights.append(edge_value)
        edge_index = torch.LongTensor(
            edge_index).t().contiguous().to(self.device)
        edge_weights = torch.FloatTensor(edge_weights).to(self.device)
        graph_data = Data(
            x=node_features, edge_index=edge_index, edge_attr=edge_weights)
        graph_embedding = self.convmodel(graph_data)
        return graph_embedding

    def forward(self, context_vector, knob_vector, pos_vector, tempature=None, historyaction=None, historysearchstate=None):
        """_summary_

        Args:
            context_vector (torch.FloatTensor): [bs, context_size].
            knob_vector (torch.LongTensor): [bs, search_space_size].
            pos_vector (torch.LongTensor): [bs, search_space_size].
            tempature (int, optional): 10 -> 1. Defaults to None.
            historyaction (list, optional): Store the history action. Defaults to None.
            historysearchstate (torch.FloatTensor, optional): Store the history search state. Defaults to None.

        Returns:
            _type_: activations, context_vector, log_probs, entropies, fusion_vector, historyloss
        """
        b, _ = context_vector.shape
        activations = [None] * len(self.search_spaces)
        entropies = [None] * len(self.search_spaces)
        log_probs = [None] * len(self.search_spaces)

        graph_embedding = self.graph_embedding()

        search_random_state = self.ema_seed.unsqueeze(
            0).expand(b, -1)
        generator_vector = torch.cat(
            [search_random_state, context_vector], dim=1)
        fusion_vector = self.context_fusion_layer(generator_vector)

        if historysearchstate is not None:
            for bi in range(b):
                if historysearchstate[bi] is not None:
                    fusion_vector[bi] = fusion_vector[bi] * (
                        1.0 - self.history_weight) + self.history_weight * historysearchstate[bi]

        input_states = self.embedding(fusion_vector, knob_vector[:, 0], pos_vector[:, 0],
                                      graph_embedding[0].unsqueeze(0).repeat(b, 1))

        generate_orders = list(range(len(self.search_spaces)))

        if len(input_states.shape) == 2:
            input_states = input_states.unsqueeze(0)

        # Generate skipnumber and scaledownresolution
        for search_i in list(range(len(self.search_spaces))):
            predicted_hidden = self.predictor(input_states, input_states)[-1]
            step_i = generate_orders[search_i]

            si, sj = sum(self.search_spaces[:step_i]), sum(
                self.search_spaces[:step_i + 1])
            logits = self.knob_layer(predicted_hidden)[
                :, si:sj] / tempature  # bs, search_space_size
            probs = F.softmax(logits, dim=-1)  # bs, search_space_size

            log_prob = F.log_softmax(logits, dim=-1)  # bs, search_space_size
            entropy = -(log_prob * probs).sum(1, keepdim=False)  # bs

            if self.training and historyaction is not None:
                firstflag = True
                historyloss = None
                for i in range(b):
                    if historyaction[i] is not None:
                        action = historyaction[i][step_i]
                        if firstflag:
                            historyloss = [self.loss_func(
                                logits[i].unsqueeze(0), action.unsqueeze(0))]
                            firstflag = False
                        else:
                            historyloss += [self.loss_func(
                                logits[i].unsqueeze(0), action.unsqueeze(0))]

            # if probs is nan or inf, replace it with 1.0 / search_space_size
            probs = replace_invalid_values(probs, self.search_spaces[step_i])

            action = probs.multinomial(num_samples=1).data  # bs, 1
            selected_log_prob = log_prob.gather(1, action)  # bs, 1

            input_action = action[:, 0] + \
                sum(self.search_spaces[:step_i])  # bs

            input_state = self.embedding(input_action,
                                         knob_vector[:, step_i +
                                                     1], pos_vector[:, step_i + 1],
                                         graph_embedding[input_action + 1])

            if len(input_state.shape) == 2:
                input_state = input_state.unsqueeze(0)

            input_states = torch.cat([input_states, input_state], dim=0)

            activations[step_i] = action[:, 0]
            entropies[step_i] = entropy
            log_probs[step_i] = selected_log_prob

        activations = torch.stack(activations).transpose(0, 1)
        if self.training:
            log_probs = torch.cat(log_probs).squeeze()
            entropies = torch.cat(entropies)
            if self.training and historyaction is not None:
                return activations, context_vector, log_probs, entropies, fusion_vector, historyloss
            else:
                return activations, context_vector, log_probs, entropies, fusion_vector
        else:
            return activations

    @torch.no_grad()
    def inference(self, context_vector, knob_vector, pos_vector, tempature=1.0,
                  historysearchstate=None):
        """_summary_

        Args:
            context_vector (torch.FloatTensor): [bs, context_size].
            knob_vector (torch.LongTensor): [bs, search_space_size].
            pos_vector (torch.LongTensor): [bs, search_space_size].
            tempature (int, optional): 10 -> 1. Defaults to None.
            historyaction (list, optional): Store the history action. Defaults to None.
            historysearchstate (torch.FloatTensor, optional): Store the history search state. Defaults to None.

        Returns:
            _type_: activations, context_vector, log_probs, entropies, fusion_vector, historyloss
        """
        b, _ = context_vector.shape
        activations = [None] * len(self.search_spaces)

        graph_embedding = self.graph_embedding()

        search_random_state = self.search_random_state.unsqueeze(
            0).expand(b, -1)
        generator_vector = torch.cat(
            [search_random_state, context_vector], dim=1)
        fusion_vector = self.context_fusion_layer(generator_vector)

        if historysearchstate is not None:
            for bi in range(b):
                if historysearchstate[bi] is not None:
                    fusion_vector[bi] = fusion_vector[bi] * \
                        (1.0 - self.history_weight) + \
                        self.history_weight * historysearchstate[bi]

        input_states = self.embedding(fusion_vector, knob_vector[:, 0], pos_vector[:, 0],
                                      graph_embedding[0].unsqueeze(0).repeat(b, 1))

        generate_orders = list(range(len(self.search_spaces)))

        if len(input_states.shape) == 2:
            input_states = input_states.unsqueeze(0)

        # Generate skipnumber and scaledownresolution
        for search_i in list(range(len(self.search_spaces))):
            predicted_hidden = self.predictor(input_states, input_states)[-1]
            step_i = generate_orders[search_i]

            si, sj = sum(self.search_spaces[:step_i]), sum(
                self.search_spaces[:step_i + 1])

            logits = self.knob_layer(predicted_hidden)[
                :, si:sj] / tempature  # bs, search_space_size
            assert sj - si == SEARCH_SPACE[step_i]
            probs = F.softmax(logits, dim=-1)  # bs, search_space_size

            # if probs is nan or inf, replace it with 1.0 / search_space_size
            # probs = replace_invalid_values(probs, self.search_spaces[step_i])

            # action = probs.multinomial(num_samples=1).data  # bs, 1
            action = torch.argmax(probs, dim=-1, keepdim=True)

            input_action = action[:, 0] + \
                sum(self.search_spaces[:step_i])  # bs

            input_state = self.embedding(input_action,
                                         knob_vector[:, step_i +
                                                     1], pos_vector[:, step_i + 1],
                                         graph_embedding[input_action + 1])

            if len(input_state.shape) == 2:
                input_state = input_state.unsqueeze(0)

            input_states = torch.cat([input_states, input_state], dim=0)

            activations[step_i] = action[:, 0]

        activations = torch.stack(activations).transpose(0, 1)
        return activations

    @torch.no_grad()
    def inference_with_mask(self, context_vector, knob_vector, pos_vector, maskinfos, tempature=1.0,
                            historysearchstate=None):
        """_summary_

        Args:
            context_vector (torch.FloatTensor): [bs, context_size].
            knob_vector (torch.LongTensor): [bs, search_space_size].
            pos_vector (torch.LongTensor): [bs, search_space_size].
            tempature (int, optional): 10 -> 1. Defaults to None.
            historyaction (list, optional): Store the history action. Defaults to None.
            historysearchstate (torch.FloatTensor, optional): Store the history search state. Defaults to None.

        Returns:
            _type_: activations, context_vector, log_probs, entropies, fusion_vector, historyloss
        """
        b, _ = context_vector.shape
        activations = [None] * len(self.search_spaces)

        graph_embedding = self.graph_embedding()

        search_random_state = self.search_random_state.unsqueeze(
            0).expand(b, -1)
        generator_vector = torch.cat(
            [search_random_state, context_vector], dim=1)
        fusion_vector = self.context_fusion_layer(generator_vector)

        if historysearchstate is not None:
            for bi in range(b):
                if historysearchstate[bi] is not None:
                    fusion_vector[bi] = fusion_vector[bi] * \
                        (1.0 - self.history_weight) + \
                        self.history_weight * historysearchstate[bi]

        input_states = self.embedding(fusion_vector, knob_vector[:, 0], pos_vector[:, 0],
                                      graph_embedding[0].unsqueeze(0).repeat(b, 1))

        generate_orders = list(range(len(self.search_spaces)))

        if len(input_states.shape) == 2:
            input_states = input_states.unsqueeze(0)

        # Generate skipnumber and scaledownresolution
        for search_i in list(range(len(self.search_spaces))):
            predicted_hidden = self.predictor(input_states, input_states)[-1]
            step_i = generate_orders[search_i]

            mask, maskvalue = False, None
            for maskinfo in maskinfos:
                maskdimension, maskvalue = maskinfo
                if maskdimension == search_i:
                    mask = True
                    break

            si, sj = sum(self.search_spaces[:step_i]), sum(
                self.search_spaces[:step_i + 1])
            logits = self.knob_layer(predicted_hidden)[
                :, si:sj] / tempature  # bs, search_space_size
            probs = F.softmax(logits, dim=-1)  # bs, search_space_size

            # if probs is nan or inf, replace it with 1.0 / search_space_size
            probs = replace_invalid_values(probs, self.search_spaces[step_i])

            # action = probs.multinomial(num_samples=1).data  # bs, 1  # bs, 1
            action = torch.argmax(probs, dim=-1, keepdim=True)

            if mask:
                action = torch.tensor([maskvalue]).to(
                    action.device).repeat(b, 1)

            input_action = action[:, 0] + \
                sum(self.search_spaces[:step_i])  # bs

            input_state = self.embedding(input_action,
                                         knob_vector[:, step_i +
                                                     1], pos_vector[:, step_i + 1],
                                         graph_embedding[input_action + 1])

            if len(input_state.shape) == 2:
                input_state = input_state.unsqueeze(0)

            input_states = torch.cat([input_states, input_state], dim=0)

            activations[step_i] = action[:, 0]

        activations = torch.stack(activations).transpose(0, 1)
        return activations


class Observer(nn.Module):
    def __init__(self, data_vector_size=512, model_vector_size=24):
        super(Observer, self).__init__()
        self.fc1 = nn.Linear(data_vector_size + model_vector_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)

    def forward(self, data_vec, model_vec):
        combined_vec = torch.cat((data_vec, model_vec), dim=-1)
        x = F.relu(self.fc1(combined_vec))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        performance = torch.sigmoid(self.fc4(x))
        return performance

# deviceid = 1
# model_device = torch.device(
#     f"cuda:{deviceid}" if torch.cuda.is_available() else "cpu")
# controller = TransController(
#     1, 32, model_device, history_weight=0.5, objects=["car", "bus", "truck"])
# controller.to(model_device)
# knob_vector = torch.LongTensor(
#     [0] + list(map(lambda x: x+1, KNOB_TYPES))).to(model_device).unsqueeze(0)
# pos_vector = torch.LongTensor(
#     [0] + list(map(lambda x: x+1, POS_TYPES))).to(model_device).unsqueeze(0)

# for _ in range(10):
#     context_vector = torch.randn(1, 512).to(model_device)
#     activations = controller.inference(
#         context_vector, knob_vector, pos_vector).squeeze().tolist()
#     print("activations: ", activations)
