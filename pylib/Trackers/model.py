import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool


class InteractionNetworkLayer(MessagePassing):
    def __init__(self, node_in_channels, edge_in_channels, node_out_channels):
        super(InteractionNetworkLayer, self).__init__(aggr='add')
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_in_channels +
                      edge_in_channels, edge_in_channels),
            nn.ReLU(),
            nn.Linear(edge_in_channels, node_out_channels)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(node_in_channels + node_out_channels, node_out_channels),
            nn.ReLU(),
            nn.Linear(node_out_channels, node_out_channels)
        )

    def forward(self, x, edge_index, edge_attr):
        edge_messages = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        node_input = torch.cat([x, edge_messages], dim=1)
        return self.node_mlp(node_input)

    def message(self, x_i, x_j, edge_attr):
        edge_input = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.edge_mlp(edge_input)


class MOTGraphModel(nn.Module):
    BATCH_SIZE = 4

    def __init__(self, num_node_features, num_edge_features):
        super(MOTGraphModel, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        self.node_update = nn.Sequential(
            nn.Linear(num_node_features + 64, 64),
            nn.ReLU(),
            nn.LayerNorm(64))

        self.edge_update = nn.Sequential(
            nn.Linear(num_edge_features, 64),
            nn.ReLU(),
            nn.LayerNorm(64))

        self.interaction_networks = nn.ModuleList([
            InteractionNetworkLayer(64, 64, 64),
            InteractionNetworkLayer(64, 64, 64),
            InteractionNetworkLayer(64, 64, 64),
            InteractionNetworkLayer(64, 64, 64)])

        self.edge_mlp = nn.Sequential(
            nn.Linear(64 * 2 + num_edge_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, graph, crop):
        x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr
        crop_feature = self.feature_extractor(crop)
        node_feature = torch.cat([x, crop_feature], dim=1)

        node_feature = self.node_update(node_feature)
        edge_feature = self.edge_update(edge_attr)

        for interaction_network in self.interaction_networks:
            node_feature = interaction_network(
                node_feature, edge_index, edge_feature)

        edge_features = torch.cat(
            [node_feature[edge_index[0]], node_feature[edge_index[1]], edge_attr], dim=1)
        edge_scores = self.edge_mlp(edge_features)
        return edge_scores

    def loss(self, pred, target):
        return F.binary_cross_entropy_with_logits(pred, target)


class RNNModel(nn.Module):
    model_size = 1
    input_sizes = [64 + 10, int(128 * model_size),
                   int(128 * model_size), int(128 * model_size)]
    output_sizes = [int(128 * model_size), int(128 * model_size),
                    int(128 * model_size), 128]

    def run_step(self, prev_state, curr_input):
        inputs = torch.cat([prev_state, curr_input], dim=1)
        for i in range(len(self.input_sizes)):
            inputs = self._fc_layers[i](inputs)
            if i < len(self.input_sizes) - 1:
                inputs = F.relu(inputs)
        return inputs[:, 0:64], inputs[:, 64:128]

    def __init__(self):
        super(RNNModel, self).__init__()
        self._fc_layers = nn.ModuleList()
        for i in range(len(self.input_sizes)):
            self._fc_layer_linear = nn.Linear(
                self.input_sizes[i], self.output_sizes[i])
            self._fc_layers.append(self._fc_layer_linear)
        self.scene_embedding = nn.Embedding(7, 64)
        self.scene_fusion = nn.Linear(128, 64)

        self.output_layer_1 = nn.Linear(
            64 + 10 + 5, int(64 * RNNModel.model_size))
        self.output_layer_2 = nn.Linear(
            int(64 * RNNModel.model_size), int(64 * RNNModel.model_size))
        self.output_layer_3 = nn.Linear(
            int(64 * RNNModel.model_size), int(64 * RNNModel.model_size))
        self.output_layer_4 = nn.Linear(int(64 * RNNModel.model_size), 1)

        # self.init_weights()

    def init_weights(self):
        for i in range(len(self.input_sizes)):
            nn.init.trunc_normal_(
                self._fc_layers[i].weight, std=math.sqrt(2.0 / self.input_sizes[i]))
            nn.init.zeros_(self._fc_layers[i].bias)
        nn.init.trunc_normal_(self.scene_embedding.weight,
                              std=math.sqrt(2.0 / (7 + 64)))
        nn.init.trunc_normal_(self.scene_fusion.weight,
                              std=math.sqrt(2.0 / (64 + 64)))
        nn.init.zeros_(self.scene_fusion.bias)
        nn.init.trunc_normal_(self.output_layer_1.weight,
                              std=math.sqrt(2.0 / (64 + 10 + 5)))
        nn.init.zeros_(self.output_layer_1.bias)
        nn.init.trunc_normal_(self.output_layer_2.weight,
                              std=math.sqrt(2.0 / 64))
        nn.init.zeros_(self.output_layer_2.bias)
        nn.init.trunc_normal_(self.output_layer_3.weight,
                              std=math.sqrt(2.0 / 64))
        nn.init.zeros_(self.output_layer_3.bias)
        nn.init.trunc_normal_(self.output_layer_4.weight,
                              std=math.sqrt(2.0 / 64))
        nn.init.zeros_(self.output_layer_4.bias)

    def forward(self,
                inputs,
                boxes,
                mask,
                targets,
                SCENE=None,
                MAX_LENGTH=None,
                BATCH_SIZE=None,
                NUM_BOXES=None):
        if SCENE is not None:
            scene_embedding = self.scene_embedding(SCENE)

        rnn_outputs = []
        for step_i in range(MAX_LENGTH):
            if step_i == 0:
                prev_state = torch.zeros(BATCH_SIZE, 64).cuda()

            if SCENE is not None:
                prev_state = torch.cat([prev_state, scene_embedding], dim=1)
                prev_state = self.scene_fusion(prev_state)

            prev_state, cur_outputs = self.run_step(
                prev_state, inputs[:, step_i, :])
            rnn_outputs.append(cur_outputs)
        rnn_outputs = torch.stack(rnn_outputs, dim=1)

        features = torch.cat([rnn_outputs, inputs], dim=2)
        features = features.unsqueeze(2)
        features_tiled = features.repeat(1, 1, NUM_BOXES, 1)
        flat_features_tiled = features_tiled.view(
            BATCH_SIZE * MAX_LENGTH * NUM_BOXES, self.input_sizes[0])
        flat_boxes = boxes.view(BATCH_SIZE * MAX_LENGTH * NUM_BOXES, 5)
        fc_cat = torch.cat([flat_features_tiled, flat_boxes], dim=1)

        fc1 = F.relu(self.output_layer_1(fc_cat))
        fc2 = F.relu(self.output_layer_2(fc1))
        fc3 = F.relu(self.output_layer_3(fc2))
        fc4 = self.output_layer_4(fc3)

        pre_outputs = fc4.view(BATCH_SIZE, MAX_LENGTH, NUM_BOXES)
        pre_outputs = torch.cat([-6 * torch.ones(BATCH_SIZE, MAX_LENGTH, 1).cuda(),
                                 pre_outputs[:, :, 1:]], dim=2)

        targets = torch.argmax(targets, dim=2)

        loss = F.cross_entropy(pre_outputs.view(-1, NUM_BOXES),
                               targets.view(-1), reduction="none").squeeze().view(BATCH_SIZE, MAX_LENGTH) * mask
        loss = torch.sum(loss, dim=1) / torch.sum(mask, dim=1)
        loss = torch.mean(loss)
        return loss, pre_outputs

    @torch.no_grad()
    def predict(self, inputs, states, boxes, SCENE=None):
        num_inputs = inputs.shape[0]
        num_bboxes = boxes.shape[0]
        if SCENE is not None:
            scene_embedding = self.scene_embedding(SCENE)
            states = torch.cat([states, scene_embedding], dim=1)
            states = self.scene_fusion(states)
        out_states, rnn_outputs = self.run_step(states, inputs)
        features = torch.cat([rnn_outputs, inputs], dim=1)
        features = features.unsqueeze(1)
        features_tiled = features.repeat(1, num_bboxes, 1)
        flat_features_tiled = features_tiled.view(
            num_inputs * num_bboxes, self.input_sizes[0])
        flat_boxes = boxes.unsqueeze(0).repeat(
            num_inputs, 1, 1).view(num_inputs * num_bboxes, 5)
        fc_cat = torch.cat([flat_features_tiled, flat_boxes], dim=1)
        fc1 = F.relu(self.output_layer_1(fc_cat))
        fc2 = F.relu(self.output_layer_2(fc1))
        fc3 = F.relu(self.output_layer_3(fc2))
        fc4 = self.output_layer_4(fc3)
        pre_outputs = fc4.view(num_inputs, num_bboxes)
        pre_outputs = torch.cat([pre_outputs[:, 0:num_bboxes-1],
                                 -6 * torch.ones(num_inputs, 1).cuda()], dim=1)
        outputs = torch.softmax(pre_outputs, dim=1)

        return outputs, out_states
