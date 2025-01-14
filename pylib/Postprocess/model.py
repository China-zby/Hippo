import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class PSRNN(nn.Module):
    def __init__(self, scene2id):
        super().__init__()
        self.rnnmodel = nn.LSTM(
            input_size=5,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.scene_embedding_h = nn.Embedding(len(scene2id.keys()), 32)
        self.scene_embedding_c = nn.Embedding(len(scene2id.keys()), 32)

        self.prefix_header = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
            nn.Sigmoid()
        )
        self.suffix_header = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
            nn.Sigmoid()
        )

    # just need the final hidden state of rnn
    def forward(self, x, lengths, sid):
        bs, sl, hs = x.shape
        h = self.scene_embedding_h(sid).squeeze()  # (batch_size, 32)
        h = h.unsqueeze(0).repeat(2, 1, 1)

        c = self.scene_embedding_c(sid).squeeze()  # (batch_size, 32)
        c = c.unsqueeze(0).repeat(2, 1, 1)

        x = x.transpose(0, 1)
        lengths, perm_idx = lengths.sort(0, descending=True)
        x = x[:, perm_idx]
        x = pack_padded_sequence(x, lengths)
        x, _ = self.rnnmodel(x, (h, c))
        x, _ = pad_packed_sequence(x)
        x = x[lengths - 1, torch.arange(x.size(1)), :]
        prefix = self.prefix_header(x)
        suffix = self.suffix_header(x)
        # remap to original order
        perm_idx = torch.argsort(perm_idx)
        prefix = prefix[perm_idx]
        suffix = suffix[perm_idx]
        return prefix, suffix

    @torch.no_grad()
    def inference(self, x, sid):
        bs, sl, hs = x.shape
        h = self.scene_embedding_h(sid)
        h = h.unsqueeze(0).repeat(2, 1, 1)
        c = self.scene_embedding_c(sid)
        c = c.unsqueeze(0).repeat(2, 1, 1)
        x, _ = self.rnnmodel(x, (h, c))
        x = x[:, -1, :]
        prefix = self.prefix_header(x)
        suffix = self.suffix_header(x)
        return prefix, suffix
