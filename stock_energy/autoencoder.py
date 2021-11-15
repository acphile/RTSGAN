# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from torch.nn import functional as F
from fastNLP import seq_len_to_mask
from basic import PositionwiseFeedForward
import random

def mean_pooling(tensor, seq_len, dim=1):
    mask = seq_len_to_mask(seq_len)
    mask = mask.view(mask.size(0), mask.size(1), -1).float()
    return torch.sum(tensor * mask, dim=dim) / seq_len.unsqueeze(-1).float()

def max_pooling(tensor, seq_len, dim=1):
    mask = seq_len_to_mask(seq_len)
    mask = mask.view(mask.size(0), mask.size(1), -1)
    mask = mask.expand(-1, -1, tensor.size(2)).float()
    return torch.max(tensor + mask.le(0.5).float() * -1e9, dim=dim)

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.rnn = nn.GRU(input_dim, hidden_dim, layers, batch_first=True, dropout=dropout)
        
        self.fc = nn.Linear(hidden_dim * 3, hidden_dim)
        #self.fc1 = nn.Linear(hidden_dim * layers, hidden_dim * layers)
        self.final = nn.LeakyReLU(0.2)
        
    def forward(self, statics, dynamics, seq_len):
        bs, max_len, _ = dynamics.size()
        #x = statics.unsqueeze(1).expand(-1, max_len, -1)
        #x = torch.cat([x, dynamics], dim=-1)
        x = dynamics
        
        packed = nn.utils.rnn.pack_padded_sequence(x, seq_len, batch_first=True, enforce_sorted=False)
        out, h = self.rnn(packed)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        #h, c = h
        h1, _ = max_pooling(out, seq_len)
        h2 = mean_pooling(out, seq_len)
        h3 = h.view(self.layers, -1, bs, self.hidden_dim)[-1].view(bs, -1)
        glob = torch.cat([h1,h2,h3], dim=-1)
        glob = self.final(self.fc(glob))
        
        h3 = h.permute(1,0,2).contiguous().view(bs, -1)
        #h3 = self.final(self.fc1(h3))
        hidden = torch.cat([glob, h3], dim=-1)
        return hidden

def apply_activation(processors, x):
    data = []
    st = 0
    for model in processors.models:
        ed = model.length + st
        if model.which == 'categorical':
            if not model.missing:
                data.append(torch.softmax(x[:, st:ed], dim=-1))
            else:
                data.append(torch.softmax(x[:, st:ed-1], dim=-1))
                data.append(torch.sigmoid(x[:, ed-1:ed]))
            st = ed
        else:
            data.append(torch.sigmoid(x[:, st:ed]))
            st = ed

    return torch.cat(data, dim=-1)

class Decoder(nn.Module):
    def __init__(self, processors, hidden_dim, statics_dim, dynamics_dim, layers, dropout):
        super(Decoder, self).__init__()
        self.s_P, self.d_P = processors
        self.hidden_dim = hidden_dim
        self.dynamics_dim = dynamics_dim
        self.layers = layers
        self.rnn = nn.GRU(hidden_dim + dynamics_dim, hidden_dim, layers, batch_first=True, dropout=dropout)
        #self.statics_fc = PositionwiseFeedForward(hidden_dim, statics_dim, 
        #    d_ff=(hidden_dim+statics_dim)//2, dropout=0)
        self.dynamics_fc = nn.Linear(hidden_dim, dynamics_dim)

    def forward(self, embed, dynamics, seq_len, forcing=0.5):
        glob, hidden = embed[:, :self.hidden_dim], embed[:, self.hidden_dim:]
        #statics_x = self.statics_fc(glob)
        glob = glob.unsqueeze(1)
        bs, max_len, _ = dynamics.size()
        x = dynamics[:,0:1,:]
        hidden = hidden.view(bs, self.layers, -1).permute(1,0,2).contiguous()
        res = []
        for i in range(max_len):
            x = torch.cat([glob, x.detach()], dim=-1)
            out, hidden = self.rnn(x, hidden)
            out = apply_activation(self.d_P, self.dynamics_fc(out).squeeze(1)).unsqueeze(1)
            #out = torch.FloatTensor(self.d_P.transform(self.d_P.inverse_transform(out.cpu().numpy()))).to(embed.device)
            if random.random()>forcing:
                x = out
            else:
                x = dynamics[:,i+1:i+2,:]
            res.append(out)
        res = torch.cat(res, dim=1)
        #print(res.size(),dynamics.size())
        return None, res

    def generate_dynamics(self, embed, max_len):
        glob, hidden = embed[:, :self.hidden_dim], embed[:, self.hidden_dim:]
        glob = glob.unsqueeze(1)
        bs = glob.size(0)
        x = torch.zeros((bs, 1, self.dynamics_dim)).to(embed.device)
        hidden = hidden.view(bs, self.layers, -1).permute(1,0,2).contiguous()
        res = []
        for i in range(max_len):
            x = torch.cat([glob, x], dim=-1)
            out, hidden = self.rnn(x, hidden)
            out = apply_activation(self.d_P, self.dynamics_fc(out).squeeze(1)).detach()
            #out = torch.FloatTensor(self.d_P.transform(self.d_P.inverse_transform(out.cpu().numpy()))).to(embed.device)
            x = out.unsqueeze(1) 
            res.append(x)
        res = torch.cat(res, dim=1)
        return res.cpu().numpy()

class Autoencoder(nn.Module):
    def __init__(self, processors, hidden_dim, embed_dim, layers, dropout=0.0):
        super(Autoencoder, self).__init__()
        statics_dim, dynamics_dim = processors[0].dim, processors[1].dim
        self.encoder = Encoder(statics_dim + dynamics_dim, hidden_dim, embed_dim, layers, dropout)
        self.decoder = Decoder(processors, hidden_dim, statics_dim, dynamics_dim, layers, dropout)

    def forward(self, statics, dynamics, seq_len):
        hidden = self.encoder(statics, dynamics, seq_len)
        input_x = torch.zeros_like(dynamics[:, 0:1, :])
        input_x = torch.cat([input_x, dynamics[:, :-1, :]], dim=1)
        return self.decoder(hidden, input_x, seq_len)

