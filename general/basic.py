import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable
from fastNLP import seq_len_to_mask

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def mean_pooling(tensor, seq_len, dim=1):
    mask = seq_len_to_mask(seq_len)
    mask = mask.view(mask.size(0), mask.size(1), -1).float()
    return torch.sum(tensor * mask, dim=dim) / seq_len.unsqueeze(-1).float()

def max_pooling(tensor, seq_len, dim=1):
    mask = seq_len_to_mask(seq_len)
    mask = mask.view(mask.size(0), mask.size(1), -1)
    mask = mask.expand(-1, -1, tensor.size(2)).float()
    return torch.max(tensor + mask.le(0.5).float() * -1e9, dim=dim)
    
class TimeEncoding(nn.Module):
    def __init__(self, d_model):
        super(TimeEncoding, self).__init__()
        self.fc = nn.Linear(1, d_model)
        self.d_model = d_model

    def forward(self, x):
        x = self.fc(x)
        x = torch.cat([x[:,:,0:1], torch.sin(x[:, :, 1:])], dim=-1)
        return x

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).unsqueeze(0)
        div_term.require_grad = False
        self.register_buffer('div_term', div_term)
        
    def forward(self, x):
        bs, lens, _ = x.size()
        x = x.view(-1, 1)
        pe = torch.zeros(x.size(0), self.d_model).to(x.device)
        x = x * 100
        pe[:, 0::2] = torch.sin(x * self.div_term)
        pe[:, 1::2] = torch.cos(x * self.div_term)
        return Variable(pe.view(bs, lens, -1), requires_grad=False)
    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, input_dim, output_dim, d_ff=None, activation=F.relu, dropout=0.0):
        super(PositionwiseFeedForward, self).__init__()
        if d_ff is None:
            d_ff = output_dim * 4
        self.act = activation
        self.w_1 = nn.Linear(input_dim, d_ff)
        self.w_2 = nn.Linear(d_ff, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.act(self.w_1(x))))

def dot_attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        #print(scores.size(),mask.size())
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value)

class BahdanauAttention(nn.Module):
    def __init__(self, d_k):
        super(BahdanauAttention, self).__init__()
        self.alignment_layer = nn.Linear(d_k, 1, bias=False)
    
    def forward(self, query, key, value, mask=None):
        query = query.unsqueeze(-2)
        key = key.unsqueeze(-3)
        scores = self.alignment_layer(query + key).squeeze(-1)
        if mask is not None:
            #print(scores.size(),mask.size())
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim = -1)
        return torch.matmul(p_attn, value)

class SelfAttention(nn.Module):
    def __init__(self, d_model, h=2, dropout=0.1):
        "Take in model size and number of heads."
        super(SelfAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = BahdanauAttention(self.d_k)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
            
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x = self.attn(query, key, value, mask=mask)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.dropout(self.linears[-1](x))
