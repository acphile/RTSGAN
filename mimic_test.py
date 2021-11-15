# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import pickle
import collections
import logging
import math
import os,sys,time
import random
import sys
sys.path.append('./general/')
import pickle
import numpy as np
import torch
import torch.nn as nn
from utils.general import init_logger, make_sure_path_exists
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from fastNLP import DataSet, DataSetIter, RandomSampler, SequentialSampler
import torch
import torch.nn as nn
from torch.nn import functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--task-name", default='', dest="task_name",
                    help="Name for this task, use a comprehensive one")   
parser.add_argument("--impute", default='zero', dest="imputation method")   
parser.add_argument("--devi", default="0", dest="devi", help="gpu")
parser.add_argument("--skip-dev", dest="skip_dev", action="store_true", help="Skip dev set during training")
options = parser.parse_args()

root_dir = os.path.join(options.task_name, "test_info")
make_sure_path_exists(root_dir)

logger = init_logger(root_dir)
# ===-----------------------------------------------------------------------===
# Log some stuff about this run
# ===-----------------------------------------------------------------------===
logger.info(' '.join(sys.argv))
logger.info('')
logger.info(options)

devices=[int(x) for x in options.devi]
device = torch.device("cuda:{}".format(devices[0]))  

ori = pickle.load(open("./data/inhospital/fullhos.pkl", "rb"))
d_P = ori["dynamic_processor"]
if options.task_name == '':
    train_set = pickle.load(open("./data/inhospital/train_clamp.pkl", "rb"))
else:
    print(options.task_name)
    train_set = pickle.load(open("{}/mimic.pkl".format(options.task_name), "rb"))

test_set=pickle.load(open("./data/inhospital/test_clamp.pkl", "rb"))

print(max(test_set["seq_len"]))
dev_set = test_set
if options.skip_dev == False:
    train_set, dev_set=train_set.split(0.1)
    dev_set.set_input("dyn", "mask", "label", "times", "lag", "seq_len", "priv", "nex")

class CLS(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, dropout):
        super(CLS, self).__init__()
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.rnn = nn.GRU(input_dim, hidden_dim, layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, dynamics, lag, mask, priv, nex, times, seq_len):
        bs, max_len, _ = dynamics.size()
        if options.impute =='zero':
            x = torch.cat([dynamics, mask, times], dim=-1)
        elif options.impute =='last':
            i = 0
            st = 0 
            d = []
            for model in d_P.models:
                if model.missing:
                    d.append(dynamics[:,:,st:st+model.tgt_len] + (1 - mask[:,:,i:i+1]) * priv[:,:,st:st+model.tgt_len])
                    i+=1
                else:
                    d.append(dynamics[:,:,st:st+model.tgt_len])
                st+=model.tgt_len
            d = torch.cat(d, dim=-1)    
            l = lag * (1 - mask)
            x = torch.cat([d, l, times], dim=-1)
        else:
            x = torch.cat([dynamics, mask, priv, lag, times], dim=-1)
            
        packed = nn.utils.rnn.pack_padded_sequence(x, seq_len, batch_first=True, enforce_sorted=False)
        out, h = self.rnn(packed)
        
        h3 = h.view(self.layers, -1, bs, self.hidden_dim)[-1].view(bs, -1)
        out = self.fc(h3)
        return torch.sigmoid(out)

if options.impute in ['zero', 'last']:
    model = CLS(d_P.tgt_dim + 1 + d_P.miss_dim, 16, 2, 0.3)
else:
    model = CLS(d_P.tgt_dim * 2 + 1 + d_P.miss_dim * 2, 16, 2, 0.3)
    
model = model.to(device)
optm = torch.optim.Adam(params=model.parameters(), lr=1e-3, betas=(0.9, 0.999))
loss_f = nn.BCELoss()

train_batch=DataSetIter(dataset=train_set, batch_size=128, sampler=RandomSampler())
dev_batch=DataSetIter(dataset=dev_set, batch_size=256, sampler=SequentialSampler())
test_batch=DataSetIter(dataset=test_set, batch_size=256, sampler=SequentialSampler())
best_acc = 0
best_auc = 0
epochs = 30
logger.info("training:{}".format(len(train_batch)))

def evaluate(dev_batch):
    model.eval()
    prob = []
    label = [] 
    for batch_x, batch_y in dev_batch:
        with torch.no_grad():
            target = batch_x["label"].to(device).float()
            dyn = batch_x["dyn"].to(device)
            mask = batch_x["mask"].to(device)
            lag = batch_x["lag"].to(device)
            priv = batch_x["priv"].to(device)
            nex = batch_x["nex"].to(device)
            times = batch_x["times"].to(device)
            seq_len = batch_x["seq_len"].to(device)
            
            bs, length, dim = dyn.size()
            
            out = model(dyn, lag, mask, priv, nex, times, seq_len)
            prob.extend(out.cpu().numpy().reshape(-1).tolist())
            label.extend(target.cpu().numpy().reshape(-1).tolist())
    prob = np.array(prob)
    preds = (prob > 0.5).astype('int')
    label = np.array(label,dtype=int)
    auc = roc_auc_score(label, prob)
    f1 = f1_score(label, preds)
    acc = accuracy_score(label, preds) 
    logger.info("{}\t{}\t{}".format(auc,f1,acc))
    return auc
               
for i in range(epochs):
    model.train()
    tot = 0
    tot_loss = 0
    t1 = time.time()
    for batch_x, batch_y in train_batch:
        model.zero_grad()
        target = batch_x["label"].to(device).float()
        dyn = batch_x["dyn"].to(device)
        mask = batch_x["mask"].to(device)
        lag = batch_x["lag"].to(device)
        priv = batch_x["priv"].to(device)
        nex = batch_x["nex"].to(device)
        times = batch_x["times"].to(device)
        seq_len = batch_x["seq_len"].to(device)
        
        bs, length, dim = dyn.size()
        
        out = model(dyn, lag, mask, priv, nex, times, seq_len)
        loss = loss_f(out, target)
        loss.backward()
        optm.step()

        tot_loss += loss.item()
        tot += 1
   
    logger.info("Epoch:{} {}\t{}".format(i+1, time.time()-t1, tot_loss/tot))
    if options.skip_dev != True:
        auc = evaluate(dev_batch)
    auc = evaluate(test_batch)
    if auc>best_auc:
        best_auc = auc
        
logger.info("Best auc: {}".format(best_auc))
