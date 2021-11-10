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
parser.add_argument("--task-name", default='../physio_data', dest="task_name",
                    help="Name for this task, use a comprehensive one")   
parser.add_argument("--impute", default='zero', dest="impute")   
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

rawset = pickle.load(open("./data/physio_data/full2012.pkl", "rb"))
if options.task_name == './data/physio_data':
    train_set=rawset["raw_set"]    
else:
    dataset = pickle.load(open(options.task_name+"/2012.pkl", "rb"))
    train_set=dataset["train_set"]
test_set=dataset["test_set"]
d_P=rawset["dynamic_processor"]
s_P=rawset["static_processor"]
assert s_P.models[5].name =='Label'
assert s_P.models[5].which =='binary'
assert s_P.models[6].name =='seq_len'
s_dim = sum([model.tgt_len for model in s_P.models[:5]])
print(s_dim)
def gen_dataset(raw):
    sta, dyn = raw
    s = s_P.transform(sta)
    seq_len = [len(x) for x in dyn]
    d_lis=[d_P.transform(ds) for ds in dyn]
    d = [x[0].tolist() for x in d_lis]
    lag = [x[1].tolist() for x in d_lis]
    mask = [x[2].tolist() for x in d_lis]
    times = [x[-1].tolist() for x in d_lis]
    priv = [x[3].tolist() for x in d_lis]
    nex = [x[4].tolist() for x in d_lis]

    dataset = DataSet({"seq_len": seq_len, 
                       "dyn": d, "lag":lag, "mask": mask,
                       "sta": s, "times":times, "priv":priv, "nex":nex
                      })
    return dataset

train_set = gen_dataset(train_set)
test_set = gen_dataset(test_set)

print(max(test_set["seq_len"]))

class CLS(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, dropout):
        super(CLS, self).__init__()
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.rnn = nn.GRU(input_dim, hidden_dim, layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, statics, dynamics, lag, mask, priv, nex, times, seq_len):
        bs, max_len, _ = dynamics.size()
        x = statics.unsqueeze(1).expand(-1, max_len, -1)
        if options.impute =='zero':
            x = torch.cat([x, dynamics, mask, times], dim=-1)
        elif options.impute =='last':
            d = dynamics + (1 - mask) * priv
            l = lag * (1 - mask)
            x = torch.cat([x, d, l, times], dim=-1)
        else:
            x = torch.cat([x, dynamics, mask, priv, lag, times], dim=-1)
            
        packed = nn.utils.rnn.pack_padded_sequence(x, seq_len, batch_first=True, enforce_sorted=False)
        out, h = self.rnn(packed)
        
        h3 = h.view(self.layers, -1, bs, self.hidden_dim)[-1].view(bs, -1)
        out = self.fc(h3)
        return torch.sigmoid(out)

best = []
last = []
for _ in range(4):
    dev_set = test_set
    if options.skip_dev == False:
        train_set, dev_set=train_set.split(0.1)
        dev_set.set_input("dyn", "mask", "sta", "times", "lag", "seq_len","priv", "nex")

    train_set.set_input("dyn", "mask", "sta", "times", "lag", "seq_len", "priv", "nex")
    test_set.set_input("dyn", "mask", "sta", "times", "lag", "seq_len","priv", "nex")
    if options.impute in ['zero', 'last']:
        model = CLS(s_dim + d_P.tgt_dim + 1 + d_P.miss_dim, 32, 2, 0.1)
    else:
        model = CLS(s_dim + d_P.tgt_dim * 3 + 1 + d_P.miss_dim, 32, 2, 0.1)

    model = model.to(device)
    optm = torch.optim.Adam(params=model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    loss_f = nn.BCELoss()

    train_batch=DataSetIter(dataset=train_set, batch_size=256, sampler=RandomSampler())
    dev_batch=DataSetIter(dataset=dev_set, batch_size=256, sampler=SequentialSampler())
    test_batch=DataSetIter(dataset=test_set, batch_size=256, sampler=SequentialSampler())
    best_acc = 0
    best_auc = 0
    epochs = 40
    logger.info("training:{}".format(len(train_batch)))

    def evaluate(dev_batch):
        model.eval()
        prob = []
        label = [] 
        for batch_x, batch_y in dev_batch:
            with torch.no_grad():
                sta = batch_x["sta"].to(device)
                dyn = batch_x["dyn"].to(device)
                mask = batch_x["mask"].to(device)
                lag = batch_x["lag"].to(device)
                priv = batch_x["priv"].to(device)
                nex = batch_x["nex"].to(device)
                times = batch_x["times"].to(device)
                seq_len = batch_x["seq_len"].to(device)

                bs, length, dim = dyn.size()
                target = sta[:, s_dim:s_dim+1]
                sta = sta[:, :s_dim]

                out = model(sta, dyn, lag, mask, priv, nex, times, seq_len)
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
            sta = batch_x["sta"].to(device)
            dyn = batch_x["dyn"].to(device)
            mask = batch_x["mask"].to(device)
            lag = batch_x["lag"].to(device)
            priv = batch_x["priv"].to(device)
            nex = batch_x["nex"].to(device)
            times = batch_x["times"].to(device)
            seq_len = batch_x["seq_len"].to(device)

            bs, length, dim = dyn.size()
            target = sta[:, s_dim:s_dim+1]
            sta = sta[:, :s_dim]

            out = model(sta, dyn, lag, mask, priv, nex, times, seq_len)
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
    best.append(str(best_auc))
    last.append(str(auc))

print("\t".join(best))
print("\t".join(last))