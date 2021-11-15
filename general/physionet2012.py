# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from aegan import AeGAN
from fastNLP import DataSet, DataSetIter, RandomSampler, SequentialSampler
from sklearn.metrics import f1_score, recall_score

class Physio2012(AeGAN):
    def synthesize(self, n, batch_size=500):
        self.ae.decoder.eval()
        self.generator.eval()
        sta = []
        seq_len = []
        dyn = []
        def _gen(n):
            with torch.no_grad():
                z = torch.randn(n, self.params['noise_dim']).to(self.device)
                hidden =self.generator(z)
                statics = self.ae.decoder.generate_statics(hidden)
                df_sta = self.static_processor.inverse_transform(statics.cpu().numpy())
                max_len = int(df_sta['seq_len'].max())
                sta.append(df_sta)
                cur_len = df_sta['seq_len'].values.astype(int).tolist()
                dynamics, missing, times = self.ae.decoder.generate_dynamics(hidden, statics, max_len)
                dynamics = dynamics.cpu().numpy()
                missing = missing.cpu().numpy()
                times = times.cpu().numpy()
                for i, length in enumerate(cur_len):
                    d = self.dynamic_processor.inverse_transform(dynamics[i,:length], missing[i,:length], times[i,:length])
                    dyn.append(d)

        tt = n // batch_size
        for i in range(tt):
            _gen(batch_size)
        res = n - tt * batch_size
        if res>0:
            _gen(res)
        sta = pd.concat(sta)
        assert len(sta)==len(dyn)
        print(sta[0:1])
        print(dyn[0])
        return (sta, dyn)
    
    def generate_ae(self, dataset):
        batch_size = self.params["gan_batch_size"]
        batch = DataSetIter(dataset=dataset, batch_size=batch_size, sampler=SequentialSampler())
        loss1 = 0
        loss2 = 0
        loss3 = 0
        tt = 0
        self.ae.eval()
        sta_lis = []
        dyn_lis = []
        acc = 0
        sl=0
        pr=[]
        s = 0
        gold=[]
        with torch.no_grad():
            for batch_x, batch_y in batch:        
                sta = batch_x["sta"].to(self.device)
                dyn = batch_x["dyn"].to(self.device)
                mask = batch_x["mask"].to(self.device)
                lag = batch_x["lag"].to(self.device)
                priv = batch_x["priv"].to(self.device)
                nex = batch_x["nex"].to(self.device)
                times = batch_x["times"].to(self.device)
                seq_len = batch_x["seq_len"].to(self.device)
                
                hidden = self.ae.encoder(sta, dyn, priv, nex, mask, times, seq_len)
                statics = self.ae.decoder.generate_statics(hidden)
                max_len = dyn.size(1)
                dynamics, missing, gt = self.ae.decoder.generate_dynamics(hidden, sta, max_len)
                loss1 += self.dyn_loss(dynamics, dyn, seq_len, mask)
                loss2 += self.missing_loss(missing, mask, seq_len)
                loss3 += self.time_loss(gt, times, seq_len)
                tt+=1
                
                thr = torch.Tensor([model.threshold for model in self.dynamic_processor.models if model.missing]).to(self.device)
                thr = thr.unsqueeze(0)
                for i in range(len(sta)):
                    length = int(seq_len[i].cpu())
                    pred = (missing[i, :length] > thr).float()
                    s+= torch.sum(pred, dim=0)
                    acc += torch.sum((pred==mask[i, :length]).float(), dim=0)
                    pr.append(pred.cpu().numpy())
                    gold.append(mask[i,:length].cpu().numpy())
                    sl +=length
                dynamics = dynamics.cpu().numpy()
                missing = missing.cpu().numpy()
                times = times.cpu().numpy()
                df_sta = self.static_processor.inverse_transform(statics.cpu().numpy())
                sta_lis.append(df_sta)
                for i, length in enumerate(seq_len.cpu().numpy()):
                    d = self.dynamic_processor.inverse_transform(dynamics[i,:length], missing[i,:length], times[i,:length])
                    dyn_lis.append(d)
        
        sta_lis = pd.concat(sta_lis)
        #self.logger.info((acc/sl).cpu().numpy())
        #self.logger.info((s/sl).cpu().numpy())
        pr = np.concatenate(pr, axis=0)
        gold = np.concatenate(gold, axis=0)
        self.logger.info(f1_score(gold, pr, average="micro"))
        self.logger.info(f1_score(gold, pr, average="macro"))
        #self.logger.info(recall_score(gold, pr, average=None))
        #self.logger.info(f1_score(gold, pr, average=None))
        self.logger.info(loss1.item()/tt)
        self.logger.info(loss2.item()/tt)
        self.logger.info(loss3.item()/tt)
        return sta_lis, dyn_lis