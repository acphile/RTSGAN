# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from aegan import AeGAN
from fastNLP import DataSet, DataSetIter, RandomSampler, SequentialSampler
from sklearn.metrics import f1_score

orders = ['Hours', 'Capillary refill rate', 'Diastolic blood pressure',
       'Fraction inspired oxygen', 'Glascow coma scale eye opening',
       'Glascow coma scale motor response', 'Glascow coma scale total',
       'Glascow coma scale verbal response', 'Glucose', 'Heart Rate', 'Height',
       'Mean blood pressure', 'Oxygen saturation', 'Respiratory rate',
       'Systolic blood pressure', 'Temperature', 'Weight', 'pH']

class Inhospital(AeGAN):
    def synthesize(self, n, batch_size=500):
        self.ae.decoder.eval()
        self.generator.eval()
        sta = []
        def _gen(n):
            with torch.no_grad():
                z = torch.randn(n, self.params['noise_dim']).to(self.device)
                hidden =self.generator(z)
                statics = self.ae.decoder.generate_statics(hidden)
                df_sta = self.static_processor.inverse_transform(statics.cpu().numpy())
                max_len = int(df_sta['seq_len'].max())
                dynamics, missing, times = self.ae.decoder.generate_dynamics(hidden, statics, max_len)
                dynamics = dynamics.cpu().numpy()
                missing = missing.cpu().numpy()
                times = times.cpu().numpy()
                
            sta.append(df_sta)
            res = []
            for i in range(n):
                length = int(df_sta['seq_len'].values[i])
                dyn = self.dynamic_processor.inverse_transform(dynamics[i,:length], missing[i,:length], times[i,:length])
                for x in ['Glascow coma scale total']:
                    dyn[x] = np.array([y if y!=y else int(round(y)) for y in dyn[x].values.astype('float')], dtype=object)
                #scale = 48 / dyn["Hours"].max()
                #dyn["Hours"] = dyn["Hours"] * scale
                if "Height" in df_sta.columns:
                    h = float(df_sta.loc[i, "Height"])
                    dyn['Height'] = np.array([float("nan")]*len(dyn), dtype=object)
                    j = 0
                    for k in range(len(dyn)):
                        x = float(dyn.loc[k, "Weight"])
                        if x==x:
                            j = k
                            break
                    dyn.loc[j, "Height"] = h
                    dyn = dyn[orders]

                res.append(dyn)

            return res

        data = []
        tt = n // batch_size
        for i in range(tt):
            data.extend(_gen(batch_size))
        res = n - tt * batch_size
        if res>0:
            data.extend(_gen(res))
        sta = pd.concat(sta, ignore_index=True)
        sta.drop(columns=["seq_len"], inplace=True)
        print(data[0])
        return sta, data
    
    def generate_ae(self, dataset):
        batch_size = self.params["gan_batch_size"]
        batch = DataSetIter(dataset=dataset, batch_size=batch_size, sampler=SequentialSampler())
        loss1 = 0
        loss2 = 0
        loss3 = 0 
        tt = 0
        self.ae.eval()
        sta_lis = []
        res = []
        acc = 0
        sl=0
        s=0
        pr=[]
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
                
                df_sta = self.static_processor.inverse_transform(statics.cpu().numpy())
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
                
                sta_lis.append(df_sta)
                for i in range(len(sta)):
                    length = int(df_sta['seq_len'].values[i])
                    dyn = self.dynamic_processor.inverse_transform(dynamics[i,:length], missing[i,:length], times[i,:length])
                    for x in ['Glascow coma scale total']:
                        dyn[x] = np.array([y if y!=y else int(round(y)) for y in dyn[x].values.astype('float')], dtype=object)

                    if "Height" in df_sta.columns:
                        h = float(df_sta.loc[i, "Height"])
                        dyn['Height'] = np.array([float("nan")]*len(dyn), dtype=object)
                        j = 0
                        for k in range(len(dyn)):
                            x = float(dyn.loc[k, "Weight"])
                            if x==x:
                                j = k
                                break
                        dyn.loc[j, "Height"] = h
                        dyn = dyn[orders]
                        
                    res.append(dyn)
                    
        #self.logger.info(acc/sl)
        #self.logger.info(s/sl)
        pr = np.concatenate(pr, axis=0)
        gold = np.concatenate(gold, axis=0)
        self.logger.info(f1_score(gold, pr, average="micro"))
        self.logger.info(f1_score(gold, pr, average="macro"))
        self.logger.info(loss1.item()/tt)
        self.logger.info(loss2.item()/tt)
        self.logger.info(loss3.item()/tt)
        
        sta = pd.concat(sta_lis, ignore_index=True)
        sta.drop(columns=["seq_len"], inplace=True)
        if len(res)>100:
            dyna = pd.concat(res)
            for col in dyna.columns:
                if col not in ["Glascow coma scale eye opening","Glascow coma scale motor response", "Glascow coma scale verbal response"]:
                    self.logger.info("{}\t{}\t{}".format(col, dyna[col].mean(), dyna[col].std()))
        return sta, res
