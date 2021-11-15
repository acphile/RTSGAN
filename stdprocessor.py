# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelBinarizer
import numpy as np
import pandas as pd

class MissingProcessor:
    def __init__(self, threshold=None, which="continuous", name=None, use_pri=None):
        self.threshold = threshold
        self.tgt_len = 0
        if which in ["binary", "categorical"]:
            self.model = LabelBinarizer()
        else: 
            self.model = StandardScaler()
        self.which = which
        self.missing = False
        self.non = None
        self.name = name
        self.use_pri = use_pri
        
    def fit(self, data):
        self.tgt_len = 0
        self.dtype = data.dtype
        if data.dtype == object:
            loc = np.array([x!=x for x in data])
            self.non = data[~loc][0]
        else:
            loc = np.isnan(data)
        if loc.any():
            self.missing = True
            self.tgt_len = int(self.use_pri is None)
            if self.threshold is None:
                self.threshold = 1.0 * sum(loc) / len(data)
            #if self.threshold < 0.2:      
            #    self.missing_regression = False
            if loc.all():
                res = self.model.fit_transform(np.zeros((1,1)))
            else:
                res = self.model.fit_transform(data[~loc].reshape(-1,1))
        else:
            res = self.model.fit_transform(data.reshape(-1,1))
        self.tgt_len += res.shape[1]

    def transform(self, data, times=None):
        if self.missing:
            tgt = np.ones((len(data), self.tgt_len))
            if self.dtype == object:
                loc = np.array([x!=x for x in data])
                filldata = np.array([self.non if x!=x else x for x in data]) 
            else:
                loc = np.isnan(data)
                filldata = np.nan_to_num(data)
                
            filldata = filldata.astype(self.dtype)
            trans = self.model.transform(filldata.reshape(-1,1))
            fea_len = trans.shape[1]
            tgt[:, :fea_len] = trans
            tgt[loc] = 0
                
            if times is not None:   
                lag = np.zeros(len(data))
                lag[0] = times[0]
                priv = np.zeros((len(data), fea_len))
                nex = np.zeros((len(data), fea_len))
                for i in range(1, len(data)):
                    if loc[i-1]:
                        #lag[i] = lag[i-1] + times[i]
                        lag[i] = lag[i-1] + times[i] - times[i-1]
                        priv[i] = priv[i-1]
                    else:
                        #lag[i] = times[i]
                        lag[i] = times[i] - times[i-1]
                        priv[i] = tgt[i-1]
                for i in range(len(data)-2, 0, -1):
                    if loc[i+1]:
                        nex[i] = nex[i+1]
                    else:
                        nex[i] = tgt[i+1]
                return tgt.astype("float32"), lag.reshape(-1,1), (1-loc).reshape(-1,1).astype('float'), priv, nex
            else:
                return tgt.astype("float32")
        else:
            trans = self.model.transform(data.reshape(-1,1))
            return trans

    def inverse_transform(self, data, miss=None):
        res = self.model.inverse_transform(data)
        if "int" in self.which:
            res =res.round()
        res = res.astype(self.dtype)
        if miss is not None and self.missing:
            loc = miss > self.threshold
            res[~loc] = np.nan

        return res.reshape(-1)

class StdProcessor:
    def __init__(self, types, use_pri=None):
        self.names = []
        self.models = []
        self.types = types
        self.miss_dim = 0
        self.tgt_dim = 0 
        self.use_pri = use_pri
        
    def fit(self, data):
        self.names = []
        self.models = []
        self.miss_dim = 0
        self.tgt_dim = 0 
        matrix = data.values
        for i, (types, col) in enumerate(zip(self.types,data.columns)):
            value = matrix[:, i]
            self.names.append(col)
            if types == 'continuous':
                if all([float(x).is_integer() for x in data[col].unique() if float(x) == float(x)]):
                    print("all values are integer")
                    types = 'int'
            model = MissingProcessor(which=types,name=col, use_pri=self.use_pri)
            if col == self.use_pri:
                tmp = value.tolist()
                tmp.append(0)
                value = np.array(tmp)
            model.fit(value)
            self.models.append(model)
            print(model.name, model.tgt_len, model.threshold, model.which, model.non)
            #if types in ['continuous', 'int']:
            #    print(model.model.data_max_, model.model.data_min_)
            #elif types == 'categorical':
            #    print(model.model.classes_)
            if col != self.use_pri:
                self.miss_dim += model.missing
                self.tgt_dim += model.tgt_len
            
    def transform(self, data):
        tgt = []
        lag = []
        mask = []
        priv = []
        nex = []
        matrix = data.values
        times = None
        if self.use_pri:
            for i, col in enumerate(data.columns):
                value = matrix[:, i]
                if col == self.use_pri:
                    model = self.models[i].model
                    #assert isinstance(model, MinMaxScaler)
                    times = model.transform(value.reshape(-1,1)).reshape(-1)
                    break
            assert times is not None
        for i, col in enumerate(data.columns):
            if col == self.use_pri: continue
            value = matrix[:, i]
            x = self.models[i].transform(value, times)
            if times is not None:
                tgt.append(x[0])
                if self.models[i].missing:
                    _,l,m,p,nx = x
                    mask.append(m)
                    lag.append(l)
                    priv.append(p)
                    nex.append(nx)
            else:
                tgt.append(x)  
                             
        tgt = np.concatenate(tgt, axis=1)
        if times is None:
            return tgt                       
        lag = np.concatenate(lag, axis=1)                       
        mask = np.concatenate(mask, axis=1)
        priv = np.concatenate(priv, axis=1)
        nex = np.concatenate(nex, axis=1)
        return tgt, lag, mask, priv, nex, times.reshape(-1,1)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
        
    def inverse_transform(self, data, miss=None, times=None):
        df = pd.DataFrame()
        j=0
        i=0
        for name, model in zip(self.names, self.models):
            value = data[:, j:j+model.tgt_len]
            if name == self.use_pri:
                x = model.inverse_transform(times.reshape(-1,1))
                df[name] = x
                continue
                
            part = None
            if model.missing:
                if miss is not None:
                    part = miss[:, i]
                    i+=1
                else:
                    assert self.use_pri is None
                    value, part = value[:,:-1], value[:, -1]
            x = model.inverse_transform(value, part)
            df[name] = x
            j+=model.tgt_len
            
        return df

    def re_transform(self, data, miss=None):
        nw = []
        i=0
        j=0
        for name, model in zip(self.names, self.models):
            value = data[:, j:j+model.tgt_len]
            if name == self.use_pri:
                continue
                
            part = None
            if model.missing:
                if miss is not None:
                    part = miss[:, i]
                    i+=1
                else:
                    assert self.use_pri is None
                    value, part = value[:,:-1], value[:, -1]
            x = model.inverse_transform(value, part)
            x = model.transform(x.reshape(-1))
            nw.append(x)
            j+=model.tgt_len
        return np.concatenate(nw, axis=1)
