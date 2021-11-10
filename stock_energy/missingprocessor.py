from sklearn.preprocessing import MinMaxScaler, RobustScaler, LabelBinarizer
import numpy as np
import pandas as pd

class MissingProcessor:
    def __init__(self, threshold=None, which="continuous"):
        self.threshold = threshold
        self.length = 0
        if which in ["binary", "categorical"]:
            self.model = LabelBinarizer()
        else: 
            self.model = MinMaxScaler()
        self.which = which
        self.missing = False
        
    def fit(self, data):
        loc = np.isnan(data)
        if loc.any():
            self.length = 1
            self.missing = True
            if self.threshold is None:
                self.threshold = 1.0 * sum(loc) / len(data)
            if loc.all():
                res = self.model.fit_transform(np.zeros((1,1)))
            else:
                res = self.model.fit_transform(data[~loc].reshape(-1,1))
        else:
            res = self.model.fit_transform(data.reshape(-1,1))
        self.length += res.shape[1]

    def transform(self, data, fillnan=np.nan):
        if self.missing:
            res = np.ones((len(data), self.length))
            loc = np.isnan(data)
            if fillnan != fillnan:
                res[:, :-1] = self.model.transform(np.nan_to_num(data).reshape(-1,1))
                res[loc] = 0
            else:
                res[:, :-1] = self.model.transform(np.nan_to_num(data, nan=fillnan).reshape(-1,1))
                res[loc, -1] = 0
                #res[loc, -1] = 1 - 1.0*sum(loc)/len(loc)
                #res[loc, -1]=np.random.rand(sum(loc))*self.threshold
            return res.astype("float32")
        else:
            return self.model.transform(data.reshape(-1,1))

    def inverse_transform(self, data):
        if self.missing:
            res = np.zeros((len(data), 1))
            loc = data[:, -1] > self.threshold
            res = self.model.inverse_transform(data[:, :-1])
            res[~loc] = np.nan
        else:
            res = self.model.inverse_transform(data)
        res = res.reshape(-1, 1)
        if "int" in self.which:
            res = res.round()
        return res

class Processor:
    def __init__(self, types):
        self.names = []
        self.models = []
        self.types = types
        self.dim = 0

    def fit(self, data):
        self.names = []
        self.models = []
        self.dim = 0
        matrix = data.values
        for i, (types, col) in enumerate(zip(self.types,data.columns)):
            value = matrix[:, i]
            self.names.append(col)
            model = MissingProcessor(which=types)
            model.fit(value)
            self.models.append(model)
            print(col, model.length, model.threshold, model.which)
            self.dim += model.length
            
    def transform(self, data, nan_lis=None):
        cols = []
        matrix = data.values
        for i, col in enumerate(data.columns):
            value = matrix[:, i]
            fillnan = np.nan if nan_lis is None else nan_lis[i]
            preprocessed_col = self.models[i].transform(value, fillnan)
            cols.append(preprocessed_col)

        return np.concatenate(cols, axis=1)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
        
    def inverse_transform(self, data):
        res = []
        j=0
        for model in self.models:
            value = data[:, j:j+model.length]
            x = model.inverse_transform(value)
            res.append(x)
            j+=model.length
        matrix = np.concatenate(res, axis=1)
        return pd.DataFrame(matrix, columns=self.names)

