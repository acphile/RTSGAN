# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import pickle
import collections
import logging
import math
import os,sys,time
import random
from sys import maxsize
import pickle
import numpy as np
import torch
import torch.nn as nn
from utils.general import init_logger, make_sure_path_exists
sys.path.append('./general/')

from inhospital import Inhospital

DEBUG_SCALE = 512
# ===-----------------------------------------------------------------------===
# Argument parsing
# ===-----------------------------------------------------------------------===
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, dest="dataset", help=".pkl file to use") 
parser.add_argument("--force", default="", dest="force", help="schedule")
parser.add_argument("--devi", default="0", dest="devi", help="gpu")
parser.add_argument("--epochs", default=800, dest="epochs", type=int,
                    help="Number of full passes through training set for autoencoder")
parser.add_argument("--iterations", default=15000, dest="iterations", type=int,
                    help="Number of iterations through training set for WGAN")
parser.add_argument("--d-update", default=5, dest="d_update", type=int,
                    help="discriminator updates per generator update")
parser.add_argument("--log-dir", default="../inhospital_result", dest="log_dir",
                    help="Directory where to write logs / serialized models")
parser.add_argument("--task-name", default=time.strftime("%Y-%m-%d-%H-%M-%S"), dest="task_name",
                    help="Name for this task, use a comprehensive one")
parser.add_argument("--python-seed", dest="python_seed", type=int, default=random.randrange(maxsize),
                    help="Random seed of Python and NumPy")
parser.add_argument("--debug", dest="debug", default=False, action="store_true", help="Debug mode")
parser.add_argument("--eval-ae", dest="eval_ae", default=False, action="store_true", help="eval mode")
parser.add_argument("--fix-ae", dest="fix_ae", default=None, help="Test mode")
parser.add_argument("--fix-gan", dest="fix_gan", default=None, help="Test mode")

parser.add_argument("--ae-batch-size", default=128, dest="ae_batch_size", type=int,
                    help="Minibatch size for autoencoder")
parser.add_argument("--gan-batch-size", default=512, dest="gan_batch_size", type=int,
                    help="Minibatch size for WGAN")
parser.add_argument("--embed-dim", default=512, dest="embed_dim", type=int, help="dim of hidden state")
parser.add_argument("--hidden-dim", default=128, dest="hidden_dim", type=int, help="dim of GRU hidden state")
parser.add_argument("--layers", default=3, dest="layers", type=int, help="layers")
parser.add_argument("--ae-lr", default=1e-3, dest="ae_lr", type=float, help="autoencoder learning rate")
parser.add_argument("--weight-decay", default=0, dest="weight_decay", type=float, help="weight decay")
parser.add_argument("--dropout", default=0.0, dest="dropout", type=float,
                    help="Amount of dropout(not keep rate, but drop rate) to apply to embeddings part of graph")

parser.add_argument("--gan-lr", default=1e-4, dest="gan_lr", type=float, help="WGAN learning rate")
parser.add_argument("--gan-alpha", default=0.99, dest="gan_alpha", type=float, help="for RMSprop")
parser.add_argument("--noise-dim", default=512, dest="noise_dim", type=int, help="dim of WGAN noise state")

options = parser.parse_args()

task_name = options.task_name
root_dir = "{}/{}".format(options.log_dir, task_name)
make_sure_path_exists(root_dir)

devices=[int(x) for x in options.devi]
device = torch.device("cuda:{}".format(devices[0]))  

# ===-----------------------------------------------------------------------===
# Set up logging
# ===-----------------------------------------------------------------------===
logger = init_logger(root_dir)

# ===-----------------------------------------------------------------------===
# Log some stuff about this run
# ===-----------------------------------------------------------------------===
logger.info(' '.join(sys.argv))
logger.info('')
logger.info(options)

if options.debug:
    print("DEBUG MODE")
    options.epochs=5
    options.iterations=1

random.seed(options.python_seed)
np.random.seed(options.python_seed % (2 ** 32 - 1))
logger.info('Python random seed: {}'.format(options.python_seed))

# ===-----------------------------------------------------------------------===
# Read in dataset
# ===-----------------------------------------------------------------------===
dataset = pickle.load(open(options.dataset, "rb"))
train_set=dataset["train_set"]
dynamic_processor=dataset["dynamic_processor"]
static_processor=dataset["static_processor"]
name_lis=dataset["name_lis"]
train_set.set_input("dyn", "mask", "sta", "times", "lag", "seq_len", "nex", "priv", "label")
                    
if options.debug:
    train_set = train_set[0:DEBUG_SCALE]
    name_lis = name_lis[0:DEBUG_SCALE]
# ===-----------------------------------------------------------------------===
# Build model and trainer
# ===-----------------------------------------------------------------------===

params=vars(options)
params["static_processor"]=static_processor
params["dynamic_processor"]=dynamic_processor
params["root_dir"]=root_dir
params["logger"]=logger
params["device"]=device
print(params.keys())

syn = Inhospital((static_processor, dynamic_processor), params)

if options.fix_ae is not None:
    syn.load_ae(options.fix_ae)
else:
    syn.train_ae(train_set, options.epochs)
    
logger.info("\n")
logger.info("Reconstructing data!")
sta, dyn = syn.generate_ae(train_set)
make_sure_path_exists("{}/reconstruction".format(root_dir))
print(sta["y_true"].value_counts())
for res, name in zip(dyn[:10], name_lis[:10]):
    res.to_csv("{}/reconstruction/{}".format(root_dir,name), sep=',', index=False)

h = syn.eval_ae(train_set)
with open("{}/hidden".format(root_dir), "wb") as f:
    pickle.dump(h, f)
    
logger.info("Finshed!")

if options.fix_gan is not None:
    syn.load_generator(options.fix_gan)
else:
    syn.train_gan(train_set, options.iterations, options.d_update)

h = syn.gen_hidden(len(train_set))
with open("{}/gen_hidden".format(root_dir), "wb") as f:
    pickle.dump(h, f)
    
logger.info("\n")
logger.info("Generating data!")
sta, dyn = syn.synthesize(len(train_set))
    
make_sure_path_exists("{}/train".format(root_dir))
print(sta["y_true"].value_counts())
for res, name in zip(dyn, name_lis):
    res.to_csv("{}/train/{}".format(root_dir,name), sep=',', index=False)
sta["stay"] = np.array(name_lis)
sta=sta[['stay','y_true']]
sta.to_csv("{}/train/listfile.csv".format(root_dir), sep=',', index=False)
