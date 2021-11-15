# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from fastNLP import DataSet, DataSetIter, RandomSampler, SequentialSampler
from fastNLP import seq_len_to_mask
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelBinarizer
from basic import max_pooling
from autoencoder import Autoencoder
from gan import Generator, Discriminator
import random
from torch import autograd
import time
    
class AeGAN:
    def __init__(self, processors, params):
        self.params = params
        if self.params.get("force") is None:
            self.params["force"]=""
        self.device = params["device"]
        self.logger = params["logger"]
        self.static_processor, self.dynamic_processor = processors

        self.ae = Autoencoder(
            processors, self.params["hidden_dim"], self.params["embed_dim"], self.params["layers"], dropout=self.params["dropout"])
        self.ae.to(self.device)
        """
        self.decoder_optm = torch.optim.Adam(
            params=self.ae.decoder.parameters(),
            lr=self.params['ae_lr'],
            betas=(0.9, 0.999),
        )
        self.encoder_optm = torch.optim.Adam(
            params=self.ae.encoder.parameters(),
            lr=self.params['ae_lr'],
            betas=(0.9, 0.999),
        )
        """
        self.ae_optm = torch.optim.Adam(
            params=self.ae.parameters(),
            lr=self.params['ae_lr'],
            betas=(0.9, 0.999),
            weight_decay=self.params["weight_decay"]
        )
        
        self.loss_con = nn.MSELoss(reduction='none')
        self.loss_dis = nn.NLLLoss(reduction='none')
        self.loss_mis = nn.BCELoss(reduction='none')
        
        self.generator = Generator(self.params["noise_dim"], self.params["hidden_dim"], self.params["layers"]).to(self.device)
        self.discriminator = Discriminator(self.params["embed_dim"]).to(self.device)
        self.discriminator_optm = torch.optim.RMSprop(
            params=self.discriminator.parameters(),
            lr=self.params['gan_lr'],
            alpha=self.params['gan_alpha'],
        )
        self.generator_optm = torch.optim.RMSprop(
            params=self.generator.parameters(),
            lr=self.params['gan_lr'],
            alpha=self.params['gan_alpha'],
        )

    def load_ae(self, pretrained_dir=None):
        if pretrained_dir is not None:
            path = pretrained_dir
        else:
            path = '{}/ae.dat'.format(self.params["root_dir"])
        self.logger.info("load: "+path)
        self.ae.load_state_dict(torch.load(path, map_location=self.device))

    def load_generator(self, pretrained_dir=None):
        if pretrained_dir is not None:
            path = pretrained_dir
        else:
            path = '{}/generator.dat'.format(self.params["root_dir"])
        self.logger.info("load: "+path)
        self.generator.load_state_dict(torch.load(path, map_location=self.device))

    def sta_loss(self, data, target):
        loss = 0
        n = len(self.static_processor.models)
        st = 0
        #print(data,target)
        for i, model in enumerate(self.static_processor.models):
            ed = st + model.tgt_len - int(model.missing)
            use = 1
            if model.missing:
                loss += 0.1 * torch.mean(self.loss_mis(data[:, ed], target[:, ed])) 
                use = target[:, ed:ed+1]
             
            if model.which == "categorical":
                loss += torch.mean(use * self.loss_dis((data[:, st:ed]+1e-8).log(), torch.argmax(target[:,st:ed], dim=-1)).unsqueeze(-1))
            elif model.which =="binary" :
                loss += torch.mean(use * self.loss_mis(data[:, st:ed], target[:, st:ed]))
            else:
                loss += torch.mean(use * self.loss_con(data[:, st:ed], target[:, st:ed]))
 
            st += model.tgt_len
        assert st == target.size(-1)
        return loss/n
    
    def dyn_loss(self, data, target, seq_len, mask):
        loss = []
        n = len(self.dynamic_processor.models)
        st = 0
        i = 0
        for model in self.dynamic_processor.models:
            if model.name == self.dynamic_processor.use_pri: continue
            ed = st + model.tgt_len
            use = 1
            if model.missing:
                use = mask[:, :, i:i+1]
                i+=1
                
            if model.which == "categorical":
                x = (data[:, :, st:ed] + 1e-8).log().transpose(1,2)
                loss.append(use * self.loss_dis(x, torch.argmax(target[:, :, st:ed], dim=-1)).unsqueeze(-1))
            elif model.which =="binary" :
                loss.append(use * self.loss_mis(data[:, :, st:ed], target[:, :, st:ed]))
            else:
                loss.append(use * 10 * self.loss_con(data[:, :, st:ed], target[:, :, st:ed]))
            st += model.tgt_len
        assert i == mask.size(-1)
        loss = torch.cat(loss, dim=-1)
        seq_mask = seq_len_to_mask(seq_len)
        loss = torch.masked_select(loss, seq_mask.unsqueeze(-1))
        return torch.mean(loss)
    
    def time_loss(self, data, target, seq_len):
        loss = self.loss_con(data, target)
        seq_mask = seq_len_to_mask(seq_len)
        loss = torch.masked_select(loss, seq_mask.unsqueeze(-1))
        return torch.mean(loss)
    
    def missing_loss(self, data, target, seq_len):
        thr = torch.Tensor([model.threshold for model in self.dynamic_processor.models if model.missing]).to(data.device)
        thr = thr.unsqueeze(0).unsqueeze(0)
        
        scale = thr * target + (1 - thr) * (1 - target)
        loss = self.loss_mis(data, target) * scale
        seq_mask = seq_len_to_mask(seq_len)
        loss = torch.masked_select(loss, seq_mask.unsqueeze(-1))
        
        mx, _ = max_pooling(data, seq_len)
        gold_mx, _ = torch.max(target, dim=1)
        loss1 = self.loss_mis(mx, gold_mx)
        return torch.mean(loss) + torch.mean(torch.masked_select(loss1, gold_mx==0))
    
    def train_ae(self, dataset, epochs=800):
        min_loss = 1e15
        best_epsilon = 0
        train_batch=DataSetIter(dataset=dataset, batch_size=self.params["ae_batch_size"],sampler=RandomSampler())
        force = 1
        for i in range(epochs):
            self.ae.train()
            tot_loss = 0
            con_loss = 0
            dis_loss = 0
            miss_loss1 = 0
            miss_loss2 = 0
            tot = 0
            t1 = time.time()
            if self.params["force"] == "linear":
                if i >= epochs / 100 and i < epochs / 2:
                    force -= 2 / epochs 
                elif i >= epochs / 2:
                    force = -1
            elif self.params["force"] == "constant":
                force = 0.5
            else :
                force = 1
            for batch_x, batch_y in train_batch:
                self.ae.zero_grad()
                sta = batch_x["sta"].to(self.device)
                dyn = batch_x["dyn"].to(self.device)
                mask = batch_x["mask"].to(self.device)
                lag = batch_x["lag"].to(self.device)
                priv = batch_x["priv"].to(self.device)
                nex = batch_x["nex"].to(self.device)
                times = batch_x["times"].to(self.device)
                seq_len = batch_x["seq_len"].to(self.device)
                
                out_sta, out_dyn, missing, gt = self.ae(sta, dyn, lag, mask, priv, nex, times, seq_len, forcing=force)
                loss3 = self.missing_loss(missing, mask, seq_len)
                miss_loss1 += loss3.item()  
                loss4 = self.time_loss(gt, times, seq_len)
                miss_loss2 += loss4.item()   
                
                loss1 = self.sta_loss(out_sta, sta)
                loss2 = self.dyn_loss(out_dyn, dyn, seq_len, mask)

                sta_num = len(self.static_processor.models)
                dyn_num = len(self.dynamic_processor.models)
                scale1 = sta_num / (sta_num + dyn_num)
                scale2 = dyn_num / (sta_num + dyn_num)
                scale3 = 0.1
                
                loss = scale1 * loss1 + scale2 * (loss2 + loss3) + scale3 * loss4
                #loss = loss1 + loss2 + loss3 + loss4
                loss.backward()
                self.ae_optm.step()

                tot_loss += loss.item()
                con_loss += loss1.item()
                dis_loss += loss2.item()
                tot += 1

            tot_loss/=tot
            if i % 5 == 0:
                self.logger.info("Epoch:{} {}\t{}\t{}\t{}\t{}\t{}".format(
                    i+1, time.time()-t1, force, con_loss/tot, dis_loss/tot, miss_loss1/tot, miss_loss2/tot))
                    
            if i % 100 == 99:      
                torch.save(self.ae.state_dict(), '{}/ae{}.dat'.format(self.params["root_dir"], i)) 
                self.generate_ae(dataset[:100])
                
        torch.save(self.ae.state_dict(), '{}/ae.dat'.format(self.params["root_dir"]))            
    
    def train_gan(self, dataset, iterations=15000, d_update=5):
        self.discriminator.train()
        self.generator.train()
        self.ae.train()
        batch_size = self.params["gan_batch_size"]
        idxs = list(range(len(dataset)))
        batch = DataSetIter(dataset=dataset, batch_size=batch_size, sampler=RandomSampler())
        min_loss = 1e15
        for iteration in range(iterations):
            avg_d_loss = 0
            t1 = time.time()
            toggle_grad(self.generator, False)
            toggle_grad(self.discriminator, True)
            
            for j in range(d_update):
                for batch_x, batch_y in batch:        
                    self.discriminator_optm.zero_grad()
                    z = torch.randn(batch_size, self.params['noise_dim']).to(self.device)

                    sta = batch_x["sta"].to(self.device)
                    dyn = batch_x["dyn"].to(self.device)
                    mask = batch_x["mask"].to(self.device)
                    lag = batch_x["lag"].to(self.device)
                    priv = batch_x["priv"].to(self.device)
                    nex = batch_x["nex"].to(self.device)
                    times = batch_x["times"].to(self.device)
                    seq_len = batch_x["seq_len"].to(self.device)
                    
                    real_rep = self.ae.encoder(sta, dyn, priv, nex, mask, times, seq_len)
                    d_real = self.discriminator(real_rep)
                    dloss_real = -d_real.mean()
                    #y = d_real.new_full(size=d_real.size(), fill_value=1)
                    #dloss_real = F.binary_cross_entropy_with_logits(d_real, y)
                    dloss_real.backward()
                    
                    """
                    dloss_real.backward(retain_graph=True)
                    reg = 10 * compute_grad2(d_real, real_rep).mean()
                    reg.backward()
                    """

                    # On fake data
                    with torch.no_grad():
                        x_fake = self.generator(z)

                    x_fake.requires_grad_()
                    d_fake = self.discriminator(x_fake)
                    dloss_fake = d_fake.mean()
                    """
                    y = d_fake.new_full(size=d_fake.size(), fill_value=0)
                    dloss_fake = F.binary_cross_entropy_with_logits(d_fake, y)
                    """
                    dloss_fake.backward()
                    """
                    dloss_fake.backward(retain_graph=True)
                    reg = 10 * compute_grad2(d_fake, x_fake).mean()
                    reg.backward()
                    """
                    reg = 10 * self.wgan_gp_reg(real_rep, x_fake)
                    reg.backward()

                    self.discriminator_optm.step()
                    d_loss = dloss_fake + dloss_real
                    avg_d_loss += d_loss.item()
                    break

            avg_d_loss/=d_update

            toggle_grad(self.generator, True)
            toggle_grad(self.discriminator, False)
            self.generator_optm.zero_grad()
            z = torch.randn(batch_size, self.params['noise_dim']).to(self.device)
            fake = self.generator(z)
            g_loss = -torch.mean(self.discriminator(fake))
            g_loss.backward()
            self.generator_optm.step()

            if iteration % 1000 == 999:
                self.logger.info('[Iteration %d/%d] [%f] [D loss: %f] [G loss: %f] [%f]' % (
                    iteration, iterations, time.time()-t1, avg_d_loss, g_loss.item(), reg.item()
                ))       
        torch.save(self.generator.state_dict(), '{}/generator.dat'.format(self.params["root_dir"])) 

    def wgan_gp_reg(self, x_real, x_fake, center=1.):
        batch_size = x_real.size(0)
        eps = torch.rand(batch_size, device=self.device).view(batch_size, -1)
        x_interp = (1 - eps) * x_real + eps * x_fake
        x_interp = x_interp.detach()
        x_interp.requires_grad_()
        d_out = self.discriminator(x_interp)

        reg = (compute_grad2(d_out, x_interp).sqrt() - center).pow(2).mean()

        return reg
    
    def synthesize(self, n, batch_size=500):
        pass

    def eval_ae(self, dataset):
        batch_size = self.params["gan_batch_size"]
        idxs = list(range(len(dataset)))
        batch = DataSetIter(dataset=dataset, batch_size=batch_size, sampler=SequentialSampler())
        res = []
        h = []
        for batch_x, batch_y in batch:
            with torch.no_grad():
                sta = batch_x["sta"].to(self.device)
                dyn = batch_x["dyn"].to(self.device)
                mask = batch_x["mask"].to(self.device)
                lag = batch_x["lag"].to(self.device)
                priv = batch_x["priv"].to(self.device)
                nex = batch_x["nex"].to(self.device)
                times = batch_x["times"].to(self.device)
                seq_len = batch_x["seq_len"].to(self.device)

                hidden = self.ae.encoder(sta, dyn, priv, nex, mask, times, seq_len)
                h.append(hidden)
        h = torch.cat(h, dim=0).cpu().numpy()
        return h
    
    def gen_hidden(self, n):
        self.ae.decoder.eval()
        self.generator.eval()
        batch_size = self.params["gan_batch_size"]
        h = []
        def _gen(n):
            with torch.no_grad():
                z = torch.randn(n, self.params['noise_dim']).to(self.device)
                hidden =self.generator(z)
                h.append(hidden)
                
        tt = n // batch_size
        for i in range(tt):
            _gen(batch_size)
        res = n - tt * batch_size
        if res>0:
            _gen(res)
        h = torch.cat(h, dim=0).cpu().numpy()
        return h
    
# Utility functions
def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg

def update_average(model_tgt, model_src, beta):
    toggle_grad(model_src, False)
    toggle_grad(model_tgt, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert(p_src is not p_tgt)
        p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)
