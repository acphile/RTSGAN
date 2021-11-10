import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from fastNLP import DataSet, DataSetIter, RandomSampler, SequentialSampler
from fastNLP import seq_len_to_mask
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelBinarizer
from autoencoder import Autoencoder
from gan import Generator, Discriminator
import random
from torch import autograd
import time
    
class AeGAN:
    def __init__(self, processors, params):
        self.params = params
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
        for model in self.static_processor.models:
            ed = st + model.length - int(model.missing)
            use = 1
            if model.missing:
                loss += torch.mean(self.loss_mis(data[:, ed], target[:, ed])) 
                use = 0.1 + target[:, ed:ed+1]
             
            if model.which == "categorical":
                loss += torch.mean(use * self.loss_dis((data[:, st:ed]+1e-8).log(), target[:,st:ed]).unsqueeze(-1))
            elif model.which =="binary" :
                loss += torch.mean(use * self.loss_mis(data[:, st:ed], target[:, st:ed]))
            else:
                loss += torch.mean(use * self.loss_con(data[:, st:ed], target[:, st:ed]))
 
            st += model.length
        return loss/n
    
    def dyn_loss(self, data, target, seq_len):
        loss = []
        n = len(self.dynamic_processor.models)
        st = 0
        for model in self.dynamic_processor.models:
            ed = st + model.length - int(model.missing)
            use = 1
            if model.missing:
                loss.append(self.loss_mis(data[:, :, ed], target[:, :, ed]).unsqueeze(-1))
                use = 0.1 + target[:, :, ed:ed+1]
                
            if model.which == "categorical":
                loss.append(use * self.loss_dis((data[:, :, st:ed]+1e-8).log(), target[:, :, st:ed]).unsqueeze(-1))
            elif model.which =="binary" :
                loss.append(use * self.loss_mis(data[:, :, st:ed], target[:, :, st:ed]))
            else:
                #print(data.size(), target.size(), st, ed)
                loss.append(use * self.loss_con(data[:, :, st:ed], target[:, :, st:ed]))
            st += model.length
        loss = torch.cat(loss, dim=-1)
        mask = seq_len_to_mask(seq_len)
        loss = torch.masked_select(loss, mask.unsqueeze(-1))
        return torch.mean(loss)
        
    def train_ae(self, dataset, epochs=1000):
        min_loss = 1e15
        best_epsilon = 0
        train_batch=DataSetIter(dataset=dataset, batch_size=self.params["ae_batch_size"],sampler=RandomSampler())
        for i in range(epochs):
            self.ae.train()
            tot_loss = 0
            con_loss = 0
            dis_loss = 0
            tot = 0
            t1 = time.time()
            for batch_x, batch_y in train_batch:
                self.ae.zero_grad()
                sta = None
                dyn = batch_x["dyn"].to(self.device)
                seq_len = batch_x["seq_len"].to(self.device)
                out_sta, out_dyn = self.ae(sta, dyn, seq_len)
                loss1 = 0 #self.sta_loss(out_sta, sta)
                loss2 = self.dyn_loss(out_dyn, dyn, seq_len)
                loss = loss2
                loss.backward()
                self.ae_optm.step()

                tot_loss += loss.item()
                con_loss += 0 #loss1.item()
                dis_loss += loss2.item()
                tot += 1

            tot_loss/=tot
            if i % 5 == 0:
                self.logger.info("Epoch:{} {}\t{}\t{}\t{}".format(i+1, time.time()-t1, (con_loss+dis_loss)/tot, con_loss/tot, dis_loss/tot))
                    
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
            self.generator.train()
            self.discriminator.train()
            
            for j in range(d_update):
                for batch_x, batch_y in batch:        
                    self.discriminator_optm.zero_grad()
                    z = torch.randn(batch_size, self.params['noise_dim']).to(self.device)

                    sta = None
                    dyn = batch_x["dyn"].to(self.device)
                    seq_len = batch_x["seq_len"].to(self.device)
                    real_rep = self.ae.encoder(sta, dyn, seq_len)
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
            self.generator.train()
            self.discriminator.train()
            self.generator_optm.zero_grad()
            z = torch.randn(batch_size, self.params['noise_dim']).to(self.device)
            fake = self.generator(z)
            g_loss = -torch.mean(self.discriminator(fake))
            """
            d_fake = self.discriminator(fake)
            y = d_fake.new_full(size=d_fake.size(), fill_value=1)
            g_loss = F.binary_cross_entropy_with_logits(d_fake, y)
            """
            g_loss.backward()
            self.generator_optm.step()

            if iteration % 1000 == 999:
                self.logger.info('[Iteration %d/%d] [%f] [D loss: %f] [G loss: %f] [%f]' % (
                    iteration, iterations, time.time()-t1, avg_d_loss, g_loss.item(), reg.item()
                ))
        torch.save(self.generator.state_dict(), '{}/generator.dat'.format(self.params["root_dir"]))              
    
    def synthesize(self, n, seq_len=24, batch_size=500):
        self.ae.decoder.eval()
        self.generator.eval()
        def _gen(n):
            with torch.no_grad():
                z = torch.randn(n, self.params['noise_dim']).to(self.device)
                hidden =self.generator(z)
                dynamics = self.ae.decoder.generate_dynamics(hidden, seq_len)
            res = []
            for i in range(n):
                #dyn = self.dynamic_processor.inverse_transform(dynamics[i]).values.tolist()
                dyn = dynamics[i].tolist()
                res.append(dyn)
            return res

        data = []
        tt = n // batch_size
        for i in range(tt):
            data.extend(_gen(batch_size))
        res = n - tt * batch_size
        if res>0:
            data.extend(_gen(res))
        return data

    def wgan_gp_reg(self, x_real, x_fake, center=1.):
        batch_size = x_real.size(0)
        eps = torch.rand(batch_size, device=self.device).view(batch_size, -1)
        x_interp = (1 - eps) * x_real + eps * x_fake
        x_interp = x_interp.detach()
        x_interp.requires_grad_()
        d_out = self.discriminator(x_interp)

        reg = (compute_grad2(d_out, x_interp).sqrt() - center).pow(2).mean()

        return reg
    
    def eval_ae(self, dataset):
        batch_size = self.params["gan_batch_size"]
        idxs = list(range(len(dataset)))
        batch = DataSetIter(dataset=dataset, batch_size=batch_size, sampler=SequentialSampler())
        res = []
        h = []
        for batch_x, batch_y in batch:
            with torch.no_grad():
                sta = None
                dyn = batch_x["dyn"].to(self.device)
                seq_len = batch_x["seq_len"].to(self.device)
                hidden = self.ae.encoder(sta, dyn, seq_len)
                dynamics = self.ae.decoder.generate_dynamics(hidden, 24)
                h.append(hidden)
                for i in range(len(dyn)):
                    #dyn = self.dynamic_processor.inverse_transform(dynamics[i]).values.tolist()
                    dyn = dynamics[i].tolist()
                    res.append(dyn)
        h = torch.cat(h, dim=0).cpu().numpy()
        assert len(h) == len(res)
        return res, h
    
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
