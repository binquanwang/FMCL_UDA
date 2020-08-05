from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable

from .evaluation_metrics import accuracy
from .utils.meters import AverageMeter
import copy
import numpy as np
from sklearn.cluster import KMeans
import visdom
import os
import torch.nn.functional as F

from collections import defaultdict

import scipy

from .loss import TripletLoss,TripletLoss2, CrossEntropyLabelSmooth, SoftTripletLoss, SoftEntropy#########new add


class Trainer(object):
    def __init__(self, model,model_ema,  model_inv, lmd=0.3):
        super(Trainer, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.model_inv = model_inv
        self.pid_criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.lmd = lmd

        self.model_ema = model_ema    #########new add
        self.alpha = 0.01#########new add
        self.beta = 0.01
        self.gama = 0.99

         
     #   self.model_ema_2= model_ema_2    #########new add
 
        self.criterion_tri = TripletLoss(margin=None).cuda()
        self.criterion_tri2 = TripletLoss2(margin=1,num_instances=2).cuda()
        
        #self.criterion_tri = SoftTripletLoss(margin=0.0).cuda() #########new add
        self.criterion_tri_soft = SoftTripletLoss(margin=None).cuda()
        

    

        

    def train(self, epoch, data_loader, target_train_loader, optimizer, print_freq=1):

        global flag
        self.set_model_train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_un = AverageMeter()
        precisions = AverageMeter()
    
        

        end = time.time()

        # Target iter
        target_iter = iter(target_train_loader)

        # Train
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            
                

            # Source inputs
            inputs, pids = self._parse_data(inputs)

            # Target inputs
            try:
                inputs_target = next(target_iter)
            except:
                target_iter = iter(target_train_loader)
                inputs_target = next(target_iter)
            inputs_target,index_target = self._parse_tgt_data(inputs_target)

            image = inputs_target.cpu()
            image_target = image.numpy()
            '''
            print(index_target[1],image_target[1])
            for i in range(128):
                scipy.misc.imsave('/data/ustc/wbq/ECN-contrast/images/'+str(index_target[i])+'.jpg', image_target[i][1])

               # scipy.misc.toimage(inputs_target[i], cmin=0, cmax=255).save('/data/ustc/wbq/ECN-contrast/images/'+str(index_target[i])+'.jpg')(str(index_target[i])+'.jpg',inputs_target[i] )
            '''
            

            # Source pid loss
            outputs = self.model(inputs)
           # outputs_source_ema = self.model_ema(inputs)
            source_pid_loss = self.pid_criterion(outputs, pids)
            '''
            loss_tri_source = self.criterion_tri(outputs,outputs, pids)
           # print(index_target)
            loss_tri_soft_source = self.criterion_tri_soft(outputs, outputs_source_ema,  pids)# + \
                           # self.criterion_tri_soft(outputs, outputs_ema_2, index_target)
            '''
           # source_pid_loss = source_pid_loss #+ 0.8*loss_tri_source + 0.8*loss_tri_soft_source
            
            #print(pids)
            prec, = accuracy(outputs.data, pids.data)
            prec1 = prec[0]


            # Target invariance loss
            outputs_target,f_target= self.model(inputs_target, 'tgt_feat')
            outputs_ema,f_ema= self.model_ema(inputs_target, 'tgt_feat') ########new add
            pro=F.softmax(f_target,dim=1)
            pro_ema=F.softmax(f_ema,dim=1)
            
           # print(pro_target,pro_target.size())

          #  outputs_ema_2 ,prob_ema2= self.model_ema_2(inputs_target, 'tgt_feat')
            

          #  outputs_mean = (outputs_ema_2)

            


            loss_un,queue,queue2,queue3 = self.model_inv(outputs_target,outputs_ema,pro, index_target, epoch=epoch)
            #loss_un_ema,queue_ema,queue2_ema,queue3_ema = self.model_inv(outputs_ema,outputs_target,pro, index_target, epoch=epoch)

           
            
           # flag = epoch
           # print(queue3,queue3.size())
            feature_all = queue3.t()

            label_target_2=feature_all.argmax(dim=1)
            label_target_2=label_target_2.cpu()
            label_target=label_target_2.numpy()
            
            feature_ema = queue.t()
            
          


           # print(outputs.size(), pids)

            loss_tri = self.criterion_tri(f_target, f_target, index_target)
            #loss_tri_ema = self.criterion_tri(f_ema, f_ema, index_target)

            loss_tri_soft = self.criterion_tri_soft(f_target, f_ema, index_target)
            
            loss = (1 - self.lmd) * (source_pid_loss)+ self.lmd *(loss_un)+0.8*(loss_tri) +0.8*loss_tri_soft
            
            '''
            psu_label = np.arange(12936)
            queue = queue.t()
           # print(psu_label)
            target_label =torch.from_numpy(psu_label).cuda()
          #  target_label=torch.from_numpy(target_label)
            '''
            
            
            
           # a=F.softmax(f_target, dim=1)

            #b,c=torch.max(a,1)

           # print(a,a.size(),c,f_target.size(),outputs_target.size())
           # print("feature size:", feature.size(), "\npsu_label size:", psu_label.size())
           
            '''
            if epoch>=0:
                loss_tri = self.criterion_tri(feature, feature_ema, index_target)
           # print(index_target)
              #  loss_tri_soft = self.criterion_tri_soft(feature, feature_ema, index_target)# + \
                           # self.criterion_tri_soft(outputs, outputs_ema_2, index_target)
            else:
                 loss_tri=torch.zeros(1).cuda()
                 loss_tri_soft=torch.zeros(1).cuda()
            '''
          #  loss_tri = self.criterion_tri(feature.clone().detach(), feature.clone().detach(), psu_label)
            
           # loss = (1 - self.lmd) * (source_pid_loss)+ self.lmd *loss_un  #+0.8*loss_tri #+1.8*loss_tri_soft#+  loss_un_ema) #+0.5*loss_tri 

            loss_print = {}
            loss_print['s_pid_loss'] = source_pid_loss.item()
          
            loss_print['t_un_loss'] = loss_un.item()
           # loss_print['t_un_loss_ema'] = loss_un_ema.item()

            loss_print['loss_tri_soft'] = loss_tri_soft.item()
            loss_print['loss_tri'] = loss_tri.item()
          #  loss_print['loss_tri_ema'] = loss_tri_ema.item()

        


            
            losses_un.update(loss_un.item(), outputs.size(0))
            
            losses.update(loss.item(), outputs.size(0))
            precisions.update(prec1, outputs.size(0))
      

            
            optimizer.zero_grad()
            
            loss.backward()
     
            optimizer.step()
            

            self._update_ema_variables(self.model, self.model_ema, self.alpha,epoch)#######################update model_ema
         #   self._update_ema_variables(self.model, self.model_ema_2, self.beta)#######################update model_ema

        

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                log = "Epoch: [{}][{}/{}], Time {:.3f} ({:.3f}), Data {:.3f} ({:.3f}), Loss {:.3f} ({:.3f}), Prec {:.2%} ({:.2%})" \
                    .format(epoch, i + 1, len(data_loader),
                            batch_time.val, batch_time.avg,
                            data_time.val, data_time.avg,
                            losses.val, losses.avg,
                            precisions.val, precisions.avg)

                for tag, value in loss_print.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = imgs.to(self.device)
        pids = pids.to(self.device)
        return inputs, pids

    def _parse_tgt_data(self, inputs_target):
        inputs, _, _, index = inputs_target
        inputs = inputs.to(self.device)
        index = index.to(self.device)
        return inputs, index

    def _update_ema_variables(self, model, ema_model, alpha,epoch):
       # alpha = min(1 - 1 / (global_step + 1), alpha)

        if epoch >25:
            alpha = 0.999
        
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):

           
            
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


    def set_model_train(self):
        self.model.train()


        # Fix first BN
        fixed_bns = []
        for idx, (name, module) in enumerate(self.model.module.named_modules()):
            if name.find("layer3") != -1:
                # assert len(fixed_bns) == 22
                break
            if name.find("bn") != -1:
                fixed_bns.append(name)
                module.eval()

                fixed_bns = []

