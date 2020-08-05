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


class pretrainer(object):
    def __init__(self, model,lmd=0.3):
        super(pretrainer, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model

        self.pid_criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.lmd = lmd


        self.alpha = 0.01#########new add
        self.beta = 0.01
        self.gama = 0.99

         


        self.criterion_ce = CrossEntropyLabelSmooth(702).cuda()
        self.criterion_ce_soft = SoftEntropy().cuda()
 
        self.criterion_tri = TripletLoss(margin=None).cuda()
        self.criterion_tri2 = TripletLoss2(margin=1,num_instances=2).cuda()
        
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
        for cnt, inputs in enumerate(data_loader):
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
            

            feature,outputs = self.model(inputs)

            source_pid_loss = self.pid_criterion(outputs, pids)
            

            loss_tri_source = self.criterion_tri(feature,feature, pids)
            source_pid_loss = source_pid_loss + loss_tri_source

            
            prec, = accuracy(outputs.data, pids.data)
            prec1 = prec[0]
         

            
            loss =  (source_pid_loss)
            #loss=Variable(loss,requires_grad=True)
            

            loss_print = {}
            loss_print['s_pid_loss'] = source_pid_loss.item()
            loss_print['loss_tri_source'] = loss_tri_source.item()
          

            
            
            
            losses.update(loss.item(), outputs.size(0))
            precisions.update(prec1, outputs.size(0))
      

            
            optimizer.zero_grad()
            
            loss.backward()
     
            optimizer.step()
            

 

        

            batch_time.update(time.time() - end)
            end = time.time()
            i = cnt
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



        self.criterion_ce = CrossEntropyLabelSmooth(702).cuda()
        self.criterion_ce_soft = SoftEntropy().cuda()
 
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
        for cnt, inputs in enumerate(data_loader):
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
            feature,outputs = self.model(inputs)
           # outputs_source_ema = self.model_ema(inputs)
            source_pid_loss = self.pid_criterion(outputs, pids)

            loss_tri_source = self.criterion_tri(feature,feature, pids)
            source_pid_loss = source_pid_loss #+ loss_tri_source
            
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
            outputs_ema,f_ema= self.model_ema(inputs_target, 'tgt_feat') ########new add, 'tgt_feat', 'tgt_feat'
            #fea,fea_label= self.model(inputs_target)

            label_target = F.softmax(f_target,dim=1)
            label_ema = F.softmax(f_ema,dim=1)
            
           # print(pro_target,pro_target.size())

          #  outputs_ema_2 ,prob_ema2= self.model_ema_2(inputs_target, 'tgt_feat')
            

          #  outputs_mean = (outputs_ema_2)

            
            
            
            loss_un,queue,queue2,queue3,queue4,label = self.model_inv(outputs_target,outputs_ema,f_target,f_ema, index_target, epoch=epoch)
            #loss_un_ema,queue_ema,queue2_ema,queue3_ema = self.model_inv(outputs_ema,outputs_target,pro, index_target, epoch=epoch)

            
            
           # flag = epoch
           # print(queue3,queue3.size())
            feature_all = queue3.t()
            feature_all_ema = queue4.t()
            
            pro=F.softmax(feature_all,dim=1)
            
            pro_ema=F.softmax(feature_all_ema,dim=1)

            
            #label_prob_ema = pro_ema[index_target]

            label_target_2=pro.argmax(dim=1)
            label_target_ema=pro_ema.argmax(dim=1)

            #print(label_target_2,label_target_2.size())
            
            label_prob = label_target_2[index_target-1]
            label_prob_ema = label_target_ema[index_target-1]
            
            label_target_2=label_target_2.cpu()
            label_target=label_target_2.numpy()+1
            
           # feature_ema = queue.t()


            '''
            if epoch <= 0:
                flag = 0
                psu_label=index_target
            
           # loss_un_ema = self.model_inv(outputs_ema,outputs_ema, index_target, epoch=epoch)
           

            flag = epoch  
            if epoch==flag and epoch>0:
                
                index_dic = defaultdict(list)
                indices = torch.randperm(703)
                pids=np.arange(0,703,1)

                
                for i in range(703):
                    a = np.where(label_target==i)
                    for j in range(len(a)):
                        
                        index_dic[i].append(a[j])
                  #  index_dic[i]=np.array(index_dic[i])
                    #index_dic[i] = index_dic[i].t()
                    
                        
              #  print(pids,indices,index_dic)

                ret = []
                for i in indices:
                    pid = pids[i]
                    t = index_dic[pid]
                   # t=torch.from_numpy(t).cuda()
                  #  print(t)
                    #print("pid =", pid, "t =", t)
                    if len(t[0]) >= 4:
                        t = np.random.choice(t[0], size=4, replace=False)
                    else:
                        if len(t[0]) == 0:
                            continue
                        t = np.random.choice(t[0], size=4, replace=True)
                    ret.extend(t)
                labels = ret[0:128]
                labels=np.array(labels)
                psu_label = torch.from_numpy(labels).cuda()
               # print(psu_label)

            
            feature = queue3[:,psu_label].t()
            feature_ema = queue4[:,psu_label].t()
            '''
            '''
            psu_label = np.arange(12936)
            queue = queue.t()
           # print(psu_label)
            target_label =torch.from_numpy(psu_label).cuda()
          #  target_label=torch.from_numpy(target_label)
            '''
            '''
            psu_label_2 = [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,9,10,10,10,10,11,11,11,11,12,12,12,12,13,13,13,13,14,14,14,14,15,15,15,15,16,16,16,16,17,17,17,17,18,18,18,18,19,19,19,19,20,20,20,20,21,21,21,21,22,22,22,22,23,23,23,23,24,24,24,24,25,25,25,25,26,26,26,26,27,27,27,27,28,28,28,28,29,29,29,29,30,30,30,30,31,31,31,31,32,32,32,32]
            psu_label = np.array(psu_label_2)
            target_label=torch.from_numpy( psu_label).cuda()
            
            '''

           # print(outputs.size(), pids)
            '''
            x =  np.arange(12936)+1
            psu_label = np.array(x)
            target_label=torch.from_numpy( psu_label).cuda()
            '''

           # loss_ce_1 = self.criterion_ce(f_target, label_prob)
            #loss_ce_ema = self.criterion_ce(f_ema, label_prob)

           # loss_ce_soft = self.criterion_ce_soft(f_target, f_ema)+self.criterion_ce_soft( f_ema,f_target)
            device = torch.device("cuda:1")
            #print("feature_all.device =", feature_all.device, "feature_all_ema.device =", feature_all_ema.device, "label_prob.device =", label_prob.device)
            
            feature_all = feature_all.to(device)
            feature_all_ema = feature_all_ema.to(device)
            #label_prob = label_prob.to(device)

           # f_target = f_target.to(device)
            #print(label,label.size())
            loss_tri = self.criterion_tri(outputs_target,outputs_target, label_prob)
            loss_tri_soft = self.criterion_tri_soft(outputs_target, outputs_ema, label_prob)
            loss_ce_soft = self.criterion_ce_soft(f_target, f_ema)
            #print("loss_tri.device =", loss_tri.device)

            
           # loss_ce_2 = self.criterion_ce(f_ema, label_prob)
            
            device = torch.device("cuda:0")
            loss_tri = loss_tri.to(device)
            loss_tri_soft = loss_tri_soft.to(device)
            loss_ce_soft = loss_ce_soft.to(device)
            #loss_ce_1 = loss_ce_1.to(device)

            loss_ce_1 = self.criterion_ce(f_target,label_prob_ema)
            loss_ce_2 = self.criterion_ce(f_ema, label_prob)

            #print(label_prob)
            
            #loss_tri_ema = self.criterion_tri(f_ema, f_ema, index_target)

           # loss_tri_soft = self.criterion_tri_soft(outputs_target, outputs_ema, label_prob)
            
            loss = (1 - self.lmd) * (source_pid_loss)+ self.lmd *(loss_un)+loss_ce_soft#+0.5*(loss_tri)#+0.5*loss_tri_soft#+0.5*loss_ce_1#0.5*loss_ce_2##+0.5*loss_tri_soft#+loss_ce_soft# +0.5*loss_tri_soft  +loss_ce_soft
           # loss = (loss_un)+0.5*loss_tri_soft+0.5*(loss_tri)#+0.5*loss_tri_soft#+loss_ce_soft#+0.5*(loss_tri) +0.5*loss_tri_soft
           # loss=Variable(loss,requires_grad=True)
            
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

            #loss_print['loss_tri_soft'] = loss_tri_soft.item()
            loss_print['loss_tri'] = loss_tri.item()

            loss_print['loss_ce_soft'] = loss_ce_soft.item()

         #   loss_print['loss_ce_1'] = loss_ce_1.item()
          #  loss_print['loss_ce_2'] = loss_ce_2.item()

            #loss_print['loss_ce_soft'] = loss_ce_soft.item()
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
            i = cnt
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

