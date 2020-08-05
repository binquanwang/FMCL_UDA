import torch
import torch.nn.functional as F
from torch import nn, autograd
from torch.autograd import Variable, Function
import numpy as np
import math
import tensorflow as tf



class ExemplarMemory(Function):
    def __init__(self, em, alpha=0.01):
        super(ExemplarMemory, self).__init__()
        self.em = em
        self.alpha = alpha

    def forward(self, inputs, targets):
        self.save_for_backward(inputs, targets)
        outputs = inputs.mm(self.em.t())
        return outputs

    def backward(self, grad_outputs):
        inputs, targets = self.saved_tensors
        grad_inputs = None
        if self.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(self.em)
        for x, y in zip(inputs, targets):
            self.em[y] = self.alpha * self.em[y] + (1. - self.alpha) * x
            self.em[y] /= self.em[y].norm()
        return grad_inputs, None


def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    print(torch.distributed.is_initialized())
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


# Invariance learning loss
class InvNet(nn.Module):
    def __init__(self, num_features, num_classes, beta=0.05, knn=6, alpha=0.01):
        super(InvNet, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_features = num_features
        self.num_classes = num_classes
        self.alpha = alpha  # Memory update rate
        self.beta = beta  # Temperature fact
        self.knn = knn  # Knn for neighborhood invariance


       # num_classes = 13056
       # num_classes = 512
       

        self.register_buffer("queue", torch.randn(num_features, num_classes))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_2", torch.randn(num_features, num_classes))
        self.queue_2 = nn.functional.normalize(self.queue_2, dim=0)
        self.register_buffer("queue_3", torch.randn(1041, num_classes))
        self.queue_3 = nn.functional.normalize(self.queue_3, dim=0)
        self.register_buffer("queue_4", torch.randn(1041, num_classes))
        self.queue_4 = nn.functional.normalize(self.queue_4, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.K = num_classes
        self.T = 0.07

        # Exemplar memory
        #self.em = nn.Parameter(torch.zeros(num_classes, num_features))


    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
      #  keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
       # print(batch_size,keys)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def _dequeue_and_enqueue_2(self, inputs_ema,targets):
        # gather keys before updating queue
      #  keys = concat_all_gather(keys)

        for x, y in zip(inputs_ema, targets):
            #self.queue.data.t()[y]= mu *self.queue.data.t()[y]+(1-mu)*x
            self.queue.t()[y] = x
            #self.queue[y] /= self.em[y].norm()

    def _dequeue_and_enqueue_3(self, inputs_ema,targets):
        # gather keys before updating queue
      #  keys = concat_all_gather(keys)

        for x2, y2 in zip(inputs_ema, targets):
            #self.queue.data.t()[y]= mu *self.queue.data.t()[y]+(1-mu)*x
            self.queue_2.t()[y2] = x2
            #self.queue[y] /= self.em[y].norm()

    def _dequeue_and_enqueue_4(self, inputs_ema,targets):
        # gather keys before updating queue
      #  keys = concat_all_gather(keys)

        for x3, y3 in zip(inputs_ema, targets):
            #self.queue.data.t()[y]= mu *self.queue.data.t()[y]+(1-mu)*x
            self.queue_3.t()[y3] = x3
            #self.queue[y] /= self.em[y].norm()

    def _dequeue_and_enqueue_5(self, inputs_ema,targets):
        # gather keys before updating queue
      #  keys = concat_all_gather(keys)

        for x4, y4 in zip(inputs_ema, targets):
            #self.queue.data.t()[y]= mu *self.queue.data.t()[y]+(1-mu)*x
            self.queue_4.t()[y4] = x4
            #self.queue[y] /= self.em[y].norm()


    def forward(self, inputs_feature, inputs_ema,prob,prob_ema, targets, epoch=None):

       # alpha = self.alpha * epoch

        #l_pos = inputs_feature.mm(inputs_ema.t())
        #inputs = inputs_feature.mm(self.queue.clone().detach())
        inputs_feature = nn.functional.normalize(inputs_feature, dim=1)
        inputs_ema = nn.functional.normalize(inputs_ema, dim=1)

        prob = nn.functional.normalize(prob, dim=1)
        prob_ema = nn.functional.normalize(prob_ema, dim=1)
        
        #inputs_memory = nn.functional.normalize(inputs_memory.detach(), dim=1)

        

        l_pos = torch.einsum('nc,nc->n', [inputs_feature, inputs_ema]).unsqueeze(-1)
       # l_pos = torch.einsum('nc,nc->n', [prob, prob_ema]).unsqueeze(-1)

        
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [inputs_feature, self.queue.clone().detach()])
       # l_neg = torch.einsum('nc,ck->nk', [prob, self.queue_4.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos,l_neg], dim=1)

        #labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        
        self._dequeue_and_enqueue_2(inputs_ema.clone().detach(),targets)
        self._dequeue_and_enqueue_3(inputs_feature.clone().detach(),targets)
        self._dequeue_and_enqueue_4(prob.clone().detach(),targets)
        self._dequeue_and_enqueue_5(prob_ema.clone().detach(),targets)
        

       # print(targets,logits)
       # mu = min(self.alpha / 100 * (epoch + 1), 1.0)


       # self._dequeue_and_enqueue_2(inputs_ema,targets)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        

       # inputs = ExemplarMemory(self.em, alpha=alpha)(inputs, targets)

       # targets = targets+1
        inputs = logits/self.beta
       # inputs = inputs_long[:,1:self.K+1]
        
     #   print(logits.size(),l_pos.size(),l_neg.size(),inputs.size())
        if self.knn > 0 and epoch >=20:
            # With neighborhood invariance
            loss,targets = self.smooth_loss(inputs, targets)
        else:
            # Without neighborhood invariance
            loss = F.cross_entropy(inputs, targets)
        return loss,self.queue.clone().detach(),self.queue_2.clone().detach(),self.queue_3.clone().detach(),self.queue_4.clone().detach(),targets
    
    def smooth_loss(self, inputs, targets):
        targets = self.smooth_hot(inputs.detach().clone(), targets.detach().clone(), self.knn)
        outputs = F.log_softmax(inputs, dim=1)
       # print(targets.size(),outputs.size())
        loss = - (targets * outputs)
        loss = loss.sum(dim=1)
        loss = loss.mean(dim=0)
        return loss,targets
    
    '''
    def smooth_loss(self, inputs, targets):
        alpha=0.5
        beta=0.5
        targets_line = self.smooth_hot(inputs.detach().clone(), targets.detach().clone(), self.knn)

        y_true_1 = targets_line
        y_pred_1 = F.softmax(inputs, dim=1)

        y_true_2 = targets_line
        y_pred_2 = F.softmax(inputs, dim=1)

        #print(y_pred_1,y_true_2,'...............')

        y_pred_1 = torch.clamp(y_pred_1, 1e-4, 1.0, out=None)
        y_true_2 = torch.clamp(y_true_2, 1e-4, 1.0, out=None)

       # print(y_pred_1,y_true_2)
            
        
        y_pred_1 = tf.clip_by_value(y_pred_1.cpu().detach(), 1e-7, 1.0)
        y_true_2 = tf.clip_by_value(y_true_2.cpu().detach(), 1e-4, 1.0)

        #print(y_pred_1.type())
        
        loss = alpha*tf.reduce_mean(-tf.reduce_sum(y_true_1 * tf.log(y_pred_1), axis = -1)) + beta*tf.reduce_mean(-tf.reduce_sum(y_pred_2 * tf.log(y_true_2), axis = -1))
        
        return loss,ks
        
        
        outputs_1 = torch.log(y_pred_1)
        loss_1 = - (y_true_1 * outputs_1)
        loss_1 = loss_1.sum(dim=1)
        loss_1 = loss_1.mean(dim=0)
        
        outputs_2 = torch.log(y_true_2)
        loss_2 = - (y_pred_2 * outputs_2)
        loss_2 = loss_2.sum(dim=1)
        loss_2 = loss_2.mean(dim=0)

        #print(loss_2)

        loss = alpha*loss_1 + beta*loss_2
        return loss
    '''
	
    


    def smooth_hot(self, inputs, targets, k=6):
        # Sort
        _, index_sorted = torch.sort(inputs, dim=1, descending=True)
       # print(index_sorted[:, 0:k],inputs)

        ones_mat = torch.ones(targets.size(0), k).to(self.device)
        targets = torch.unsqueeze(targets, 1)
        targets_onehot = torch.zeros(inputs.size()).to(self.device)
      #  print(ones_mat,targets_onehot,targets_onehot.size())
        f = torch.ones(targets.size(0), k).to(self.device)
        for i in range(targets.size(0)):
            f[i,:]=inputs[i, index_sorted[i, 1:k+1]]

       #  weights = F.softmax(ones_mat, dim=1)
       # print(ones_mat.size(),inputs[:,index_sorted[:, 0:k]].size())
        # print(f)
        weights = F.softmax(f, dim=1)
        # print(weights)
        targets_onehot.scatter_(1, index_sorted[:, 1:k+1], ones_mat*weights)

       #targets_onehot = targets_onehot*3.5/(k*np.log(k))
        

        
       # print(targets_onehot)
        targets_onehot.scatter_(1, targets, float(1))

        #labels = torch.zeros(inputs.shape[0], dtype=torch.long).cuda()
        #labels = torch.unsqueeze(labels, 1)
        #targets_onehot.scatter_(1, labels, float(1))
        #print(targets_onehot)
       # print('.........................',targets_onehot,targets_onehot.size(),index_sorted[:, 0:k],index_sorted)

      #  for i in range(128):
       #     print(targets_onehot[i, index_sorted[i, 0:k]])
      #  print(inputs[:, index_sorted[:, 0:k]],inputs[:, index_sorted[:, 0:k]].size())



       # print(f,'ffffffffff',f.size())
        

        return targets_onehot



