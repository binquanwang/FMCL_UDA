from __future__ import absolute_import

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable



def euclidean_dist(x, y):
	m, n = x.size(0), y.size(0)
	xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
	yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
	dist = xx + yy
	dist.addmm_(1, -2, x, y.t())
	dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
	return dist

def cosine_dist(x, y):
	bs1, bs2 = x.size(0), y.size(0)
	frac_up = torch.matmul(x, y.transpose(0, 1))
	frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2), 1))).view(bs1, 1).repeat(1, bs2) * \
	            (torch.sqrt(torch.sum(torch.pow(y, 2), 1))).view(1, bs2).repeat(bs1, 1)
	cosine = frac_up / frac_down
	return 1-cosine

def _batch_hard(mat_distance, mat_similarity, indice=False):
	sorted_mat_distance, positive_indices = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1, descending=True)
	hard_p = sorted_mat_distance[:, 0]
	hard_p_indice = positive_indices[:, 0]
	sorted_mat_distance, negative_indices = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
	#index = np.random.randint(0, 127, 1)
	#print(index)
	#hard_p = sorted_mat_distance[:, 0]
	#hard_p_indice = positive_indices[:, 0]
	hard_n = sorted_mat_distance[:, 0]
	hard_n_indice = negative_indices[:, 0]
	'''
	for i in range(len(index)):
                hard_n[i] = sorted_mat_distance[i, index[i]]
                hard_n_indice[i] = negative_indices[i, index[i]]
                
	#hard_n = sorted_mat_distance[:, index]
	#hard_n = hard_n.mean(dim=1)
	#hard_n_indice = negative_indices[:, index]
	'''
	#print(hard_p,'............', hard_n)
	if(indice):
		return hard_p, hard_n, hard_p_indice, hard_n_indice
	return hard_p, hard_n

class TripletLoss(nn.Module):
	'''
	Compute Triplet loss augmented with Batch Hard
	Details can be seen in 'In defense of the Triplet Loss for Person Re-Identification'
	'''

	def __init__(self, margin, normalize_feature=False):
		super(TripletLoss, self).__init__()
		self.margin = margin
		self.normalize_feature = normalize_feature
		self.margin_loss = nn.MarginRankingLoss(margin=margin).cuda()
		self.ranking_loss = nn.SoftMarginLoss().cuda()

	def forward(self, emb,emb2, label):
		if self.normalize_feature:
			# equal to cosine similarity
			emb = F.normalize(emb)
		mat_dist = euclidean_dist(emb, emb2)
		#print(mat_dist)
		# mat_dist = cosine_dist(emb, emb)
		assert mat_dist.size(0) == mat_dist.size(1)
		N = mat_dist.size(0)
		mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()

		dist_ap, dist_an = _batch_hard(mat_dist, mat_sim)
		#print(dist_ap, dist_an)
		assert dist_an.size(0)==dist_ap.size(0)
		y = torch.ones_like(dist_ap)

		if self.margin is not None:
                      loss = self.margin_loss(dist_an, dist_ap, y)
		else:
                      loss = self.ranking_loss(dist_an - dist_ap, y)
		
		#loss = self.margin_loss(dist_an, dist_ap, y)
		prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
		return loss

class SoftTripletLoss(nn.Module):

	def __init__(self, margin=None, normalize_feature=True):
		super(SoftTripletLoss, self).__init__()
		self.margin = margin
		self.normalize_feature = normalize_feature

	def forward(self, emb1, emb2, label):
		if self.normalize_feature:
			# equal to cosine similarity
			emb1 = F.normalize(emb1)
			emb2 = F.normalize(emb2)

		mat_dist = euclidean_dist(emb1, emb1)
		assert mat_dist.size(0) == mat_dist.size(1)
		N = mat_dist.size(0)
		mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()

		dist_ap, dist_an, ap_idx, an_idx = _batch_hard(mat_dist, mat_sim, indice=True)
		assert dist_an.size(0)==dist_ap.size(0)
		triple_dist = torch.stack((dist_ap, dist_an), dim=1)
		triple_dist = F.log_softmax(triple_dist, dim=1)
		if (self.margin is not None):
			loss = (- self.margin * triple_dist[:,0] - (1 - self.margin) * triple_dist[:,1]).mean()
			return loss

		mat_dist_ref = euclidean_dist(emb2, emb2)
		dist_ap_ref = torch.gather(mat_dist_ref, 1, ap_idx.view(N,1).expand(N,N))[:,0]
		dist_an_ref = torch.gather(mat_dist_ref, 1, an_idx.view(N,1).expand(N,N))[:,0]
		triple_dist_ref = torch.stack((dist_ap_ref, dist_an_ref), dim=1)
		triple_dist_ref = F.softmax(triple_dist_ref, dim=1).detach()

		loss = (- triple_dist_ref * triple_dist).mean(0).sum()
		return loss


class TripletLoss2(nn.Module):
    def __init__(self, margin=0, num_instances=0, use_semi=True):
        super(TripletLoss2, self).__init__()
        self.margin = margin
        self.use_semi = use_semi
        #self.ranking_loss = nn.MarginRankingLoss(margin=self.margin)
        self.ranking_loss = nn.SoftMarginLoss().cuda()
        self.K = num_instances

    def forward(self, inputs, targets, epoch, w=None):
        # if w is not None:
        #     inputs = inputs * w.unsqueeze(1)
        n = inputs.size(0)
        P = n // self.K
        t0 = 20.0
        t1 = 40.0
      
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        if False: ######## curriculum sampling
            mean = max(124-6.0*epoch, 0.0)
            #std = 12*0.001**((epoch-t0)/t0) if epoch >= t0 else 12
            std = 15*0.001**(max((epoch-t0)/(t1-t0), 0.0))
            neg_probs = norm(mean, std).pdf(np.linspace(0,123,124))
            neg_probs = torch.from_numpy(neg_probs).clamp(min=3e-5, max=20.0)
            for i in range(P):
                for j in range(self.K):
                    neg_examples = dist[i*self.K+j][mask[i*self.K+j] == 0]
                    #sort_neg_examples = torch.topk(neg_examples, k=80, largest=False)[0]
                    sort_neg_examples = torch.sort(neg_examples)[0]
                    for pair in range(j+1,self.K):
                        dist_ap.append(dist[i*self.K+j][i*self.K+pair])
                        choosen_neg = sort_neg_examples[torch.multinomial(neg_probs,1).cuda()]
                        dist_an.append(choosen_neg)
        elif self.use_semi:  ######## semi OHEM
            for i in range(P):
                for j in range(self.K):
                    neg_examples = dist[i*self.K+j][mask[i*self.K+j] == 0]
                    for pair in range(j+1,self.K):
                        ap = dist[i*self.K+j][i*self.K+pair]
                        dist_ap.append(ap.view(1))
                        dist_an.append(neg_examples.min().view(1))
        else:  ##### OHEM
            for i in range(n):
                dist_ap.append(dist[i][mask[i]].max().view(1))
                dist_an.append(dist[i][mask[i] == 0].min().view(1))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y) 
        if w is not None:
            loss = 0.
            for i in range(dist_an.size(0)):
                loss += self.ranking_loss(dist_an[i].unsqueeze(0), dist_ap, y)
            loss /= dist_an.size(0)
        else:
            #loss = self.ranking_loss(dist_an, dist_ap, y)
            loss = self.ranking_loss(dist_an-dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss
