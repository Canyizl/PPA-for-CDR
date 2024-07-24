from pickle import NONE

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import repeat

class TransformerEncoder(nn.Module):
    # sa_sizes, hidden_dim=32, norm_in=True, attend_heads=1
    def __init__(self, obs_dim, hidden_dim, opt):
        super(TransformerEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        # self.out_hidden_dim = args.out_hid_size
        self.attend_heads = 8
        self.maxlen = opt["maxlen"]
        self.task = opt["task"]
        assert (self.hidden_dim % self.attend_heads) == 0
        self.n_layers = 1
        self.obs_dim = obs_dim

        self.layernorm = nn.LayerNorm(self.hidden_dim)
        self.hid_activation = nn.LeakyReLU(0.2)

        self.hypernet = nn.Linear(1,self.hidden_dim) # 8 16
        self.hyper_fc = nn.Linear(self.hidden_dim,1)
        self.lamb_fc = nn.Linear(self.hidden_dim,1)
        self.fc_1 = nn.Bilinear(self.hidden_dim, self.hidden_dim, 1)
        self.hid_act = nn.Tanh() #nn.Sigmoid() 
        self.W_agg = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
    
    def mean_pooling(self, sequence_emb):
        mask = sequence_emb.sum(dim=-1) != 0
        mean = sequence_emb.sum(dim=1) / (mask.float().sum(dim=-1, keepdim=True) + 1.e-12)
        return mean
    
    def item_similarity_pooling(self, sequence_emb, score):
        if len(score.size()) != 2:
            score = score.view(score.size(0), -1)
        score = F.softmax(score, dim = -1)
        score = score.unsqueeze(-1)
        ans = (score * sequence_emb).sum(dim=1)
        return self.W_agg(ans)

    def forward(self, user_emb, sequence_emb, score):
        # obs : (b*n, self.obs_num, self.obs_dim)
        # hidden_state : (b, n, self.hidden_dim )
        # agent_index : (b*n, 1)
        # mask : (b*n, 1, self.obs_num + 1)
        b,n,_ = sequence_emb.shape


        x = sequence_emb
        if "multi" in self.task:
            if "user" in self.task:
                x = sequence_emb.transpose(1,2)
                if len(score.size()) != 2:
                    score = score.view(score.size(0), -1)
                score = F.softmax(score, dim = -1)
                score = score.unsqueeze(-1)
                score = self.hypernet(score).abs()
                output = self.hid_activation(torch.matmul(x,score))
                output = self.hyper_fc(output).squeeze(-1)
                output = self.W_agg(output)
            if "item" in self.task:
                output = self.item_similarity_pooling(sequence_emb, score)
        else:
            output = self.mean_pooling(x)
        

        # mean
        # for layer in self.attn_layers:
        #     x = layer(x, mask)
        # output = (x * final_mask).sum(dim=1) / final_mask.sum(dim=1)
        return output