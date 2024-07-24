import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import repeat

class TransformerDecoder(nn.Module):
    # sa_sizes, hidden_dim=32, norm_in=True, attend_heads=1
    def __init__(self, opt):
        super(TransformerDecoder, self).__init__()
        self.hid_size = opt["latent_dim"]
        self.task = opt["task"]
        self.epsilon = 1e-4
        self.Softmax = nn.Softmax(dim=-1)
        self.output_dim = 2
        self.num_prototypes_per_class = 24 #4
        self.num_prototypes = self.output_dim * self.num_prototypes_per_class
        self.prototype_shape = (self.num_prototypes, self.hid_size)
        self.prototype_vectors = nn.Parameter(torch.randn(self.num_prototypes, self.hid_size))
        self.last_layer = nn.Linear(self.num_prototypes, self.output_dim,
                                    bias=False)  # do not use bias

        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    self.output_dim)
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // self.num_prototypes_per_class] = 1
        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)

        #self.conv_global = nn.Linear(self.hid_size*2, self.hid_size)

        self.conv_global = nn.Sequential(
            nn.Linear(self.hid_size, self.hid_size),
            nn.ReLU(),
            nn.Linear(self.hid_size, self.hid_size)
        )

        self.relu = nn.ReLU()
        self.fc_layer = nn.Sequential(
            nn.Linear(self.hid_size, 1))
        self.fc_test = nn.Bilinear(self.hid_size, self.hid_size, 1)
        self.fc_test_1 = nn.Bilinear(self.hid_size, self.hid_size, self.hid_size)
        self.fc_test_2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hid_size,1)
        )
        

    def mean_pooling(self, sequence_emb):
        mask = sequence_emb.sum(dim=-1) != 0
        mean = sequence_emb.sum(dim=1) / (mask.float().sum(dim=-1, keepdim=True) + 1.e-12)
        return mean

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def prototype_distances(self, x):
        xp = torch.mm(x, torch.t(self.prototype_vectors))
        distance = -2 * xp + torch.sum(x ** 2, dim=1, keepdim=True) + torch.t(
            torch.sum(self.prototype_vectors ** 2, dim=1, keepdim=True))
        similarity = torch.log((distance+1) / (distance + self.epsilon))
        return similarity, distance

    def prototype_subgraph_distances(self, x, prototype):
        b = x.size(0)
        all_sim = torch.empty(b,self.num_prototypes_per_class,1)
        for j in range(self.num_prototypes_per_class):
            p = prototype[:,j].unsqueeze(1)
            distance = torch.norm(x - p, p=2, dim=-1, keepdim=True) ** 2
            similarity = torch.mean(torch.log((distance + 1) / (distance + self.epsilon)), dim=1)
            all_sim[:,j,:] = similarity
        return all_sim, None

    def forward(self, specific_user, shared_user, flag):
        b,h = shared_user.shape
        if flag:
            idx = torch.LongTensor([1.]).to(shared_user.device)
        else:
            idx = torch.LongTensor([0.]).to(shared_user.device)
        prototype_activations, min_distances = self.prototype_distances(specific_user)
        logits = self.last_layer(prototype_activations)
        select_proto = self.prototype_vectors.reshape(self.output_dim,self.num_prototypes_per_class,self.hid_size).to(shared_user.device)
        sub_prototype = torch.index_select(select_proto, dim=0, index=idx).to(shared_user.device).view(1,-1, self.hid_size).repeat(b,1,1)
        similarity, _ = self.prototype_subgraph_distances(specific_user, sub_prototype)
        chosen_idx = torch.argmax(similarity, dim=1, keepdim=True).to(shared_user.device)
        chosen_idx = chosen_idx.expand(-1, 1, self.hid_size)
        chosen_proto = torch.gather(sub_prototype, index=chosen_idx.long(), dim=1).squeeze(1)
        x_glb = shared_user
        pos = self.fc_test(chosen_proto, shared_user)
        return pos, x_glb, logits, min_distances