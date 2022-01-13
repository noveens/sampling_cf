import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_utils import LongTensor, FloatTensor

class BaseMF(nn.Module):
    def __init__(self, hyper_params, keep_gamma = True):
        super(BaseMF, self).__init__()
        self.hyper_params = hyper_params

        # Declaring alpha, beta, gamma
        self.global_bias = nn.Parameter(FloatTensor([ 4.0 if hyper_params['task'] == 'explicit' else 0.5 ]))
        self.user_bias = nn.Parameter(FloatTensor([ 0.0 for _ in range(hyper_params['total_users']) ]))
        self.item_bias = nn.Parameter(FloatTensor([ 0.0 for _ in range(hyper_params['total_items']) ]))
        if keep_gamma:
            self.user_embedding = nn.Embedding(hyper_params['total_users'], hyper_params['latent_size'])
            self.item_embedding = nn.Embedding(hyper_params['total_items'], hyper_params['latent_size'])

        # For faster evaluation
        self.all_items_vector = LongTensor(
            list(range(hyper_params['total_items']))
        )

    def get_score(self, data):
        pass # Virtual function, implement in all sub-classes

    def forward(self, data, eval = False):
        user_id, pos_item_id, neg_items = data

        # Evaluation -- Rank all items
        if pos_item_id is None: 
            ret = []
            for b in range(user_id.shape[0]):
                ret.append(self.get_score(
                    user_id[b].unsqueeze(-1).repeat(1, self.hyper_params['total_items']).view(-1), 
                    self.all_items_vector.view(-1)
                ).view(1, -1))
            return torch.cat(ret)
        
        # Explicit feedback
        if neg_items is None: return self.get_score(user_id, pos_item_id.squeeze(-1))
        
        # Implicit feedback
        return self.get_score(
            user_id.unsqueeze(-1).repeat(1, pos_item_id.shape[1]).view(-1), 
            pos_item_id.view(-1)
        ).view(pos_item_id.shape), self.get_score(
            user_id.unsqueeze(-1).repeat(1, neg_items.shape[1]).view(-1), 
            neg_items.view(-1)
        ).view(neg_items.shape)

class MF(BaseMF):
    def __init__(self, hyper_params):
        keep_gamma = hyper_params['model_type'] != 'bias_only'

        super(MF, self).__init__(hyper_params, keep_gamma = keep_gamma)
        if keep_gamma: self.dropout = nn.Dropout(hyper_params['dropout'])

        if hyper_params['model_type'] == 'MF':
            latent_size = hyper_params['latent_size']

            self.projection = nn.Sequential(
                nn.Dropout(hyper_params['dropout']),
                nn.Linear(2 * latent_size, latent_size),
                nn.ReLU(),
                nn.Linear(latent_size, latent_size)
            )
            for m in self.projection:
                if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight)

            self.final = nn.Linear(2 * latent_size, 1)
            self.sigmoid = nn.Sigmoid()
            self.relu = nn.ReLU()

    def get_score(self, user_id, item_id):
        # For the FM
        user_bias = self.user_bias.gather(0, user_id.view(-1)).view(user_id.shape)
        item_bias = self.item_bias.gather(0, item_id.view(-1)).view(item_id.shape)

        if self.hyper_params['model_type'] == 'bias_only': 
            return user_bias + item_bias + self.global_bias

        # Embed Latent space
        user = self.dropout(self.user_embedding(user_id.view(-1))) # [bsz x 32]
        item = self.dropout(self.item_embedding(item_id.view(-1))) # [bsz x 32]

        # Dot product
        if self.hyper_params['model_type'] == 'MF_dot':
            rating = torch.sum(user * item, dim = -1).view(user_id.shape)
            return user_bias + item_bias + self.global_bias + rating

        mf_vector = user * item
        cat = torch.cat([ user, item ], dim = -1)
        mlp_vector = self.projection(cat)

        # Concatenate and get single score
        cat = torch.cat([ mlp_vector, mf_vector ], dim = -1)
        rating = self.final(cat)[:, 0].view(user_id.shape) # [bsz]

        return user_bias + item_bias + self.global_bias + rating
