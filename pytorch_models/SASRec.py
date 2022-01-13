import torch
import numpy as np
import torch.nn as nn

from torch_utils import LongTensor, BoolTensor, is_cuda_available

class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

class SASRec(torch.nn.Module):
    def __init__(self, hyper_params):
        super(SASRec, self).__init__()
        self.hyper_params = hyper_params
        self.item_num = hyper_params['total_items']

        self.item_emb = torch.nn.Embedding(self.item_num+1, hyper_params['latent_size'], padding_idx=self.item_num)
        self.pos_emb = torch.nn.Embedding(hyper_params['max_seq_len'], hyper_params['latent_size']) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=hyper_params['dropout'])

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(hyper_params['latent_size'], eps=1e-8)

        for _ in range(hyper_params['num_blocks']):
            new_attn_layernorm = torch.nn.LayerNorm(hyper_params['latent_size'], eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(
                hyper_params['latent_size'],
                hyper_params['num_heads'],
                hyper_params['dropout']
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(hyper_params['latent_size'], eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(hyper_params['latent_size'], hyper_params['dropout'])
            self.forward_layers.append(new_fwd_layer)

    def log2feats(self, log_seqs):
        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(LongTensor(positions))
        seqs = self.emb_dropout(seqs)

        timeline_mask = BoolTensor(log_seqs == self.item_num)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        temp = torch.ones((tl, tl), dtype=torch.bool)
        if is_cuda_available: temp = temp.cuda()
        attention_mask = ~torch.tril(temp)

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def get_score(self, log_feats, items):
        embs = self.item_emb(items)
        return (log_feats * embs).sum(dim=-1)

    def forward(self, data, eval = False):
        log_seqs, pos_seqs, neg_seqs = data

        # Embed sequence
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        if eval:
            log_feats = log_feats[:, -1, :]

            # Rank all items
            if pos_seqs is None: 
                return torch.matmul(
                    log_feats, 
                    self.item_emb.weight.transpose(0, 1)
                )[:, :-1]
            
            # Sampled evaluation
            orig_shape = neg_seqs.shape
            return self.get_score(log_feats.unsqueeze(1), pos_seqs), self.get_score(
                log_feats.unsqueeze(1).repeat(1, orig_shape[1], 1), neg_seqs
            ).view(orig_shape)

        # Dot product
        orig_shape = neg_seqs.shape
        return self.get_score(log_feats, pos_seqs).unsqueeze(-1).repeat(1, 1, orig_shape[2]), \
        self.get_score(
            log_feats.unsqueeze(2).repeat(1, 1, orig_shape[2], 1).view(orig_shape[0], orig_shape[1] * orig_shape[2], -1), 
            neg_seqs.view(orig_shape[0], orig_shape[1] * orig_shape[2])
        ).view(orig_shape)
