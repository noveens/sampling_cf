import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

from torch_utils import is_cuda_available

class Encoder(nn.Module):
    def __init__(self, hyper_params):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(
            hyper_params['total_items'], hyper_params['latent_size']
        )
        nn.init.xavier_normal_(self.linear1.weight)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        return x

class Decoder(nn.Module):
    def __init__(self, hyper_params):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(hyper_params['latent_size'], hyper_params['latent_size'])
        self.linear2 = nn.Linear(hyper_params['latent_size'], hyper_params['total_items'])
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

class MVAE(nn.Module):
    def __init__(self, hyper_params):
        super(MVAE, self).__init__()
        self.hyper_params = hyper_params
        
        self.encoder = Encoder(hyper_params)
        self.decoder = Decoder(hyper_params)
        
        self.linear1 = nn.Linear(hyper_params['latent_size'], 2 * hyper_params['latent_size'])
        nn.init.xavier_normal_(self.linear1.weight)
        
        self.tanh = nn.Tanh()
        
    def sample_latent(self, h_enc):
        temp_out = self.linear1(h_enc)
        
        mu = temp_out[:, :self.hyper_params['latent_size']]
        log_sigma = temp_out[:, self.hyper_params['latent_size']:]
        
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()
        if is_cuda_available: std_z = std_z.cuda()

        return mu + sigma * Variable(std_z, requires_grad=False), mu, log_sigma  # Reparameterization trick

    def forward(self, data, eval = False):
        x, _, _ = data
        enc_out = self.encoder(x)
        sampled_z, z_mean, z_log_sigma = self.sample_latent(enc_out)
        dec_out = self.decoder(sampled_z)
                              
        return dec_out, z_mean, z_log_sigma
