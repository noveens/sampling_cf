import torch
import torch.nn.functional as F

from torch_utils import is_cuda_available

class CustomLoss(torch.nn.Module):
    def __init__(self, hyper_params):
        super(CustomLoss, self).__init__()
        self.forward = {
            'explicit': self.mse,
            'implicit': self.bpr,
            'sequential': self.bpr,
        }[hyper_params['task']]

        if hyper_params['model_type'] == "MVAE": self.forward = self.vae_loss
        if hyper_params['model_type'] == "SVAE": self.forward = self.svae_loss
        if hyper_params['model_type'] == "SASRec": self.forward = self.bce_sasrec

        self.torch_bce = torch.nn.BCEWithLogitsLoss()
        self.anneal_val = 0.0
        self.hyper_params = hyper_params

    def mse(self, output, y, return_mean = True):
        mse = torch.pow(output - y, 2)
                
        if return_mean: return torch.mean(mse)
        return mse

    def bce_sasrec(self, output, pos, return_mean = True):
        pos_logits, neg_logits = output
        pos_labels, neg_labels = torch.ones(pos_logits.shape), torch.zeros(neg_logits.shape)
        if is_cuda_available: pos_labels, neg_labels = pos_labels.cuda(), neg_labels.cuda()

        indices = pos != self.hyper_params['total_items']

        loss = self.torch_bce(pos_logits[indices], pos_labels[indices])
        loss += self.torch_bce(neg_logits[indices], neg_labels[indices])
        return loss

    def bpr(self, output, y, return_mean = True):
        pos_output, neg_output = output
        pos_output = pos_output.repeat(1, neg_output.shape[1]).view(-1)
        neg_output = neg_output.view(-1)
        
        loss = -F.logsigmoid(pos_output - neg_output)
                
        if return_mean: return torch.mean(loss)
        return loss

    def anneal(self, step_size):
        self.anneal_val += step_size
        self.anneal_val = max(self.anneal_val, 0.2)

    def vae_loss(self, output, y_true_s, return_mean = True):
        decoder_output, mu_q, logvar_q = output

        # Calculate KL Divergence loss
        kld = torch.mean(torch.sum(0.5 * (-logvar_q + torch.exp(logvar_q) + mu_q**2 - 1), -1))
    
        # Calculate Likelihood
        decoder_output = F.log_softmax(decoder_output, -1)
        likelihood = torch.sum(-1.0 * y_true_s * decoder_output, -1)
        
        final = (self.anneal_val * kld) + (likelihood)
        
        if return_mean: return torch.mean(final)
        return final

    def svae_loss(self, output, y, return_mean = True):
        decoder_output, mu_q, logvar_q = output
        dec_shape = decoder_output.shape # [batch_size x seq_len x total_items]

        # Calculate KL Divergence loss
        kld = torch.mean(torch.sum(0.5 * (-logvar_q + torch.exp(logvar_q) + mu_q**2 - 1), -1))
    
        # Don't compute loss on padded items
        y_true_s, y_indices = y
        keep_indices = y_indices != self.hyper_params['total_items']
        y_true_s = y_true_s[keep_indices]
        decoder_output = decoder_output[keep_indices]

        # Calculate Likelihood
        decoder_output = F.log_softmax(decoder_output, -1)
        likelihood = torch.sum(-1.0 * y_true_s * decoder_output)
        likelihood = likelihood / float(dec_shape[0] * self.hyper_params['num_next'])
        
        final = (self.anneal_val * kld) + (likelihood)
        
        if return_mean: return torch.mean(final)
        return final
