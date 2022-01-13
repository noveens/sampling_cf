import torch
import numpy as np

from data_loaders.base import BaseTrainDataset, BaseTestDataset
from torch_utils import LongTensor, is_cuda_available

class TrainDataset(BaseTrainDataset):
    def __init__(self, data, hyper_params, track_events):
        super(TrainDataset, self).__init__(data, hyper_params)
        self.shuffle_allowed = not track_events

        self.user_history = LongTensor(self.sequential_pad(self.user_history, hyper_params))
        self.num_interactions = self.num_users

        self.init_background_sampler(
            lambda : torch.LongTensor(self.sample_negatives(
                self.num_users, (self.hyper_params['max_seq_len'] - 1) * self.hyper_params['num_train_negs'],
                lambda point, random_neg: random_neg not in self.user_history_set[point]
            )).view(self.num_users, self.hyper_params['max_seq_len'] - 1, self.hyper_params['num_train_negs'])
        )

    def __iter__(self):
        # Important for optimal and stable performance
        indices = np.arange(self.num_interactions)
        if self.shuffle_allowed: 
            np.random.shuffle(indices)
            temp = self.user_history[indices]

        negatives = self.result_queue.get()[indices]
        if is_cuda_available: negatives = negatives.cuda()
        self.event.set()

        for u in range(0, self.num_interactions, self.batch_size):
            sequence = temp[u:u+self.batch_size] # self.user_history
            x = sequence[:, :-1]
            y = sequence[:, 1:]
            neg = negatives[u:u+self.batch_size, :, :]

            yield [ x, y, neg ], y

class TestDataset(BaseTestDataset):
    def __init__(self, data, train_data, hyper_params, val_data = None, test_set = False):
        super(TestDataset, self).__init__(data, train_data, hyper_params, val_data)
        self.test_set = test_set

        # Padding for easier scattering
        self.train_user_history_full = list(map(lambda x: LongTensor(x), self.train_user_history))
        self.train_user_history = LongTensor(self.sequential_pad(self.train_user_history, hyper_params))
        self.test_user_history = LongTensor(self.pad(self.test_user_history))

        # Total number of interactions
        self.num_interactions = self.num_users

        self.partial_eval = (not self.test_set) and self.hyper_params['partial_eval']

        def one_sample():
            negatives = self.sample_negatives(
                self.num_users, self.hyper_params['num_test_negs'],
                lambda point, random_neg: random_neg not in self.train_user_history_set[point] and \
                                          random_neg not in self.test_user_history_set[point]
            )
            if self.partial_eval: negatives = torch.LongTensor(negatives) # Sampled ranking
            else: negatives = np.array(negatives) # Sampled AUC
            return negatives

        self.init_background_sampler(one_sample)

    def __iter__(self):
        negatives = self.result_queue.get() ; self.event.set()
        if self.partial_eval and is_cuda_available: negatives = negatives.cuda()

        for u in range(0, self.num_interactions, self.batch_size):
            train_positive      = self.train_user_history[u:u+self.batch_size]
            train_positive_full = self.train_user_history_full[u:u+self.batch_size]
            test_positive       = self.test_user_history[u:u+self.batch_size]
            test_positive_set   = self.test_user_history_set[u:u+self.batch_size]
            test_negative       = negatives[u:u+self.batch_size]

            yield [ train_positive, test_positive if self.partial_eval else None, test_negative ], [ 
                train_positive_full,
                test_positive_set,
            ]
