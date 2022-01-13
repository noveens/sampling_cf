import torch
import numpy as np

from data_loaders.base import BaseTrainDataset, BaseTestDataset
from torch_utils import LongTensor, FloatTensor, is_cuda_available

class TrainDataset(BaseTrainDataset):
    def __init__(self, data, hyper_params, track_events):
        super(TrainDataset, self).__init__(data, hyper_params)
        self.shuffle_allowed = not track_events

        # Copying ENTIRE dataset to GPU
        self.users_cpu = list(map(lambda x: x[0], data))
        self.users = LongTensor(self.users_cpu)
        self.items = LongTensor(list(map(lambda x: x[1], data)))
        self.ratings = FloatTensor(list(map(lambda x: x[2], data)))

        self.num_interactions = len(data)

        self.init_background_sampler(
            lambda : torch.LongTensor(self.sample_negatives(
                len(self.data), self.hyper_params['num_train_negs'],
                lambda point, random_neg: random_neg not in self.user_history_set[self.users_cpu[point]]
            ))
        )

    def __iter__(self):
        # Important for optimal and stable performance
        indices = np.arange(self.num_interactions)
        if self.shuffle_allowed: np.random.shuffle(indices)
        temp_users = self.users[indices] ; temp_items = self.items[indices] ; temp_ratings = self.ratings[indices]

        if self.implicit_task: 
            negatives = self.result_queue.get()[indices]
            if is_cuda_available: negatives = negatives.cuda()
            self.event.set()

        for i in range(0, self.num_interactions, self.batch_size):
            yield [ 
                temp_users[i:i+self.batch_size], 
                temp_items[i:i+self.batch_size].unsqueeze(-1), 
                negatives[i:i+self.batch_size] if self.implicit_task else None, 
            ], temp_ratings[i:i+self.batch_size]

class TestDataset(BaseTestDataset):
    def __init__(self, data, train_data, hyper_params, val_data = None, test_set = False):
        super(TestDataset, self).__init__(data, train_data, hyper_params, val_data)
        self.test_set = test_set

        if self.implicit_task:
            # Padding for easier scattering
            self.test_user_history = LongTensor(self.pad(self.test_user_history))
            self.train_user_history = list(map(lambda x: LongTensor(x), self.train_user_history))

            # Copying all user-IDs to GPU
            self.all_users = LongTensor(list(range(self.num_users)))

            self.partial_eval = (not test_set) and hyper_params['partial_eval']

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

        else:
            self.users = LongTensor(list(map(lambda x: x[0], data)))
            self.items = LongTensor(list(map(lambda x: x[1], data)))
            self.ratings = FloatTensor(list(map(lambda x: x[2], data)))

        self.num_interactions = self.num_users if self.implicit_task else len(data)

    def __iter__(self):
        if self.implicit_task:
            negatives = self.result_queue.get() ; self.event.set()
            if self.partial_eval and is_cuda_available: negatives = negatives.cuda()

        for u in range(0, self.num_interactions, self.batch_size):
            if self.implicit_task:
                batch             = self.all_users[u:u+self.batch_size]
                train_positive    = self.train_user_history[u:u+self.batch_size]
                test_positive     = self.test_user_history[u:u+self.batch_size]
                test_positive_set = self.test_user_history_set[u:u+self.batch_size]
                test_negative     = negatives[u:u+self.batch_size]

                yield [ batch, test_positive if self.partial_eval else None, test_negative ], [ 
                    train_positive,
                    test_positive_set,
                ]
            else:
                yield [ 
                    self.users[u:u+self.batch_size], 
                    self.items[u:u+self.batch_size].unsqueeze(-1), 
                    None, 
                ], self.ratings[u:u+self.batch_size]
