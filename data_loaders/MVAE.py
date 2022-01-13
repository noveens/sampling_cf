import numpy as np

from data_loaders.base import BaseTrainDataset, BaseTestDataset
from torch_utils import LongTensor, FloatTensor

class TrainDataset(BaseTrainDataset):
    def __init__(self, data, hyper_params, track_events):
        super(TrainDataset, self).__init__(data, hyper_params)
        self.shuffle_allowed = not track_events

        self.user_history = np.array(self.pad(self.user_history))
        self.num_interactions = self.num_users

    def __iter__(self):
        # Important for optimal and stable performance
        indices = np.arange(self.num_interactions)
        if self.shuffle_allowed: np.random.shuffle(indices)
        temp = self.user_history[indices].tolist()

        for u in range(0, self.num_interactions, self.batch_size):
            batch = temp[u:u+self.batch_size]
            x_and_y = self.scatter(batch, FloatTensor, self.num_items)
            yield [ x_and_y, None, None ], x_and_y

class TestDataset(BaseTestDataset):
    def __init__(self, data, train_data, hyper_params, val_data = None, test_set = False):
        super(TestDataset, self).__init__(data, train_data, hyper_params, val_data)
        self.test_set = test_set

        # Padding for easier scattering
        self.train_user_history_full = list(map(lambda x: LongTensor(x), self.train_user_history))
        self.train_user_history = self.pad(self.train_user_history)
        self.num_interactions = self.num_users

        self.init_background_sampler(
            lambda : np.array(self.sample_negatives(
                self.num_users, self.hyper_params['num_test_negs'],
                lambda point, random_neg: random_neg not in self.train_user_history_set[point] and \
                                          random_neg not in self.test_user_history_set[point]
            ))
        )

    def __iter__(self):
        ## No sampled ranking required as model by default needs to compute score over all items
        ## Will only be used for AUC computation
        negatives = self.result_queue.get() ; self.event.set()

        for u in range(0, self.num_interactions, self.batch_size):
            train_positive_full = self.train_user_history_full[u:u+self.batch_size]
            train_positive      = self.train_user_history[u:u+self.batch_size]
            test_positive_set   = self.test_user_history_set[u:u+self.batch_size]
            test_negative       = negatives[u:u+self.batch_size]

            yield [ self.scatter(train_positive, FloatTensor, self.num_items), None, test_negative ], [ 
                train_positive_full,
                test_positive_set,
            ]
