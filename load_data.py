import h5py
import numpy as np

from utils import get_data_loader_class
from data_path_constants import get_data_path, get_index_path

def load_data(hyper_params, track_events = False):
    rating_data_path = get_data_path(hyper_params)
    index_path = get_index_path(hyper_params)

    data_holder = DataHolder(rating_data_path, index_path)
    print("# of users: {}\n# of items: {}".format(data_holder.num_users, data_holder.num_items))

    hyper_params['total_users']  = data_holder.num_users
    hyper_params['total_items']  = data_holder.num_items
    # Do a partial item-space evaluation (only on the validation set)
    # if the dataset has too many items
    hyper_params['partial_eval'] = hyper_params['total_items'] > 1_000

    train_loader_class, test_loader_class = get_data_loader_class(hyper_params)
    
    send_val = hyper_params['model_type'] in [ 'SASRec', 'SVAE', 'MVAE' ]

    return train_loader_class(data_holder.train, hyper_params, track_events), test_loader_class(
        data_holder.test, data_holder.train, hyper_params, test_set = True,
        val_data = data_holder.val if send_val else None
    ), test_loader_class(data_holder.val, data_holder.train, hyper_params), hyper_params

class DataHolder:
    def __init__(self, rating_data_path, index_path):
        with h5py.File(rating_data_path + "total_data.hdf5", 'r') as f:
            self.data = list(zip(f['user'][:], f['item'][:], f['rating'][:]))

        self.index = np.load(index_path + "/index.npz")['data']
        self.remap()

    def remap(self):
        ## Counting number of unique users/items before
        valid_users, valid_items = set(), set()
        for at, (u, i, r) in enumerate(self.data):
            if self.index[at] != -1:
                valid_users.add(u)
                valid_items.add(i)

        ## Map creation done!
        user_map = dict(zip(list(valid_users), list(range(len(valid_users)))))
        item_map = dict(zip(list(valid_items), list(range(len(valid_items)))))

        new_data, new_index = [], []
        for at, (u, i, r) in enumerate(self.data):
            if self.index[at] == -1: continue
            new_data.append([ user_map[u], item_map[i], r ])
            new_index.append(self.index[at])

        self.data = new_data
        self.index = new_index
        self.num_users = len(valid_users)
        self.num_items = len(valid_items)

    def select(self, index_val):
        ret = []
        for at, tup in enumerate(self.data):
            if self.index[at] == index_val: ret.append(tup)
        return ret

    @property
    def train(self): return self.select(0)

    @property
    def val(self): return self.select(1)

    @property
    def test(self): return self.select(2)

    @property
    def num_train_interactions(self): return int(sum(map(lambda x: x == 0, self.index)))

    @property
    def num_val_interactions(self): return int(sum(map(lambda x: x == 1, self.index)))

    @property
    def num_test_interactions(self): return int(sum(map(lambda x: x == 2, self.index)))