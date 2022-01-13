import torch
import numpy as np
from collections import defaultdict
from torch.multiprocessing import Process, Queue, Event

class CombinedBase:
    def __init__(self): pass

    def __len__(self): return (self.num_interactions // self.batch_size) + 1

    def __del__(self):
        try:
            self.p.terminate() ; self.p.join()
        except: pass

    def make_user_history(self, data):
        user_history = [ [] for _ in range(self.num_users) ]
        for u, i, r in data: user_history[u].append(i)
        return user_history

    def pad(self, arr, max_len = None, pad_with = -1, side = 'right'):
        seq_len = max_len if max_len is not None else max(map(len, arr))
        seq_len = min(seq_len, 200) # You don't need more than this

        for i in range(len(arr)):
            while len(arr[i]) < seq_len: 
                pad_elem = arr[i][-1] if len(arr[i]) > 0 else 0
                pad_elem = pad_elem if pad_with == -1 else pad_with
                if side == 'right': arr[i].append(pad_elem)
                else: arr[i] = [ pad_elem ] + arr[i]
            arr[i] = arr[i][-seq_len:] # Keep last `seq_len` items
        return arr

    def sequential_pad(self, arr, hyper_params):
        # Padding left side so that we can simply take out [:, -1, :] in the output
        return self.pad(
            arr, max_len = hyper_params['max_seq_len'], 
            pad_with = hyper_params['total_items'], side = 'left'
        )

    def scatter(self, batch, tensor_kind, last_dimension):
        ret = tensor_kind(len(batch), last_dimension).zero_()

        if not torch.is_tensor(batch):
            if ret.is_cuda: batch = torch.cuda.LongTensor(batch)
            else: batch = torch.LongTensor(batch)

        return ret.scatter_(1, batch, 1)

    # NOTE: is_negative(user, item) is a function which tells 
    # if the item is a negative item for the user
    def sample_negatives(self, num_points, num_negs, is_negative):
        # Sample all the random numbers you need at once as this is much faster than 
        # calling random.randint() once everytime
        random_numbers = np.random.randint(
            self.num_items, 
            size = int(num_points * num_negs * 1.5)
        )

        negatives, at = [], 0
        for u in range(num_points):
            temp_negatives = []
            while len(temp_negatives) < num_negs:
                ## Negatives not possible
                if at >= len(random_numbers):
                    temp_negatives.append(0)
                    continue

                random_item = random_numbers[at] ; at += 1
                if is_negative(u, random_item):
                    # allowing duplicates, rare possibility
                    temp_negatives.append(random_item)
            negatives.append(temp_negatives)

        return negatives

    # So that training, GPU copying etc. 
    # doesn't have to wait for negative sampling
    def init_background_sampler(self, function):
        self.event = Event()
        self.result_queue = Queue(maxsize=4)
        
        def sample(result_queue):
            try:
                while True:
                    result_queue.put(function())
                    self.event.wait()
            except KeyboardInterrupt: pass
        self.p = Process(target = sample, args=(self.result_queue, ))
        self.p.daemon = True ; self.p.start()

class BaseTrainDataset(CombinedBase):
    def __init__(self, data, hyper_params):
        self.hyper_params = hyper_params
        self.batch_size = hyper_params['batch_size']
        self.implicit_task = hyper_params['task'] in [ 'implicit', 'sequential' ]
        self.data = data
        self.num_users, self.num_items = hyper_params['total_users'], hyper_params['total_items']
        
        ## Making user histories because sequential models require this
        self.user_history = self.make_user_history(data)
        
        ## Making sets of history for easier finding
        self.user_history_set = list(map(set, self.user_history))

        ## For computing PSP-metrics
        self.item_propensity = self.get_item_propensity()

    def get_item_count_map(self):
        item_count = defaultdict(int)
        for u, i, r in self.data: item_count[i] += 1
        return item_count

    def get_item_propensity(self, A = 0.55, B = 1.5):
        item_freq_map = self.get_item_count_map()
        item_freq = [ item_freq_map[i] for i in range(self.num_items) ]
        num_instances = len(self.data)

        C = (np.log(num_instances)-1)*np.power(B+1, A)
        wts = 1.0 + C*np.power(np.array(item_freq)+B, -A)
        return np.ravel(wts)

class BaseTestDataset(CombinedBase):
    def __init__(self, data, train_data, hyper_params, val_data):
        self.hyper_params = hyper_params
        self.batch_size = hyper_params['batch_size']
        self.implicit_task = hyper_params['task'] in [ 'implicit', 'sequential' ]
        self.data, self.train_data = data, train_data
        self.num_users, self.num_items = hyper_params['total_users'], hyper_params['total_items']
        
        ## Making user histories because sequential models require this
        self.train_user_history = self.make_user_history(train_data)
        if val_data is not None: 
            self.val_user_history = self.make_user_history(val_data)
            for u in range(self.num_users): self.train_user_history[u] += self.val_user_history[u]
        self.test_user_history = self.make_user_history(data)

        ## Making sets of history for easier finding
        self.train_user_history_set = list(map(set, self.train_user_history))
        self.test_user_history_set = list(map(set, self.test_user_history))
