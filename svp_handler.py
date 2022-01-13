import numpy as np
from collections import defaultdict

from main import main_pytorch
from data_path_constants import get_svp_log_file_path, get_svp_model_file_path

class SVPHandler:
    def __init__(self, model_type, loss_type, hyper_params):
        hyper_params['model_type'] = model_type
        hyper_params['task'] = loss_type
        hyper_params['num_train_negs'] = 1
        hyper_params['num_test_negs'] = 100

        hyper_params['latent_size'] = 10
        hyper_params['dropout'] = 0.3
        hyper_params['weight_decay'] = float(1e-6)
        hyper_params['lr'] = 0.006
        hyper_params['epochs'] = 50
        hyper_params['validate_every'] = 5000
        hyper_params['batch_size'] = 1024
        self.hyper_params = hyper_params
        self.hyper_params['log_file'] = self.log_file
        self.hyper_params['model_path'] = self.model_file

        self.train_model()

    def train_model(self): 
        _, self.forgetted_count = main_pytorch(self.hyper_params, track_events = True, eval_full = False)

    def forgetting_events(self, percent, data, index):
        # Keep those points which have the maximum forgetted count
        # => Remove those points which have the minimum forgetted count
        index_map = []
        for at, i in enumerate(index):
            if i == 0: index_map.append(at)

        split_point = int(float(len(self.forgetted_count)) * (float(percent) / 100.0))
        order = np.argsort(self.forgetted_count)
        order = list(map(lambda x: index_map[x], order))
        remove_indices = order[:split_point] # If greedy

        for i in remove_indices: index[i] = -1 # Remove
        return index

    def forgetting_events_user(self, percent, data, index):
        # Keep those users which have the maximum forgetted count
        # Remove those users which have the minimum forgetted count
        index_map, user_map, hist, at, total = [], [], {}, 0, 0
        for u in range(len(data)):
            for i, r, t in data[u]:
                if index[at] == 0:
                    index_map.append(at)
                    user_map.append(u)
                    if u not in hist: hist[u] = 0
                    hist[u] += 1
                    total += 1
                at += 1

        user_forgetted_count = defaultdict(list)
        for train_at, cnt in enumerate(self.forgetted_count):
            user_forgetted_count[user_map[train_at]].append(cnt)
        user_forgetted_count = sorted(list(dict(user_forgetted_count).items()), key = lambda x: np.mean(x[1]))

        interactions_to_remove, removed, users_to_remove = total * (float(percent) / 100.0), 0, set()
        for u, _ in user_forgetted_count:
            if removed >= interactions_to_remove: break
            users_to_remove.add(u)
            removed += hist[u]

        for train_at in range(len(user_map)):
            if user_map[train_at] in users_to_remove: index[index_map[train_at]] = -1

        return index

    def compute_freq(self, data, index, freq_type):
        freq, at = defaultdict(int), 0
        for u in range(len(data)):
            for i, r, t in data[u]:
                if index[at] == 0:
                    to_count = [ u, i ][freq_type]
                    freq[to_count] += 1
                at += 1

        valid_users = list(freq.keys())
        return list(map(lambda x: freq[x], valid_users)), dict(zip(valid_users, list(range(len(freq)))))

    def compute_prop(self, freq_vector, num_instances, A = 0.55, B = 1.5):
        C = (np.log(num_instances)-1)*np.power(B+1, A)
        wts = 1.0 + C*np.power(np.array(freq_vector)+B, -A)
        return np.ravel(wts)

    def forgetting_events_propensity(self, percent, data, index, pooling_method = 'max'):
        # Keep those points which have the maximum forgetted count
        # Remove those points which have the minimum forgetted count

        num_interactions = len(self.forgetted_count)
        user_freq, user_map = self.compute_freq(data, index, 0)
        user_propensity_vector = self.compute_prop(user_freq, num_interactions)
        item_freq, item_map = self.compute_freq(data, index, 1)
        item_propensity_vector = self.compute_prop(item_freq, num_interactions)
        interaction_propensity, at = [], 0
        freq, at = defaultdict(int), 0
        
        def pool(prop_u, prop_i):
            if pooling_method == 'sum': return prop_u + prop_i
            elif pooling_method == 'max': return max(prop_u, prop_i)

        for u in range(len(data)):
            for i, r, t in data[u]:
                if index[at] == 0:
                    interaction_propensity.append(
                        pool(user_propensity_vector[user_map[u]], item_propensity_vector[item_map[i]])
                    )
                at += 1
        assert len(interaction_propensity) == num_interactions

        # interaction_propensity actually estimates the `inverse` propensity, hence multiply
        updated_count = np.array(self.forgetted_count) * np.array(interaction_propensity)

        index_map = []
        for at, i in enumerate(index):
            if i == 0: index_map.append(at)

        split_point = int(float(len(updated_count)) * (float(percent) / 100.0))
        order = np.argsort(updated_count)
        order = list(map(lambda x: index_map[x], order))
        remove_indices = order[:split_point] # If greedy

        for i in remove_indices: index[i] = -1 # Remove
        return index

    def forgetting_events_user_propensity(self, percent, data, index):
        # Keep those users which have the maximum forgetted count
        # Keep those users which have the maximum propensity --> minimum frequency
        # Remove those users which have the minimum forgetted count

        num_interactions = len(self.forgetted_count)
        user_freq, user_index_map = self.compute_freq(data, index, 0)
        user_propensity_vector = self.compute_prop(user_freq, num_interactions)

        index_map, user_map, hist, at, total = [], [], {}, 0, 0
        for u in range(len(data)):
            for i, r, t in data[u]:
                if index[at] == 0:
                    index_map.append(at)
                    user_map.append(u)
                    if u not in hist: hist[u] = 0
                    hist[u] += 1
                    total += 1
                at += 1

        user_forgetted_count = defaultdict(list)
        for train_at, cnt in enumerate(self.forgetted_count):
            u = user_map[train_at]
            user_forgetted_count[u].append(cnt * user_propensity_vector[user_index_map[u]])
        user_forgetted_count = sorted(list(dict(user_forgetted_count).items()), key = lambda x: np.mean(x[1]))

        interactions_to_remove, removed, users_to_remove = total * (float(percent) / 100.0), 0, set()
        for u, _ in user_forgetted_count:
            if removed >= interactions_to_remove: break
            users_to_remove.add(u)
            removed += hist[u]

        for train_at in range(len(user_map)):
            if user_map[train_at] in users_to_remove: index[index_map[train_at]] = -1

        return index

    @property
    def model_file(self): 
        return get_svp_model_file_path(self.hyper_params)

    @property
    def log_file(self): 
        return get_svp_log_file_path(self.hyper_params)
