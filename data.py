import os
import h5py
import json
import math
import random
import numpy as np
from collections import defaultdict

import networkx as nx
import networkit as nk
nk.setNumberOfThreads(16)

from graph_sampling.ForestFire import ForestFireSampler
from graph_sampling.RW import RandomWalkWithRestartSampler

class rating_data:
    def __init__(self, data):
        self.data = data

        self.index = [] # 0: train, 1: validation, 2: test, -1: removed/ignore
        for user_data in self.data:
            for _ in range(len(user_data)): self.index.append(42)

        self.complete_data_stats = None

    def train_test_split(self, split_type):
        at = 0

        for user in range(len(self.data)):
            if split_type == "20_percent_hist": 
                first_split_point = int(0.8 * len(self.data[user]))
                second_split_point = int(0.9 * len(self.data[user]))

                indices = np.arange(len(self.data[user]))
                np.random.shuffle(indices)

                for timestep, (item, rating, time) in enumerate(self.data[user]):
                    if len(self.data[user]) < 3: self.index[at] = -1
                    else:
                        # Force atleast one element in user history to be in test
                        if timestep == indices[0]: self.index[at] = 2
                        else:
                            if timestep in indices[:first_split_point]: self.index[at] = 0
                            elif timestep in indices[first_split_point:second_split_point]: self.index[at] = 1
                            else: self.index[at] = 2
                    at += 1
            
            elif split_type == "leave_2":
                for timestep, (item, rating, time) in enumerate(self.data[user]):
                    if len(self.data[user]) < 3: self.index[at] = -1
                    else:
                        if timestep <= len(self.data[user]) - 3: self.index[at] = 0
                        elif timestep == len(self.data[user]) - 2: self.index[at] = 1
                        else: self.index[at] = 2
                    at += 1

        assert at == len(self.index)
        self.complete_data_stats = None

    def interaction_random_sample(self, percent):
        active, at = set(), 0
        for u in range(len(self.data)):
            for i, r, t in self.data[u]:
                # NOTE: only sample on the train-set
                if self.index[at] == 0: active.add(at)
                at += 1
        active = list(active)

        # Remove `percent`% at random
        remove_mask = {}
        for i in active: remove_mask[i] = False
        random.shuffle(active)
        split_point = int(float(len(active)) * (float(percent) / 100.0))
        for i in active[:split_point]: remove_mask[i] = True

        at = 0
        for u in range(len(self.data)):
            for i, r, t in self.data[u]:
                if remove_mask.get(at, False) and self.index[at] == 0: self.index[at] = -1
                at += 1
        assert at == len(self.index)

    def frequency_sample(self, percent, sample_type):
        hist, at = {}, 0
        for u in range(len(self.data)):
            for i, r, t in self.data[u]:
                key = [ u, i ][sample_type]
                if key not in hist: hist[key] = []
                # NOTE: only sample on the train-set
                if self.index[at] == 0: hist[key].append(at) 
                at += 1

        # Remove `percent`% at random
        remove_mask = {}
        for key in hist:
            interactions = hist[key]
            random.shuffle(interactions)
            split_point = math.ceil(float(len(interactions)) * (float(percent) / 100.0))
            for i in interactions[:split_point]: remove_mask[i] = True

        at = 0
        for u in range(len(self.data)):
            for i, r, t in self.data[u]:
                if remove_mask.get(at, False) and self.index[at] == 0: self.index[at] = -1
                at += 1
        assert at == len(self.index)

    def user_random_sample(self, percent):
        hist, at, total = {}, 0, 0
        for u in range(len(self.data)):
            for i, r, t in self.data[u]:
                # NOTE: only sample on the train-set
                if self.index[at] == 0: 
                    if u not in hist: hist[u] = 0
                    hist[u] += 1
                    total += 1
                at += 1

        # Remove `percent`% at random
        user_freqs = list(hist.items()) ; np.random.shuffle(user_freqs)
        interactions_to_remove, removed, users_to_remove = total * (float(percent) / 100.0), 0, set()
        for u, cnt in user_freqs:
            if removed >= interactions_to_remove: break
            users_to_remove.add(u)
            removed += cnt

        at = 0
        for u in range(len(self.data)):
            for i, r, t in self.data[u]:
                if u in users_to_remove and self.index[at] == 0: self.index[at] = -1
                at += 1
        assert at == len(self.index)

    def temporal_sample(self, percent):
        hist, at = {}, 0
        for u in range(len(self.data)):
            for i, r, t in self.data[u]:
                if u not in hist: hist[u] = []
                # NOTE: only sample on the train-set
                if self.index[at] == 0: hist[u].append(at) 
                at += 1

        # Remove first `percent`% interactions for each user
        remove_mask = {}
        for u in hist:
            interactions = hist[u]
            # random.shuffle(interactions) ### No shuffling, remove first % interactions
            split_point = math.ceil(float(len(interactions)) * (float(percent) / 100.0))
            for i in interactions[:split_point]: remove_mask[i] = True

        at = 0
        for u in range(len(self.data)):
            for i, r, t in self.data[u]:
                if remove_mask.get(at, False) and self.index[at] == 0: self.index[at] = -1
                at += 1
        assert at == len(self.index)

    def tail_user_remove(self, percent):
        hist, at, total = {}, 0, 0
        for u in range(len(self.data)):
            for i, r, t in self.data[u]:
                # NOTE: only count on the train-set
                if self.index[at] == 0: 
                    if u not in hist: hist[u] = 0
                    hist[u] += 1
                    total += 1
                at += 1

        user_freqs = sorted(list(hist.items()), key = lambda x: x[1])
        interactions_to_remove, removed, users_to_remove = total * (float(percent) / 100.0), 0, set()
        for u, cnt in user_freqs:
            if removed >= interactions_to_remove: break
            users_to_remove.add(u)
            removed += cnt

        at = 0
        for u in range(len(self.data)):
            for i, r, t in self.data[u]:
                if u in users_to_remove and self.index[at] == 0: self.index[at] = -1
                at += 1
        assert at == len(self.index)

    def svp_sample(self, percent, svp_handler, sampling_type):
        self.index = {
            'forgetting_events': svp_handler.forgetting_events,
            'forgetting_events_user': svp_handler.forgetting_events_user,
            'forgetting_events_propensity': svp_handler.forgetting_events_propensity,
            'forgetting_events_user_propensity': svp_handler.forgetting_events_user_propensity,
        }[sampling_type](percent, self.data, self.index)

    def construct_nx_graph(self):
        # Make graph
        g = nx.Graph()

        # Add nodes & edges
        user_map, item_map, rev_user_map, rev_item_map, at, node_num = {}, {}, {}, {}, 0, 0
        user_actions, item_actions, total = defaultdict(list), defaultdict(list), 0

        for u in range(len(self.data)):
            for i, r, t in self.data[u]:
                # NOTE: only sample on the train-set
                if self.index[at] == 0: 
                    total += 1
                    user_actions[u].append(at)
                    item_actions[i].append(at)

                    if u not in user_map:
                        user_map[u] = node_num
                        rev_user_map[node_num] = u
                        g.add_node(node_num)
                        node_num += 1
                    if i not in item_map:
                        item_map[i] = node_num
                        rev_item_map[node_num] = i
                        g.add_node(node_num)
                        node_num += 1
                    g.add_edge(user_map[u], item_map[i])
                at += 1
        assert node_num == g.number_of_nodes()
        return g, rev_user_map, rev_item_map, user_actions, item_actions

    def pagerank_sample(self, percent):
        # networkx graph
        g, rev_user_map, rev_item_map, user_actions, item_actions = self.construct_nx_graph()

        # Convert to networkit
        nk_g = nk.nxadapter.nx2nk(g)

        # Run pagerank
        pr = nk.centrality.PageRank(nk_g, 1e-6) ; pr.run()

        # Remove `percent`% acc to pagerank scores
        # THOUGHT: the nodes with the least pagerank scores will most probably be the tail users/items
        interactions_to_remove, removed = nk_g.numberOfEdges() * (float(percent) / 100.0), 0
        for node, _ in pr.ranking()[::-1]:
            if removed >= interactions_to_remove: break
            
            if node in rev_user_map: 
                for at in user_actions[rev_user_map[node]]:
                    if self.index[at] != -1: removed += 1
                    self.index[at] = -1
            else: 
                for at in item_actions[rev_item_map[node]]:
                    if self.index[at] != -1: removed += 1
                    self.index[at] = -1

    def random_walk_sample(self, percent):
        at, total = 0, 0
        for u in range(len(self.data)):
            for i, r, t in self.data[u]:
                if self.index[at] == 0: total += 1
                at += 1

        interactions_to_remove, removed = float(total) * (float(percent) / 100.0), 0
        while removed < interactions_to_remove:
            # networkx graph
            nx_g, rev_user_map, rev_item_map, user_actions, item_actions = self.construct_nx_graph()
            
            # Create sampler ## Nodes to keep
            sampler = RandomWalkWithRestartSampler(number_of_nodes = int(nx_g.number_of_nodes() * (float(100 - percent) / 100.0)))
            sampler._create_initial_node_set(nx_g, None)

            # Sample
            while len(sampler._sampled_nodes) < sampler.number_of_nodes:
                sampler._do_a_step(nx_g)

            # Remove from the main graph
            ## `sampler._sampled_nodes` are the nodes that are kept, not removed
            nodes_to_remove = list(sampler._set_of_nodes.difference(sampler._sampled_nodes))

            for node in nodes_to_remove:
                if removed >= interactions_to_remove: break

                if node in rev_user_map: 
                    for at in user_actions[rev_user_map[node]]:
                        if self.index[at] != -1: removed += 1
                        self.index[at] = -1
                else: 
                    for at in item_actions[rev_item_map[node]]:
                        if self.index[at] != -1: removed += 1
                        self.index[at] = -1

    def forest_fire_sample(self, percent):
        at, total = 0, 0
        for u in range(len(self.data)):
            for i, r, t in self.data[u]:
                if self.index[at] == 0: total += 1
                at += 1

        interactions_to_remove, removed = float(total) * (float(percent) / 100.0), 0
        while removed < interactions_to_remove:
            # networkx graph
            nx_g, rev_user_map, rev_item_map, user_actions, item_actions = self.construct_nx_graph()
            
            # Create sampler ## Nodes to keep
            sampler = ForestFireSampler(number_of_nodes = int(nx_g.number_of_nodes() * (float(100 - percent) / 100.0)))
            sampler._create_node_sets(nx_g)

            # Sample
            while len(sampler._sampled_nodes) < sampler.number_of_nodes: 
                sampler._start_a_fire(nx_g)

            # Remove from the main graph
            ## `sampler._sampled_nodes` are the nodes that are kept, not removed
            nodes_to_remove = list(sampler._set_of_nodes.difference(sampler._sampled_nodes))

            for node in nodes_to_remove:
                if removed >= interactions_to_remove: break

                if node in rev_user_map: 
                    for at in user_actions[rev_user_map[node]]:
                        if self.index[at] != -1: removed += 1
                        self.index[at] = -1
                else: 
                    for at in item_actions[rev_item_map[node]]:
                        if self.index[at] != -1: removed += 1
                        self.index[at] = -1

    def measure_data_stats(self):
        num_users, num_items, num_interactions, num_test, num_val = set(), set(), 0, 0, 0
        at = 0
        for u in range(len(self.data)):
            for i, _, _ in self.data[u]:
                if self.index[at] == 0: num_interactions += 1
                if self.index[at] == 1: num_val += 1
                if self.index[at] == 2: num_test += 1

                if self.index[at] != -1:
                    num_users.add(u)
                    num_items.add(i)
                at += 1

        data_stats = {}
        data_stats["num_users"] = len(num_users)
        data_stats["num_items"] = len(num_items)
        data_stats["num_train_interactions"] = num_interactions
        data_stats["num_test"] = num_test
        data_stats["num_val"] = num_val

        return data_stats

    def save_index(self, path, statistics = True):
        os.makedirs(path, exist_ok = True)
        with open(path + "/index.npz", "wb") as f: np.savez_compressed(f, data = self.index)

        if statistics:
            data_stats = self.measure_data_stats() 
            if self.complete_data_stats is None: print("FULL DATA:", data_stats)
            else: 
                def convert(key): return round(100.0 - (100.0 * float(data_stats[key] / float(self.complete_data_stats[key]))), 2)
                print("SAMPLE SIZE: {}% users ; {}% items ; {}% train interactions ; {}% test interactions removed".format(
                    convert('num_users'), convert('num_items'), convert('num_train_interactions'), convert('num_test')
                ))
            with open(path + "/data_stats.json", 'w') as f: json.dump(data_stats, f)

    def load_index(self, path):
        self.index = np.load(path + "/index.npz")['data']
        if self.complete_data_stats is None: self.complete_data_stats = self.measure_data_stats()

    def save_data(self, path):
        flat_data = []
        for u in range(len(self.data)):
            flat_data += list(map(lambda x: [ u ] + x, self.data[u]))
        flat_data = np.array(flat_data)

        shape = [ len(flat_data) ]

        os.makedirs(path, exist_ok = True)
        with h5py.File(path + '/total_data.hdf5', 'w') as file:
            dset = {}
            dset['user'] = file.create_dataset("user", shape, dtype = 'i4', maxshape = shape, compression="gzip")
            dset['item'] = file.create_dataset("item", shape, dtype = 'i4', maxshape = shape, compression="gzip")
            dset['rating'] = file.create_dataset("rating", shape, dtype = 'f', maxshape = shape, compression="gzip")
            dset['time'] = file.create_dataset("time", shape, dtype = 'i4', maxshape = shape, compression="gzip")

            dset['user'][:] = flat_data[:, 0]
            dset['item'][:] = flat_data[:, 1]
            dset['rating'][:] = flat_data[:, 2]
            dset['time'][:] = flat_data[:, 3]
