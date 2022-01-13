import random
import numpy as np
from collections import deque

class ForestFireSampler:
    """An implementation of forest fire sampling. The procedure is a stochastic
    snowball sampling method where the expansion is proportional to the burning probability. 
    `"For details about the algorithm see this paper." <https://cs.stanford.edu/people/jure/pubs/sampling-kdd06.pdf>`_

    Inspiration credit: 
        littleballoffur
        https://github.com/benedekrozemberczki/littleballoffur

    Args:
        number_of_nodes (int): Number of sampled nodes. Default is 100.
        p (float): Burning probability. Default is 0.4.
        seed (int): Random seed. Default is 42.
    """
    def __init__(self, number_of_nodes: int=100, p: float=0.4, seed: int=42, max_visited_nodes_backlog: int=100,
                 restart_hop_size: int = 10):
        self.number_of_nodes = number_of_nodes
        self.p = p
        self.seed = seed
        self._set_seed() 
        self.restart_hop_size = restart_hop_size
        self.max_visited_nodes_backlog = max_visited_nodes_backlog

    def _set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)

    def _create_node_sets(self, graph):
        """
        Create a starting set of nodes.
        """
        self._sampled_nodes = set()
        self._set_of_nodes = set(range(graph.number_of_nodes()))
        self._visited_nodes = deque(maxlen=self.max_visited_nodes_backlog)

    def get_neighbors(self, graph, node):
        return list(graph.neighbors(node))

    def _start_a_fire(self, graph):
        """
        Starting a forest fire from a single node.
        """
        remaining_nodes = list(self._set_of_nodes.difference(self._sampled_nodes))
        seed_node = random.choice(remaining_nodes)
        self._sampled_nodes.add(seed_node)
        node_queue = deque([seed_node])
        while len(self._sampled_nodes) < self.number_of_nodes:
            if len(node_queue) == 0:
                node_queue = deque([self._visited_nodes.popleft()
                              for k in range(min(self.restart_hop_size, len(self._visited_nodes)))])
                if len(node_queue) == 0:
                    # print('Warning: could not collect the required number of nodes. The fire could not find enough nodes to burn.')
                    break
            top_node = node_queue.popleft()
            self._sampled_nodes.add(top_node)
            neighbors = set(self.get_neighbors(graph, top_node))
            unvisited_neighbors = neighbors.difference(self._sampled_nodes)
            score = np.random.geometric(self.p)
            count = min(len(unvisited_neighbors), score)
            burned_neighbors = random.sample(unvisited_neighbors, count)
            self._visited_nodes.extendleft(unvisited_neighbors.difference(set(burned_neighbors)))
            for neighbor in burned_neighbors:
                if len(self._sampled_nodes) >= self.number_of_nodes:
                    break
                node_queue.extend([neighbor])
