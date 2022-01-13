import random

class RandomWalkWithRestartSampler:
    """An implementation of node sampling by random walks with restart. The 
    process is a discrete random walker on nodes which teleports back to the
    staring node with a fixed probability. This results in a connected subsample
    from the original input graph. `"For details about the algorithm see this 
    paper." <https://cs.stanford.edu/people/jure/pubs/sampling-kdd06.pdf>`_

    Inspiration credit: 
        littleballoffur
        https://github.com/benedekrozemberczki/littleballoffur

    Args:
        number_of_nodes (int): Number of nodes. Default is 100.
        seed (int): Random seed. Default is 42.
        p (float): Restart probability. Default is 0.1.
    """
    def __init__(self, number_of_nodes: int=100, seed: int=42, p: float=0.1):
        self.number_of_nodes = number_of_nodes
        self.seed = seed
        self.p = p
        self._set_seed()

    def _set_seed(self):
        random.seed(self.seed)

    def get_neighbors(self, graph, node):
        return list(graph.neighbors(node))

    def get_random_neighbor(self, graph, node):
        return random.choice(self.get_neighbors(graph, node))

    def get_nodes(self, graph):
        return list(graph.nodes)

    def get_number_of_nodes(self, graph):
        return graph.number_of_nodes()

    def _create_initial_node_set(self, graph, start_node):
        """
        Choosing an initial node.
        """
        self._set_of_nodes = set(self.get_nodes(graph))

        if start_node is not None:
            if start_node >= 0 and start_node < self.get_number_of_nodes(graph):
                self._current_node = start_node
                self._sampled_nodes = set([self._current_node])
            else:
                raise ValueError("Starting node index is out of range.")
        else:
            self._current_node = random.choice(range(self.get_number_of_nodes(graph)))
            self._sampled_nodes = set([self._current_node])
        self._initial_node = self._current_node

    def _do_a_step(self, graph):
        """
        Doing a single random walk step.
        """
        score = random.uniform(0, 1)
        if score < self.p:
            self._current_node = self._initial_node
        else:
            new_node = self.get_random_neighbor(graph, self._current_node)
            self._sampled_nodes.add(new_node)
            self._current_node = new_node
