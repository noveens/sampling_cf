from torch_utils import FloatTensor

class PopRec:
    def __init__(self, hyper_params, item_count):
        self.hyper_params = hyper_params
        self.top_items = FloatTensor([ item_count[i] for i in range(hyper_params['total_items']) ]).unsqueeze(0)

    def __call__(self, data, eval = False):
        users, _, _ = data
        return self.top_items.repeat(users.shape[0], 1)

    def eval(self):
    	pass
