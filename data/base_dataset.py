#   Taken from Deep Blackbox Graph Matching: https://github.com/martius-lab/blackbox-deep-graph-matching.
class BaseDataset:
    def __init__(self):
        pass

    def get_k_samples(self, idx, k, mode, cls=None, shuffle=True):
        raise NotImplementedError
