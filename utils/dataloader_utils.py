import torch
import numpy as np
from torch_geometric.data import Data, Batch
import torch.nn.functional as F
import random
from utils.config import cfg
import torchvision.models as tvm

def collate_fn(data: list):
    """
    Create mini-batch data for training.
    :param data: data dict
    :return: mini-batch
    """

    def pad_tensor(inp):
        assert type(inp[0]) == torch.Tensor
        it = iter(inp)
        t = next(it)
        max_shape = list(t.shape)
        while True:
            try:
                t = next(it)
                for i in range(len(max_shape)):
                    max_shape[i] = int(max(max_shape[i], t.shape[i]))
            except StopIteration:
                break
        max_shape = np.array(max_shape)

        padded_ts = []
        for t in inp:
            pad_pattern = np.zeros(2 * len(max_shape), dtype=np.int64)
            pad_pattern[::-2] = max_shape - np.array(t.shape)
            pad_pattern = tuple(pad_pattern.tolist())
            padded_ts.append(F.pad(t, pad_pattern, "constant", 0))

        return padded_ts

    def stack(inp):
        
        # print(f"type(inp[0]): {type(inp[0])}")
        if type(inp[0]) == list:
            # print(f"inp[0]: {inp[0]}")
            ret = []
            for vs in zip(*inp):
                ret.append(stack(vs))
        elif type(inp[0]) == dict:
            # print(f"inp[0].keys(): {inp[0].keys()}")
            ret = {}
            for kvs in zip(*[x.items() for x in inp]):
                ks, vs = zip(*kvs)
                for k in ks:
                    assert k == ks[0], "Key value mismatch."
                # print(f"k: {k}")
                ret[k] = stack(vs)
        elif type(inp[0]) == torch.Tensor:
            new_t = pad_tensor(inp)
            ret = torch.stack(new_t, 0)
        elif type(inp[0]) == np.ndarray:
            new_t = pad_tensor([torch.from_numpy(x) for x in inp])
            ret = torch.stack(new_t, 0)
        elif type(inp[0]) == str:
            ret = inp
        elif type(inp[0]) == Data:  # Graph from torch.geometric, create a batch
            ret = Batch.from_data_list(inp)
        else:
            raise ValueError("Cannot handle type {}".format(type(inp[0])))
        return ret
    
    ret = stack(data)
    return ret
    # return data[0]

def worker_init_fix(worker_id):
    """
    Init dataloader workers with fixed seed.
    """
    random.seed(cfg.RANDOM_SEED + worker_id)
    np.random.seed(cfg.RANDOM_SEED + worker_id)

def worker_init_rand(worker_id):
    """
    Init dataloader workers with torch.initial_seed().
    torch.initial_seed() returns different seeds when called from different dataloader threads.
    """
    random.seed(torch.initial_seed())
    np.random.seed(torch.initial_seed() % 2 ** 32)
