import os, random
import numpy as np
import torch

REGION_CODES = (1, 2, 3, 4, 5, 6)
REGION_NAMES = {1:"NC", 2:"NE", 3:"NW", 4:"SE", 5:"SS", 6:"SW"}

def set_global_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

def client_round_seed(base_seed, region_code, server_round):
    return int((int(base_seed)*1_000_003 + int(region_code)*10_007 + int(server_round)) % (2**31-1))
