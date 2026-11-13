import os, random
import numpy as np
import torch

REGION_CODES = (1, 2, 3, 4, 5, 6)
REGION_NAMES = {1:"NC", 2:"NE", 3:"NW", 4:"SE", 5:"SS", 6:"SW"}

def set_global_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)

def client_round_seed(base_seed, region_code, server_round):
    return int((int(base_seed)*1_000_003 + int(region_code)*10_007 + int(server_round)) % (2**31-1))

def flower_partition_to_region(partition_id, num_partitions):
    if int(num_partitions) != 6:
        raise ValueError(f"Expected exactly 6 Flower partitions, got {num_partitions}")
    if int(partition_id) not in range(6):
        raise ValueError(f"Expected partition-id 0..5, got {partition_id}")
    return int(partition_id) + 1
