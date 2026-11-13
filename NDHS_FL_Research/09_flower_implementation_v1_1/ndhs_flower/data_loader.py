from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_partition(npz_path, region_code):
    path=Path(npz_path)
    if not path.is_file(): raise FileNotFoundError(path)
    # Trusted local project NPZ: source_row_id is object dtype, used only for audit.
    with np.load(path,allow_pickle=True) as z:
        required={"X","y","source_row_id","v024"}
        if not required.issubset(z.files): raise KeyError(f"Missing arrays: {required-set(z.files)}")
        X=np.asarray(z["X"],dtype=np.float32); y=np.asarray(z["y"],dtype=np.float32)
        ids=np.asarray(z["source_row_id"]); regions=np.asarray(z["v024"],dtype=np.int64)
    if X.ndim!=2 or y.ndim!=1 or len(X)!=len(y) or len(ids)!=len(y) or len(regions)!=len(y):
        raise ValueError("Inconsistent archive dimensions")
    if not np.isfinite(X).all() or not np.isfinite(y).all(): raise ValueError("Non-finite model values")
    if not set(np.unique(regions)).issubset({1,2,3,4,5,6}): raise ValueError("Unexpected v024 code")
    mask=regions==int(region_code)
    if not mask.any(): raise ValueError(f"No records for v024={region_code}")
    return X[mask],y[mask],ids[mask]

def make_loader(X,y,batch_size,seed):
    gen=torch.Generator().manual_seed(int(seed))
    return DataLoader(TensorDataset(torch.from_numpy(X),torch.from_numpy(y)),
        batch_size=int(batch_size),shuffle=True,generator=gen,num_workers=0,drop_last=False)
