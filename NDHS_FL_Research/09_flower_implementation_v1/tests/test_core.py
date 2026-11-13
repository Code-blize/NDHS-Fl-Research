import numpy as np, torch
from ndhs_flower.model import NDHSMLP,get_parameters,set_parameters
from ndhs_flower.training import train_local
from ndhs_flower.data_loader import load_partition

def test_parameter_roundtrip():
    a=NDHSMLP(15); b=NDHSMLP(15); set_parameters(b,get_parameters(a))
    assert all(torch.equal(x,y) for x,y in zip(a.state_dict().values(),b.state_dict().values()))

def test_mu_zero_deterministic_equivalence():
    X=np.array([[.1,.2],[.2,.3],[.8,.7],[.9,.8]],dtype=np.float32)
    y=np.array([0,0,1,1],dtype=np.float32)
    a,b=NDHSMLP(2),NDHSMLP(2); b.load_state_dict(a.state_dict())
    train_local(a,X,y,1,2,.001,0.0,777); train_local(b,X,y,1,2,.001,0.0,777)
    assert all(torch.equal(x,y) for x,y in zip(a.state_dict().values(),b.state_dict().values()))

def test_region_isolation(tmp_path):
    p=tmp_path/"toy.npz"
    np.savez(p,X=np.ones((6,2),np.float32),y=np.array([0,1,0,1,0,1],np.float32),
      source_row_id=np.array(["a","b","c","d","e","f"],dtype=object),v024=np.array([1,1,2,2,3,3]))
    X,y,ids=load_partition(p,2)
    assert len(X)==2 and set(ids)=={"c","d"}
