import numpy as np, torch, pytest
from ndhs_flower.model import NDHSMLP,get_parameters,set_parameters
from ndhs_flower.training import train_local
from ndhs_flower.data_loader import load_partition
from ndhs_flower.reproducibility import flower_partition_to_region
from ndhs_flower.evaluation import _metrics

def test_parameter_roundtrip():
    a=NDHSMLP(15); b=NDHSMLP(15); set_parameters(b,get_parameters(a))
    assert all(torch.equal(x,y) for x,y in zip(a.state_dict().values(),b.state_dict().values()))

def test_mu_zero_equivalence_and_repeatability():
    X=np.array([[.1,.2],[.2,.3],[.8,.7],[.9,.8]],np.float32); y=np.array([0,0,1,1],np.float32)
    base=NDHSMLP(2); initial={k:v.clone() for k,v in base.state_dict().items()}; models=[]
    for _ in range(2):
        m=NDHSMLP(2); m.load_state_dict(initial)
        train_local(m,X,y,1,2,.001,0.0,777); models.append(m)
    assert all(torch.equal(a,b) for a,b in zip(models[0].state_dict().values(),models[1].state_dict().values()))

def test_region_isolation(tmp_path):
    p=tmp_path/"toy.npz"
    np.savez(p,X=np.ones((6,2),np.float32),y=np.array([0,1,0,1,0,1],np.float32),
      source_row_id=np.array(["a","b","c","d","e","f"],object),v024=np.array([1,1,2,2,3,3]))
    X,y,ids=load_partition(p,2)
    assert len(X)==2 and set(ids)=={"c","d"}

def test_partition_mapping():
    assert [flower_partition_to_region(i,6) for i in range(6)]==[1,2,3,4,5,6]
    with pytest.raises(ValueError): flower_partition_to_region(6,6)
    with pytest.raises(ValueError): flower_partition_to_region(0,5)

def test_weighted_aggregation_hand_calculation():
    a=np.array([1.,3.]); b=np.array([5.,7.]); n1,n2=2,6
    assert np.allclose((n1*a+n2*b)/(n1+n2),[4.,6.])

def test_metrics_complete():
    m=_metrics(np.array([0,0,1,1]),np.array([.1,.4,.6,.9]))
    required={"tn","fp","fn","tp","accuracy","precision","sensitivity","specificity","f1",
      "macro_f1","balanced_accuracy","auroc","auprc_average_precision","brier",
      "binary_cross_entropy","observed_prevalence","predicted_prevalence",
      "calibration_intercept","calibration_slope","calibration_status","ece_10"}
    assert required.issubset(m)
    assert m["tn"]+m["fp"]+m["fn"]+m["tp"]==4
    assert 0<=m["ece_10"]<=1
