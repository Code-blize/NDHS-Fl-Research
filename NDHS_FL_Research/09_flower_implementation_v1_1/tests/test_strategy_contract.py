from ndhs_flower.strategy import SixRegionFedAvg

def test_strategy_contract():
    s=SixRegionFedAvg(fraction_train=1.0,fraction_evaluate=0.0,
      min_train_nodes=6,min_available_nodes=6,weighted_by_key="num-examples")
    assert s.weighted_by_key=="num-examples"
    assert s.min_train_nodes==6 and s.min_available_nodes==6
