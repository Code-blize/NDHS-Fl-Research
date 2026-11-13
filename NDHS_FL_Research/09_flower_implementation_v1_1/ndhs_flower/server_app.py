from pathlib import Path
import csv,torch
from flwr.serverapp import ServerApp
from flwr.common import ArrayRecord,ConfigRecord,MetricRecord
from .model import NDHSMLP
from .evaluation import evaluate_pooled_and_regions
from .strategy import SixRegionFedAvg
from .reproducibility import set_global_seed

app=ServerApp()

@app.main()
def main(grid,context):
    cfg=context.run_config
    feature=str(cfg["feature-config"]); input_dim={"baseline":15,"expanded":51}[feature]
    algorithm=str(cfg["algorithm"]).lower()
    if algorithm not in {"fedavg","fedprox"}: raise ValueError("algorithm must be fedavg/fedprox")
    mu=0.0 if algorithm=="fedavg" else float(cfg["proximal-mu"])
    set_global_seed(int(cfg["seed"]))
    model=NDHSMLP(input_dim); initial=ArrayRecord(model.state_dict())
    train_cfg=ConfigRecord({"train-data":str(cfg["train-data"]),
      "local-epochs":int(cfg["local-epochs"]),"batch-size":int(cfg["batch-size"]),
      "learning-rate":float(cfg["learning-rate"]),"proximal-mu":mu,"seed":int(cfg["seed"])})
    strategy=SixRegionFedAvg(fraction_train=1.0,fraction_evaluate=0.0,
      min_train_nodes=6,min_available_nodes=6,weighted_by_key="num-examples")
    out=Path(str(cfg["output-dir"])); out.mkdir(parents=True,exist_ok=True); rows=[]
    def server_eval(server_round,arrays):
        current=evaluate_pooled_and_regions(arrays.to_numpy_ndarrays(),str(cfg["validation-data"]),input_dim)
        for r in current:
            r.update({"server_round":int(server_round),"algorithm":algorithm,"feature_config":feature,
              "local_epochs":int(cfg["local-epochs"]),"seed":int(cfg["seed"])})
            rows.append(r)
        pooled=current[0]
        return MetricRecord({"validation-accuracy":float(pooled["accuracy"]),
          "validation-macro-f1":float(pooled["macro_f1"]),"validation-auroc":float(pooled["auroc"])})
    result=strategy.start(grid=grid,initial_arrays=initial,train_config=train_cfg,
      num_rounds=int(cfg["num-server-rounds"]),evaluate_fn=server_eval)
    if rows:
        with (out/"validation_metrics_pooled_and_regional.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    torch.save(result.arrays.to_torch_state_dict(),out/"final_model_state_dict.pt")
