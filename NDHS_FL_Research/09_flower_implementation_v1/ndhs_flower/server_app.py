from pathlib import Path
import csv, torch
from flwr.serverapp import ServerApp
from flwr.serverapp.strategy import FedAvg
from flwr.common import ArrayRecord, ConfigRecord
from .model import NDHSMLP
from .evaluation import evaluate_pooled

app=ServerApp()

@app.main()
def main(grid,context):
    cfg=context.run_config
    feature=str(cfg["feature-config"]); input_dim={"baseline":15,"expanded":51}[feature]
    algorithm=str(cfg["algorithm"]).lower()
    if algorithm not in ("fedavg","fedprox"): raise ValueError("algorithm must be fedavg/fedprox")
    mu=0.0 if algorithm=="fedavg" else float(cfg["proximal-mu"])
    model=NDHSMLP(input_dim)
    initial=ArrayRecord(model.state_dict())
    train_cfg=ConfigRecord({
        "train-data":str(cfg["train-data"]),"local-epochs":int(cfg["local-epochs"]),
        "batch-size":int(cfg["batch-size"]),"learning-rate":float(cfg["learning-rate"]),
        "proximal-mu":mu,"seed":int(cfg["seed"]),"server-round":1})
    strategy=FedAvg(fraction_train=float(cfg["fraction-train"]),fraction_evaluate=0.0,
                    min_train_nodes=6,min_available_nodes=6,weighted_by_key="num-examples")
    out=Path(str(cfg["output-dir"])); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    def server_eval(server_round, arrays):
        metrics=evaluate_pooled(arrays.to_numpy_ndarrays(),str(cfg["validation-data"]),input_dim)
        metrics.update({"server_round":int(server_round),"algorithm":algorithm,
                        "feature_config":feature,"local_epochs":int(cfg["local-epochs"]),
                        "seed":int(cfg["seed"])})
        rows.append(metrics)
        return metrics
    result=strategy.start(grid=grid,initial_arrays=initial,train_config=train_cfg,
                          num_rounds=int(cfg["num-server-rounds"]),evaluate_fn=server_eval)
    if rows:
        with (out/"server_validation_metrics.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    torch.save(result.arrays.to_torch_state_dict(),out/"final_model_state_dict.pt")
