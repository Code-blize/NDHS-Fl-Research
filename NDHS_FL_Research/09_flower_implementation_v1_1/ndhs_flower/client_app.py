from flwr.client import ClientApp
from flwr.common import ArrayRecord,MetricRecord,Message,RecordDict
from .model import NDHSMLP,set_parameters
from .data_loader import load_partition
from .training import train_local
from .reproducibility import client_round_seed,flower_partition_to_region

app=ClientApp()

@app.train()
def train(msg:Message,context):
    cfg=msg.content["config"]; arrays=msg.content["arrays"]
    pid=int(context.node_config["partition-id"]); n_parts=int(context.node_config["num-partitions"])
    region=flower_partition_to_region(pid,n_parts)
    X,y,ids=load_partition(cfg["train-data"],region)
    model=NDHSMLP(X.shape[1]); set_parameters(model,arrays.to_numpy_ndarrays())
    seed=client_round_seed(int(cfg["seed"]),region,int(cfg["server-round"]))
    loss=train_local(model,X,y,int(cfg["local-epochs"]),int(cfg["batch-size"]),
        float(cfg["learning-rate"]),float(cfg["proximal-mu"]),seed)
    return Message(content=RecordDict({
      "arrays":ArrayRecord(model.state_dict()),
      "metrics":MetricRecord({"num-examples":int(len(y)),"train-loss":float(loss),
        "region-code":int(region),"partition-id":int(pid),"unique-source-ids":int(len(set(ids.tolist())))})
    }),reply_to=msg)
