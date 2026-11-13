from flwr.client import ClientApp
from flwr.common import ArrayRecord, MetricRecord, Message, RecordDict
from .model import NDHSMLP, set_parameters
from .data_loader import load_partition
from .training import train_local
from .reproducibility import client_round_seed

app = ClientApp()

@app.train()
def train(msg: Message, context):
    cfg=msg.content["config"]; arrays=msg.content["arrays"]
    # The simulation must expose an explicit 1..6 partition-id mapping.
    region=int(context.node_config["partition-id"])
    if region not in (1,2,3,4,5,6):
        raise ValueError(f"Expected region partition-id 1..6; got {region}")
    X,y,_ids=load_partition(cfg["train-data"],region)
    model=NDHSMLP(X.shape[1]); set_parameters(model,arrays.to_numpy_ndarrays())
    seed=client_round_seed(cfg["seed"],region,cfg["server-round"])
    loss=train_local(model,X,y,cfg["local-epochs"],cfg["batch-size"],
                     cfg["learning-rate"],cfg["proximal-mu"],seed)
    return Message(content=RecordDict({
        "arrays":ArrayRecord(model.state_dict()),
        "metrics":MetricRecord({"num-examples":int(len(y)),"train-loss":float(loss),
                                "region-code":int(region)})}), reply_to=msg)
