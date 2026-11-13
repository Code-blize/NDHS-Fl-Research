import argparse
from flwr.simulation import run_simulation
from ndhs_flower.client_app import app as client_app
from ndhs_flower.server_app import app as server_app

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--nodes",type=int,default=6)
    p.add_argument("--cpus-per-client",type=float,default=1.0)
    args=p.parse_args()
    if args.nodes != 6: raise ValueError("The design requires six regional clients.")
    run_simulation(server_app=server_app,client_app=client_app,num_supernodes=6,
        backend_name="ray",
        backend_config={"client_resources":{"num_cpus":args.cpus_per_client,"num_gpus":0}})
if __name__=="__main__": main()
