import torch
from torch import nn
from .data_loader import make_loader
from .reproducibility import set_global_seed

def train_local(model, X, y, epochs, batch_size, lr, proximal_mu, seed):
    set_global_seed(seed)
    loader = make_loader(X,y,batch_size,seed)
    criterion = nn.BCEWithLogitsLoss()
    # New Adam instance for every client-round; moments are reset each round.
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    # Snapshot global parameters before local updates (model is loaded from server arrays).
    global_params = [p.detach().clone() for p in model.parameters()]
    model.train(); total=0.0; n_total=0
    for _ in range(int(epochs)):
        for xb,yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb),yb)
            if float(proximal_mu) > 0:
                penalty = sum(torch.sum((p-gp)**2) for p,gp in zip(model.parameters(),global_params))
                loss = loss + (float(proximal_mu)/2.0)*penalty
            loss.backward(); optimizer.step()
            n=len(yb); total += float(loss.detach())*n; n_total += n
    return total/max(n_total,1)
