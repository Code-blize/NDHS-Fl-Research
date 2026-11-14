import torch
from torch import nn

class NDHSMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)

def get_parameters(model):
    return [v.detach().cpu().numpy().copy() for v in model.state_dict().values()]

def set_parameters(model, parameters):
    state = model.state_dict()
    if len(parameters) != len(state):
        raise ValueError("Parameter count mismatch")
    loaded = {}
    for (key, tensor), array in zip(state.items(), parameters):
        value = torch.as_tensor(array, dtype=tensor.dtype)
        if tuple(value.shape) != tuple(tensor.shape):
            raise ValueError(f"Shape mismatch for {key}: {tuple(value.shape)} != {tuple(tensor.shape)}")
        loaded[key] = value
    model.load_state_dict(loaded, strict=True)
