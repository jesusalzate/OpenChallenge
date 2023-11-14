import torch

st = torch.load("model_best_weights.pth", map_location=torch.device("cpu"))[
    "state_dict"
]

torch.save(st, "model_best_state.pth")
