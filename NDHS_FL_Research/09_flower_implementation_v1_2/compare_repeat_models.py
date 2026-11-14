import torch

original_path = (
    "outputs/integration_smoke/baseline_fedavg_e1_s42/"
    "final_model_state_dict.pt"
)
repeat_path = (
    "outputs/reproducibility/baseline_fedavg_e1_s42_repeat1/"
    "final_model_state_dict.pt"
)

original = torch.load(original_path, map_location="cpu", weights_only=True)
repeat = torch.load(repeat_path, map_location="cpu", weights_only=True)

print("Same parameter keys:", original.keys() == repeat.keys())

max_abs_diff = 0.0
different_tensors = 0

for key in original:
    a = original[key].float()
    b = repeat[key].float()

    diff = (a - b).abs()
    tensor_max = diff.max().item() if diff.numel() else 0.0

    max_abs_diff = max(max_abs_diff, tensor_max)

    if not torch.equal(original[key], repeat[key]):
        different_tensors += 1

    print(
        f"{key}: exact_match={torch.equal(original[key], repeat[key])}, "
        f"max_abs_diff={tensor_max:.12g}"
    )

print("\nDifferent tensors:", different_tensors)
print("Maximum absolute parameter difference:", f"{max_abs_diff:.12g}")
