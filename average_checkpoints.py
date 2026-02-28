import torch
import os
from collections import OrderedDict


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

source_folder = os.path.join(BASE_DIR, 'database/model/')
starts_with = "step"
ends_with   = ".pth.tar"

checkpoint_names = [f for f in os.listdir(source_folder) if f.startswith(starts_with) and f.endswith(ends_with)]
assert len(checkpoint_names) > 0, "Did not find any checkpoint names"

average_params = OrderedDict()
for checkpoint_name in checkpoint_names:
    checkpoint = torch.load(checkpoint_name)
    checkpoint_params = checkpoint.state_dict()
    checkpoint_params_name = checkpoint_params.key()
    for param_name in checkpoint_params_name:
        if param_name not in average_params :
            average_params[param_name] = checkpoint_params[param_name].clone() * 1.0 / len(checkpoint_names)
        else :
            average_params[param_name] += checkpoint_params[param_name].clone() * 1.0 / len(checkpoint_names)

average_checkpoint = torch.load(checkpoint_names[0])['model']
for param_name in average_params:
    assert param_name in average_params

average_checkpoint.load_state_dict(average_params)
torch.save({'model': average_checkpoint}, "averaged_transformer_checkpoint.pth.tar")


