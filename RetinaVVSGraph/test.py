from RetinaVVSGraph_class import RetinaVVSGraph
import torch

vvs_graph = {
    '0': [1, 2, 4, "out"],
    '1': [2, 3, 4, "out"],
    '2': [3, 4, "out"],
    '3': [4, "out"],
    '4': ["out"]
}

hparams = {
        'input_shape': (1, 32, 32),
        'batch_size': 32,
        'ret_channels': 32,
        'vvs_graph': vvs_graph,
        'dropout': 0.0,
        'model_class': "RetinaVVSGraph",
        'lr': 1e-3
    }
model = RetinaVVSGraph(hparams)
input_t = torch.randn((hparams["batch_size"], 1, 32, 32))

print(f"Test model: {model(input_t).shape}")
