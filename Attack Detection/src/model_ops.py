from torch import autograd
from torch import nn
import numpy as np
import torch

def load_server(load_location='', model_class=None):
	net = model_class()
	net.load_state_dict(torch.load(load_location))
	return net

def model_handle(oracle: nn.Module) -> callable:
	def predict(img_query: np.ndarray) -> np.ndarray:
		print("The input query shape is:", img_query.shape, "and the size is:", img_query.size)
		img_torch = torch.from_numpy(img_query).view(1, 3, 32, 32).float()
		img_torch = to_range(img_torch, img_torch.min(), img_torch.max(), -1., 1.)
		return oracle(autograd.Variable(img_torch)).data.cpu().numpy()
	return predict

def to_range(x, old_min, old_max, new_min, new_max):
	old_range = (old_max - old_min)
	new_range = (new_max - new_min)
	return (((x - old_min) * new_range) / old_range) + new_min
