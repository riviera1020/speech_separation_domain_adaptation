
from torch.nn.utils import spectral_norm, weight_norm

def apply_norm(layer, norm_type):
    if norm_type == 'weight_norm':
        return weight_norm(layer)
    elif norm_type == 'spectral_norm':
        return spectral_norm(layer)
    else:
        return layer
