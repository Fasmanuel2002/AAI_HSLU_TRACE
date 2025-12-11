import torch
from typing import Optional

def normalize_features(features : torch.Tensor, feature_mask : Optional[torch.Tensor] = None, eps : float = 1e-6) -> torch.Tensor:
    if feature_mask is None:
        return (features - features.mean()) / (features.std() + eps)           
    else:
        mean = (features * feature_mask).sum(1, keepdim=True) / feature_mask.sum(1, keepdim=True).clamp(min=1)
        std  = torch.sqrt(((features - mean) ** 2 * feature_mask).sum(1, keepdim=True) / feature_mask.sum(1, keepdim=True).clamp(min=1)+ eps)
        return torch.where(feature_mask, (features - mean) / std, torch.zeros_like(features)) 

