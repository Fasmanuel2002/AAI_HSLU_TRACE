import torch
from typing import Optional

def normalize_features(features: torch.Tensor, 
                       feature_mask: Optional[torch.Tensor] = None,
                       eps: float = 1e-6) -> torch.Tensor:
    """
    Normalize input timestamps as described in Section 2.2 (Feature Engineering) of the TRACE paper.
    A feature mask is applied to exclude padded values, since padding introduces zeros that would
    otherwise distort the mean and variance during normalization.
    """
    
    #If there is not any zero do the normal normalization
    if feature_mask is None:
        return (features - features.mean()) / (features.std() + eps)
    
    mask_bool = feature_mask.bool()
    
    mask_float = mask_bool.float()
    
    mean = (features * mask_float).sum(dim=1, keepdim=True) / mask_float.sum(dim=1, keepdim=True).clamp(min=1)
    variance  = ((features - mean) ** 2 * mask_float).sum(dim=1, keepdim=True) / mask_float.sum(dim=1, keepdim=True).clamp(min=1)
    
    std  = torch.sqrt(variance + eps)
    
    normalized = (features - mean) / std
    
    return torch.where(mask_bool, normalized, torch.zeros_like(features))
