import torch
from utils.normalization import normalize_features

def get_elapsed_feature(timestamps: torch.Tensor) -> torch.Tensor:
    valid_mask = timestamps != 0

    first_idx = valid_mask.float().argmax(dim=1)
    last_idx  = valid_mask.sum(dim=1) - 1

    ts_first = timestamps[torch.arange(timestamps.size(0)), first_idx]

    ts_last  = timestamps[torch.arange(timestamps.size(0)), last_idx]

    delta_elapsed = torch.clamp(ts_last - ts_first, min=0)

    log_delta_elapsed = torch.log1p(delta_elapsed)

    return normalize_features(log_delta_elapsed)
    
def get_between_features(timestamps: torch.Tensor) -> torch.Tensor:
    delta_between = timestamps[:, 1:] - timestamps[:, :-1]

    valid_mask = (timestamps[:, 1:] > 0) & (timestamps[:, :-1] > 0)

    delta_between = torch.clamp(delta_between, min=0)

    log_delta_between = torch.log1p(delta_between)

    delta_norm_between = normalize_features(log_delta_between, valid_mask)

    return delta_norm_between