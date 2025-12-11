import torch
from utils.normalization import normalize_features

def get_elapsed_feature(timestamps : torch.Tensor) -> torch.Tensor:
    zero_mask = timestamps != 0 
    last_indices = zero_mask.sum(dim=1) - 1 
    ts_first = timestamps[:, 0]
    ts_last = timestamps[torch.arange(timestamps.size(0)), last_indices]

    log_delta_elapsed = torch.log1p(torch.clamp(ts_last - ts_first, min=0))

    # Jan: Reuse code when you can ... (normalization is something so typical, you might need it again)
    # ... so separate it so that you can reuse in the future easily
    delta_elapsed_normalized = normalize_features(log_delta_elapsed)

    return delta_elapsed_normalized
    
def get_delta_features(timestamps : torch.Tensor) -> torch.Tensor:
    #Delta Between, after - before
    delta_between = timestamps[:, 1:] - timestamps[:, :-1]
    mask_zero = delta_between > 0 #True if is bigger than 0
    
    log_delta_between = torch.log1p(delta_between.clamp(min=0))

    # Jan: Reuse code when you can ... (normalization is something so typical, you might need it again)
    # ... so separate it so that you can reuse in the future easily
    delta_norm_between = normalize_features(log_delta_between, mask_zero)

    return delta_norm_between
