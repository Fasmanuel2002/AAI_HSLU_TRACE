from sklearn.metrics import f1_score
from typing import Tuple, List
import torch

def search_best_f1_thr(val_probs, val_true, thresholds) -> Tuple[float, float, float]: 
    """
    # Convert continuous probabilities [0, 1] into binary predictions [0 or 1]
    # based on the current threshold candidate 't'
    # If this threshold results in a better F1 score, update our best values
    """
    best_thr, best_f1, best_macro_f1 = 0.5, 0.0, 0.0
    for t in thresholds:
        pred = (val_probs >= t).astype(int)
        f1 = f1_score(val_true, pred, zero_division=0)
        macro_f1 = f1_score(val_true, pred ,zero_division=0, average="macro")
        
        if f1 > best_f1:
            best_f1 = f1
            best_thr = t
            best_macro_f1 = macro_f1
    return float(best_f1), float(best_macro_f1), float(best_thr)



def update_binary_metrics(
    logit : torch.Tensor,
    targets : torch.Tensor,
    correct_predictions : int,
    total_predictions : int,
    y_true_list : list,
    y_pred_list : list,
    threshold : float = 0.5
    ) -> Tuple[int,int]:
    
    probs = torch.sigmoid(logit)
    preds = (probs >= threshold).float()
    
    correct_predictions += (preds == targets).sum().item() # type: ignore
    total_predictions += targets.numel()
    
    y_true_list.append(targets.detach().cpu())
    y_pred_list.append(preds.detach().cpu())
    
    return correct_predictions, total_predictions


def append_probs_and_true(
    logits: torch.Tensor,
    targets: torch.Tensor,
    probs_list: List[torch.Tensor],
    true_list: List[torch.Tensor],
) -> None:
    """
    Appends sigmoid probabilities and targets (both moved to CPU) to lists.
    logits/targets expected shape: (B, 1) for binary task.
    """
    probs = torch.sigmoid(logits)
    probs_list.append(probs.detach().cpu())
    true_list.append(targets.detach().cpu())