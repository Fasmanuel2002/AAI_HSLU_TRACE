from sklearn.metrics import f1_score
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader
from torch import Tensor
from Otto_TRACE.model.trace import TRACE
from Otto_TRACE.dataset.otto_final import TraceOttoDataset
from torch import device as torch_device

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


def initialize_TRACE_model(dataset_processed : TraceOttoDataset, num_classes: int, device : torch_device) -> TRACE:
    """
    Initialize the TRACE paper from the paper part 2.3 (Model Architecture)
    """
    max_aid = max(
        session[0]["aid"].max().item()
        for session in dataset_processed
    )
    max_type = max(
        session[0]["type"].max().item()
        for session in dataset_processed
    )

    num_embeddings_aid = max_aid + 1  
    num_embeddings_event_type = max_type + 1
    trace_model = TRACE(
        num_embeddings_aid=num_embeddings_aid,
        num_embeddings_event_type=num_embeddings_event_type,
        embedding_dim=32,
        num_classes=num_classes
    )  
      

    trace_model = trace_model.to(device)
    
    return trace_model


def update_binary_metrics(logit : torch.Tensor,
                          targets : torch.Tensor,
                          correct_predictions : int,
                          total_predictions : int,
                          y_true_list : list,
                          y_pred_list : list,
                          threshold : float = 0.5) -> Tuple[int,int]:
    """
    Convert logits to binary predictions, update accuracy counters, and store
    true/predicted labels for later metric computation.
    """
    probs = torch.sigmoid(logit)
    preds = (probs >= threshold).float()
    
    correct_predictions += (preds == targets).sum().item() # type: ignore
    total_predictions += targets.numel()
    
    y_true_list.append(targets.detach().cpu())
    y_pred_list.append(preds.detach().cpu())
    
    return correct_predictions, total_predictions


def append_probs_and_true(logits: torch.Tensor,
                          targets: torch.Tensor,
                          probs_list: List[torch.Tensor],
                          true_list: List[torch.Tensor]):
    """
    Appends sigmoid probabilities and targets (both moved to CPU) to lists.
    logits/targets expected shape: (B, 1) for binary task. 
    """
    probs = torch.sigmoid(logits)
    probs_list.append(probs.detach().cpu())
    true_list.append(targets.detach().cpu())
    
    
def compute_f1_tasks(y_true_batches: List[torch.Tensor], 
                     y_pred_batches: List[torch.Tensor],
                     zero_division: int = 0) -> float:
    """
    Concatenates batch-level tensors and computes F1 score.
    """
    y_true = torch.cat(y_true_batches).detach().cpu().numpy().ravel()
    y_pred = torch.cat(y_pred_batches).detach().cpu().numpy().ravel()
    return float(f1_score(y_true, y_pred, zero_division=zero_division))



def ratio_finder_single_task(train_loader : DataLoader, task_train : str , device : torch_device) -> Tuple[Tensor, Tensor]:
    """
    Computes the ratio between positive (1) and negative (0) samples for a single task.
    """

    labels_list = []
    for _, targets in train_loader:
        labels_list.append(targets[task_train].view(-1)) #(Batch, )
        
    labels = torch.cat(labels_list, dim=0) #(N, )           
    
    #Number of positives in the train_loader
    num_pos = (labels == 1).sum().item()
    
    #Number of Negatives in the train_loader
    num_neg = (labels == 0).sum().item()
    
    ratio = num_neg / max(num_pos, 1)

    print("Train pos/neg:", num_pos, num_neg)

    w_pos = torch.tensor([ratio], device=device).float() 
    
    w_neg = torch.tensor([1.0], device=device).float()
    
    return (w_pos, w_neg)



def ratios_finder_multi_task(train_loader : DataLoader, device: torch_device) -> Tuple[Tensor, Tensor, Tensor]: 
    """
    Computes the ratio between positive (1) and negative (0) samples for  Multi task Learning
    """
    labels_list_ATC = []
    labels_list_SAT = []
    labels_list_MAP = []
    
    for _, targets in train_loader:
        labels_list_ATC.append(targets["ATC"].view(-1)) #(Batch, )
        labels_list_SAT.append(targets["SAT"].view(-1)) #(Batch, )
        labels_list_MAP.append(targets["MAP"].view(-1)) # (Batch, )
        
    labels_ATC = torch.cat(labels_list_ATC, dim=0)
    labels_SAT = torch.cat(labels_list_SAT, dim=0)
    labels_MAP = torch.cat(labels_list_MAP,dim=0)
    
    num_pos_ATC = (labels_ATC == 1).sum().item()
    num_neg_ATC = (labels_ATC == 0).sum().item()
    
    num_pos_SAT = (labels_SAT == 1).sum().item()
    num_neg_SAT = (labels_SAT == 0).sum().item()
    
    num_pos_MAP = (labels_MAP == 1).sum().item()
    num_neg_MAP = (labels_MAP == 0).sum().item()
    
    
    ratio_ATC = num_neg_ATC / max(num_pos_ATC, 1)
    
    ratio_SAT = num_neg_SAT / max(num_pos_SAT, 1)
    
    ratio_MAP = num_neg_MAP / max(num_pos_MAP, 1)
    
        
    print("ATC Train pos/neg:", num_pos_ATC, num_neg_ATC)
    print("SAT Train pos/neg:", num_pos_SAT, num_neg_SAT)
    print("MAP Train pos/neg:", num_pos_MAP, num_neg_MAP)
    
    w_pos_ATC = torch.tensor([ratio_ATC], device=device).float()
    w_pos_SAT = torch.tensor([ratio_SAT], device=device).float()
    w_pos_MAP = torch.tensor([ratio_MAP], device=device).float()
    
    return (w_pos_ATC, w_pos_SAT, w_pos_MAP)

def concate_probs_true(val_probs : List[Tensor] , val_true : List[Tensor]) -> Tuple:
    """
    Concatonate the Probabilities and true labels
    """
    val_probs = torch.cat(val_probs).detach().cpu().numpy().ravel()
    val_true = torch.cat(val_true).detach().cpu().numpy().ravel()
    return val_probs, val_true