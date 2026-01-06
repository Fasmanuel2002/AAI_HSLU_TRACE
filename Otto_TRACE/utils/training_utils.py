from sklearn.metrics import f1_score
from typing import Tuple

def search_best_f1_thr(val_probs, val_true, thresholds) -> Tuple[float, float]: 
    """
    # Convert continuous probabilities [0, 1] into binary predictions [0 or 1]
    # based on the current threshold candidate 't'
    # If this threshold results in a better F1 score, update our best values
    """
    best_thr, best_f1 = 0.5, 0.0
    for t in thresholds:
        pred = (val_probs >= t).astype(int)
        f1 = f1_score(val_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, t
    return best_thr, float(best_f1)
