import numpy as np
from dataset.otto_final import TraceOttoDataset
from typing import Tuple


def mean_input_target_clicks_per_session(dataset : TraceOttoDataset) -> Tuple[float, float]:

    clicks_per_session_input = []
    clicks_per_session_target = []

    for session in dataset.session:
        input_part, target_part = dataset.__split_input_target__(session)
    
        type_input = np.asarray(input_part["type"])
        type_target = np.asarray(target_part["type"])
        clicks_input = np.sum(type_input == 1)   
        clicks_target = np.sum(type_target == 1)   
        clicks_per_session_input.append(clicks_input)
        clicks_per_session_target.append(clicks_target)
        

    return (float(np.mean(clicks_per_session_input)), float(np.mean(clicks_per_session_target)))
    
