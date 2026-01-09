import numpy as np
from dataset.otto_final import TraceOttoDataset
from typing import Tuple

def see_target_temporal_windows(dataset):
    target_lengths = []
    temporal_gaps = []

    for session in dataset.session:
        input_part, target_part = dataset.__split_input_target__(session)

        target_lengths.append(len(target_part["timestamps"]))
        gap = target_part["timestamps"][0] - input_part["timestamps"][-1]
        temporal_gaps.append(gap)

    print("Target length:")
    print("min of timestamps in the target:", min(target_lengths))
    print("mean: of timestamps in the target", sum(target_lengths) / len(target_lengths))
    print("max: of timestamps in the target", max(target_lengths))

    print("Temporal gap in miliseconds:")
    print("min:", min(temporal_gaps))
    print("mean:", sum(temporal_gaps) / len(temporal_gaps))
    print("max:", max(temporal_gaps))

def inspect_purchase_temporal_windows(dataset):
    purchase_counts = []
    purchase_gaps = []

    for session in dataset.session:
        input_part, target_part = dataset.__split_input_target__(session)

        target_orders = np.asarray(target_part["type"])
        input_ts = np.asarray(input_part["timestamps"])
        target_ts = np.asarray(target_part["timestamps"]) 
        purchase_ts = target_ts[(target_orders == 3)]
        
        
        
        if purchase_ts.size == 0:            
            continue

        purchase_counts.append(int(purchase_ts.size))
        purchase_gaps.append(int(purchase_ts[0] - input_ts[-1]))

    if not purchase_counts:
        print("No purchases found in any target window.")
        return

    print("Purchase count per target:")
    print("min:", min(purchase_counts))
    print("mean:", np.mean(purchase_counts))
    print("max:", max(purchase_counts))

    print("\nPurchase temporal gap:")
    print("min:", min(purchase_gaps))
    print("mean:", np.mean((purchase_gaps)))
    print("max:", max(purchase_gaps))
    
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
    
