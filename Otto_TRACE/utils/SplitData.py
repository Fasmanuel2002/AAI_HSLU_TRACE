import os
import sys
from typing import Tuple
from dataset.otto_final import TraceOttoDataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import torch

def split_data_Train_Val_Test(data_set : TraceOttoDataset, batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    generator = torch.Generator().manual_seed(42)
    
    # Data splitting train/test/val
    dataset_size = len(data_set)
    
    train_size = int(0.80 * dataset_size)
    
    val_size = int(0.10 * dataset_size)
    
    test_size = dataset_size - train_size - val_size
    
    train_data, val_data, test_data = random_split(dataset=data_set, lengths=[train_size, val_size, test_size],generator=generator)
    
    
    """
    Talk With Jan
    labels = []
    for i in range(len(train_data)):
        inputs, targets = train_data[i]
        labels.append(int(targets["PD1"]))
    
    labels = torch.tensor(labels, dtype=torch.int64)
    
    num_ones = (labels == 1).sum().item()
    
    num_zeros = (labels == 0).sum().item()
    
    weights_ones = 1.0 / max(num_ones, 1)
    
    weights_zeros = 1.0 / max(num_zeros, 1)
    
    class_weights = torch.tensor([weights_zeros, weights_ones], dtype=torch.double) 
    
    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(sample_weights, 
                                    num_samples=len(sample_weights),
                                    replacement=True)
    """
    #TRAIN SET
    train_loader = DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    shuffle=True
    )
    #VALIDATION SET
    validation_loader = DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=False,
    )

    #TEST SET
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, validation_loader, test_loader