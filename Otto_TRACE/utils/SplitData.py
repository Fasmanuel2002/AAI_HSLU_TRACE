from typing import Tuple
from dataset.otto_final import TraceOttoDataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch

def split_data_Train_Val_Test(data_set : TraceOttoDataset, batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Split the dataset into three DataLoaders:
    - Training set: 80% of the full dataset
    - Validation set: 10% of the full dataset
    - Test set: 10% of the full dataset
    """

    generator = torch.Generator().manual_seed(42)
    
    # Data splitting train/test/val
    dataset_size = len(data_set)
    
    train_size = int(0.80 * dataset_size)
    
    val_size = int(0.10 * dataset_size)
    
    test_size = dataset_size - train_size - val_size
    
    train_data, val_data, test_data = random_split(dataset=data_set, lengths=[train_size, val_size, test_size],generator=generator)

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