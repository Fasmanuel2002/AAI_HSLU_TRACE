from typing import Tuple
from dataset.otto_final import TraceOttoDataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch
from typing import Dict
import torch
from functools import partial

def split_data_Train_Val_Test(data_set : TraceOttoDataset, batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Split the dataset into three DataLoaders:
    - Training set: 80% of the full dataset
    - Validation set: 10% of the full dataset
    - Test set: 10% of the full dataset
    
    FOR TRACE MODEL
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





def collate_fn_lstm(batch):
    """
    Computes true session lengths for the multi-task Bi-LSTM, so padded timesteps are ignored.
    """
    aids_list, types_list, ts_list, targets_list, lengths_list = [], [], [], [], []

    for inputs_batch, target_batch in batch:
        aid_batch = inputs_batch["aid"]
        type_batch = inputs_batch["type"]
        timestamps = inputs_batch["timestamps"]

        length = int((type_batch != 0).sum().item())
        lengths_list.append(max(1, length))

        aids_list.append(aid_batch)
        types_list.append(type_batch)
        ts_list.append(timestamps)
        targets_list.append(target_batch)

    inputs_batch = {
        "aid": torch.stack(aids_list).long(),
        "type": torch.stack(types_list).long(),
        "timestamps": torch.stack(ts_list).float(),
    }
    lengths_batch = torch.tensor(lengths_list, dtype=torch.long)

    targets_batch = {
        "ATC": torch.stack([t["ATC"] for t in targets_list]).float().view(-1, 1),
        "SAT": torch.stack([t["SAT"] for t in targets_list]).float().view(-1, 1),
        "MAP": torch.stack([t["MAP"] for t in targets_list]).float().view(-1, 1),
    }
    return inputs_batch, lengths_batch, targets_batch



def split_data_Train_Val_Test_LSTM(data_set, batch_size: int = 32):
    """
    Split the dataset into three DataLoaders:
    - Training set: 80% of the full dataset
    - Validation set: 10% of the full dataset
    - Test set: 10% of the full dataset
    
    FOR MLT BI-LSTM
    Added the collate_fn_lstm to ignore padded timestamps that no have any valuable data 
    """
    generator = torch.Generator().manual_seed(42)

    dataset_size = len(data_set)
    train_size = int(0.80 * dataset_size)
    val_size = int(0.10 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_data, val_data, test_data = random_split(
        dataset=data_set,
        lengths=[train_size, val_size, test_size],
        generator=generator
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn_lstm)
    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn_lstm)
    test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn_lstm)

    return train_loader, val_loader, test_loader
