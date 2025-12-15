import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter # type: ignore
import time
import numpy as np
from model.trace import TRACE
from dataset.otto_trace import TraceOttoDataSet
from utils.feature_engineering import get_between_features, get_elapsed_feature

from utils.EarlyStopping import EarlyStopping

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def main():
        #DataSet    
    dataset_processed = TraceOttoDataSet(
        file_name='train.jsonl',
        input_seq_len=64,
        min_timestamps_per_sample=16,
        max_samples=1000000
    )

        
    #See if the Lenght of the new inputs are the Lenght Sequence
    
    # Data splitting train/val
    train_size = int(0.8 * len(dataset_processed))
    val_size = len(dataset_processed) - train_size
    
    train_data, val_data = random_split(
        dataset_processed,
        [train_size, val_size]
    )



    #DEV_SET
    """
    validation_loader = DataLoader(
        dataset=val_data,
        batch_size=32,
        collate_fn=custom_collate,
        shuffle=False
    )
    """

    #TRAIN SET
    #TRAIN SET
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=32,
        shuffle=True,
        pin_memory=True,
        num_workers=0
    )
    #val SET
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=32,
        shuffle=False,
        pin_memory=True,
        num_workers=0
    )    
    
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
        num_classes=1
    )  
      
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trace_model = trace_model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(trace_model.parameters(), lr=3e-5, weight_decay=1e-6)
    early_stopping = EarlyStopping(patience=6, min_delta=1e-4, mode="min", path="best_TRACE_model.pt")
    
    num_epochs = 10

    #Summary Writer for tensorBoard
    tensor_board_writer = SummaryWriter(log_dir=f"runs/Debugging_RA1/exp_{time.time()}")
    trace_model.train()

    for epoch in range(num_epochs):
        # -------------------------------TRAINING ---------------------------
        trace_model.train()
        epoch_loss = 0.0
        correct_train_RA1 = 0

        
        total_train_RA1 = 0

        for inputs_train, targets_train in train_loader:

            label_train_RA1 = targets_train["RA1"].unsqueeze(1).to(device)

            inputs_train = {
                k: v.to(device)
                for k, v in inputs_train.items()
            }

            delta_elapsed = get_elapsed_feature(inputs_train["timestamps"]).to(device)
            delta_between = get_between_features(inputs_train["timestamps"]).to(device)

            logits_val = trace_model(
                inputs_train["aid"],
                inputs_train["type"],
                delta_elapsed,
                delta_between
            )

            
            pred_RA1 = logits_val
            
            loss_RA1 = criterion(pred_RA1, label_train_RA1.float())

            optimizer.zero_grad()
            
            loss_training = loss_RA1
            
            loss_training.backward()
            
            optimizer.step()

            epoch_loss += loss_training.item()


            # ============ RA1 ============
            probs_RA1 = torch.sigmoid(pred_RA1)
            preds_RA1 = (probs_RA1 >= 0.5).float()
            correct_train_RA1 += (preds_RA1 == label_train_RA1).sum().item()
            total_train_RA1 += label_train_RA1.numel()

        train_loss = epoch_loss / len(train_loader)

        train_acc_RA1 = correct_train_RA1 / max(total_train_RA1, 1)

        tensor_board_writer.add_scalar("Training/Loss", train_loss, epoch)

        tensor_board_writer.add_scalar("Train/Acc_RA1", train_acc_RA1, epoch)

        # -------------------------------VALIDATION---------------------------
        trace_model.eval()
        val_loss = 0.0

        correct_val_RA1 = 0

        total_val_RA1 = 0

        with torch.no_grad():
            for inputs_val, targets_val in val_loader:
                
                label_val_RA1 = targets_val["RA1"].unsqueeze(1).to(device)

                inputs_val = {
                    k: v.to(device)
                    for k, v in inputs_val.items()
                }

                delta_elapsed = get_elapsed_feature(inputs_val["timestamps"]).to(device)
                delta_between = get_between_features(inputs_val["timestamps"]).to(device)

                logits_val = trace_model(
                    inputs_val["aid"],
                    inputs_val["type"],
                    delta_elapsed,
                    delta_between
                )

            
                
                pred_RA1_val = logits_val
                
                loss_RA1_val = criterion(pred_RA1_val, label_val_RA1.float())

                val_loss += loss_RA1_val.item()

                #RA1 Debuggin
                probs_RA1_val = torch.sigmoid(pred_RA1_val)
                preds_RA1_val = (probs_RA1_val >= 0.5).float()
                correct_val_RA1 += (preds_RA1_val == label_val_RA1).sum().item()
                total_val_RA1 += label_val_RA1.numel()

        val_loss /= len(val_loader)

        
        val_acc_RA1 = correct_val_RA1 / max(total_val_RA1, 1)

        tensor_board_writer.add_scalar("Val/Loss", val_loss, epoch)
        tensor_board_writer.add_scalar("Val/Acc_RA1", val_acc_RA1, epoch)
        
            
        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc_RA1:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc_RA1:.4f}"
        )
        

        early_stopping(val_loss, trace_model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    tensor_board_writer.close()
   
"""
    #Load the Best model
    early_stopping.load_best_weights(trace_model)

    ## Save all the model(only to resume training)
    torch.save({
        "epoch": epoch,
        "model_state_dict": trace_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }, "checkpoint_TRACE.pt")
"""

        
if __name__ == "__main__":
    main()