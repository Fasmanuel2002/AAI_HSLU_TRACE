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
from utils.feature_engineering import get_delta_features, get_elapsed_feature

from utils.EarlyStopping import EarlyStopping

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def main():
        #DataSet    
    dataset_processed = TraceOttoDataSet(
        file_name='train.jsonl',
        input_seq_len=64,
        min_timestamps_per_sample=16,
    )

        
    #See if the Lenght of the new inputs are the Lenght Sequence
    sample = dataset_processed[0]
    assert len(sample[0]["timestamps"]) == 64

    
    # Data splitting train/test
    train_size = int(0.8 * len(dataset_processed))
    test_size = len(dataset_processed) - train_size
    
    train_data, test_data = random_split(
        dataset_processed,
        [train_size, test_size]
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
    #TEST SET
    test_loader = DataLoader(
        dataset=test_data,
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
        num_classes=4#4 # Jan: You are doing only one class here, so not 4 classes...
    )  
      
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trace_model = trace_model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(trace_model.parameters(), lr=3e-5, weight_decay=1e-6)
    early_stopping = EarlyStopping(
    patience=6,
    min_delta=1e-4,
    mode="min",
    path="best_TRACE_model.pt"
    )
    num_epochs = 35

    #Summary Writer for tensorBoard
    tensor_board_writer = SummaryWriter(log_dir=f"runs/exp_{time.time()}")
    trace_model.train()

    for epoch in range(num_epochs):
        # -------------------------------TRAINING ---------------------------
        trace_model.train()
        epoch_loss = 0.0

        correct_train_ATC = 0
        correct_train_SAT = 0
        correct_train_PD1 = 0
        correct_train_RA1 = 0

        total_train_ATC = 0
        total_train_SAT = 0
        total_train_PD1 = 0
        total_train_RA1 = 0

        for inputs_train, targets_train in train_loader:

           
            label_train_ATC = targets_train["ATC"].unsqueeze(1).to(device)
            label_train_SAT = targets_train["SAT"].unsqueeze(1).to(device)
            label_train_PD1 = targets_train["PD1"].unsqueeze(1).to(device)
            label_train_RA1 = targets_train["RA1"].unsqueeze(1).to(device)

   
            inputs_train = {
                k: v.to(device)
                for k, v in inputs_train.items()
            }

            delta_elapsed = get_elapsed_feature(inputs_train["timestamps"]).to(device)
            delta_between = get_delta_features(inputs_train["timestamps"]).to(device)

            logits_val = trace_model(
                inputs_train["aid"],
                inputs_train["type"],
                delta_elapsed,
                delta_between
            )

            pred_ATC = logits_val[:, 0:1] #ATC
            pred_SAT = logits_val[:, 1:2] #SAT
            pred_PD1 = logits_val[:, 2:3] #PD1
            pred_RA1 = logits_val[:, 3:4] #RA1

            loss_ATC = criterion(pred_ATC, label_train_ATC.float())
            loss_SAT = criterion(pred_SAT, label_train_SAT.float())
            loss_PD1 = criterion(pred_PD1, label_train_PD1.float())
            loss_RA1 = criterion(pred_RA1, label_train_RA1.float())

            optimizer.zero_grad()
            
            loss_training = loss_ATC + loss_SAT + loss_PD1 + loss_RA1
            
            loss_training.backward()
            
            optimizer.step()

            epoch_loss += loss_training.item()

            # ============ ATC ============
            probs_ATC = torch.sigmoid(pred_ATC)
            preds_ATC = (probs_ATC >= 0.5).float()
            correct_train_ATC += (preds_ATC == label_train_ATC).sum().item()
            total_train_ATC += label_train_ATC.numel()

            # ============ SAT ============
            probs_SAT = torch.sigmoid(pred_SAT)
            preds_SAT = (probs_SAT >= 0.5).float()
            correct_train_SAT += (preds_SAT == label_train_SAT).sum().item()
            total_train_SAT += label_train_SAT.numel()

            # ============ PD1 ============
            probs_PD1 = torch.sigmoid(pred_PD1)
            preds_PD1 = (probs_PD1 >= 0.5).float()
            correct_train_PD1 += (preds_PD1 == label_train_PD1).sum().item()
            total_train_PD1 += label_train_PD1.numel()

            # ============ RA1 ============
            probs_RA1 = torch.sigmoid(pred_RA1)
            preds_RA1 = (probs_RA1 >= 0.5).float()
            correct_train_RA1 += (preds_RA1 == label_train_RA1).sum().item()
            total_train_RA1 += label_train_RA1.numel()

        train_loss = epoch_loss / len(train_loader)

        train_acc_ATC = correct_train_ATC / max(total_train_ATC, 1)
        train_acc_SAT = correct_train_SAT / max(total_train_SAT, 1)
        train_acc_PD1 = correct_train_PD1 / max(total_train_PD1, 1)
        train_acc_RA1 = correct_train_RA1 / max(total_train_RA1, 1)

        tensor_board_writer.add_scalar("Training/Loss", train_loss, epoch)
        tensor_board_writer.add_scalar("Train/Acc_ATC", train_acc_ATC, epoch)
        tensor_board_writer.add_scalar("Train/Acc_SAT", train_acc_SAT, epoch)
        tensor_board_writer.add_scalar("Train/Acc_PD1", train_acc_PD1, epoch)
        tensor_board_writer.add_scalar("Train/Acc_RA1", train_acc_RA1, epoch)

        # -------------------------------VALIDATION---------------------------
        trace_model.eval()
        val_loss = 0.0

        correct_test_ATC = 0
        correct_test_SAT = 0
        correct_test_PD1 = 0
        correct_test_RA1 = 0

        total_test_ATC = 0
        total_test_SAT = 0
        total_test_PD1 = 0
        total_test_RA1 = 0

        with torch.no_grad():
            for inputs_test, targets_test in test_loader:

                label_test_ATC = targets_test["ATC"].unsqueeze(1).to(device)
                label_test_SAT = targets_test["SAT"].unsqueeze(1).to(device)
                label_test_PD1 = targets_test["PD1"].unsqueeze(1).to(device)
                label_test_RA1 = targets_test["RA1"].unsqueeze(1).to(device)

                inputs_test = {
                    k: v.to(device)
                    for k, v in inputs_test.items()
                }

                delta_elapsed = get_elapsed_feature(inputs_test["timestamps"]).to(device)
                delta_between = get_delta_features(inputs_test["timestamps"]).to(device)

                logits_test = trace_model(
                    inputs_test["aid"],
                    inputs_test["type"],
                    delta_elapsed,
                    delta_between
                )

                pred_ATC_test = logits_test[:, 0:1]
                pred_SAT_test = logits_test[:, 1:2]
                pred_PD1_test = logits_test[:, 2:3]
                pred_RA1_test = logits_test[:, 3:4]
                
                loss_ATC_val = criterion(pred_ATC_test, label_test_ATC.float())
                loss_SAT_val = criterion(pred_SAT_test, label_test_SAT.float())
                loss_PD1_val = criterion(pred_PD1_test, label_test_PD1.float())
                loss_RA1_val = criterion(pred_RA1_test, label_test_RA1.float())

                loss_validation = loss_ATC_val + loss_SAT_val + loss_PD1_val + loss_RA1_val
                
                

                val_loss += loss_validation.item()

                
                probs_ATC_test = torch.sigmoid(pred_ATC_test)
                preds_ATC_test = (probs_ATC_test >= 0.5).float()
                correct_test_ATC += (preds_ATC_test == label_test_ATC).sum().item()
                total_test_ATC += label_test_ATC.numel()

                probs_SAT_test = torch.sigmoid(pred_SAT_test)
                preds_SAT_test = (probs_SAT_test >= 0.5).float()
                correct_test_SAT += (preds_SAT_test == label_test_SAT).sum().item()
                total_test_SAT += label_test_SAT.numel()

                probs_PD1_test = torch.sigmoid(pred_PD1_test)
                preds_PD1_test = (probs_PD1_test >= 0.5).float()
                correct_test_PD1 += (preds_PD1_test == label_test_PD1).sum().item()
                total_test_PD1 += label_test_PD1.numel()

                probs_RA1_test = torch.sigmoid(pred_RA1_test)
                preds_RA1_test = (probs_RA1_test >= 0.5).float()
                correct_test_RA1 += (preds_RA1_test == label_test_RA1).sum().item()
                total_test_RA1 += label_test_RA1.numel()

        val_loss /= len(test_loader)

        val_acc_ATC = correct_test_ATC / max(total_test_ATC, 1)
        val_acc_SAT = correct_test_SAT / max(total_test_SAT, 1)
        val_acc_PD1 = correct_test_PD1 / max(total_test_PD1, 1)
        val_acc_RA1 = correct_test_RA1 / max(total_test_RA1, 1)

        tensor_board_writer.add_scalar("Val/Loss", val_loss, epoch)
        tensor_board_writer.add_scalar("Val/Acc_ATC", val_acc_ATC, epoch)
        tensor_board_writer.add_scalar("Val/Acc_SAT", val_acc_SAT, epoch)
        tensor_board_writer.add_scalar("Val/Acc_PD1", val_acc_PD1, epoch)
        tensor_board_writer.add_scalar("Val/Acc_RA1", val_acc_RA1, epoch)

        early_stopping(val_loss, trace_model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    tensor_board_writer.close()
   
        
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


        
if __name__ == "__main__":
    main()