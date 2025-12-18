import torch
from torch import nn
import torch.optim as optim
from utils.SplitData import split_data_Train_Val_Test
from torch.utils.tensorboard import SummaryWriter # type: ignore
import numpy as np
from model.trace import TRACE
from dataset.otto_trace import TraceOttoDataSet

from utils.EarlyStopping import EarlyStopping
from utils.feature_engineering import get_between_features, get_elapsed_feature
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
        #DataSet    
    dataset_processed = TraceOttoDataSet(
        file_name='train.jsonl',
        input_seq_len=64,
        min_timestamps_per_sample=16
    )
    
    

    train_loader, validation_loader, test_loader = split_data_Train_Val_Test(dataset_processed)
    #See if the Lenght of the new inputs are the Lenght Sequence

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
    
    trace_model = trace_model.to(device)
    optimizer = optim.AdamW(trace_model.parameters(), lr=3e-5, weight_decay=1e-6)
    early_stopping = EarlyStopping(
    patience=6,
    min_delta=1e-4,
        mode="min",
        path="best_Check_TRACE_PD1_model.pt"
        )
    num_epochs = 40

        #Summary Writer for tensorBoard
    tensor_board_writer = SummaryWriter(log_dir=f"runs/Final/PD1_MODEL_SingleTask")
    print("Started the Training")
    #Figthing Data Imbalanced
    pos_weight = 4.0
    neg_weight = 1.0

    for epoch in range(num_epochs):
            # -------------------------------TRAINING ---------------------------
        trace_model.train()
        epoch_loss = 0.0
        correct_train_PD1 = 0
        total_train_PD1 = 0

        for inputs_train, targets_train in train_loader:

            label_train_PD1 = targets_train["PD1"].unsqueeze(1).to(device)

            inputs_train = {
                k: v.to(device)
                    for k, v in inputs_train.items()
                }

            delta_elapsed = get_elapsed_feature(inputs_train["timestamps"]).to(device)
            delta_between = get_between_features(inputs_train["timestamps"]).to(device)

                
            logits_train = trace_model(
                    inputs_train["aid"],
                    inputs_train["type"],
                    delta_elapsed,
                    delta_between
                )

                
            
            weights = torch.where(label_train_PD1.float() == 0, neg_weight, pos_weight)
            loss_training = F.binary_cross_entropy_with_logits(logits_train, label_train_PD1.float(), weight=weights)   
            

            optimizer.zero_grad()
            loss_training.backward()
                
            optimizer.step()

            epoch_loss += loss_training.item()


                # ============ PD1 ============
            probs_PD1 = torch.sigmoid(logits_train)
            preds_PD1 = (probs_PD1 >= 0.5).float()
            correct_train_PD1 += (preds_PD1 == label_train_PD1).sum().item()
            total_train_PD1 += label_train_PD1.numel()

        train_loss = epoch_loss / len(train_loader)

        train_acc_PD1 = correct_train_PD1 / max(total_train_PD1, 1)

        tensor_board_writer.add_scalar("Training/Loss", train_loss, epoch)

        tensor_board_writer.add_scalar("Train/Acc_PD1", train_acc_PD1, epoch)

            # -------------------------------VALIDATION---------------------------
        trace_model.eval()
        val_loss = 0.0

        correct_val_PD1 = 0

        total_val_PD1 = 0

        with torch.no_grad():
            for inputs_val, targets_val in validation_loader:
                    
                label_val_PD1 = targets_val["PD1"].unsqueeze(1).to(device)

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

                

                weights = torch.where(label_val_PD1.float() == 0, neg_weight, pos_weight)
                loss_PD1_val = F.binary_cross_entropy_with_logits(logits_val, label_val_PD1.float(), weight=weights)   

                val_loss += loss_PD1_val.item()

                    #PD1 Debuggin
                probs_PD1_val = torch.sigmoid(logits_val)
                preds_PD1_val = (probs_PD1_val >= 0.5).float()
                correct_val_PD1 += (preds_PD1_val == label_val_PD1).sum().item()
                total_val_PD1 += label_val_PD1.numel()

        val_loss /= len(validation_loader)

            
        val_acc_PD1 = correct_val_PD1 / max(total_val_PD1, 1)
        tensor_board_writer.add_scalar("Val/Loss", val_loss, epoch)
        tensor_board_writer.add_scalar("Val/Acc_PD1", val_acc_PD1, epoch)
            
                
                
        print(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc_PD1:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc_PD1:.4f}"
            )
            

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
    }, "Final_PD1_ALLmodel.pt")


        
if __name__ == "__main__":
    main()