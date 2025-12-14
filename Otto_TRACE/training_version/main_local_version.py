import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
from model.trace import TRACE
from dataset.otto_trace import TraceOttoDataSet
from utils.feature_engineering import get_delta_features, get_elapsed_feature

TF_ENABLE_ONEDNN_OPTS=0

def main():
        #DataSet    
    dataset_processed = TraceOttoDataSet(
        file_name='train.jsonl',
        input_seq_len=64,
        min_timestamps_per_sample=16,
        max_samples=1000
    )

        
    #See if the Lenght of the new inputs are the Lenght Sequence
    for i,sample in enumerate(dataset_processed):
        len_sample = sample["inputs"]
        print(f"Len of the timestamps after the L sequence cut {len(len_sample["timestamps"])}")
        assert len(len_sample["timestamps"]) == 64 
        
        if i == 0:
            break    


    print("================================================ (Logits part) ===================================================")
    print("Logits Data_set_processed the ATC (Add to the Cart)")
    print(dataset_processed.__ATC_task_logit__())
        
    print("Logits for SAT4 (Seeing the same Aid 4 times)")
    print(dataset_processed.__SAT__task_logit__())
        
    print("Logits for PD1 (Make any Purchase within a day)")
    print(dataset_processed.__PD1_task_logit___())
        
    print("Logits for RA1 (Return to the same Aid in 1 days)")
    print(dataset_processed.__RA1_task_logit___())
    
    # Data splitting train/test
    train_portion = 0.80
    train_data, test_data = random_split(dataset=dataset_processed, lengths=[train_portion, 1 - train_portion])


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
        num_workers=2
    )
    #TEST SET
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=32,
        shuffle=False,
        pin_memory=True,
        num_workers=2
    )    
    
    max_aid = max(
        session["inputs"]["aid"].max().item()
        for session in dataset_processed
    )
    max_type = max(
        session["inputs"]["type"].max().item()
        for session in dataset_processed
    )

    num_embeddings_aid = max_aid + 1  
    num_embeddings_event_type = max_type + 1
    trace_model = TRACE(
        num_embeddings_aid=num_embeddings_aid,
        num_embeddings_event_type=num_embeddings_event_type,
        embedding_dim=32,
        num_classes=1 #4 # Jan: You are doing only one class here, so not 4 classes...
    )
    
    for batch_training in train_loader:
        sample = batch_training["inputs"]

        print(
            f"Shape Aids: {sample['aid'].shape}, "
            f"Shape Timestamps: {sample['timestamps'].shape}, "
            f"Shape Type: {sample['type'].shape}"
        )
        break  
    
        
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(trace_model.parameters(), lr=1e-5, weight_decay=1e-6)

    num_epochs = 15



    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(trace_model.parameters(), lr=1e-5, weight_decay=1e-6)

    num_epochs = 15

    #Summary Writer for tensorBoard
    tensor_board_writer = SummaryWriter(log_dir=f"runs/exp_{time.time()}")
    trace_model.train()
    for epoch in range(num_epochs):
        # -------------------------------TRAINING ---------------------------
        trace_model.train()
        epoch_loss = 0.0

        correct_train = 0
        total_train = 0

        for batch_trainer in train_loader:
            evidence = batch_trainer["inputs"]
            label_train = batch_trainer["targets"]["ATC"]

            delta_elapsed = get_elapsed_feature(evidence["timestamps"])
            delta_between = get_delta_features(evidence["timestamps"])

            pred = trace_model(
                evidence["aid"],
                evidence["type"],
                delta_elapsed,
                delta_between
            )

            loss = criterion(pred, label_train.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            probs = torch.sigmoid(pred)
            preds = (probs >= 0.5).float()

            correct_train += (preds == label_train).sum().item()
            total_train += label_train.numel()

        train_loss = epoch_loss / len(train_loader)
        train_acc = correct_train / max(total_train, 1)
        
        #TensorBoard Writer
        tensor_board_writer.add_scalar("Training/Loss", train_loss, epoch)
        tensor_board_writer.add_scalar("Training/Accuracy", train_acc, epoch)
        
        # ------------------------------- VALIDATION ---------------------------

        trace_model.eval()
        val_loss = 0.0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for batch_test in test_loader:
                evidence = batch_test["inputs"]
                label_test = batch_test["targets"]["ATC"].unsqueeze_(1)

                delta_elapsed = get_elapsed_feature(evidence["timestamps"])
                delta_between = get_delta_features(evidence["timestamps"])

                pred_test = trace_model(
                    evidence["aid"],
                    evidence["type"],
                    delta_elapsed,
                    delta_between
                )

                loss_v = criterion(pred_test, label_test.float())
                val_loss += loss_v.item()

                probs_test = torch.sigmoid(pred_test)
                preds_test = (probs_test >= 0.5).float()

                correct_test += (preds_test == label_test).sum().item()
                total_test += label_test.numel()

        val_loss /= len(test_loader)
        val_acc = correct_test / max(total_test, 1)
        
        #TensorBoard Writer
        tensor_board_writer.add_scalar("Val/Loss", val_loss, epoch)
        tensor_board_writer.add_scalar("Val/Accuracy", val_acc, epoch)
        
        
        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )
        
        
    #Saving the model TRACE
    torch.save({
        "epoch": epoch,
        "model_state_dict": trace_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }, "model_TRACE.pt")


        
if __name__ == "__main__":
    main()