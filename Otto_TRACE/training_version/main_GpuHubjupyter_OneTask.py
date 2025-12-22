import torch
import torch.optim as optim
from utils.SplitData import split_data_Train_Val_Test
from torch.utils.tensorboard import SummaryWriter # type: ignore

from model.trace import TRACE
from dataset.otto_trace import TraceOttoDataSet

from utils.EarlyStopping import EarlyStopping
from utils.feature_engineering import get_between_features, get_elapsed_feature
from sklearn.metrics import f1_score
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    
    print("Beginning")
    #DataSet    
    dataset_processed = TraceOttoDataSet(
        file_name='train.jsonl',
        input_seq_len=64,
        min_timestamps_per_sample=16
    )
    
    #Split the Data into Training_loader, Validation_loader and test_loaders
    train_loader, validation_loader, test_loader = split_data_Train_Val_Test(dataset_processed, batch_size=16)
    
    #calling the max aid and type for combating the Out of Range Error -> Learning Embeddings
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
    
    #Initialize the Model TRACE(Model Architecture TRACE paper Part 2.3)
    trace_model = TRACE(
        num_embeddings_aid=num_embeddings_aid,
        num_embeddings_event_type=num_embeddings_event_type,
        embedding_dim=32,
        num_classes=1
    )  
    
    trace_model = trace_model.to(device)
    optimizer = optim.AdamW(trace_model.parameters(), lr=3e-5, weight_decay=1e-6)
    early_stopping = EarlyStopping(patience=6,min_delta=5e-4,mode="max",path="best_CheckPoint_batch16_PD1_model_lr3-e5_wd1e-6_earlystopping.pt")
    num_epochs = 40

    #Summary Writer for tensorBoard
    tensor_board_writer = SummaryWriter(log_dir=f"runs/HyperParameterTuning_lr3-e5_wd1e-6_earlystopping")
    
    print("Started the Training")
    
    #Figthing Data Imbalanced
    pos_weight = torch.tensor([3.0], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    #Learning Rate Scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode="max",factor=0.5,patience=2,min_lr=1e-6)
    best_val_f1 = -1.0
    for epoch in range(num_epochs):
        #F1 Score training
        all_train_y_true = []
        all_train_y_pred = []
        
        #F1 Score validation
        all_val_y_true = []
        all_val_y_pred = []
        
        
        # -------------------------------TRAINING ---------------------------
        #Initializing the training variables
        trace_model.train()
        epoch_loss = 0.0
        correct_train_PD1 = 0
        total_train_PD1 = 0
        
        for inputs_train, targets_train in train_loader:
            
            label_train_PD1 = targets_train["PD1"].unsqueeze(1).to(device)
            #Changing the Inputs -> to have GPU for JupyterGPUHub
            inputs_train = {
                k: v.to(device)
                    for k, v in inputs_train.items()
                }
            #Calculation of the timestamps(Feature Engineer Trace Paper Part 2.2)
            delta_elapsed = get_elapsed_feature(inputs_train["timestamps"]).to(device)
            delta_between = get_between_features(inputs_train["timestamps"]).to(device)
            
            optimizer.zero_grad()
            #Predictions of the model
            logits_train = trace_model(
                    inputs_train["aid"],
                    inputs_train["type"],
                    delta_elapsed,
                    delta_between
                )
            
            #Calculation loss for Training using BCEWithLogitsLoss
            loss_training = criterion(logits_train,label_train_PD1.float())
            loss_training.backward()
            optimizer.step()
            
            epoch_loss += loss_training.item()
            
            # ============ PD1(Prediction, calculation of Accuracy) ============
            probs_PD1 = torch.sigmoid(logits_train)
            preds_PD1 = (probs_PD1 >= 0.5).float()
            correct_train_PD1 += (preds_PD1 == label_train_PD1).sum().item()
            total_train_PD1 += label_train_PD1.numel()
            
            #F1 Score For training 
            all_train_y_true.append(label_train_PD1.detach().cpu())
            all_train_y_pred.append(preds_PD1.detach().cpu())

        #Training Loss and Accuracy 
        train_loss = epoch_loss / len(train_loader)
        train_acc_PD1 = correct_train_PD1 / max(total_train_PD1, 1)
        
        #F1 Score for training
        all_train_y_true = torch.cat(all_train_y_true).numpy()
        all_train_y_pred = torch.cat(all_train_y_pred).numpy()
        train_f1_PD1 = f1_score(all_train_y_true, all_train_y_pred, zero_division=0)
        
        #TensorBoard Writing
        tensor_board_writer.add_scalar("Train/F1_PD1", train_f1_PD1, epoch)
        tensor_board_writer.add_scalar("Training/Loss", train_loss, epoch)
        tensor_board_writer.add_scalar("Train/Acc_PD1", train_acc_PD1, epoch)
        
        current_lr = optimizer.param_groups[0]["lr"]
        tensor_board_writer.add_scalar("LR", current_lr, epoch)


        # -------------------------------VALIDATION---------------------------
        #Initializing the validation variables    
        trace_model.eval()
        val_loss = 0.0
        correct_val_PD1 = 0
        total_val_PD1 = 0
        

        with torch.no_grad():
            for inputs_val, targets_val in validation_loader:
                    
                label_val_PD1 = targets_val["PD1"].unsqueeze(1).to(device)

                #Changing the Inputs -> to have GPU for JupyterGPUHub
                inputs_val = {
                    k: v.to(device)
                    for k, v in inputs_val.items()
                }
                #Calculation of the timestamps(Part of Feature Engineer Trace Paper part 2.2)
                delta_elapsed = get_elapsed_feature(inputs_val["timestamps"]).to(device)
                delta_between = get_between_features(inputs_val["timestamps"]).to(device)

                #Predictions of the model
                logits_val = trace_model(
                    inputs_val["aid"],
                    inputs_val["type"],
                    delta_elapsed,
                    delta_between
                    )
            
                #Calculation loss for validation using BCEWithLogitsLoss
                loss_validation = criterion(logits_val,label_val_PD1.float())
                val_loss += loss_validation.item()

                # ============ PD1(Prediction, calculation of Accuracy) ============
                probs_PD1_val = torch.sigmoid(logits_val)
                preds_PD1_val = (probs_PD1_val >= 0.5).float()
                correct_val_PD1 += (preds_PD1_val == label_val_PD1).sum().item()
                total_val_PD1 += label_val_PD1.numel()
                
                #F1 Score for validation
                all_val_y_true.append(label_val_PD1.detach().cpu())
                all_val_y_pred.append(preds_PD1_val.detach().cpu())

        
        #F1 Score for Validation
        all_val_y_true = torch.cat(all_val_y_true).numpy()
        all_val_y_pred = torch.cat(all_val_y_pred).numpy()
        val_f1_PD1 = f1_score(all_val_y_true, all_val_y_pred, zero_division=0)
        if val_f1_PD1 > best_val_f1:
            best_val_f1 = val_f1_PD1
        
        #Validation Loss and Accuracy 
        val_loss /= len(validation_loader)
        val_acc_PD1 = correct_val_PD1 / max(total_val_PD1, 1)
        
        #TensorBoard
        tensor_board_writer.add_scalar("Val/Loss", val_loss, epoch)
        tensor_board_writer.add_scalar("Val/Acc_PD1", val_acc_PD1, epoch)
        tensor_board_writer.add_scalar("Val/F1_PD1", val_f1_PD1, epoch)
        lr_scheduler.step(val_f1_PD1)
                
                
        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc_PD1:.4f} | Train F1: {train_f1_PD1:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc_PD1:.4f} | Val F1: {val_f1_PD1:.4f}"
        )
        
        
        #Early Stopping
        early_stopping(val_f1_PD1, trace_model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    tensor_board_writer.close()
   
    #Load the Best Checkpoint model
    early_stopping.load_best_weights(trace_model)
    
    #
    torch.save({
        "model_state_dict": trace_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_f1": best_val_f1,
    }, "Final_PD1_16Batch_LrSchedulee_model.pt")



        
if __name__ == "__main__":
    main()