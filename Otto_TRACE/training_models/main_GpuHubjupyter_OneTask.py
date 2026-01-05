import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import numpy as np
import torch
import torch.optim as optim
from utils.SplitData import split_data_Train_Val_Test
from torch.utils.tensorboard import SummaryWriter # type: ignore

from model.trace import TRACE
from dataset.otto_final import TraceOttoDataset
from utils.feature_engineering import get_between_features, get_elapsed_feature
from utils.EarlyStopping import EarlyStopping
from sklearn.metrics import f1_score,precision_score,recall_score
import torch.nn.functional as F
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print("Beginning")
    #DataSet    
    dataset_processed = TraceOttoDataset(
        file_name='../train.jsonl',
        input_seq_len=64,
        min_timestamps_per_sample=16,
        max_samples=1000
    )
    
    #Split the Data into Training_loader, Validation_loader and test_loaders
    train_loader, validation_loader, test_loader = split_data_Train_Val_Test(dataset_processed, batch_size=32)
    
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
    optimizer = optim.AdamW(trace_model.parameters(), lr=1e-4, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                        cooldown=1,
                                                        mode="max",
                                                        factor=0.5,
                                                        patience=2,
                                                        min_lr=1e-6)
    early_stopping = EarlyStopping(patience=4,
                                   min_delta=1e-4,
                                   mode="max",
                                   path=f"ModelTrace_MoreNeurons_lossWeighted_5_1_2026_CheckPoint.pt")
    
    #Summary Writer for tensorBoard
    tensor_board_writer = SummaryWriter(log_dir=f"runs/Testing_HyperparameterTuning_5_01_2026_version1_MoreNeurons_lossWeighted1.5")
    
    
    
    print("Started the Training")
    #Figthing Data Imbalanced
    labels_list = []
    for inputs, targets in train_loader:
        labels_list.append(targets["PD1"].view(-1)) #(Batch, )
        
    labels = torch.cat(labels_list, dim=0) #(N, )           
    
    #Number of positives in the train_loader
    num_pos = (labels == 1).sum().item()
    
    #Number of Negatives in the train_loader
    num_neg = (labels == 0).sum().item()
    
    ratio = num_neg / max(num_pos, 1)
    
    #smoothed_weight = torch.tensor([ratio], device=device)
    
    print("Train pos/neg:", num_pos, num_neg)
    
    ##adding for smoothing the weights only for Training 
    #criterion_train = torch.nn.BCEWithLogitsLoss(pos_weight=smoothed_weight) 
    
    criterion_validation = torch.nn.BCEWithLogitsLoss()
    
    
    w_pos = torch.tensor([1.5], device=device).float()
    w_neg = torch.tensor([1.0], device=device).float()

    print("w_pos and w_neg", w_pos, w_neg)
    #Learning Rate Scheduler
    #To Save the Best F1 for the Model
    best_val_f1 = -1.0
    best_global_thr = 0.5
    
    
    num_epochs = 40
    for epoch in range(num_epochs):
        
        #F1 Score training
        all_train_y_true = []
        all_train_y_pred = []
        all_val_y_true = []
        all_val_probs = []
            
        # -------------------------------TRAINING ---------------------------
        #Initializing the training variables
        trace_model.train()
        epoch_loss = 0.0
        correct_train_PD1 = 0
        total_train_PD1 = 0
            
        for inputs_train, targets_train in train_loader:
                
            target_train_PD1 = targets_train["PD1"].unsqueeze(1).to(device)
            #Changing the Inputs -> to have GPU for JupyterGPUHub
            inputs_train = {
                k: v.to(device)
                    for k, v in inputs_train.items()
                }
            #Calculation of the timestamps(Feature Engineer Trace Paper Part 2.2)
            delta_elapsed = get_elapsed_feature(inputs_train["timestamps"]).to(device)
            delta_between = get_between_features(inputs_train["timestamps"]).to(device)
                
            optimizer.zero_grad(set_to_none=True)
            #Predictions of the model
            logits_train = trace_model(
                    inputs_train["aid"],
                    inputs_train["type"],
                    delta_elapsed,
                    delta_between
                )
            
            weights = torch.where(target_train_PD1 == 1, w_pos, w_neg)
            print(logits_train.shape, target_train_PD1.shape, weights.shape)

            #Calculation loss for Training using BCEWithLogitsLoss
            loss_training = F.binary_cross_entropy_with_logits(logits_train,target_train_PD1.float(), weight=weights)
            loss_training.backward()
            optimizer.step()
                
            epoch_loss += loss_training.item()
                
            # ============ PD1(Prediction, calculation of Accuracy) ============
            probs_PD1 = torch.sigmoid(logits_train)
            preds_PD1 = (probs_PD1 >= 0.5).float()
            correct_train_PD1 += (preds_PD1 == target_train_PD1).sum().item()
            total_train_PD1 += target_train_PD1.numel()
                
            #F1 Score For training 
            all_train_y_true.append(target_train_PD1.detach().cpu())
            all_train_y_pred.append(preds_PD1.detach().cpu())
    
        #Training Loss and Accuracy 
        train_loss = epoch_loss / len(train_loader)
        train_acc_PD1 = correct_train_PD1 / max(total_train_PD1, 1)
            
        #F1 Score for training
        all_train_y_true = torch.cat(all_train_y_true).numpy().ravel()
        all_train_y_pred = torch.cat(all_train_y_pred).numpy().ravel()
        train_f1_PD1 = f1_score(all_train_y_true, all_train_y_pred, zero_division=0)
            
        #TensorBoard Writing
        tensor_board_writer.add_scalar("Train/F1_PD1", train_f1_PD1, epoch)
        tensor_board_writer.add_scalar("Training/Loss", train_loss, epoch)
        tensor_board_writer.add_scalar("Train/Acc_PD1", train_acc_PD1, epoch)
            
    
    
        # -------------------------------VALIDATION---------------------------
        #Initializing the validation variables    
        trace_model.eval()
        val_loss = 0.0
        correct_val_PD1 = 0
        total_val_PD1 = 0
            
        all_val_probs = []
        all_val_y_true = []
    
        with torch.no_grad():
            for inputs_val, targets_val in validation_loader:
                target_val_PD1 = targets_val["PD1"].unsqueeze(1).to(device)
    
                inputs_val = {k: v.to(device) for k, v in inputs_val.items()}
    
                delta_elapsed = get_elapsed_feature(inputs_val["timestamps"]).to(device)
                delta_between = get_between_features(inputs_val["timestamps"]).to(device)
    
                logits_val = trace_model(
                    inputs_val["aid"],
                    inputs_val["type"],
                    delta_elapsed,
                    delta_between
                )
                
                loss_validation = criterion_validation(logits_val, target_val_PD1.float())
                val_loss += loss_validation.item()

                #Logits converted to sigmoid
                probs_PD1_val = torch.sigmoid(logits_val)
                preds_PD1_val = (probs_PD1_val >= 0.5).float()
                correct_val_PD1 += (preds_PD1_val == target_val_PD1).sum().item()
                total_val_PD1 += target_val_PD1.numel()
    
                all_val_y_true.append(target_val_PD1.detach().cpu())
                all_val_probs.append(probs_PD1_val.detach().cpu())
    
        # ----Concatonate the Probabilities and true labels ----
        all_val_y_true = torch.cat(all_val_y_true).numpy().ravel()
        all_val_probs = torch.cat(all_val_probs).numpy().ravel()
    
        #Generate 81 possible threshold values from 0.1 to 0.99 (steps of 0.01).
        thresholds = np.linspace(0.01, 0.99, 99)
        #Normal Threshold
        best_thr = 0.5
        best_f1 = 0.0
        for t in thresholds:
            # Convert continuous probabilities [0, 1] into binary predictions [0 or 1]
            # based on the current threshold candidate 't'
            preds_thr = (all_val_probs >= t).astype(int)
            f1 = f1_score(all_val_y_true, preds_thr, zero_division=0)
            
            # If this threshold results in a better F1 score, update our best values
            if f1 > best_f1:
                best_f1 = f1
                best_thr = t
        
        #Looking for the Best F1 Score and threshold
        val_f1_PD1 = best_f1
        threshold = best_thr
        
        # Generate final predictions using the newly discovered optimal threshold
        val_pred = (all_val_probs >= threshold).astype(int)
        
        # Calculate additional metrics (Precision and Recall) at this specific threshold
        val_precision = precision_score(all_val_y_true, val_pred, zero_division=0)
        val_recall = recall_score(all_val_y_true, val_pred, zero_division=0)
        
        # If the F1 score of this epoch is the best seen so far across all epochs,
        # we update the global "Best Model" variables to ensure we save the right threshold.
        if val_f1_PD1 > best_val_f1:
            best_val_f1 = val_f1_PD1
            best_global_thr = threshold
    
        
        #Validation Loss
        val_loss /= len(validation_loader)
        #calculates the optimized Accuracy based on the best threshold found
        val_acc_best_thr = ((all_val_probs >= threshold).astype(int) == all_val_y_true.astype(int)).mean()
        #TensorBoard
        tensor_board_writer.add_scalar("Val/Loss", val_loss, epoch)
        tensor_board_writer.add_scalar("Val/Acc_PD1_best_thr", val_acc_best_thr, epoch)
        tensor_board_writer.add_scalar("Val/F1_PD1", val_f1_PD1, epoch)
        tensor_board_writer.add_scalar("Val/Best_Threshold", threshold, epoch)
        tensor_board_writer.add_scalar("Val/Best_Global_Threshold", best_global_thr, epoch)
        tensor_board_writer.add_scalar("Val/Precision_PD1", val_precision, epoch)
        tensor_board_writer.add_scalar("Val/Recall_PD1", val_recall, epoch)

        lr_scheduler.step(val_f1_PD1)
                            
        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc_PD1:.4f} | Train F1: {train_f1_PD1:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val F1: {val_f1_PD1:.4f} | "
            f"BestThr: {threshold:.3f} | Val Acc best threshold: {val_acc_best_thr:.4f} "
            f"Val Precision: {val_precision} | Val Recall {val_recall} "
        )

        #Print the Current Learning rate after the Lr
        current_lr = optimizer.param_groups[0]["lr"]
        tensor_board_writer.add_scalar("LR", current_lr, epoch)
        print(f"This is the LR: {current_lr}")
            
        #Early Stopping
        early_stopping(val_f1_PD1, trace_model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
        
    tensor_board_writer.close()
    #Saves the Model CheckPoint if the JupyterGpuHub the session expires
    early_stopping.load_best_weights(trace_model)
    
    #Save the total Model after the training
    torch.save({
        "model_state_dict": trace_model.state_dict(),
        "best_val_f1": best_val_f1,
        "best_global_threshold": best_global_thr,
    }, "ModelTrace_MoreNeurons_lossWeighted4_version_3_1_2026.pt")



        
if __name__ == "__main__":
    main()