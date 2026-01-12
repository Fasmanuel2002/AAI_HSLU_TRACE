import os
import numpy as np
import torch
import torch.optim as optim
from utils.SplitData import split_data_Train_Val_Test
from torch.utils.tensorboard import SummaryWriter # type: ignore
import torch.nn as nn
from dataset.otto_final import TraceOttoDataset
from utils.feature_engineering import get_between_features, get_elapsed_feature
from utils.EarlyStopping import EarlyStopping
import torch.nn.functional as F
from utils.training_utils import search_best_f1_thr, update_binary_metrics, append_probs_and_true, ratios_finder_multi_task, compute_f1_tasks, initialize_TRACE_model


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    
    print("Beginning")
    #DataSet    
    dataset_processed = TraceOttoDataset(
        file_name='train.jsonl',
        input_seq_len=64,
        min_timestamps_per_sample=32,
        max_samples=200000
    )
    train_loader, validation_loader, _ = split_data_Train_Val_Test(dataset_processed, batch_size=128)
    
    trace_model = initialize_TRACE_model(dataset_processed, num_classes=3,device=device)
    
    
    optimizer = optim.AdamW(trace_model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    #Learning Rate Scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                        cooldown=1,
                                                        mode="max",
                                                        factor=0.5,
                                                        patience=2,
                                                        min_lr=1e-6)
    early_stopping = EarlyStopping(patience=7,
                                   min_delta=1e-4,
                                   mode="max",
                                   path=f"Model_TRACE_checkpoint_ATC_task.pt")
    
    #Summary Writer for tensorBoard
    tensor_board_writer = SummaryWriter(log_dir=f"runs/MLT_task_ATC_SAT_MAP/")
    
    
    print("Started the Training")
    
    w_pos_ATC, w_pos_SAT, w_pos_MAP = ratios_finder_multi_task(train_loader, device)

    w_neg = torch.tensor([1.0], device=device).float()
    
    criterion_validation = nn.BCEWithLogitsLoss()
    
    num_epochs = 40

    best_val_f1 = -1.0
    
    best_global_thr = {"ATC": 0.5, "SAT": 0.5, "MAP": 0.5}

    for epoch in range(num_epochs):
        #F1 Score for Training ATC
        train_y_true_ATC = []
        train_y_pred_ATC = []

        #F1 Score training for SAT
        train_y_true_SAT = []
        train_y_pred_SAT = []
        
        #F1 Score for Training MAP
        train_y_true_MAP = []
        train_y_pred_MAP = []
            
        
        # -------------------------------TRAINING ---------------------------
        trace_model.train()
        epoch_loss = 0.0

        correct_train_ATC = 0
        correct_train_SAT = 0
        correct_train_MAP = 0

        total_train_ATC = 0
        total_train_SAT = 0
        total_train_MAP = 0
    

        for inputs_train, targets_train in train_loader:
            
            target_train_ATC = targets_train["ATC"].unsqueeze(1).to(device).float()
            target_train_SAT = targets_train["SAT"].unsqueeze(1).to(device).float()
            target_train_MAP = targets_train["MAP"].unsqueeze(1).to(device).float()
            
   
            inputs_train = {
                k: v.to(device)
                for k, v in inputs_train.items()
            }

            delta_elapsed = get_elapsed_feature(inputs_train["timestamps"]).to(device).float()
            delta_between = get_between_features(inputs_train["timestamps"]).to(device).float()

            logits_train = trace_model(
                inputs_train["aid"],
                inputs_train["type"],
                delta_elapsed,
                delta_between
            )

            pred_ATC = logits_train[:, 0:1] #ATC
            pred_SAT = logits_train[:, 1:2] #SAT
            pred_MAP = logits_train[:, 2:3] #MAP
            
            
            weights_ATC = torch.where(target_train_ATC == 1.0, w_pos_ATC, w_neg)
            weights_SAT = torch.where(target_train_SAT == 1.0, w_pos_SAT, w_neg)
            weights_MAP = torch.where(target_train_MAP == 1.0, w_pos_MAP, w_neg)

            loss_ATC = F.binary_cross_entropy_with_logits(pred_ATC, target_train_ATC, weight=weights_ATC)
            loss_SAT = F.binary_cross_entropy_with_logits(pred_SAT, target_train_SAT, weight=weights_SAT)
            loss_MAP = F.binary_cross_entropy_with_logits(pred_MAP, target_train_MAP, weight=weights_MAP)
            optimizer.zero_grad()
            
            loss_training = (loss_ATC + loss_SAT + loss_MAP) 
            
            loss_training.backward()
            
            optimizer.step()

            epoch_loss += loss_training.item()

            # ============ ATC ============
            correct_train_ATC, total_train_ATC = update_binary_metrics(pred_ATC,target_train_ATC,correct_train_ATC,total_train_ATC,train_y_true_ATC,train_y_pred_ATC)
            # ============ SAT ============
            correct_train_SAT, total_train_SAT = update_binary_metrics(pred_SAT, target_train_SAT, correct_train_SAT, total_train_SAT, train_y_true_SAT, train_y_pred_SAT)
            # ============ MAP ===========
            correct_train_MAP, total_train_MAP = update_binary_metrics(pred_MAP, target_train_MAP, correct_train_MAP, total_train_MAP, train_y_true_MAP,train_y_pred_MAP)
            
            
        train_loss = epoch_loss / len(train_loader)

        train_acc_ATC = correct_train_ATC / max(total_train_ATC, 1)
        train_acc_SAT = correct_train_SAT / max(total_train_SAT, 1)
        train_acc_MAP = correct_train_MAP / max(total_train_MAP, 1)
        
        
        #F1 Score for training ATC
        train_f1_ATC = compute_f1_tasks(train_y_true_ATC, train_y_pred_ATC)
        
        #F1 Score for training SAT
        train_f1_SAT = compute_f1_tasks(train_y_true_SAT, train_y_pred_SAT)
        
        #F1 Score for training MAP
        train_f1_MAP = compute_f1_tasks(train_y_true_MAP, train_y_pred_MAP)
        
        #TensorBoard Writing
        tensor_board_writer.add_scalar("Training/Loss", train_loss, epoch)
        tensor_board_writer.add_scalar("Train/Acc_ATC", train_acc_ATC, epoch)
        tensor_board_writer.add_scalar("Train/Acc_SAT", train_acc_SAT, epoch)
        tensor_board_writer.add_scalar("Train/Acc_MAP", train_acc_MAP, epoch)
        tensor_board_writer.add_scalar("Train/F1_ATC", train_f1_ATC, epoch)
        tensor_board_writer.add_scalar("Train/F1_SAT", train_f1_SAT, epoch)
        tensor_board_writer.add_scalar("Train/F1_MAP", train_f1_MAP, epoch)
        

        # -------------------------------VALIDATION---------------------------
        trace_model.eval()
        val_loss = 0.0
        val_probs_ATC, val_true_ATC = [], []
        val_probs_SAT, val_true_SAT = [], []
        val_probs_MAP, val_true_MAP = [], []

        with torch.no_grad():
            for inputs_val, targets_val in validation_loader:

                target_val_ATC = targets_val["ATC"].unsqueeze(1).to(device).float()
                target_val_SAT = targets_val["SAT"].unsqueeze(1).to(device).float()
                target_val_MAP = targets_val["MAP"].unsqueeze(1).to(device).float()
        
                inputs_val = {
                    k: v.to(device)
                    for k, v in inputs_val.items()
                }

                delta_elapsed = get_elapsed_feature(inputs_val["timestamps"]).to(device).float()
                delta_between = get_between_features(inputs_val["timestamps"]).to(device).float()
                
                logits_val = trace_model(
                    inputs_val["aid"],
                    inputs_val["type"],
                    delta_elapsed,
                    delta_between
                )

                logits_ATC_val = logits_val[:, 0:1]
                logits_SAT_val = logits_val[:, 1:2]
                logits_MAP_val = logits_val[:, 2:3]
            
                
                loss_ATC_val = criterion_validation(logits_ATC_val, target_val_ATC)
                loss_SAT_val = criterion_validation(logits_SAT_val, target_val_SAT)
                loss_MAP_val = criterion_validation(logits_MAP_val, target_val_MAP)
            
                
                loss_validation = (loss_ATC_val + loss_SAT_val + loss_MAP_val)
                val_loss += loss_validation.item()

                append_probs_and_true(logits_ATC_val, target_val_ATC, val_probs_ATC, val_true_ATC)
                append_probs_and_true(logits_SAT_val, target_val_SAT, val_probs_SAT, val_true_SAT)
                append_probs_and_true(logits_MAP_val, target_val_MAP, val_probs_MAP, val_true_MAP)
         
         
         
        val_probs_ATC = torch.cat(val_probs_ATC).numpy().ravel()
        val_true_ATC  = torch.cat(val_true_ATC).numpy().ravel()

        val_probs_SAT = torch.cat(val_probs_SAT).numpy().ravel()
        val_true_SAT  = torch.cat(val_true_SAT).numpy().ravel()
        
        val_probs_MAP = torch.cat(val_probs_MAP).numpy().ravel()
        val_true_MAP  = torch.cat(val_true_MAP).numpy().ravel()

        
        thresholds = np.linspace(0.01,0.99, 99)
        
        val_f1_ATC, val_macro_f1_ATC, threshold_ATC = search_best_f1_thr(val_probs_ATC, val_true_ATC, thresholds)
        
        val_f1_SAT, val_macro_f1_SAT, threshold_SAT = search_best_f1_thr(val_probs_SAT, val_true_SAT, thresholds)
        
        val_f1_MAP, val_macro_f1_MAP, threshold_MAP = search_best_f1_thr(val_probs_MAP, val_true_MAP, thresholds)
        
        val_f1_mean = (val_f1_ATC + val_f1_SAT + val_f1_MAP) / 3
        val_macro_f1_mean = (val_macro_f1_ATC + val_macro_f1_SAT + val_macro_f1_MAP) / 3

        if val_f1_mean > best_val_f1:
            best_val_f1 = val_f1_mean
            best_global_thr = {"ATC": threshold_ATC, "SAT": threshold_SAT, "MAP": threshold_MAP}

        
        
        
        val_loss /= len(validation_loader)
        val_acc_best_thr_ATC = ((val_probs_ATC >= threshold_ATC).astype(int) == val_true_ATC.astype(int)).mean()
        val_acc_best_thr_SAT = ((val_probs_SAT >= threshold_SAT).astype(int) == val_true_SAT.astype(int)).mean()
        val_acc_best_thr_MAP = ((val_probs_MAP >= threshold_MAP).astype(int) == val_true_MAP.astype(int)).mean()

        
        tensor_board_writer.add_scalar("Val/Loss", val_loss, epoch)
        tensor_board_writer.add_scalar("Val/ATC_F1", val_f1_ATC, epoch)
        tensor_board_writer.add_scalar("Val/SAT_F1", val_f1_SAT, epoch)
        tensor_board_writer.add_scalar("Val/MAP_F1", val_f1_MAP, epoch)
       
        tensor_board_writer.add_scalar("Val/Acc_ATC_best_thr", val_acc_best_thr_ATC, epoch)
        tensor_board_writer.add_scalar("Val/Acc_SAT_best_thr", val_acc_best_thr_SAT, epoch)
        tensor_board_writer.add_scalar("Val/Acc_MAP_best_thr", val_acc_best_thr_MAP, epoch)
        
        tensor_board_writer.add_scalar("Val/f1_mean", val_f1_mean, epoch)

        lr_scheduler.step(val_f1_mean)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Acc ATC : {train_acc_ATC:.4f} | Train F1 ATC: {train_f1_ATC:.4f} | Val F1 ATC: {val_f1_ATC:.4f} | Val Macro F1 ATC {val_macro_f1_ATC} ")
        print(f"Train Acc SAT : {train_acc_SAT:.4f} | Train F1 SAT: {train_f1_SAT:.4f} | Val F1 SAT: {val_f1_SAT:.4f} | Val Macro F1 SAT {val_macro_f1_SAT} ")
        print(f"Train Acc MAP : {train_acc_MAP:.4f} | Train F1 MAP: {train_f1_MAP:.4f} | Val F1 MAP: {val_f1_MAP:.4f} | Val Macro F1 MAP {val_macro_f1_MAP} ")
        print(f"Val F1 mean: {val_f1_mean:.4f}  | Val Macro F1: {val_macro_f1_mean:.4f}| Thr: ATC: {threshold_ATC:.2f}/SAT:{threshold_SAT:.2f} MAP:{threshold_MAP:.2f}/")
        
        #Print the Current Learning rate after the Lr
        current_lr = optimizer.param_groups[0]["lr"]
        tensor_board_writer.add_scalar("LR", current_lr, epoch)
        print(f"This is the LR: {current_lr}")
        

        early_stopping(float(val_f1_mean), trace_model)
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
    }, "Model_TRACE_MLT_ATC_SAT_MLP_FinalVersion_SingleTask.pt")
       
if __name__ == "__main__":
    main()