import os
import numpy as np
import torch
import torch.optim as optim
from utils.SplitData import split_data_Train_Val_Test
from torch.utils.tensorboard import SummaryWriter # type: ignore
import torch.nn as nn
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
        file_name='train.jsonl',
        input_seq_len=64,
        min_timestamps_per_sample=16
    )
    train_loader, validation_loader, test_loader = split_data_Train_Val_Test(dataset_processed, batch_size=128)
    
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
        num_classes=2
    )  
      

    trace_model = trace_model.to(device)
    optimizer = optim.AdamW(trace_model.parameters(), lr=1e-4, weight_decay=1e-4)
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
    tensor_board_writer = SummaryWriter(log_dir=f"runs/MLT_task_ATC_SAT/")
    
    

    #Summary Writer for tensorBoard
    print("Started the Training")
    labels_list_ATC = []
    labels_list_SAT = []
    for inputs, targets in train_loader:
        labels_list_ATC.append(targets["ATC"].view(-1)) #(Batch, )
        labels_list_SAT.append(targets["SAT"].view(-1)) #(Batch, )
    labels_ATC = torch.cat(labels_list_ATC, dim=0)
    labels_SAT = torch.cat(labels_list_SAT, dim=0)
    
    num_pos_ATC = (labels_ATC == 1).sum().item()
    num_neg_ATC = (labels_ATC == 0).sum().item()
    
    num_pos_SAT = (labels_SAT == 1).sum().item()
    num_neg_SAT = (labels_SAT == 0).sum().item()
    
    
    ratio_ATC = num_neg_ATC / max(num_pos_ATC, 1)
    ratio_ATC = min(ratio_ATC, 30.0)
    
    
    ratio_SAT = num_neg_SAT / max(num_pos_SAT, 1)
    ratio_SAT = min(ratio_SAT, 30.0)
    print("ATC Train pos/neg:", num_pos_ATC, num_neg_ATC)
    print("SAT Train pos/neg:", num_pos_SAT, num_neg_SAT)
    
    w_pos_ATC = torch.tensor([ratio_ATC], device=device).float()
    w_pos_SAT = torch.tensor([ratio_SAT], device=device).float()
    w_neg = torch.tensor([1.0], device=device).float()
    criterion_validation = nn.BCEWithLogitsLoss()
    
    num_epochs = 40

    best_val_f1 = -1.0
    best_global_thr = {"ATC": 0.5, "SAT": 0.5}

    for epoch in range(num_epochs):
        #F1 Score for Training ATC
        all_train_y_true_ATC = []
        all_train_y_pred_ATC = []

        #F1 Score training for SAT
        all_train_y_true_SAT = []
        all_train_y_pred_SAT = []
            
            
        
        # -------------------------------TRAINING ---------------------------
        trace_model.train()
        epoch_loss = 0.0

        correct_train_ATC = 0
        correct_train_SAT = 0
        #correct_train_PD1 = 0
        #correct_train_RA1 = 0

        total_train_ATC = 0
        total_train_SAT = 0
        #total_train_PD1 = 0
        #total_train_RA1 = 0

        for inputs_train, targets_train in train_loader:

           
            target_train_ATC = targets_train["ATC"].unsqueeze(1).to(device).float()
            target_train_SAT = targets_train["SAT"].unsqueeze(1).to(device).float()
            """target_train_PD1 = targets_train["PD1"].unsqueeze(1).to(device)
            target_train_RA1 = targets_train["RA1"].unsqueeze(1).to(device)
            """
   
            inputs_train = {
                k: v.to(device)
                for k, v in inputs_train.items()
            }

            delta_elapsed = get_elapsed_feature(inputs_train["timestamps"]).to(device).float()
            delta_between = get_between_features(inputs_train["timestamps"]).to(device).float()

            logits_val = trace_model(
                inputs_train["aid"],
                inputs_train["type"],
                delta_elapsed,
                delta_between
            )

            pred_ATC = logits_val[:, 0:1] #ATC
            pred_SAT = logits_val[:, 1:2] #SAT
            #pred_PD1 = logits_val[:, 2:3] #PD1
            #pred_RA1 = logits_val[:, 3:4] #RA1
            
            weights_ATC = torch.where(target_train_ATC == 1.0, w_pos_ATC, w_neg)
            weights_SAT = torch.where(target_train_SAT == 1.0, w_pos_SAT, w_neg)

            loss_ATC = F.binary_cross_entropy_with_logits(pred_ATC, target_train_ATC, weight=weights_ATC)
            loss_SAT = F.binary_cross_entropy_with_logits(pred_SAT, target_train_SAT, weight=weights_SAT)
            optimizer.zero_grad()
            
            loss_training = (loss_ATC + loss_SAT) #+ loss_PD1 + loss_RA1
            
            loss_training.backward()
            
            optimizer.step()

            epoch_loss += loss_training.item()

            # ============ ATC ============
            probs_ATC = torch.sigmoid(pred_ATC)
            preds_ATC = (probs_ATC >= 0.5).float()
            correct_train_ATC += (preds_ATC == target_train_ATC).sum().item()
            total_train_ATC += target_train_ATC.numel()
            
            all_train_y_true_ATC.append(target_train_ATC.detach().cpu())
            all_train_y_pred_ATC.append(preds_ATC.detach().cpu())
            

            # ============ SAT ============
            probs_SAT = torch.sigmoid(pred_SAT)
            preds_SAT = (probs_SAT >= 0.5).float()
            correct_train_SAT += (preds_SAT == target_train_SAT).sum().item()
            total_train_SAT += target_train_SAT.numel()
            
            all_train_y_true_SAT.append(target_train_SAT.detach().cpu())
            all_train_y_pred_SAT.append(preds_SAT.detach().cpu())
            

            """# ============ PD1 ============
            probs_PD1 = torch.sigmoid(pred_PD1)
            preds_PD1 = (probs_PD1 >= 0.5).float()
            correct_train_PD1 += (preds_PD1 == label_train_PD1).sum().item()
            total_train_PD1 += label_train_PD1.numel()

            # ============ RA1 ============
            probs_RA1 = torch.sigmoid(pred_RA1)
            preds_RA1 = (probs_RA1 >= 0.5).float()
            correct_train_RA1 += (preds_RA1 == label_train_RA1).sum().item()
            total_train_RA1 += label_train_RA1.numel()
            """
            
            
        train_loss = epoch_loss / len(train_loader)

        train_acc_ATC = correct_train_ATC / max(total_train_ATC, 1)
        train_acc_SAT = correct_train_SAT / max(total_train_SAT, 1)
        """train_acc_PD1 = correct_train_PD1 / max(total_train_PD1, 1)
        train_acc_RA1 = correct_train_RA1 / max(total_train_RA1, 1)
        """
        
        #F1 Score for training ATC
        all_train_y_true_ATC = torch.cat(all_train_y_true_ATC).numpy().ravel()
        all_train_y_pred_ATC = torch.cat(all_train_y_pred_ATC).numpy().ravel()
        train_f1_ATC = f1_score(all_train_y_true_ATC, all_train_y_pred_ATC, zero_division=0)
        
        
        #F1 Score for training ATC
        all_train_y_true_SAT = torch.cat(all_train_y_true_SAT).numpy().ravel()
        all_train_y_pred_SAT = torch.cat(all_train_y_pred_SAT).numpy().ravel()
        train_f1_SAT = f1_score(all_train_y_true_SAT, all_train_y_pred_SAT, zero_division=0)
        
        #TensorBoard Writing
        tensor_board_writer.add_scalar("Training/Loss", train_loss, epoch)
        tensor_board_writer.add_scalar("Train/Acc_ATC", train_acc_ATC, epoch)
        tensor_board_writer.add_scalar("Train/Acc_SAT", train_acc_SAT, epoch)
        tensor_board_writer.add_scalar("Train/F1_ATC", train_f1_ATC, epoch)
        tensor_board_writer.add_scalar("Train/F1_SAT", train_f1_SAT, epoch)
        #tensor_board_writer.add_scalar("Train/Acc_PD1", train_acc_PD1, epoch)
        #tensor_board_writer.add_scalar("Train/Acc_RA1", train_acc_RA1, epoch)

        # -------------------------------VALIDATION---------------------------
        trace_model.eval()
        val_loss = 0.0

        correct_val_ATC = 0
        correct_val_SAT = 0
        #correct_val_PD1 = 0
        #correct_val_RA1 = 0

        total_val_ATC = 0
        total_val_SAT = 0
        #total_val_PD1 = 0
        #total_val_RA1 = 0

        all_val_probs_ATC = []
        all_val_y_true_ATC = []
        all_val_probs_SAT = []
        all_val_y_true_SAT = []

        with torch.no_grad():
            for inputs_val, targets_val in validation_loader:

                target_val_ATC = targets_val["ATC"].unsqueeze(1).to(device).float()
                target_val_SAT = targets_val["SAT"].unsqueeze(1).to(device).float()
                #target_val_PD1 = targets_val["PD1"].unsqueeze(1).to(device)
                #target_val_RA1 = targets_val["RA1"].unsqueeze(1).to(device)

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

                pred_ATC_val = logits_val[:, 0:1]
                pred_SAT_val = logits_val[:, 1:2]
                #pred_PD1_val = logits_val[:, 2:3]
                #pred_RA1_val = logits_val[:, 3:4]
                
                loss_ATC_val = criterion_validation(pred_ATC_val, target_val_ATC)
                loss_SAT_val = criterion_validation(pred_SAT_val, target_val_SAT)
                #loss_PD1_val = criterion_validation(pred_PD1_val, target_val_PD1.float())
                #loss_RA1_val = criterion_validation(pred_RA1_val, target_val_RA1.float())

                loss_validation = loss_ATC_val + loss_SAT_val #+ loss_PD1_val + loss_RA1_val
                val_loss += loss_validation.item()

                
                probs_ATC_val = torch.sigmoid(pred_ATC_val)
                probs_SAT_val = torch.sigmoid(pred_SAT_val)
                
                probs_SAT_0_5 = (probs_SAT_val >= 0.5).float()
                probs_ATC_0_5 = (probs_ATC_val >= 0.5).float()
                
                correct_val_ATC += (probs_ATC_0_5 == target_val_ATC).sum().item()
                total_val_ATC += target_val_ATC.numel()
                
                correct_val_SAT += (probs_SAT_0_5 == target_val_SAT).sum().item()
                total_val_SAT += target_val_SAT.numel()
                
                      
                all_val_y_true_ATC.append(target_val_ATC.detach().cpu())
                all_val_probs_ATC.append(probs_ATC_val.detach().cpu())
            
                all_val_y_true_SAT.append(target_val_SAT.detach().cpu())
                all_val_probs_SAT.append(probs_SAT_val.detach().cpu())

                """
                probs_PD1_val = torch.sigmoid(pred_PD1_val)
                probs_PD1_val = (probs_PD1_val >= 0.5).float()
                correct_val_PD1 += (probs_PD1_val == target_val_PD1).sum().item()
                total_val_PD1 += target_val_PD1.numel()

                probs_RA1_val = torch.sigmoid(pred_RA1_val)
                probs_RA1_val = (probs_RA1_val >= 0.5).float()
                correct_val_RA1 += (probs_RA1_val == target_val_RA1).sum().item()
                total_val_RA1 += target_val_RA1.numel()

                """
        
        val_acc_ATC_05 = correct_val_ATC / max(total_val_ATC, 1)
        val_acc_SAT_05 = correct_val_SAT / max(total_val_SAT, 1)

        val_probs_ATC = torch.cat(all_val_probs_ATC).numpy().ravel()
        val_true_ATC  = torch.cat(all_val_y_true_ATC).numpy().ravel()

        val_probs_SAT = torch.cat(all_val_probs_SAT).numpy().ravel()
        val_true_SAT  = torch.cat(all_val_y_true_SAT).numpy().ravel()
        
        thresholds = np.linspace(0.01,0.99, 99)
        
        
        best_thr_ATC, best_f1_ATC = 0.5, 0.0
        for t in thresholds:
            pred = (val_probs_ATC >= t).astype(int)
            f1 = f1_score(val_true_ATC, pred, zero_division=0)
            if f1 > best_f1_ATC:
                best_f1_ATC, best_thr_ATC = f1, t

        best_thr_SAT, best_f1_SAT = 0.5, 0.0
        for t in thresholds:
            pred = (val_probs_SAT >= t).astype(int)
            f1 = f1_score(val_true_SAT, pred, zero_division=0)
            if f1 > best_f1_SAT:
                best_f1_SAT, best_thr_SAT = f1, t
            
        val_f1_ATC = best_f1_ATC
        threshold_ATC = best_thr_ATC
        
        
        val_f1_SAT = best_f1_SAT
        threshold_SAT = best_thr_SAT
                        
        val_f1_mean = 0.5 * (val_f1_ATC + val_f1_SAT)

        if val_f1_mean > best_val_f1:
            best_val_f1 = val_f1_mean
            best_global_thr = {"ATC": threshold_ATC, "SAT": threshold_SAT}

        
        
        
        val_loss /= len(validation_loader)
        val_acc_best_thr_ATC = ((val_probs_ATC >= threshold_ATC).astype(int) == val_true_ATC.astype(int)).mean()
        val_acc_best_thr_SAT = ((val_probs_SAT >= threshold_SAT).astype(int) == val_true_SAT.astype(int)).mean()

        
        tensor_board_writer.add_scalar("Val/Loss", val_loss, epoch)
        tensor_board_writer.add_scalar("Val/ATC_F1", val_f1_ATC, epoch)
        tensor_board_writer.add_scalar("Val/SAT_F1", val_f1_SAT, epoch)
        tensor_board_writer.add_scalar("Val/Acc_ATC_best_thr", val_acc_best_thr_ATC, epoch)
        tensor_board_writer.add_scalar("Val/Acc_sat_best_thr", val_acc_best_thr_SAT, epoch)
        tensor_board_writer.add_scalar("Val/Acc_ATC_0.5", val_acc_ATC_05, epoch)
        tensor_board_writer.add_scalar("Val/Acc_SAT_0.5", val_acc_SAT_05, epoch)
        tensor_board_writer.add_scalar("Val/f1_mean", val_f1_mean, epoch)

        #tensor_board_writer.add_scalar("Val/Acc_SAT", val_acc_SAT, epoch)
        #tensor_board_writer.add_scalar("Val/Acc_PD1", val_acc_PD1, epoch)
        #tensor_board_writer.add_scalar("Val/Acc_RA1", val_acc_RA1, epoch)
        
        lr_scheduler.step(val_f1_mean)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Acc ATC : {train_acc_ATC:.4f} | Train F1 ATC: {train_f1_ATC:.4f} | Val F1 ATC: {val_f1_ATC:.4f} | ")
        print(f"Train Acc SAT : {train_acc_SAT:.4f} | Train F1 SAT: {train_f1_SAT:.4f} | Val F1 SAT: {val_f1_SAT:.4f} | ")
        print(f"Val F1 mean: {val_f1_mean:.4f} | Thr: {threshold_ATC:.2f}/{threshold_SAT:.2f}")
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
    }, "Model_TRACE_MLT_ATC_SAT_FinalVersion_SingleTask.pt")



        
if __name__ == "__main__":
    main()