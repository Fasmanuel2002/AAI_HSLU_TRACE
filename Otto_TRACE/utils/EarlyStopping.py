import numpy as np
import torch

class EarlyStopping:
    def __init__(self,
                    patience: int = 5, #How many consecutive periods can they go without improvement before stopping training 
                    min_delta: float = 0.0, #The minimum improvement we require to say "this is an improvement".
                    mode : str = "min", #val_loss, 
                    restore_best_weights : bool = True, #Save the best model in the run
                    path: str = "best_TRACE_model.pt",
                    verbose: bool = True #print
                    ):
        
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.path = path
        self.verbose = verbose
        
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
        if mode == "min":
            self.compare = lambda current_epoch, best_epoch : current_epoch < best_epoch - min_delta
        else: #mode == "max"
            self.compare = lambda current_epoch, best_epoch : current_epoch > best_epoch + min_delta

    def __call__(self, metric_model : float, trace_model):
            if self.best_score is None:
                self.best_score = metric_model
                self.save_checkpoint(metric_model, trace_model)
                return
            if self.compare(metric_model, self.best_score):
                self.best_score = metric_model
                self.counter = 0
                self.save_checkpoint(metric_model, trace_model)
            
            else:
                self.counter += 1
                if self.verbose == True:
                    print(f"EarlyStopping counter {self.counter}/{self.patience}")
                
                if self.counter >= self.patience:
                    self.early_stop = True
                    
    def save_checkpoint(self, metric_model, trace_model):
        if self.verbose == True:
            print(f"New best metric for trace model {metric_model:.4f} -> New model") 
        torch.save(trace_model.state_dict(), self.path)
    
    
    def load_best_weights(self, model):
        if self.restore_best_weights:
            model.load_state_dict(torch.load(self.path))