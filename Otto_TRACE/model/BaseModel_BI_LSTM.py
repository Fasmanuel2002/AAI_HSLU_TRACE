import torch 
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Bi_LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super(Bi_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        self.ATC_head = nn.Linear(hidden_size*2, 1)
        self.SAT_head = nn.Linear(hidden_size*2, 1)
        self.MAP_head = nn.Linear(hidden_size*2, 1)
        
    def forward(self, x, lengths):
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # LSTM forward
        packed_out, _ = self.lstm(packed_x)
        
        # Unpack
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        
        batch_size = x.size(0)
        
        h = torch.zeros(batch_size, self.hidden_size * 2).to(x.device)
        
        for i, lenght in enumerate(lengths):
            h[i] = out[i, lenght - 1, :]
            
            
        h = self.layer_norm(h)
        
        return {
            "ATC":self.ATC_head(h),
            "SAT":self.SAT_head(h),
            "MAP":self.MAP_head(h)
        }