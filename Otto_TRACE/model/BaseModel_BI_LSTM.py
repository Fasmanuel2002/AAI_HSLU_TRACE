import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Dict

class Bi_LSTM(nn.Module):
    def __init__(self, num_embeddings_aid: int, 
                 num_embeddings_event_type: int,
                 embedding_dim: int = 32,
                 hidden_size: int = 128,
                 num_layers: int = 2):
        super().__init__()

        self.aid_emb = nn.Embedding(num_embeddings_aid, embedding_dim, padding_idx=0)
        self.type_emb = nn.Embedding(num_embeddings_event_type, embedding_dim, padding_idx=0)

        input_size = embedding_dim * 2 + 2  # delta_elapsed + delta_between
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)

        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        self.ATC_head = nn.Linear(hidden_size * 2, 1) #(2 hidden_size -> Bidirectional, 1 logit)
        self.SAT_head = nn.Linear(hidden_size * 2, 1) #(2 hidden_size -> Bidirectional, 1 logit)
        self.MAP_head = nn.Linear(hidden_size * 2, 1) #(2 hidden_size -> Bidirectional, 1 logit)

    def forward(self, aid_ids: torch.Tensor, type_ids: torch.Tensor,delta_elapsed: torch.Tensor, delta_between: torch.Tensor,lengths: torch.Tensor) -> Dict:

        x = torch.cat([
            self.aid_emb(aid_ids),      
            self.type_emb(type_ids),    
            delta_elapsed,              
            delta_between               
        ], dim=-1)

        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)  # Packs padded batch into a PackedSequence so the LSTM ignores padded timesteps.
        
        packed_out, _ = self.lstm(packed_x)  # Run the Bi-LSTM over the packed sequences.
        
        out, _ = pad_packed_sequence(packed_out, batch_first=True) # Unpack back to a padded tensor
            

        idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).to(out.device) #Index of the last element of the real elements
        
        h = out.gather(1, idx.expand(-1, 1, out.size(-1))).squeeze(1)  #Gather this last element and puts it on a (B, 2 hidden_size -> Bidirectional)

        h = self.layer_norm(h) #Layer of normalization

        return {
            "ATC": self.ATC_head(h), #(2 hidden_size -> Bidirectional, 1 logit)
            "SAT": self.SAT_head(h), #(2 hidden_size -> Bidirectional, 1 logit)
            "MAP": self.MAP_head(h), #(2 hidden_size -> Bidirectional, 1 logit)
        }
