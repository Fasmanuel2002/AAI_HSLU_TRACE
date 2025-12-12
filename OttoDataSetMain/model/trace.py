import torch
import torch.nn as nn
import math

class MLP(nn.Module):
    def __init__(self, input_channels : int, output_channels : int):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Linear(input_channels, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, output_channels)
            )
    def forward(self, x):
            return self.layers(x)
         
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.d_model = d_model

        
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)  

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x
    

class TRACE(nn.Module):
    def __init__(self, num_embeddings_aid : int, 
                 num_embeddings_event_type : int,
                 embedding_dim : int,
                 num_classes : int = 4
               ):
        
        super(TRACE, self).__init__()
        
        self.D_model = embedding_dim * 2 + 2 # 66
        
        
        self.embedding_aid = nn.Embedding(num_embeddings=num_embeddings_aid,
                                          embedding_dim=embedding_dim)
        
        self.embedding_eventtype = nn.Embedding(num_embeddings=num_embeddings_event_type,
                                                embedding_dim=embedding_dim) 

        self.positional_embedding = PositionalEncoding(d_model=self.D_model)
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.D_model, 
                                                        nhead=6, # 8
                                                        dim_feedforward=128,
                                                        dropout=0.2,
                                                        activation="relu",
                                                        batch_first=True)
        
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer,
                                             num_layers=1)
        
                
        self.GBAP = nn.AdaptiveMaxPool1d(output_size=1)
        
        # Jan: Why this? You have parameter num_classes, use it!
        #self.__ATC__ = MLP(input_channels=self.D_model, output_channels=1)
        
        self.MLP_layer = MLP(input_channels=self.D_model, 
                             output_channels=num_classes)
        
        
    def forward(self, aids_ids: torch.Tensor, type_ids: torch.Tensor, delta_elapsed : torch.Tensor, delta_between : torch.Tensor) -> torch.Tensor:
        
        #Categorical Learning Embeddings: 32 + 32
        aid_emb = self.embedding_aid(aids_ids)
        type_emb = self.embedding_eventtype(type_ids)
        
        B, L, _ = aid_emb.shape #(Batch, L_seq , 1)
        
        
        #Time Learning Embeddings: 1 + 1
        delta_between = delta_between.unsqueeze(-1) # (Batch, L_seq, 1)


        delta_between = torch.cat([delta_between, delta_between[:, -1:, :]],dim=1)   #(Batch, L_Seq - 1, 1)  
                    
        delta_elapsed = delta_elapsed.unsqueeze(-1).unsqueeze(-1) #(B,) --> #(B, 1, 1)
        
        
        delta_elapsed = delta_elapsed.expand(-1, L, -1) #(B, L_Seq, 1)
          

        x = torch.cat(
            [aid_emb,
            type_emb,
            delta_between,
            delta_elapsed],
            dim=-1)  
            
        
        positional_embed = self.positional_embedding(x)
        
        encoder = self.encoder(positional_embed)
        
        global_avarage_pooling = self.GBAP(encoder.transpose(1,2)).squeeze(-1)
              
        logits = self.MLP_layer(global_avarage_pooling)
        
        return logits  