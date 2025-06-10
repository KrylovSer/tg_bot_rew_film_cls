import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

HIDDEN_SIZE = 32
BATCH_SIZE = 1024
SEQ_LEN = 135
VOCAB_SIZE = 3957
EMBEDDING_DIM = 64

class BahdanauAttention(nn.Module):
    def __init__(
            self, 
            hidden_size: int = HIDDEN_SIZE
            ) -> None:
        
        super().__init__()
        self.hidden_size = hidden_size
        self.linear_w_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_w_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_w_v = nn.Linear(self.hidden_size, 1)
        self.tanh = nn.Tanh()

    def forward(
            self, 
            lstm_outputs: torch.Tensor, # BATCH_SIZE x SEQ_LEN x HIDDEN_SIZE
            final_hidden: torch.Tensor  # BATCH_SIZE x HIDDEN_SIZE
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        keys = self.linear_w_k(lstm_outputs)                              # (B, S, H)
        query = self.linear_w_q((final_hidden).unsqueeze(1))              # (B, 1, H)

        combined = self.tanh(keys + query)                                # (B, S, H)

        w_v = self.linear_w_v(combined).squeeze(-1)                       # (B, S)
        att_weights = F.softmax(w_v, dim=1)                               # (B, S)
        cntxt = torch.bmm(att_weights.unsqueeze(1), keys)                 # (B, 1, H)

        return cntxt.squeeze(1), att_weights                              # (B, H), (B, S)

class LSTMBahdanauAttentionEmb(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)

        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_SIZE, batch_first=True)
        self.attn = BahdanauAttention(HIDDEN_SIZE)
        self.clf = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 128),
            nn.Dropout(),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        embeddings = self.embedding(x)
        outputs, (h_n, _) = self.lstm(embeddings)
        att_hidden, att_weights = self.attn(outputs, h_n.squeeze(0))
        out = self.clf(att_hidden)
        return out, att_weights