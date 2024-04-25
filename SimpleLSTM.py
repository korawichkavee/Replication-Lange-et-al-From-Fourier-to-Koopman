import torch
import torch.nn as nn
import numpy as np

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device='cpu'):
        super(SimpleLSTM, self).__init__()
        self.device = torch.device(device)  # Ensure device is a torch.device object
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.to(self.device)  # Move the whole module's parameters and buffers to the specified device

    def forward(self, x):
        # Initial hidden and cell states
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        # LSTM output
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Using the last output state
        return out