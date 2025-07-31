import torch
import torch.nn as nn
import torch.nn.functional as F

class TLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input gate
        self.Wi = nn.Linear(input_dim, hidden_dim)
        self.Ui = nn.Linear(hidden_dim, hidden_dim)
        # Forget gate
        self.Wf = nn.Linear(input_dim, hidden_dim)
        self.Uf = nn.Linear(hidden_dim, hidden_dim)
        # Output gate
        self.Wog = nn.Linear(input_dim, hidden_dim)
        self.Uog = nn.Linear(hidden_dim, hidden_dim)
        # Cell candidate
        self.Wc = nn.Linear(input_dim, hidden_dim)
        self.Uc = nn.Linear(hidden_dim, hidden_dim)
        # Decomposition
        self.W_decomp = nn.Linear(hidden_dim, hidden_dim)
        # Time mapping
        self.c2 = 2.7183

    def map_elapse_time(self, t):
        # t: [batch, 1]
        T = 1.0 / t
        T = T.repeat(1, self.hidden_dim)  # [batch, hidden_dim]
        return T

    def forward(self, x, t, prev_hidden, prev_cell):
        # x: [batch, input_dim], t: [batch, 1]
        # prev_hidden, prev_cell: [batch, hidden_dim]
        T = self.map_elapse_time(t)
        C_ST = torch.tanh(self.W_decomp(prev_cell))
        C_ST_dis = T * C_ST
        prev_cell = prev_cell - C_ST + C_ST_dis

        i = torch.sigmoid(self.Wi(x) + self.Ui(prev_hidden))
        f = torch.sigmoid(self.Wf(x) + self.Uf(prev_hidden))
        o = torch.sigmoid(self.Wog(x) + self.Uog(prev_hidden))
        C = torch.tanh(self.Wc(x) + self.Uc(prev_hidden))
        Ct = f * prev_cell + i * C
        current_hidden = o * torch.tanh(Ct)
        return current_hidden, Ct

# 使用範例
batch_size, input_dim, hidden_dim = 32, 8, 16
cell = TLSTMCell(input_dim, hidden_dim)
x = torch.randn(batch_size, input_dim)
t = torch.abs(torch.randn(batch_size, 1)) + 1e-2  # 避免 log(0)
h0 = torch.zeros(batch_size, hidden_dim)
c0 = torch.zeros(batch_size, hidden_dim)
h1, c1 = cell(x, t, h0, c0)
print(h1.shape, c1.shape)  # (32, 16)