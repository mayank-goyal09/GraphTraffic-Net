import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialGraphConv, self).__init__()
        # 3. Add Spatial "Attention" (The GAT Layer) 🧠
        # GAT with 1 head for simplicity, or multi-head if needed later
        self.gat_conv = GATConv(in_channels, out_channels, heads=1, concat=True)

    def forward(self, x, edge_index, edge_weight):
        # x shape: [Batch * Nodes, Features]
        # GAT handles edge weights differently, but basic usage matches
        # Note: GATConv expects edge_index
        return self.gat_conv(x, edge_index)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TemporalConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels * 2, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [Batch * Nodes, Channels, Time]
        out = self.conv(x)
        # GLU Logic: Split the channels into 'data' and 'gate'
        p, q = torch.split(out, out.shape[1] // 2, dim=1)
        return p * self.sigmoid(q)        