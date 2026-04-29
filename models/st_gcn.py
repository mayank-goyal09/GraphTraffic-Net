import torch
import torch.nn as nn
from models.layers import TemporalConv, SpatialGraphConv

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, spatial_channels, out_channels, num_nodes):
        super(STGCNBlock, self).__init__()
        # 1. Temporal Conv Layer
        self.tmp_conv1 = TemporalConv(in_channels, out_channels)
        # 2. Spatial Graph Conv Layer
        self.sdp_conv = SpatialGraphConv(out_channels, spatial_channels)
        # 3. Second Temporal Conv Layer
        self.tmp_conv2 = TemporalConv(spatial_channels, out_channels)
        # Batch Norm for stability
        self.batch_norm = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_weight):
        # x shape: [Batch, Time, Nodes, Channels]
        batch_size, time_steps, num_nodes, in_channels = x.shape
        
        # 1. First Temporal Layer
        x = x.permute(0, 2, 3, 1).reshape(-1, in_channels, time_steps)
        x = self.tmp_conv1(x) 
        
        # 2. Spatial Layer (GAT)
        # Reshape for GNN: [Batch, Time, Nodes, out_channels] -> [Batch * Time, Nodes, out_channels]
        x = x.reshape(batch_size, num_nodes, -1, x.shape[-1]).permute(0, 3, 1, 2)
        curr_time_steps = x.shape[1]
        
        # Prepare for PyG
        x_batched = x.reshape(-1, num_nodes, x.shape[-1]) 
        num_batches = x_batched.shape[0]
        x_pyg = x_batched.reshape(-1, x_batched.shape[-1])
        
        # Vectorized Edge Index Construction
        num_edges = edge_index.shape[1]
        edge_index_batched = edge_index.repeat(1, num_batches)
        offsets = torch.arange(num_batches, device=edge_index.device) * num_nodes
        offsets = offsets.view(-1, 1).repeat(1, num_edges).view(1, -1)
        edge_index_batched += offsets
            
        # Run GAT (No edge_weight needed for basic GAT)
        x = self.sdp_conv(x_pyg, edge_index_batched, edge_weight=None)
        
        # Reshape back 
        x = x.reshape(batch_size, curr_time_steps, num_nodes, -1)
        
        # 3. Second Temporal Layer
        x = x.permute(0, 2, 3, 1).reshape(batch_size * num_nodes, -1, curr_time_steps)
        x = self.tmp_conv2(x)
        
        # Reshape & Norm
        x = x.reshape(batch_size, num_nodes, -1, time_steps)
        x = x.permute(0, 2, 1, 3) 
        x = self.batch_norm(x)
        x = x.permute(0, 3, 2, 1)
        return self.relu(x)

class TrafficForecaster(nn.Module):
    def __init__(self, num_nodes, num_features, num_timesteps_input, num_timesteps_output):
        super(TrafficForecaster, self).__init__()
        # 1. Increase Model Depth (Stacking) 🏗️
        # We add a 3rd block as requested
        self.block1 = STGCNBlock(num_features, 16, 64, num_nodes)
        self.block2 = STGCNBlock(64, 16, 64, num_nodes)
        self.block3 = STGCNBlock(64, 16, 64, num_nodes)
        
        self.fully_connected = nn.Linear(64 * num_timesteps_input * num_nodes, num_timesteps_output * num_nodes)
        self.num_timesteps_output = num_timesteps_output
        self.num_nodes = num_nodes

    def forward(self, x, edge_index, edge_weight):
        # x shape: [Batch, Time, Nodes, Features]
        x = self.block1(x, edge_index, edge_weight)
        x = self.block2(x, edge_index, edge_weight)
        x = self.block3(x, edge_index, edge_weight)
        
        x = x.reshape(x.size(0), -1)
        x = self.fully_connected(x)
        return x.reshape(-1, self.num_timesteps_output, self.num_nodes)

