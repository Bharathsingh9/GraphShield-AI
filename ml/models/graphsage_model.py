import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv

class HeteroGraphSAGE(nn.Module):
    """
    Heterogeneous GraphSAGE model for transaction edge classification.
    
    This model:
    1. Projects diverse node features into a uniform hidden dimension.
    2. Runs 2 layers of SAGEConv convolutions across heterogeneous relations to learn structural embeddings.
    3. Leverages custom EdgeClassifier heads to predict fraud probabilities on transaction edges.
    """
    def __init__(self, hidden_channels=64, edge_dim=4):
        super().__init__()
        
        # 1. Node Feature Projection Layers
        # Maps input shapes (Customer:4, Account:3, Merchant:2, Device:2) to hidden_channels
        self.proj = nn.ModuleDict({
            "customer": nn.Linear(4, hidden_channels),
            "account": nn.Linear(3, hidden_channels),
            "merchant": nn.Linear(2, hidden_channels),
            "device": nn.Linear(2, hidden_channels)
        })
        
        # 2. GNN Layer 1
        self.conv1 = HeteroConv({
            ("customer", "owns", "account"): SAGEConv((-1, -1), hidden_channels),
            ("account", "uses", "device"): SAGEConv((-1, -1), hidden_channels),
            ("account", "performs", "account"): SAGEConv((-1, -1), hidden_channels),
            ("account", "paid_to", "merchant"): SAGEConv((-1, -1), hidden_channels)
        }, aggr="mean")
        
        # 3. GNN Layer 2
        self.conv2 = HeteroConv({
            ("customer", "owns", "account"): SAGEConv((-1, -1), hidden_channels),
            ("account", "uses", "device"): SAGEConv((-1, -1), hidden_channels),
            ("account", "performs", "account"): SAGEConv((-1, -1), hidden_channels),
            ("account", "paid_to", "merchant"): SAGEConv((-1, -1), hidden_channels)
        }, aggr="mean")
        
        # 4. Edge Classification Heads
        # P2P Transfers: Concatenates [src_account, dst_account, edge_attributes]
        self.performs_classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2 + edge_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1)
        )
        
        # Merchant Payments: Concatenates [src_account, dst_merchant, edge_attributes]
        self.paid_to_classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2 + edge_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1)
        )
        
    def forward(self, x_dict, edge_index_dict):
        # Project node features to hidden dimensions
        h_dict = {
            node_type: F.relu(self.proj[node_type](x)) 
            for node_type, x in x_dict.items()
        }
        
        # GNN Layer 1
        h_dict_new = self.conv1(h_dict, edge_index_dict)
        # Apply activation and retain projections for nodes that did not receive updates
        h_dict = {
            node_type: F.relu(h_dict_new.get(node_type, h_dict[node_type]))
            for node_type in h_dict
        }
        
        # GNN Layer 2
        h_dict_new = self.conv2(h_dict, edge_index_dict)
        h_dict = {
            node_type: h_dict_new.get(node_type, h_dict[node_type])
            for node_type in h_dict
        }
        
        return h_dict
        
    def classify_performs(self, h_dict, edge_index, edge_attr):
        """Classifies 'performs' (Account -> Account) transfer edges."""
        src, dst = edge_index[0], edge_index[1]
        src_emb = h_dict["account"][src]
        dst_emb = h_dict["account"][dst]
        
        # Concatenate source node, destination node, and transaction attributes
        feat = torch.cat([src_emb, dst_emb, edge_attr], dim=-1)
        return self.performs_classifier(feat).squeeze(-1)
        
    def classify_paid_to(self, h_dict, edge_index, edge_attr):
        """Classifies 'paid_to' (Account -> Merchant) purchase edges."""
        src, dst = edge_index[0], edge_index[1]
        src_emb = h_dict["account"][src]
        dst_emb = h_dict["merchant"][dst]
        
        # Concatenate source account node, merchant node, and transaction attributes
        feat = torch.cat([src_emb, dst_emb, edge_attr], dim=-1)
        return self.paid_to_classifier(feat).squeeze(-1)
