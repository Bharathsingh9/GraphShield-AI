import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import HeteroData

# Add model path to system path for imports
import sys
sys.path.append("d:/fraud_detection")

from ml.models.graphsage_model import HeteroGraphSAGE

# Configuration
GRAPH_DATA_PATH = "d:/fraud_detection/data/processed/graph_data.pt"
MODEL_SAVE_DIR = "d:/fraud_detection/ml/saved_models"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "graphsage_model.pth")
EPOCHS = 40
LEARNING_RATE = 0.005
HIDDEN_CHANNELS = 64

def train_model():
    print("Loading heterogeneous graph dataset...")
    if not os.path.exists(GRAPH_DATA_PATH):
        raise FileNotFoundError(f"Graph file not found: {GRAPH_DATA_PATH}")
        
    data = torch.load(GRAPH_DATA_PATH, weights_only=False)
    print(data)
    
    # 1. Retrieve Node features and Edge structure
    x_dict = data.x_dict
    edge_index_dict = data.edge_index_dict
    
    # 2. Extract Performs (transfers) training data
    perf_y = data["account", "performs", "account"].y
    perf_train_mask = data["account", "performs", "account"].train_mask
    perf_train_y = perf_y[perf_train_mask]
    
    # 3. Extract Paid_to (payments) training data
    paid_y = data["account", "paid_to", "merchant"].y
    paid_train_mask = data["account", "paid_to", "merchant"].train_mask
    paid_train_y = paid_y[paid_train_mask]
    
    # 4. Calculate class weights to handle imbalance (95:5)
    # performs weight
    perf_pos = (perf_train_y == 1).sum().item()
    perf_neg = (perf_train_y == 0).sum().item()
    perf_pos_weight = torch.tensor([perf_neg / max(1, perf_pos)], dtype=torch.float)
    
    # paid_to weight
    paid_pos = (paid_train_y == 1).sum().item()
    paid_neg = (paid_train_y == 0).sum().item()
    paid_pos_weight = torch.tensor([paid_neg / max(1, paid_pos)], dtype=torch.float)
    
    print(f"\nClass imbalance stats (Training set):")
    print(f"- P2P Transfers (performs): Positives (Fraud) = {perf_pos}, Negatives (Genuine) = {perf_neg}, Class Weight = {perf_pos_weight.item():.2f}")
    print(f"- Payments (paid_to): Positives (Fraud) = {paid_pos}, Negatives (Genuine) = {paid_neg}, Class Weight = {paid_pos_weight.item():.2f}")

    # Set up loss functions with balancing weights
    criterion_perf = nn.BCEWithLogitsLoss(pos_weight=perf_pos_weight)
    criterion_paid = nn.BCEWithLogitsLoss(pos_weight=paid_pos_weight)
    
    # Initialize GNN Model
    model = HeteroGraphSAGE(hidden_channels=HIDDEN_CHANNELS, edge_dim=4)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    print("\nStarting GNN training loop...")
    model.train()
    
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        
        # 1. GNN forward pass to compute structural node embeddings
        h_dict = model(x_dict, edge_index_dict)
        
        # 2. Get predictions for performs edges (transfers)
        perf_edge_idx = data["account", "performs", "account"].edge_index[:, perf_train_mask]
        perf_edge_attr = data["account", "performs", "account"].edge_attr[perf_train_mask]
        perf_pred = model.classify_performs(h_dict, perf_edge_idx, perf_edge_attr)
        
        # 3. Get predictions for paid_to edges (payments)
        paid_edge_idx = data["account", "paid_to", "merchant"].edge_index[:, paid_train_mask]
        paid_edge_attr = data["account", "paid_to", "merchant"].edge_attr[paid_train_mask]
        paid_pred = model.classify_paid_to(h_dict, paid_edge_idx, paid_edge_attr)
        
        # 4. Compute weighted losses
        loss_perf = criterion_perf(perf_pred, perf_train_y.float())
        loss_paid = criterion_paid(paid_pred, paid_train_y.float())
        
        # Combined loss
        loss = loss_perf + loss_paid
        
        # 5. Backpropagation
        loss.backward()
        optimizer.step()
        
        # Print epoch metrics
        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Total Loss: {loss.item():.4f} | P2P Loss: {loss_perf.item():.4f} | Purchase Loss: {loss_paid.item():.4f}")
            
    # Save the trained model checkpoint
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "hidden_channels": HIDDEN_CHANNELS,
        "edge_dim": 4
    }, MODEL_SAVE_PATH)
    print(f"\nTrained GraphSAGE model saved successfully to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()
