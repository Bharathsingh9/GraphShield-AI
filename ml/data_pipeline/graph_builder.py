import os
import json
import pandas as pd
import numpy as np

# Paths
RAW_DIR = "d:/fraud_detection/data/raw"
PROCESSED_DIR = "d:/fraud_detection/data/processed"
GRAPH_DATA_PATH = os.path.join(PROCESSED_DIR, "graph_data.pt")
MAPPING_DIR = os.path.join(PROCESSED_DIR, "mappings")

os.makedirs(MAPPING_DIR, exist_ok=True)

def load_datasets():
    customers = pd.read_csv(os.path.join(RAW_DIR, "customers.csv"))
    accounts = pd.read_csv(os.path.join(RAW_DIR, "accounts.csv"))
    merchants = pd.read_csv(os.path.join(RAW_DIR, "merchants.csv"))
    devices = pd.read_csv(os.path.join(RAW_DIR, "devices.csv"))
    transactions = pd.read_csv(os.path.join(PROCESSED_DIR, "engineered_transactions.csv"))
    logins = pd.read_csv(os.path.join(RAW_DIR, "logins.csv"))
    
    # Ensure datetimes
    transactions["timestamp"] = pd.to_datetime(transactions["timestamp"])
    accounts["created_at"] = pd.to_datetime(accounts["created_at"])
    customers["created_at"] = pd.to_datetime(customers["created_at"])
    logins["timestamp"] = pd.to_datetime(logins["timestamp"])
    
    return customers, accounts, merchants, devices, transactions, logins

def build_mappings(customers, accounts, merchants, devices):
    print("Building node-to-index mappings...")
    
    # Map node IDs to 0-indexed integers
    cust_map = {id_: idx for idx, id_ in enumerate(customers["customer_id"].unique())}
    acc_map = {id_: idx for idx, id_ in enumerate(accounts["account_id"].unique())}
    merch_map = {id_: idx for idx, id_ in enumerate(merchants["merchant_id"].unique())}
    dev_map = {id_: idx for idx, id_ in enumerate(devices["device_id"].unique())}
    
    # Save mappings to JSON for reference
    with open(os.path.join(MAPPING_DIR, "customer_mapping.json"), "w") as f:
        json.dump(cust_map, f)
    with open(os.path.join(MAPPING_DIR, "account_mapping.json"), "w") as f:
        json.dump(acc_map, f)
    with open(os.path.join(MAPPING_DIR, "merchant_mapping.json"), "w") as f:
        json.dump(merch_map, f)
    with open(os.path.join(MAPPING_DIR, "device_mapping.json"), "w") as f:
        json.dump(dev_map, f)
        
    print(f"Mapped: {len(cust_map)} Customers, {len(acc_map)} Accounts, {len(merch_map)} Merchants, {len(dev_map)} Devices")
    return cust_map, acc_map, merch_map, dev_map

def extract_node_features(customers, accounts, merchants, devices):
    print("Extracting and scaling node features...")
    
    # 1. Customer Features
    # Age (min-max scale), Risk Category (encoded), City (one-hot or encoded)
    customers["age_scaled"] = (customers["age"] - 18) / (85 - 18)
    risk_mapping = {"LOW": 0.0, "MEDIUM": 1.0, "HIGH": 2.0}
    customers["risk_encoded"] = customers["risk_category"].map(risk_mapping)
    
    # Encode city
    cities = sorted(customers["city"].unique())
    city_mapping = {city: float(idx) for idx, city in enumerate(cities)}
    customers["city_encoded"] = customers["city"].map(city_mapping)
    
    # Occupation encoding
    occs = sorted(customers["occupation"].unique())
    occ_mapping = {occ: float(idx) for idx, occ in enumerate(occs)}
    customers["occ_encoded"] = customers["occupation"].map(occ_mapping)
    
    customer_features = customers[["age_scaled", "risk_encoded", "city_encoded", "occ_encoded"]].values
    
    # 2. Account Features
    # Balance (log1p scaling), type, status
    accounts["balance_log"] = np.log1p(accounts["balance"].clip(lower=0))
    type_mapping = {"CURRENT": 0.0, "SAVINGS": 1.0, "BUSINESS": 2.0}
    accounts["type_encoded"] = accounts["account_type"].map(type_mapping)
    status_mapping = {"ACTIVE": 0.0, "DORMANT": 1.0}
    accounts["status_encoded"] = accounts["status"].map(status_mapping)
    
    account_features = accounts[["balance_log", "type_encoded", "status_encoded"]].values
    
    # 3. Merchant Features
    # Risk score, Category
    cat_mapping = {"RETAIL": 0.0, "TRAVEL": 1.0, "CRYPTO_EXCHANGE": 2.0, "GAMING": 3.0, "E_COMMERCE": 4.0}
    merchants["category_encoded"] = merchants["merchant_category"].map(cat_mapping)
    
    merchant_features = merchants[["risk_score", "category_encoded"]].values
    
    # 4. Device Features
    # Device Type, OS encoding
    type_map = {"MOBILE": 0.0, "LAPTOP": 1.0, "TABLET": 2.0}
    devices["type_encoded"] = devices["device_type"].map(type_map)
    
    os_list = sorted(devices["operating_system"].unique())
    os_map = {os: float(idx) for idx, os in enumerate(os_list)}
    devices["os_encoded"] = devices["operating_system"].map(os_map)
    
    device_features = devices[["type_encoded", "os_encoded"]].values
    
    return customer_features, account_features, merchant_features, device_features

def build_edges_and_attributes(transactions, logins, accounts, cust_map, acc_map, merch_map, dev_map):
    print("Building adjacency list and edge attributes...")
    
    # 1. OWNS Edges: Customer -> Account
    owns_src = []
    owns_dst = []
    for idx, row in accounts.iterrows():
        c_id = row["customer_id"]
        a_id = row["account_id"]
        if c_id in cust_map and a_id in acc_map:
            owns_src.append(cust_map[c_id])
            owns_dst.append(acc_map[a_id])
            
    owns_edge_index = np.array([owns_src, owns_dst], dtype=np.int64)
    
    # 2. USES Edges: Account -> Device (derived from logins and transaction logs)
    # Logins connect customer_id -> device_id. We map customer_id to their accounts.
    uses_src = []
    uses_dst = []
    
    # Map customer to accounts
    cust_to_accs = {}
    for a_id, c_id in zip(accounts["account_id"], accounts["customer_id"]):
        cust_to_accs.setdefault(c_id, []).append(a_id)
        
    for idx, row in logins.iterrows():
        c_id = row["customer_id"]
        d_id = row["device_id"]
        if c_id in cust_to_accs and d_id in dev_map:
            for a_id in cust_to_accs[c_id]:
                uses_src.append(acc_map[a_id])
                uses_dst.append(dev_map[d_id])
                
    # Also transactions connect sender_account_id -> device_id
    for idx, row in transactions.iterrows():
        a_id = row["sender_account_id"]
        d_id = row["device_id"]
        if not pd.isna(a_id) and a_id != "" and not pd.isna(d_id) and d_id != "":
            if a_id in acc_map and d_id in dev_map:
                uses_src.append(acc_map[a_id])
                uses_dst.append(dev_map[d_id])
                
    # Deduplicate USES edges
    uses_edges = list(set(zip(uses_src, uses_dst)))
    uses_edge_index = np.array([[e[0] for e in uses_edges], [e[1] for e in uses_edges]], dtype=np.int64)

    # 3. PERFORMS Edges: Account -> Account (P2P Transfers)
    performs_src = []
    performs_dst = []
    performs_attrs = []
    performs_labels = []
    performs_timestamps = []
    
    # 4. PAID_TO Edges: Account -> Merchant (Purchases / payments)
    paid_to_src = []
    paid_to_dst = []
    paid_to_attrs = []
    paid_to_labels = []
    paid_to_timestamps = []
    
    # Normalize amount features for edge attributes
    transactions["amount_log"] = np.log1p(transactions["amount"])
    
    for idx, row in transactions.iterrows():
        sender = row["sender_account_id"]
        receiver = row["receiver_account_id"]
        merch = row["merchant_id"]
        
        # Edge attributes: log_amount, txns_last_1h, txns_last_24h, geo_anomaly
        edge_feat = [
            row["amount_log"],
            float(row["txns_last_1h"]),
            float(row["txns_last_24h"]),
            float(row["geo_anomaly"])
        ]
        
        if not pd.isna(sender) and sender != "":
            # Check if transfer or purchase
            if not pd.isna(receiver) and receiver != "" and sender in acc_map and receiver in acc_map:
                performs_src.append(acc_map[sender])
                performs_dst.append(acc_map[receiver])
                performs_attrs.append(edge_feat)
                performs_labels.append(int(row["fraud_label"]))
                performs_timestamps.append(row["timestamp"])
            elif not pd.isna(merch) and merch != "" and sender in acc_map and merch in merch_map:
                paid_to_src.append(acc_map[sender])
                paid_to_dst.append(merch_map[merch])
                paid_to_attrs.append(edge_feat)
                paid_to_labels.append(int(row["fraud_label"]))
                paid_to_timestamps.append(row["timestamp"])
                
    performs_edge_index = np.array([performs_src, performs_dst], dtype=np.int64)
    performs_edge_attr = np.array(performs_attrs, dtype=np.float32)
    performs_labels = np.array(performs_labels, dtype=np.int64)
    
    paid_to_edge_index = np.array([paid_to_src, paid_to_dst], dtype=np.int64)
    paid_to_edge_attr = np.array(paid_to_attrs, dtype=np.float32)
    paid_to_labels = np.array(paid_to_labels, dtype=np.int64)
    
    return (
        owns_edge_index, uses_edge_index,
        performs_edge_index, performs_edge_attr, performs_labels, performs_timestamps,
        paid_to_edge_index, paid_to_edge_attr, paid_to_labels, paid_to_timestamps
    )

def main():
    print("Loading engineered features and relational datasets...")
    customers, accounts, merchants, devices, transactions, logins = load_datasets()
    
    cust_map, acc_map, merch_map, dev_map = build_mappings(customers, accounts, merchants, devices)
    
    cust_feats, acc_feats, merch_feats, dev_feats = extract_node_features(customers, accounts, merchants, devices)
    
    (
        owns_idx, uses_idx,
        perf_idx, perf_attr, perf_y, perf_time,
        paid_idx, paid_attr, paid_y, paid_time
    ) = build_edges_and_attributes(transactions, logins, accounts, cust_map, acc_map, merch_map, dev_map)
    
    print("\nSummary of Extracted Graph Structure:")
    print(f"- Customer Nodes: {cust_feats.shape} features shape")
    print(f"- Account Nodes: {acc_feats.shape} features shape")
    print(f"- Merchant Nodes: {merch_feats.shape} features shape")
    print(f"- Device Nodes: {dev_feats.shape} features shape")
    print(f"- 'owns' Edges (Customer -> Account): {owns_idx.shape[1]:,}")
    print(f"- 'uses' Edges (Account -> Device): {uses_idx.shape[1]:,}")
    print(f"- 'performs' Edges (Account -> Account): {perf_idx.shape[1]:,}")
    print(f"- 'paid_to' Edges (Account -> Merchant): {paid_idx.shape[1]:,}")
    
    # Temporal Train-Test Split (Split on May 24th, 2026)
    split_date = pd.Timestamp("2026-05-24 00:00:00")
    
    perf_train_mask = np.array([ts < split_date for ts in perf_time])
    perf_test_mask = ~perf_train_mask
    
    paid_train_mask = np.array([ts < split_date for ts in paid_time])
    paid_test_mask = ~paid_train_mask
    
    print(f"\nPerforming temporal edge split (split date: {split_date}):")
    print(f"- P2P Transfers (performs): Train: {perf_train_mask.sum():,} ({perf_train_mask.mean()*100:.1f}%), Test: {perf_test_mask.sum():,} ({perf_test_mask.mean()*100:.1f}%)")
    print(f"- Payments (paid_to): Train: {paid_train_mask.sum():,} ({paid_train_mask.mean()*100:.1f}%), Test: {paid_test_mask.sum():,} ({paid_test_mask.mean()*100:.1f}%)")

    # Attempt to build PyTorch Geometric HeteroData object
    try:
        import torch
        from torch_geometric.data import HeteroData
        
        print("\nPyTorch and PyTorch Geometric detected. Creating PyG HeteroData object...")
        data = HeteroData()
        
        # Node features
        data["customer"].x = torch.tensor(cust_feats, dtype=torch.float)
        data["account"].x = torch.tensor(acc_feats, dtype=torch.float)
        data["merchant"].x = torch.tensor(merch_feats, dtype=torch.float)
        data["device"].x = torch.tensor(dev_feats, dtype=torch.float)
        
        # Edge indices
        data["customer", "owns", "account"].edge_index = torch.tensor(owns_idx, dtype=torch.long)
        data["account", "uses", "device"].edge_index = torch.tensor(uses_idx, dtype=torch.long)
        data["account", "performs", "account"].edge_index = torch.tensor(perf_idx, dtype=torch.long)
        data["account", "paid_to", "merchant"].edge_index = torch.tensor(paid_idx, dtype=torch.long)
        
        # Edge attributes and targets
        data["account", "performs", "account"].edge_attr = torch.tensor(perf_attr, dtype=torch.float)
        data["account", "performs", "account"].y = torch.tensor(perf_y, dtype=torch.long)
        data["account", "performs", "account"].train_mask = torch.tensor(perf_train_mask, dtype=torch.bool)
        data["account", "performs", "account"].test_mask = torch.tensor(perf_test_mask, dtype=torch.bool)
        
        data["account", "paid_to", "merchant"].edge_attr = torch.tensor(paid_attr, dtype=torch.float)
        data["account", "paid_to", "merchant"].y = torch.tensor(paid_y, dtype=torch.long)
        data["account", "paid_to", "merchant"].train_mask = torch.tensor(paid_train_mask, dtype=torch.bool)
        data["account", "paid_to", "merchant"].test_mask = torch.tensor(paid_test_mask, dtype=torch.bool)
        
        # Save PyG object
        torch.save(data, GRAPH_DATA_PATH)
        print(f"Successfully serialized PyG HeteroData object to: {GRAPH_DATA_PATH}")
        print("Graph object structure:")
        print(data)
        
    except ImportError as e:
        print(f"\n[Notice] PyTorch or PyG was not found ({str(e)}). Saving graph arrays to NPZ file instead...")
        # Save as standard numpy structure for flexibility
        npz_path = os.path.join(PROCESSED_DIR, "graph_data.npz")
        np.savez_compressed(
            npz_path,
            customer_x=cust_feats,
            account_x=acc_feats,
            merchant_x=merch_feats,
            device_x=dev_feats,
            owns_edge_index=owns_idx,
            uses_edge_index=uses_idx,
            performs_edge_index=perf_idx,
            performs_edge_attr=perf_attr,
            performs_y=perf_y,
            performs_train_mask=perf_train_mask,
            performs_test_mask=perf_test_mask,
            paid_to_edge_index=paid_idx,
            paid_to_edge_attr=paid_attr,
            paid_to_y=paid_y,
            paid_to_train_mask=paid_train_mask,
            paid_to_test_mask=paid_test_mask
        )
        print(f"Serialized graph structures to NPZ archive: {npz_path}")

if __name__ == "__main__":
    main()
