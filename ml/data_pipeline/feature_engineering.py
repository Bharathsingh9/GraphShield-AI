import os
import pandas as pd
import numpy as np

# Paths
INPUT_DIR = "d:/fraud_detection/data/raw"
OUTPUT_DIR = "d:/fraud_detection/data/processed"

def load_raw_data():
    customers = pd.read_csv(os.path.join(INPUT_DIR, "customers.csv"))
    accounts = pd.read_csv(os.path.join(INPUT_DIR, "accounts.csv"))
    merchants = pd.read_csv(os.path.join(INPUT_DIR, "merchants.csv"))
    devices = pd.read_csv(os.path.join(INPUT_DIR, "devices.csv"))
    transactions = pd.read_csv(os.path.join(INPUT_DIR, "transactions.csv"))
    
    # Parse datetimes
    customers["created_at"] = pd.to_datetime(customers["created_at"])
    accounts["created_at"] = pd.to_datetime(accounts["created_at"])
    transactions["timestamp"] = pd.to_datetime(transactions["timestamp"])
    
    return customers, accounts, merchants, devices, transactions

def engineer_features():
    print("Starting Fraud Feature Engineering...")
    customers, accounts, merchants, devices, transactions = load_raw_data()
    
    # Sort transactions chronologically for rolling window calculations
    transactions = transactions.sort_values("timestamp").reset_index(drop=True)
    
    # Create lookup dictionaries for fast mapping
    cust_city = customers.set_index("customer_id")["city"].to_dict()
    acc_cust = accounts.set_index("account_id")["customer_id"].to_dict()
    acc_created = accounts.set_index("account_id")["created_at"].to_dict()
    dev_loc = devices.set_index("device_id")["location"].to_dict()
    merch_risk = merchants.set_index("merchant_id")["risk_score"].to_dict()
    
    # Calculate account sharing per device (how many accounts have used each device)
    # We will count how many unique accounts are associated with each device in transactions
    dev_share_counts = transactions.groupby("device_id")["sender_account_id"].nunique().to_dict()
    
    # Calculate how many accounts are owned by each customer
    cust_acc_counts = accounts.groupby("customer_id")["account_id"].nunique().to_dict()
    
    # Lists to store computed features
    txns_last_1h = []
    txns_last_24h = []
    avg_amount_history = []
    device_sharing = []
    linked_accounts = []
    merchant_exposure = []
    account_age_days = []
    geo_anomaly = []
    time_since_prev_sec = []
    
    # Tracker for rolling features: dictionary of account_id -> list of timestamps and amounts
    acc_history = {}
    
    print("Computing rolling window features and relational indicators...")
    for idx, row in transactions.iterrows():
        sender = row["sender_account_id"]
        timestamp = row["timestamp"]
        amount = row["amount"]
        merch_id = row["merchant_id"]
        dev_id = row["device_id"]
        
        # 1. Rolling windows (1h and 24h transaction counts)
        # 2. Historical average transaction amount
        # 3. Time since previous transaction (velocity)
        if pd.isna(sender) or sender == "":
            txns_last_1h.append(0)
            txns_last_24h.append(0)
            avg_amount_history.append(0.0)
            time_since_prev_sec.append(86400) # Default to 1 day if no prior txn
        else:
            if sender not in acc_history:
                acc_history[sender] = []
            
            # Remove timestamps older than 24 hours to keep the window memory efficient
            cutoff_24h = timestamp - pd.Timedelta(hours=24)
            acc_history[sender] = [x for x in acc_history[sender] if x[0] > cutoff_24h]
            
            # Count transactions in windows
            t_1h = 0
            t_24h = 0
            total_history_amt = 0.0
            history_cnt = 0
            cutoff_1h = timestamp - pd.Timedelta(hours=1)
            
            for ts, amt in acc_history[sender]:
                if ts > cutoff_1h:
                    t_1h += 1
                if ts > cutoff_24h:
                    t_24h += 1
            
            txns_last_1h.append(t_1h)
            txns_last_24h.append(t_24h)
            
            # Calculate time since previous transaction (velocity)
            if len(acc_history[sender]) > 0:
                last_ts = acc_history[sender][-1][0]
                td = (timestamp - last_ts).total_seconds()
                time_since_prev_sec.append(max(0.0, td))
            else:
                time_since_prev_sec.append(86400.0)
            
            # Append current to history AFTER computing window counts (excl. current transaction)
            acc_history[sender].append((timestamp, amount))
            
            # Average amount of all transactions seen so far for this account
            # We can use the moving average of the 24h window or a cumulative average
            # Let's use the average of the 24h window for better responsiveness to sudden shocks
            all_amts = [x[1] for x in acc_history[sender]]
            avg_amount_history.append(np.mean(all_amts) if all_amts else amount)
            
        # 4. Device sharing count
        if pd.isna(dev_id) or dev_id == "":
            device_sharing.append(0)
        else:
            device_sharing.append(dev_share_counts.get(dev_id, 0))
            
        # 5. Number of linked accounts for the customer
        cust_id = acc_cust.get(sender) if not pd.isna(sender) else None
        if cust_id:
            linked_accounts.append(cust_acc_counts.get(cust_id, 1))
        else:
            linked_accounts.append(0)
            
        # 6. Merchant Risk Exposure
        if not pd.isna(merch_id) and merch_id != "":
            merchant_exposure.append(merch_risk.get(merch_id, 0.0))
        else:
            merchant_exposure.append(0.0)
            
        # 7. Account Age (Days) at transaction time
        created_time = acc_created.get(sender) if not pd.isna(sender) else None
        if created_time:
            age_days = (timestamp - created_time).total_seconds() / 86400.0
            account_age_days.append(max(0.0, age_days))
        else:
            account_age_days.append(0.0)
            
        # 8. Geographical Anomaly
        # Compare Device IP location with Customer Home City
        c_city = cust_city.get(cust_id) if cust_id else None
        d_location = dev_loc.get(dev_id) if not pd.isna(dev_id) and dev_id != "" else None
        if c_city and d_location:
            # Anomaly is 1 if they do not match
            geo_anomaly.append(1 if c_city != d_location else 0)
        else:
            geo_anomaly.append(0)
            
    # Add computed features to transactions DataFrame
    transactions["txns_last_1h"] = txns_last_1h
    transactions["txns_last_24h"] = txns_last_24h
    transactions["avg_amount_history_24h"] = avg_amount_history
    transactions["device_sharing_count"] = device_sharing
    transactions["linked_accounts_count"] = linked_accounts
    transactions["merchant_risk_score"] = merchant_exposure
    transactions["account_age_days"] = account_age_days
    transactions["geo_anomaly"] = geo_anomaly
    transactions["time_since_prev_sec"] = time_since_prev_sec
    
    # Write feature descriptions and why they help detect fraud
    explanations = """# Advanced Fraud Feature Engineering Documentation

The feature engineering pipeline computes key behavioural, temporal, and spatial metrics to identify anomalous transactions.

### 1. `txns_last_1h` & `txns_last_24h` (Velocity Indicators)
- **Why it helps:** Fraudsters (or automated scripts) drain compromised accounts using multiple transactions in minutes. Genuine users rarely make more than 3-5 transfers in an hour.

### 2. `avg_amount_history_24h` (Amount Deviation)
- **Why it helps:** Sudden high-value transactions that deviate significantly from the account's historical average signal potential Account Takeover (ATO) or Money Mule cash-out activities.

### 3. `time_since_prev_sec` (Timing Anomaly)
- **Why it helps:** Millisecond or second-level gaps between consecutive transfers are indicative of scripted API abuse and rapid transfer chaining.

### 4. `device_sharing_count` (Graph-based Bipartite Feature)
- **Why it helps:** A device used by only 1-2 accounts is normal. A device used by 10+ distinct customer accounts indicates a central organizer running a mule network or a credential stuffing attack from a single machine.

### 5. `linked_accounts_count` (Profile Structural Constraint)
- **Why it helps:** Synthetic fraudsters often open multiple current, savings, and credit accounts rapidly to maximize their bust-out limits.

### 6. `merchant_risk_score` (Entity Exposure)
- **Why it helps:** Merchant categories like Cryptocurrency Exchanges and Gaming sites carry a higher likelihood of liquidation. Transactions routed to high-risk merchants require closer inspection.

### 7. `account_age_days` (Establishment History)
- **Why it helps:** Synthetic identity profiles and mule accounts are typically very new. Long-established accounts have a lower probability of initiating first-party bust-out fraud, though they can be victims of ATO.

### 8. `geo_anomaly` (Spatial Outlier)
- **Why it helps:** A user accessing the banking app from a device located in Kiev, Ukraine, while their home city is London, UK, indicates impossible travel or credential compromise.
"""
    
    # Save explanations
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "feature_explanations.md"), "w", encoding="utf-8") as f:
        f.write(explanations)
        
    # Save engineered transactions
    engineered_path = os.path.join(OUTPUT_DIR, "engineered_transactions.csv")
    transactions.to_csv(engineered_path, index=False)
    print(f"Engineered transactions saved to: {engineered_path}")
    print(f"Explanations document saved to: {os.path.join(OUTPUT_DIR, 'feature_explanations.md')}")

if __name__ == "__main__":
    engineer_features()
