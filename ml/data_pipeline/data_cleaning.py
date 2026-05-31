import os
import pandas as pd
import numpy as np

# Paths
DATA_DIR = "d:/fraud_detection/data/raw"
REPORT_PATH = "d:/fraud_detection/docs/Data_Quality_Report.md"

def load_data():
    files = {
        "customers": "customers.csv",
        "accounts": "accounts.csv",
        "merchants": "merchants.csv",
        "devices": "devices.csv",
        "transactions": "transactions.csv",
        "logins": "logins.csv",
        "beneficiaries": "beneficiaries.csv"
    }
    dfs = {}
    for name, filename in files.items():
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            dfs[name] = pd.read_csv(path)
            print(f"Loaded {filename}: {len(dfs[name])} rows")
        else:
            raise FileNotFoundError(f"Missing file: {path}")
    return dfs

def run_validation():
    print("Running Data Quality Pipeline...")
    dfs = load_data()
    
    customers = dfs["customers"]
    accounts = dfs["accounts"]
    merchants = dfs["merchants"]
    devices = dfs["devices"]
    transactions = dfs["transactions"]
    logins = dfs["logins"]
    beneficiaries = dfs["beneficiaries"]
    
    report = []
    report.append("# GraphShield AI: Data Quality Validation Report")
    report.append(f"**Date Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\nThis report summarizes the data quality checks executed by the validation pipeline on the simulated banking datasets.")
    
    # 1. Row Counts Summary
    report.append("\n## 1. Dataset Row Counts")
    count_summary = []
    count_summary.append("| Dataset Name | Row Count | columns |")
    count_summary.append("| :--- | :---: | :--- |")
    for name, df in dfs.items():
        count_summary.append(f"| {name} | {len(df):,} | {', '.join(df.columns)} |")
    report.extend(count_summary)
    
    # 2. Missing Values Check
    report.append("\n## 2. Missing Values (Nulls) Analysis")
    report.append("Checking for unexpected null values across all tables. (Note: nullable foreign keys in transactions are permitted under specific transaction types).")
    
    null_tables = []
    null_tables.append("| Table | Field Name | Null Count | Null % | Status |")
    null_tables.append("| :--- | :--- | :---: | :---: | :--- |")
    
    has_null_errors = False
    
    for name, df in dfs.items():
        null_counts = df.isnull().sum()
        for col, count in null_counts.items():
            pct = (count / len(df)) * 100
            # Define fields that are allowed to have nulls
            allowed_null = False
            if name == "transactions" and col in ["sender_account_id", "receiver_account_id", "merchant_id", "device_id"]:
                allowed_null = True
            
            if count > 0:
                status = "⚠️ Nulls Allowed (Nullable FK)" if allowed_null else "❌ ERROR: Unexpected Nulls"
                if not allowed_null:
                    has_null_errors = True
                null_tables.append(f"| {name} | {col} | {count:,} | {pct:.2f}% | {status} |")
            else:
                pass
                
    if len(null_tables) <= 2:
         report.append("\n> [!NOTE]\n> No missing values found in any required fields across all tables.")
    else:
         report.extend(null_tables)

    # 3. Duplicate Primary Keys Check
    report.append("\n## 3. Primary Key Uniqueness Check")
    pks = {
        "customers": "customer_id",
        "accounts": "account_id",
        "merchants": "merchant_id",
        "devices": "device_id",
        "transactions": "transaction_id",
        "logins": "login_id",
        "beneficiaries": "beneficiary_id"
    }
    
    pk_table = []
    pk_table.append("| Table | Primary Key Column | Duplicate Count | Status |")
    pk_table.append("| :--- | :--- | :---: | :--- |")
    
    has_pk_duplicates = False
    for name, col in pks.items():
        dups = dfs[name][col].duplicated().sum()
        if dups > 0:
            status = "❌ ERROR: Duplicate Keys Found"
            has_pk_duplicates = True
        else:
            status = "✅ Clean"
        pk_table.append(f"| {name} | {col} | {dups:,} | {status} |")
        
    report.extend(pk_table)

    # 4. Broken Foreign Keys Check
    report.append("\n## 4. Referential Integrity (Foreign Key) Check")
    fk_checks = [
        ("accounts", "customer_id", "customers", "customer_id"),
        ("transactions", "sender_account_id", "accounts", "account_id"),
        ("transactions", "receiver_account_id", "accounts", "account_id"),
        ("transactions", "merchant_id", "merchants", "merchant_id"),
        ("transactions", "device_id", "devices", "device_id"),
        ("logins", "customer_id", "customers", "customer_id"),
        ("logins", "device_id", "devices", "device_id"),
        ("beneficiaries", "account_id", "accounts", "account_id"),
        ("beneficiaries", "beneficiary_account_id", "accounts", "account_id")
    ]
    
    fk_table = []
    fk_table.append("| Source Table | Foreign Key Field | Reference Table | Reference Key Field | Broken Count | Status |")
    fk_table.append("| :--- | :--- | :--- | :--- | :---: | :--- |")
    
    has_fk_broken = False
    for src_t, src_f, ref_t, ref_f in fk_checks:
        src_df = dfs[src_t]
        ref_df = dfs[ref_t]
        
        # Extract non-null, non-empty foreign key values
        src_keys = src_df[src_f].dropna()
        if src_keys.dtype == object:
            src_keys = src_keys[src_keys != ""]
            src_keys = src_keys[src_keys.astype(str).str.strip() != ""]
            
        ref_keys = set(ref_df[ref_f].dropna().unique())
        
        broken = src_keys[~src_keys.isin(ref_keys)]
        broken_count = len(broken)
        
        if broken_count > 0:
            status = "❌ ERROR: Broken References Found"
            has_fk_broken = True
            # Log sample broken keys
            print(f"Broken FK: {src_t}.{src_f} referencing {ref_t}.{ref_f} has {broken_count} broken rows. Sample: {broken.head().values}")
        else:
            status = "✅ Integrity Maintained"
            
        fk_table.append(f"| {src_t} | {src_f} | {ref_t} | {ref_f} | {broken_count:,} | {status} |")
        
    report.extend(fk_table)

    # 5. Invalid Values and Impossible Balances Check
    report.append("\n## 5. Domain Boundary and Value Limits")
    
    # Check negative balances for savings and current accounts (typically overdraft is current only, savings should be positive)
    neg_savings = accounts[(accounts["account_type"] == "SAVINGS") & (accounts["balance"] < 0)]
    neg_current = accounts[(accounts["account_type"] == "CURRENT") & (accounts["balance"] < 0)]
    neg_business = accounts[(accounts["account_type"] == "BUSINESS") & (accounts["balance"] < 0)]
    
    report.append("\nChecking account balances for impossible configurations:")
    balance_checks = []
    balance_checks.append(f"- Savings Accounts with negative balance: {len(neg_savings)} rows (Status: {'✅ OK' if len(neg_savings)==0 else '❌ ERROR: Impossible Savings Balance'})")
    balance_checks.append(f"- Current Accounts with negative balance: {len(neg_current)} rows (Status: {'✅ OK' if len(neg_current)==0 else '❌ Warning: Overdraft Allowed'})")
    balance_checks.append(f"- Business Accounts with negative balance: {len(neg_business)} rows (Status: {'✅ OK' if len(neg_business)==0 else '❌ Warning: Overdraft Allowed'})")
    report.extend(balance_checks)
    
    # Check transaction amounts
    zero_or_neg_txns = transactions[transactions["amount"] <= 0]
    report.append(f"- Transactions with zero or negative amounts: {len(zero_or_neg_txns)} rows (Status: {'✅ OK' if len(zero_or_neg_txns)==0 else '❌ ERROR: Zero/Negative Transaction'})")

    # 6. Relationship Consistency Check
    report.append("\n## 6. Time and Relational Consistency Checks")
    
    # Convert created_at/timestamp columns to datetime
    for name, df in dfs.items():
        for col in ["created_at", "timestamp"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                
    # Check if Account was created before Customer Profile
    acc_cust = accounts.merge(customers, on="customer_id", suffixes=("_acc", "_cust"))
    early_acc = acc_cust[acc_cust["created_at_acc"] < acc_cust["created_at_cust"]]
    report.append(f"- Accounts opened prior to customer registration: {len(early_acc)} instances (Status: {'✅ OK' if len(early_acc)==0 else '❌ ERROR: Out of order creation'})")
    
    # Check if Transactions occurred after Sender/Receiver Account was opened
    # Map account creation timestamps
    acc_created = accounts.set_index("account_id")["created_at"]
    
    tx_sender = transactions.merge(accounts, left_on="sender_account_id", right_on="account_id", suffixes=("_tx", "_acc"))
    early_sender_tx = tx_sender[tx_sender["timestamp"] < tx_sender["created_at"]]
    
    tx_receiver = transactions.merge(accounts, left_on="receiver_account_id", right_on="account_id", suffixes=("_tx", "_acc"))
    early_receiver_tx = tx_receiver[tx_receiver["timestamp"] < tx_receiver["created_at"]]
    
    report.append(f"- Transactions occurring before sending account was opened: {len(early_sender_tx)} instances (Status: {'✅ OK' if len(early_sender_tx)==0 else '❌ ERROR: Time travel transaction'})")
    report.append(f"- Transactions occurring before receiving account was opened: {len(early_receiver_tx)} instances (Status: {'✅ OK' if len(early_receiver_tx)==0 else '❌ ERROR: Time travel transaction'})")
    
    # Logins before customer profile created
    login_cust = logins.merge(customers, on="customer_id", suffixes=("_log", "_cust"))
    early_log = login_cust[login_cust["timestamp"] < login_cust["created_at"]]
    report.append(f"- Logins occurring before customer registration: {len(early_log)} instances (Status: {'✅ OK' if len(early_log)==0 else '❌ ERROR: Time travel login'})")

    # Overall Status Summary
    overall_ok = not (has_null_errors or has_pk_duplicates or has_fk_broken or len(zero_or_neg_txns) > 0 or len(early_acc) > 0 or len(early_sender_tx) > 0 or len(early_log) > 0)
    
    report.append("\n## 7. Overall Pipeline Status")
    if overall_ok:
        report.append("> [!TIP]\n> **PASSED**: The entire data suite passes LBG referential integrity, domain boundary, and temporal consistency checks. The data is ready for feature engineering and Graph Neural Network ingestion.")
    else:
        report.append("> [!WARNING]\n> **FAILED**: One or more critical data quality checks failed. See detail tables above to isolate anomalies.")
        
    # Write report
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
        
    print(f"Data validation report written to: {REPORT_PATH}")
    return overall_ok

if __name__ == "__main__":
    success = run_validation()
    import sys
    sys.exit(0 if success else 1)
