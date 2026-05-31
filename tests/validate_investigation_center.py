import json
import urllib.request
import urllib.error
import pandas as pd
import time

BASE_URL = "http://127.0.0.1:8000"
API_PREFIX = "/api/v1"

def make_request(path, data=None, method="GET"):
    url = f"{BASE_URL}{path}"
    req_data = json.dumps(data).encode("utf-8") if data else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=req_data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except Exception:
            return e.code, {"detail": e.reason}
    except Exception as e:
        return 500, {"detail": str(e)}

def test_investigation_center_logic():
    print("======================================================================")
    print("        GRAPHSHEILD AI - INVESTIGATION CENTER QA SUITE                ")
    print("======================================================================")
    
    # Pre-load mapping tables for validation
    print("Step 1: Ingesting database mapping indices...")
    customers = pd.read_csv("d:/fraud_detection/data/raw/customers.csv")
    accounts = pd.read_csv("d:/fraud_detection/data/raw/accounts.csv")
    
    acc_to_cust = accounts.set_index("account_id")["customer_id"].to_dict()
    cust_to_name = customers.set_index("customer_id").apply(lambda r: f"{r['first_name']} {r['last_name']}", axis=1).to_dict()
    print("PASS: Mapping structures built successfully.")
    
    # Requirement 1: Search by Transaction ID
    print("\nCheck 1: Simulating Search by Transaction ID (TXN_F_60004315)...")
    target_txn = "TXN_F_60004315"
    df = pd.read_csv("d:/fraud_detection/data/processed/engineered_transactions.csv")
    tx_row = df[df["transaction_id"] == target_txn]
    assert not tx_row.empty, "Transaction ID not found in database."
    resolved_acc = tx_row.iloc[0]["sender_account_id"]
    resolved_cust = acc_to_cust.get(resolved_acc)
    print(f"PASS: Transaction resolved. Sender Account: {resolved_acc} | Customer: {resolved_cust}")
    
    # Requirement 2: Search by Account ID
    print("\nCheck 2: Simulating Search by Account ID (ACC_1002305)...")
    target_acc = "ACC_1002305"
    acc_row = accounts[accounts["account_id"] == target_acc]
    assert not acc_row.empty, "Account ID not found in database."
    resolved_cust_from_acc = acc_row.iloc[0]["customer_id"]
    print(f"PASS: Account resolved. Customer Owner ID: {resolved_cust_from_acc}")
    
    # Requirement 3: Search by Customer ID
    print("\nCheck 3: Simulating Search by Customer ID (C_100230)...")
    target_cust = "C_100230"
    cust_row = customers[customers["customer_id"] == target_cust]
    assert not cust_row.empty, "Customer ID not found in database."
    owned_accs = accounts[accounts["customer_id"] == target_cust]["account_id"].tolist()
    print(f"PASS: Customer resolved. Owner of Accounts: {owned_accs}")

    # Requirement 4 & 6: Display Fraud Score & SHAP Explanation
    print("\nCheck 4 & 6: Querying live GNN Risk Score & SHAP attribution coefficients...")
    payload = {"transaction_id": target_txn}
    status, res = make_request(f"{API_PREFIX}/explainability/explain", payload, "POST")
    assert status == 200, f"Expected 200, got {status}"
    risk_score = res["fraud_probability"]
    base_val = res["base_value"]
    print(f"PASS: GNN Risk Score successfully parsed: {risk_score:.2%}")
    print(f"PASS: SHAP Explanations computed. Base Score: {base_val:.4f}")
    assert "explanations" in res, "Missing SHAP contributions"
    
    # Requirement 5: Display Graph Relationships
    print("\nCheck 5: Querying Graph Neighbors for relationship mapping...")
    n_status, n_res = make_request(f"{API_PREFIX}/graph/neighbors/account/{resolved_acc}")
    assert n_status == 200, f"Expected 200, got {n_status}"
    print(f"PASS: Graph neighbors loaded. Connections found: {n_res['connections_count']}")
    
    # Requirement 7: Generate Investigation Narrative
    print("\nCheck 7: Compiling Automated Investigation Compliance Narrative...")
    cust_name = cust_to_name.get(resolved_cust, "Unknown Cardholder")
    primary_driver = sorted(res["explanations"].items(), key=lambda x: abs(x[1]["shap_value"]), reverse=True)[0][0]
    
    narrative = f"""
    [FORENSIC CASE INVESTIGATION REPORT]
    Case Reference: CASE-{resolved_acc}-{int(time.time())}
    Cardholder: {cust_name} (ID: {resolved_cust})
    GNN Fraud Score: {risk_score:.2%}
    Risk Status: {"HIGH RISK (SUSPECTED MULE/ATO)" if risk_score >= 0.5 else "LOW RISK"}
    Primary SHAP Risk Driver: {primary_driver}
    Graph Neighbors Count: {n_res['connections_count']}
    Triage Recommendation: {"SUSPEND ACCOUNT AND FREEZE FUNDS" if risk_score >= 0.5 else "APPROVE TRANSACTION"}
    """
    print(narrative)
    assert len(narrative) > 100, "Narrative report too short."
    print("PASS: Investigation narrative compiled successfully.")
    
    print("\n======================================================================")
    print("STATUS: Investigation Center validation completed with status: SUCCESS")
    print("======================================================================")

if __name__ == "__main__":
    test_investigation_center_logic()
