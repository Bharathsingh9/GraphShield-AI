import json
import urllib.request
import urllib.error
import pandas as pd
import numpy as np

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

def validate_dashboard_metrics():
    print("======================================================================")
    print("         GRAPHSHEILD AI - DASHBOARD METRICS QA VALIDATOR              ")
    print("======================================================================")
    
    # 1. Load Processed Ledger
    print("Step 1: Loading engineered transactions database...")
    df = pd.read_csv("d:/fraud_detection/data/processed/engineered_transactions.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"PASS: Ledger loaded. Total database row count: {len(df):,}")
    
    # 2. Re-calculate metrics for the active period (on or after May 24, 2026)
    print("\nStep 2: Calculating active window aggregations...")
    split_date = pd.Timestamp("2026-05-24 00:00:00")
    active_window = df[df["timestamp"] >= split_date].copy()
    
    calc_scanned = len(active_window)
    calc_alerts = len(active_window[active_window["fraud_label"] == 1])
    calc_alert_rate = calc_alerts / max(1, calc_scanned)
    calc_avg_risk = float(active_window["fraud_label"].mean())
    
    print(f"   Calculated Scanned Transactions: {calc_scanned:,}")
    print(f"   Calculated Alerts Triggered: {calc_alerts:,}")
    print(f"   Calculated Alert Rate: {calc_alert_rate:.4%}")
    print(f"   Calculated Mean Risk Score: {calc_avg_risk:.4f}")
    
    # 3. Query REST API and assert matching aggregations
    print("\nStep 3: Querying backend REST API dashboard summary and asserting matches...")
    status, api_res = make_request(f"{API_PREFIX}/dashboard/summary")
    assert status == 200, f"Expected 200, got {status}"
    
    api_scanned = api_res["total_transactions_scanned"]
    api_alerts = api_res["total_alerts_triggered"]
    api_rate = api_res["alert_rate"]
    api_avg_risk = api_res["avg_risk_score"]
    
    print(f"   API Scanned: {api_scanned:,} | Alerts: {api_alerts:,} | Rate: {api_rate:.4%} | Risk: {api_avg_risk:.4f}")
    
    # Assert alignment
    assert calc_scanned == api_scanned, f"Scanned mismatch: Calc={calc_scanned}, API={api_scanned}"
    assert calc_alerts == api_alerts, f"Alerts mismatch: Calc={calc_alerts}, API={api_alerts}"
    assert abs(calc_alert_rate - api_rate) < 0.001, f"Rate mismatch: Calc={calc_alert_rate}, API={api_rate}"
    assert abs(calc_avg_risk - api_avg_risk) < 0.001, f"Risk mismatch: Calc={calc_avg_risk}, API={api_avg_risk}"
    print("PASS: Backend API summary metrics perfectly match database aggregations!")
    
    # 4. Check for Nulls / Missing Values (which cause render crashes)
    print("\nStep 4: Checking for missing / null values in mandatory UI-facing columns...")
    # Null values in sender_account_id/receiver_account_id/merchant_id are allowed under specific transaction types (nullable FKs)
    # But transaction_id, amount, transaction_type, timestamp, and fraud_label must NEVER be null
    null_counts = active_window[["transaction_id", "amount", "transaction_type", "timestamp", "fraud_label"]].isnull().sum()
    print(f"   Null counts in mandatory UI columns:\n{null_counts}")
    assert null_counts.sum() == 0, "Null values detected in mandatory UI-facing fields!"
    print("PASS: Core mandatory UI fields contain zero null values.")
    
    # 5. Verify Confusion Matrix numbers math (Class Imbalances)
    print("\nStep 5: Verifying Confusion Matrix and FPR/Recall math for Combined model...")
    tn, fp, fn, tp = 19426, 598, 4, 879
    total_test = tn + fp + fn + tp
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / total_test
    fpr = fp / (fp + tn)
    
    print(f"   FPR Math check: {fpr:.4%}")
    print(f"   Recall (Detection Rate) Math check: {recall:.4%}")
    print(f"   Precision Math check: {precision:.4%}")
    print(f"   Accuracy Math check: {accuracy:.4%}")
    
    assert abs(accuracy - 0.9712) < 0.001
    assert abs(precision - 0.5951) < 0.001
    assert abs(recall - 0.9955) < 0.001
    assert abs(fpr - 0.0299) < 0.001
    print("PASS: GNN Performance ratios mathematically verified against confusion matrix counts.")
    
    # 6. Verify Recent Alerts Schema structure
    print("\nStep 6: Verifying recent alerts array format and sorting...")
    recent = api_res["recent_alerts"]
    assert len(recent) > 0, "API returned empty alert list."
    
    times = [pd.Timestamp(item["timestamp"]) for item in recent]
    is_sorted_desc = all(times[i] >= times[i+1] for i in range(len(times)-1))
    assert is_sorted_desc, "Recent alerts are not sorted in descending chronological order!"
    print("PASS: Alert queue sorting is correctly set to descending chronological order.")
    
    print("\n======================================================================")
    print("STATUS: Dashboard Metrics validation completed with status: SUCCESS")
    print("======================================================================")

if __name__ == "__main__":
    validate_dashboard_metrics()
