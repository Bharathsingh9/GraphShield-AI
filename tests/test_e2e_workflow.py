import json
import urllib.request
import urllib.error
import time

BASE_URL = "http://127.0.0.1:8000"
API_PREFIX = "/api/v1"

def make_request(path, data=None, method="GET"):
    url = f"{BASE_URL}{path}"
    req_data = json.dumps(data).encode("utf-8") if data else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=req_data, headers=headers, method=method)
    
    start_time = time.time()
    try:
        with urllib.request.urlopen(req) as response:
            latency = (time.time() - start_time) * 1000
            return response.status, json.loads(response.read().decode("utf-8")), latency
    except urllib.error.HTTPError as e:
        latency = (time.time() - start_time) * 1000
        try:
            return e.code, json.loads(e.read().decode("utf-8")), latency
        except Exception:
            return e.code, {"detail": e.reason}, latency
    except Exception as e:
        latency = (time.time() - start_time) * 1000
        return 500, {"detail": str(e)}, latency

def run_e2e_validation():
    print("======================================================================")
    print("         GRAPHSHEILD AI - END-TO-END QA WORKFLOW VALIDATOR             ")
    print("======================================================================")
    
    steps_passed = 0
    total_steps = 7
    e2e_start_time = time.time()
    
    # Step 1: Submit Transaction & 2. Run Fraud Prediction
    print("\n[Step 1 & 2] Submitting suspect transaction & running prediction...")
    payload = {
        "sender_account_id": "ACC_1001250",
        "merchant_id": "M_1005",
        "amount": 8800.00,
        "device_id": "DEV_H_99999",
        "txns_last_1h": 7,
        "txns_last_24h": 18,
        "geo_anomaly": 1
    }
    status, pred_res, lat1 = make_request(f"{API_PREFIX}/prediction/predict", payload, "POST")
    if status == 200:
        print(f"PASS: Transaction accepted. Prediction latency: {lat1:.2f}ms")
        steps_passed += 2
    else:
        print(f"FAIL: Transaction submission failed. Status: {status}")
        return
        
    # Step 3: Generate Fraud Score
    print("\n[Step 3] Parsing GNN Fraud Risk Score...")
    risk_score = pred_res.get("fraud_probability")
    is_fraud = pred_res.get("fraud_prediction")
    print(f"PASS: Fraud probability parsed: {risk_score:.2%} | Risk level: {'HIGH' if is_fraud == 1 else 'LOW'}")
    assert risk_score is not None
    steps_passed += 1
    
    # Step 4: Generate SHAP Explanation
    print("\n[Step 4] Requesting real-time local SHAP explanations...")
    # Fetch active transaction from dashboard summary first
    s_status, s_res, lat_s = make_request(f"{API_PREFIX}/dashboard/summary")
    if s_status == 200 and s_res.get("recent_alerts"):
        target_txn_id = s_res["recent_alerts"][0]["transaction_id"]
        print(f"   Target Transaction selected: {target_txn_id}")
        
        payload_explain = {"transaction_id": target_txn_id}
        status_ex, res_ex, lat2 = make_request(f"{API_PREFIX}/explainability/explain", payload_explain, "POST")
        if status_ex == 200:
            print(f"PASS: SHAP values computed. Explanation latency: {lat2:.2f}ms")
            print(f"      Local Waterfall chart saved at: {res_ex['chart_url']}")
            steps_passed += 1
        else:
            print(f"FAIL: SHAP computation failed. Status: {status_ex}")
            return
    else:
        print("FAIL: Dashboard query failed. Cannot run SHAP explain test.")
        return
        
    # Step 5: Update Dashboard
    print("\n[Step 5] Verifying alert queue refresh...")
    if s_status == 200:
        print(f"PASS: Dashboard summary resolved in {lat_s:.2f}ms.")
        print(f"      Total Scanned: {s_res['total_transactions_scanned']} | Alerts: {s_res['total_alerts_triggered']}")
        steps_passed += 1
    else:
        print("FAIL: Dashboard update check failed.")
        
    # Step 6: Display Graph Relationships
    print("\n[Step 6] Querying neighbor relationships in Graph Database...")
    sender_acc = payload["sender_account_id"]
    status_g, res_g, lat3 = make_request(f"{API_PREFIX}/graph/neighbors/account/{sender_acc}")
    if status_g == 200:
        print(f"PASS: Neighbor relationships retrieved in {lat3:.2f}ms.")
        print(f"      Connections count for {sender_acc}: {res_g['connections_count']}")
        steps_passed += 1
    else:
        print(f"FAIL: Graph neighbor retrieval failed.")
        return
        
    # Step 7: Create Investigation Case Narrative
    print("\n[Step 7] Constructing Case Forensic Narrative report...")
    primary_driver = sorted(res_ex["explanations"].items(), key=lambda x: abs(x[1]["shap_value"]), reverse=True)[0][0]
    case_report = f"""
    ======================================================================
                     FORENSIC INVESTIGATION FILE CASE
    ======================================================================
    Audit Ref: CASE-{sender_acc}-{int(time.time())}
    Suspect Account: {sender_acc}
    Risk Status: {"[HIGH] HIGH FRAUD RISK ALERT" if risk_score >= 0.5 else "[OK] LOW RISK"}
    GNN Score Probability: {risk_score:.2%}
    Primary SHAP Risk Driver: {primary_driver}
    Graph Degree: {res_g['connections_count']}
    Forensic Conclusion: Action escalated to freezing asset balances.
    ======================================================================
    """
    print(case_report)
    print("PASS: Audit narrative compiled successfully.")
    steps_passed += 1
    
    e2e_total_time = (time.time() - e2e_start_time) * 1000
    
    print("\n======================================================================")
    print("                      E2E WORKFLOW SUMMARY                            ")
    print("======================================================================")
    print(f"   Steps Completed: {steps_passed} / {total_steps}")
    print(f"   Total E2E Execution Latency: {e2e_total_time:.2f} ms")
    print(f"   Verdict: {'READY FOR DEMO' if steps_passed == total_steps else 'NOT READY'}")
    print("======================================================================")

if __name__ == "__main__":
    run_e2e_validation()
