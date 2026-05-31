import json
import urllib.request
import urllib.error
import time
import os

BASE_URL = "http://127.0.0.1:8000"
API_PREFIX = "/api/v1"

def make_request(path, data=None, method="GET"):
    url = f"{BASE_URL}{path}"
    req_data = json.dumps(data).encode("utf-8") if data else None
    headers = {"Content-Type": "application/json"} if data else {}
    
    req = urllib.request.Request(url, data=req_data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as response:
            res_body = response.read().decode("utf-8")
            return response.status, json.loads(res_body)
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")
            return e.code, json.loads(err_body)
        except Exception:
            return e.code, {"detail": e.reason}
    except Exception as e:
        return 500, {"detail": str(e)}

def test_root_endpoint():
    print("Testing Root Health Endpoint [/]...")
    status, res = make_request("/")
    assert status == 200, f"Expected 200, got {status}"
    assert res["status"] == "HEALTHY", "Expected status to be HEALTHY"
    print("PASS: Root Endpoint Passed!")

def test_graph_stats():
    print("\nTesting Graph Stats Endpoint [/api/v1/graph/stats]...")
    status, res = make_request(f"{API_PREFIX}/graph/stats")
    assert status == 200, f"Expected 200, got {status}"
    assert "node_counts" in res, "Missing node_counts in response"
    assert "edge_counts" in res, "Missing edge_counts in response"
    assert "density" in res, "Missing density in response"
    print("PASS: Graph Stats Endpoint Passed!")
    print(f"   Node Counts: {res['node_counts']}")
    print(f"   Edge Counts: {res['edge_counts']}")

def test_dashboard_summary():
    print("\nTesting Dashboard Summary Endpoint [/api/v1/dashboard/summary]...")
    status, res = make_request(f"{API_PREFIX}/dashboard/summary")
    assert status == 200, f"Expected 200, got {status}"
    assert "total_transactions_scanned" in res
    assert "total_alerts_triggered" in res
    assert "alert_rate" in res
    assert "recent_alerts" in res
    print("PASS: Dashboard Summary Endpoint Passed!")
    print(f"   Recent Alerts Count: {len(res['recent_alerts'])}")

def test_predict_success():
    print("\nTesting Prediction Predict Success (Genuine P2P Transfer)...")
    payload = {
        "sender_account_id": "ACC_1000001",
        "receiver_account_id": "ACC_1000005",
        "amount": 150.00,
        "device_id": "DEV_10000",
        "txns_last_1h": 1,
        "txns_last_24h": 3,
        "geo_anomaly": 0
    }
    status, res = make_request(f"{API_PREFIX}/prediction/predict", payload, "POST")
    assert status == 200, f"Expected 200, got {status}"
    assert res["fraud_prediction"] == 0, "Expected genuine prediction (0)"
    assert "fraud_probability" in res
    print("PASS: Prediction Predict Success Passed!")
    print(f"   Fraud Probability: {res['fraud_probability']:.4f}")

def test_predict_validation_error():
    print("\nTesting Prediction Request Schema Validation (Negative Amount)...")
    payload = {
        "sender_account_id": "ACC_1000001",
        "amount": -50.00,  # Invalid: must be > 0.0
        "txns_last_1h": 1,
        "txns_last_24h": 3,
        "geo_anomaly": 0
    }
    status, res = make_request(f"{API_PREFIX}/prediction/predict", payload, "POST")
    assert status == 422, f"Expected 422 Unprocessable Entity, got {status}"
    print("PASS: Request Schema Validation Error Handling Passed!")

def test_explain_success():
    print("\nTesting Explainability Endpoint [/api/v1/explainability/explain]...")
    # Fetch recent alert transaction ID from dashboard summary
    d_status, d_res = make_request(f"{API_PREFIX}/dashboard/summary")
    if d_status == 200 and d_res["recent_alerts"]:
        target_txn_id = d_res["recent_alerts"][0]["transaction_id"]
        print(f"   Target Transaction selected: {target_txn_id}")
        
        payload = {"transaction_id": target_txn_id}
        status, res = make_request(f"{API_PREFIX}/explainability/explain", payload, "POST")
        assert status == 200, f"Expected 200, got {status}"
        assert "explanations" in res, "Missing explanations dict in response"
        assert "chart_url" in res, "Missing chart_url in response"
        print("PASS: Explainability Endpoint Success Passed!")
        print(f"   Chart URL: {res['chart_url']}")
    else:
        print("WARN: Skipped explain test: No active alerts found in summary.")

def test_explain_not_found():
    print("\nTesting Explainability Error Handling (Non-existent Transaction ID)...")
    payload = {"transaction_id": "TXN_INVALID_99999"}
    status, res = make_request(f"{API_PREFIX}/explainability/explain", payload, "POST")
    assert status == 404, f"Expected 404 Not Found, got {status}"
    print("PASS: Explainability Error Handling Passed!")
    print(f"   API Message: {res['detail']}")

def test_graph_neighbors_success():
    print("\nTesting Graph Neighbors Success...")
    status, res = make_request(f"{API_PREFIX}/graph/neighbors/account/ACC_1000001")
    assert status == 200, f"Expected 200, got {status}"
    assert "connections" in res
    print("PASS: Graph Neighbors Success Passed!")
    print(f"   Neighbor count: {res['connections_count']}")

def test_graph_neighbors_not_found():
    print("\nTesting Graph Neighbors Error Handling (Invalid Node ID)...")
    status, res = make_request(f"{API_PREFIX}/graph/neighbors/account/ACC_INVALID_NODE")
    assert status == 404, f"Expected 404 Not Found, got {status}"
    print("PASS: Graph Neighbors Error Handling Passed!")
    print(f"   API Message: {res['detail']}")

def test_model_train_trigger():
    print("\nTesting Model Train Trigger [/api/v1/prediction/train]...")
    status, res = make_request(f"{API_PREFIX}/prediction/train", method="POST")
    assert status == 202, f"Expected 202 Accepted, got {status}"
    assert res["status"] == "ACCEPTED"
    assert "job_id" in res
    print("PASS: Model Train Trigger Passed!")
    print(f"   Job ID: {res['job_id']}")

def run_all_qa_validations():
    print("======================================================================")
    print("             GRAPHSHEILD AI - BACKEND API QA VALIDATION SUITE          ")
    print("======================================================================")
    
    start_time = time.time()
    try:
        test_root_endpoint()
        test_graph_stats()
        test_dashboard_summary()
        test_predict_success()
        test_predict_validation_error()
        test_explain_success()
        test_explain_not_found()
        test_graph_neighbors_success()
        test_graph_neighbors_not_found()
        test_model_train_trigger()
        
        print("\n======================================================================")
        print(f"ALL TESTS PASSED SUCCESSFULLY! Total time: {time.time() - start_time:.2f} seconds.")
        print("======================================================================")
    except AssertionError as ae:
        print("\n======================================================================")
        print(f"FAIL: TEST FAILED: {str(ae)}")
        print("======================================================================")

if __name__ == "__main__":
    run_all_qa_validations()
