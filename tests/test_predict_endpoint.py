import json
import urllib.request
import urllib.error
import time

BASE_URL = "http://127.0.0.1:8000/api/v1/prediction/predict"

def make_request(payload, method="POST"):
    req_data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(BASE_URL, data=req_data, headers=headers, method=method)
    
    start_time = time.time()
    try:
        with urllib.request.urlopen(req) as response:
            latency = (time.time() - start_time) * 1000 # convert to ms
            res_body = response.read().decode("utf-8")
            return response.status, json.loads(res_body), latency
    except urllib.error.HTTPError as e:
        latency = (time.time() - start_time) * 1000
        try:
            err_body = e.read().decode("utf-8")
            return e.code, json.loads(err_body), latency
        except Exception:
            return e.code, {"detail": e.reason}, latency
    except Exception as e:
        latency = (time.time() - start_time) * 1000
        return 500, {"detail": str(e)}, latency

def test_predict_endpoint():
    print("======================================================================")
    print("        GRAPHSHEILD AI - /predict ENDPOINT QA TEST SUITE              ")
    print("======================================================================")
    
    latencies = []
    
    # 1. Valid Genuine Payload (P2P Transfer)
    print("\n[TEST 1] Sending Valid Genuine P2P Transfer Payload...")
    payload_genuine_p2p = {
        "sender_account_id": "ACC_1000001",
        "receiver_account_id": "ACC_1000005",
        "amount": 100.00,
        "device_id": "DEV_10000",
        "txns_last_1h": 0,
        "txns_last_24h": 1,
        "geo_anomaly": 0
    }
    status, res, lat = make_request(payload_genuine_p2p)
    latencies.append(lat)
    print(f"Status: {status} | Latency: {lat:.2f}ms")
    print(f"Response: {res}")
    assert status == 200, f"Expected 200, got {status}"
    assert "transaction_type" in res
    assert "fraud_probability" in res
    assert "fraud_prediction" in res
    assert "recommendation" in res
    assert res["fraud_prediction"] == 0, "Expected genuine prediction (0)"
    
    # 2. Valid Fraudulent Payload (High-Risk Merchant Payment)
    print("\n[TEST 2] Sending Fraudulent-like Merchant Payment Payload...")
    payload_fraud_merchant = {
        "sender_account_id": "ACC_1001250",
        "merchant_id": "M_1005",
        "amount": 9500.00,  # High amount
        "device_id": "DEV_H_99999",
        "txns_last_1h": 8,  # High velocity
        "txns_last_24h": 22,
        "geo_anomaly": 1  # Location anomaly
    }
    status, res, lat = make_request(payload_fraud_merchant)
    latencies.append(lat)
    print(f"Status: {status} | Latency: {lat:.2f}ms")
    print(f"Response: {res}")
    assert status == 200, f"Expected 200, got {status}"
    assert res["fraud_prediction"] == 1, "Expected fraud prediction (1)"
    assert res["fraud_probability"] > 0.8, f"Expected high fraud probability, got {res['fraud_probability']}"

    # 3. Invalid Payload: Missing Required Fields
    print("\n[TEST 3] Sending Invalid Payload (Missing sender_account_id)...")
    payload_missing_field = {
        "amount": 150.00,
        "txns_last_1h": 0
    }
    status, res, lat = make_request(payload_missing_field)
    latencies.append(lat)
    print(f"Status: {status} | Latency: {lat:.2f}ms")
    print(f"Response: {res}")
    assert status == 422, f"Expected 422, got {status}"
    
    # 4. Invalid Payload: Negative Amount Bounds
    print("\n[TEST 4] Sending Invalid Payload (Negative Amount)...")
    payload_neg_amount = {
        "sender_account_id": "ACC_1000001",
        "amount": -20.00,
        "txns_last_1h": 0,
        "txns_last_24h": 0,
        "geo_anomaly": 0
    }
    status, res, lat = make_request(payload_neg_amount)
    latencies.append(lat)
    print(f"Status: {status} | Latency: {lat:.2f}ms")
    print(f"Response: {res}")
    assert status == 422, f"Expected 422, got {status}"
    
    # 5. Invalid Payload: Data Type Mismatch (String instead of Int)
    print("\n[TEST 5] Sending Invalid Payload (Data Type Mismatch: geo_anomaly as string)...")
    payload_type_mismatch = {
        "sender_account_id": "ACC_1000001",
        "amount": 100.00,
        "txns_last_1h": 0,
        "txns_last_24h": 0,
        "geo_anomaly": "HIGH"  # Should be 0 or 1
    }
    status, res, lat = make_request(payload_type_mismatch)
    latencies.append(lat)
    print(f"Status: {status} | Latency: {lat:.2f}ms")
    print(f"Response: {res}")
    assert status == 422, f"Expected 422, got {status}"
    
    # Print statistics
    avg_lat = sum(latencies) / len(latencies)
    max_lat = max(latencies)
    print("\n======================================================================")
    print("                     API LATENCY STATISTICS                           ")
    print("======================================================================")
    print(f"   Number of Calls: {len(latencies)}")
    print(f"   Average Latency: {avg_lat:.2f} ms")
    print(f"   Maximum Latency: {max_lat:.2f} ms")
    print("======================================================================")
    print("PASS: /predict endpoint validation completed successfully.")
    print("======================================================================")

if __name__ == "__main__":
    test_predict_endpoint()
