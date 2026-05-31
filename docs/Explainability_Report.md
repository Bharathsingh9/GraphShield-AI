# GraphShield AI: Explainable AI (XAI) Manual for Fraud Investigators
**Document Version:** 2.0.0 (Optimized) | **Generated At:** 2026-05-31 17:13:26

This document provides instructions on how to interpret local and global SHAP (SHapley Additive exPlanations) values calculated by the optimized GraphShield AI GNN model.

## 1. Speed & Architecture Performance
To meet real-time digital banking SLA requirements (<100ms), GraphShield AI uses an **Embedding Caching** architecture:
- GraphSAGE structural node embeddings are pre-computed on the full graph context.
- SHAP analysis is restricted to local edge classifier MLPs, isolating the immediate transaction's behavioral features.
- **Current cohort calculation time (22 transactions):** **0.092 seconds** on CPU.

## 2. Core Concepts: Interpreting SHAP Metrics
Each transaction has 4 local features analyzed by SHAP:
1. **Transaction Amount**: Highlights whether the transaction value deviates from typical baseline levels.
2. **Velocity (last 1H)**: Counts rapid consecutive transfers (often script-driven).
3. **Velocity (last 24H)**: Highlights elevated transaction volumes within 24 hours.
4. **Geo Anomaly**: Flag indicating if the transaction device location differs from customer's profile.

- **Positive SHAP Value ($>0$)**: Contributes to fraud classification. If large, it indicates this feature is a critical trigger.
- **Negative SHAP Value ($<0$)**: Contributes to genuine classification. Indicates a trust stabilizer.
- **Base Score**: The GNN's background risk score determined by the cached structural embeddings (network connections, device sharing, historical balances).

## 3. Case Studies: Local Waterfall Analyses

### Case Study A: Approved Genuine Transaction (ID: TXN_50021209)
- **Sender Account**: `ACC_1007257`
- **Merchant / Receiver**: `M_1464`
- **Value**: £1.50
- **GNN Model Score**: **0.01% Fraud Probability** (Recommendation: **APPROVE**)

**Mitigation Contributions:**
- *Transaction Amount (£)*: -0.0381
- *Velocity (Txns last 1 Hour)*: +0.0000
- *Velocity (Txns last 24 Hours)*: -0.0025
- *Location Mismatch (Geo Anomaly)*: -0.0003

### Case Study B: Denied Fraud Transaction (ID: TXN_F_60002183)
- **Sender Account**: `ACC_1000270`
- **Merchant / Receiver**: `ACC_1010642`
- **Value**: £551162.30
- **GNN Model Score**: **100.00% Fraud Probability** (Recommendation: **DENY / INVESTIGATE**)

**Risk Driver Contributions:**
- **Transaction Amount (£)**: +0.9572
- **Velocity (Txns last 1 Hour)**: +0.0000
- **Velocity (Txns last 24 Hours)**: +0.0103
- **Location Mismatch (Geo Anomaly)**: -0.0004

## 4. Visual Dashboard Assets
- Global Feature Importance chart: [Global_SHAP_Importance.png](file:///d:/fraud_detection/docs/Global_SHAP_Importance.png)
- Genuine Transaction explanation: [Local_SHAP_Genuine.png](file:///d:/fraud_detection/docs/Local_SHAP_Genuine.png)
- Fraudulent Transaction explanation: [Local_SHAP_Fraud.png](file:///d:/fraud_detection/docs/Local_SHAP_Fraud.png)