# GraphShield AI: GNN Model Evaluation Report
**Date Generated:** 2026-05-31T11:24:45

This report evaluates the trained heterogeneous GraphSAGE model on the out-of-time test transactions (timestamps starting May 24th, 2026).

## 1. Summary of Performance Metrics
| Transaction Type | Test Count | Fraud Count | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **P2P Transfers (`performs`)** | 3,869 | 547 | 0.9667 | 0.8129 | 0.9927 | 0.8938 | 0.9975 |
| **Merchant Payments (`paid_to`)** | 17,038 | 336 | 0.9722 | 0.4153 | 1.0000 | 0.5869 | 0.9996 |
| **Combined Total** | 20,907 | 883 | 0.9712 | 0.5951 | 0.9955 | 0.7449 | 0.9985 |

## 2. Key Findings & Analytics Insights
- **Class Imbalance Mitigation**: The weighted loss function (`BCEWithLogitsLoss(pos_weight=...)`) successfully guides the GNN to learn positive fraud signatures despite the 95:5 genuine-to-fraud ratio.
- **Recall Performance**: In banking risk management, high recall (detecting actual fraud) is prioritized over precision to block financial leakage, which is reflected in our robust Recall results.
- **ROC-AUC Value**: The high area-under-curve score shows the model has powerful discriminative capabilities in ranking fraudulent transactions above genuine ones.