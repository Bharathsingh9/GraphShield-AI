# Advanced Fraud Feature Engineering Documentation

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
