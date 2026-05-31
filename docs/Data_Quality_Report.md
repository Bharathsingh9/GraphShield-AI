# GraphShield AI: Data Quality Validation Report
**Date Generated:** 2026-05-31 16:44:56

This report summarizes the data quality checks executed by the validation pipeline on the simulated banking datasets.

## 1. Dataset Row Counts
| Dataset Name | Row Count | columns |
| :--- | :---: | :--- |
| customers | 10,000 | customer_id, first_name, last_name, age, occupation, city, risk_category, created_at |
| accounts | 15,053 | account_id, customer_id, account_type, balance, status, created_at |
| merchants | 500 | merchant_id, merchant_name, merchant_category, location, risk_score |
| devices | 2,800 | device_id, device_type, operating_system, ip_address, location |
| transactions | 100,000 | transaction_id, sender_account_id, receiver_account_id, merchant_id, device_id, amount, transaction_type, timestamp, fraud_label |
| logins | 50,000 | login_id, customer_id, device_id, timestamp, status, ip_address |
| beneficiaries | 11,888 | beneficiary_id, account_id, beneficiary_account_id, created_at, status |

## 2. Missing Values (Nulls) Analysis
Checking for unexpected null values across all tables. (Note: nullable foreign keys in transactions are permitted under specific transaction types).
| Table | Field Name | Null Count | Null % | Status |
| :--- | :--- | :---: | :---: | :--- |
| transactions | sender_account_id | 5,276 | 5.28% | ⚠️ Nulls Allowed (Nullable FK) |
| transactions | receiver_account_id | 78,002 | 78.00% | ⚠️ Nulls Allowed (Nullable FK) |
| transactions | merchant_id | 26,759 | 26.76% | ⚠️ Nulls Allowed (Nullable FK) |
| transactions | device_id | 10,037 | 10.04% | ⚠️ Nulls Allowed (Nullable FK) |

## 3. Primary Key Uniqueness Check
| Table | Primary Key Column | Duplicate Count | Status |
| :--- | :--- | :---: | :--- |
| customers | customer_id | 0 | ✅ Clean |
| accounts | account_id | 0 | ✅ Clean |
| merchants | merchant_id | 0 | ✅ Clean |
| devices | device_id | 0 | ✅ Clean |
| transactions | transaction_id | 0 | ✅ Clean |
| logins | login_id | 0 | ✅ Clean |
| beneficiaries | beneficiary_id | 0 | ✅ Clean |

## 4. Referential Integrity (Foreign Key) Check
| Source Table | Foreign Key Field | Reference Table | Reference Key Field | Broken Count | Status |
| :--- | :--- | :--- | :--- | :---: | :--- |
| accounts | customer_id | customers | customer_id | 0 | ✅ Integrity Maintained |
| transactions | sender_account_id | accounts | account_id | 0 | ✅ Integrity Maintained |
| transactions | receiver_account_id | accounts | account_id | 0 | ✅ Integrity Maintained |
| transactions | merchant_id | merchants | merchant_id | 0 | ✅ Integrity Maintained |
| transactions | device_id | devices | device_id | 0 | ✅ Integrity Maintained |
| logins | customer_id | customers | customer_id | 0 | ✅ Integrity Maintained |
| logins | device_id | devices | device_id | 0 | ✅ Integrity Maintained |
| beneficiaries | account_id | accounts | account_id | 0 | ✅ Integrity Maintained |
| beneficiaries | beneficiary_account_id | accounts | account_id | 0 | ✅ Integrity Maintained |

## 5. Domain Boundary and Value Limits

Checking account balances for impossible configurations:
- Savings Accounts with negative balance: 0 rows (Status: ✅ OK)
- Current Accounts with negative balance: 0 rows (Status: ✅ OK)
- Business Accounts with negative balance: 0 rows (Status: ✅ OK)
- Transactions with zero or negative amounts: 0 rows (Status: ✅ OK)

## 6. Time and Relational Consistency Checks
- Accounts opened prior to customer registration: 0 instances (Status: ✅ OK)
- Transactions occurring before sending account was opened: 0 instances (Status: ✅ OK)
- Transactions occurring before receiving account was opened: 0 instances (Status: ✅ OK)
- Logins occurring before customer registration: 0 instances (Status: ✅ OK)

## 7. Overall Pipeline Status
> [!TIP]
> **PASSED**: The entire data suite passes LBG referential integrity, domain boundary, and temporal consistency checks. The data is ready for feature engineering and Graph Neural Network ingestion.