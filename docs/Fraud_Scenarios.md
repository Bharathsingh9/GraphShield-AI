# GraphShield AI: Banking Fraud Scenarios & Graph Topology
**Role: Banking Fraud Investigation Expert**
**Document Version: 1.0.0**

This document designs and details the eight core banking fraud scenarios simulated for training and testing **GraphShield AI**. Each scenario describes the real-world pattern, involved actors, graph topological structures, digital indicators, and labeling strategies.

---

## 1. Money Mule Networks

### Description
Illegally obtained funds are transferred to one or more intermediary accounts (mules) who then quickly transfer the funds further or cash out. This fragments and hides the origin of the money.

### Actors Involved
- **Victim / High-risk source**: Source of the stolen funds.
- **Mule Account(s)**: Intermediate accounts (often students or low-income individuals recruited via social media).
- **Master Account / Cash-out**: Final destination, which converts the funds into cash (ATM, crypto exchange, or international transfer).

### Graph Relationships
- Structure: A star-like or tree-like out-and-in network.
- Topology: `Victim Account` $\rightarrow$ `Mule Account A`, `Mule Account B`, `Mule Account C` $\rightarrow$ `Master Account`.
- High degree of connectivity (many-to-one or one-to-many-to-one).

### Indicators
- Sudden high-value incoming transfers into previously dormant or low-activity current accounts.
- Rapid outgoing transfers of matching value within minutes of receipt, leaving a minimal residual balance.
- Multiple mules transferring funds to the same destination account.

### Example Transactions
- **Txn 1**: External wire of £5,000 into Mule Current Account `ACC-MULE-1`.
- **Txn 2**: 5 minutes later, Transfer of £4,950 from `ACC-MULE-1` to `ACC-MASTER-9` (cashing out).

### Fraud Label Strategy
- Label the intermediate transfer edges (`performs`) as `fraud_label = 1`.
- Label the involved mule accounts as high risk.

---

## 2. Account Takeover (ATO)

### Description
An unauthorized party gains control of a legitimate customer's account credentials and drains the funds.

### Actors Involved
- **Legitimate Customer**: Owner of the compromised account.
- **Fraudster**: Attacker controlling the credentials.
- **Compromised Account**: The target account.
- **New Device**: The hacker's phone/computer.

### Graph Relationships
- A new `Device` node attaches to a mature `Customer`/`Account` node.
- The new `Device` executes transactions to previously unrelated `Account` or `Merchant` nodes.
- Short path: `Account` $\rightarrow$ `USES` $\rightarrow$ `New Device` $\rightarrow$ `PERFORMS` $\rightarrow$ `Unrelated/High-risk Account`.

### Indicators
- Login from a new device ID with a different OS and an IP address location that is geographically distant from the customer's home city (impossible travel).
- Login attempts fail several times before success, followed by immediate password/email updates, and then large-value transfers.

### Example Transactions
- **Login Activity**: Failed login from `DEV-LEGIT` at 10:00. Successful login from `DEV-HACK` (IP in Romania) at 10:05.
- **Txn 1**: Transfer of £2,500 from `ACC-COMPROMISED` to `ACC-OUTSIDE-9` at 10:07.

### Fraud Label Strategy
- Label the transaction edges originating from the new device during the takeover window as `fraud_label = 1`.

---

## 3. Synthetic Identity Fraud

### Description
Fraudsters create entirely fake identities using a combination of real (stolen National Insurance numbers) and fake information to open new accounts, build credit, and eventually "bust out" by draining maximum credit/overdrafts.

### Actors Involved
- **Synthetic Customer**: A fabricated entity.
- **Synthetic Account**: Newly opened account.

### Graph Relationships
- Newly registered nodes in the graph with very sparse historic connections.
- Often, multiple synthetic customers share the same phone number, address, or device (`Customer` $\rightarrow$ `USES` $\rightarrow$ `Device` $\leftarrow$ `USES` $\leftarrow$ `Customer 2`).

### Indicators
- No credit history or a sudden thin credit profile.
- High risk score at onboarding.
- Multiple accounts opened within a short period sharing a single device ID or residential IP address.
- Dormant behavior initially (to build credit), followed by a sudden spike in maximum transaction value (overdraft drainage).

### Example Transactions
- **Onboarding**: Customer profile created at 2026-05-01. Account opened with £0 balance.
- **Txn 1**: Direct debit setup.
- **Txn 2**: High-value cash-out purchase of £10,000 (going deep into overdraft) at a crypto exchange merchant.

### Fraud Label Strategy
- Label the final bust-out transactions as `fraud_label = 1`.

---

## 4. Shared Device Fraud

### Description
A single physical device (mobile phone, laptop, or emulator) is used to access and conduct transactions across a large number of distinct customer accounts.

### Actors Involved
- **Fraud Ring Operator**: The central actor.
- **Multiple Compromised/Mule Accounts**: The targets accessed via the device.
- **Shared Device**: The hardware node.

### Graph Relationships
- A single `Device` node acts as a hub, connected via `USES` edges to 10+ different `Account` or `Customer` nodes.
- High degree of bipartite connectivity: `Account 1..N` $\rightarrow$ `USES` $\rightarrow$ `Shared Device`.

### Indicators
- A single hardware device ID logging into multiple accounts within a 24-hour period.
- Accounts are registered to different customer names, ages, and locations, but use the same IP address or device fingerprint.

### Example Transactions
- **Login**: `DEV-SHARED` logs into `ACC-A` (Manchester, age 60), `ACC-B` (London, age 22), and `ACC-C` (Edinburgh, age 45) within 2 hours.
- **Txn**: Transfers from all three accounts to the same merchant or external beneficiary.

### Fraud Label Strategy
- Label all transaction edges executed by the shared device during the overlap period as `fraud_label = 1`.

---

## 5. Rapid Transaction Fraud (Velocity)

### Description
High-frequency automated transfers where money is rapidly funneled through a chain of accounts to evade real-time monitoring rules.

### Actors Involved
- **Source Account**: The origin.
- **Intermediate Accounts**: Chain links.
- **Destination Account**: The endpoint.

### Graph Relationships
- Path structure: `ACC-1` $\rightarrow$ `ACC-2` $\rightarrow$ `ACC-3` $\rightarrow$ `ACC-4` $\rightarrow$ `ACC-5` in a directed acyclic chain.
- Temporal properties: Timestamps along the path are extremely close (seconds or minutes apart).

### Indicators
- Transaction execution occurs in milliseconds or seconds.
- Cumulative amount remains constant or slightly decreases (fee leakage).
- Loop closures (money returns to the source).

### Example Transactions
- **Txn 1**: `ACC-1` $\rightarrow$ `ACC-2` (£1,000) at 12:00:00.
- **Txn 2**: `ACC-2` $\rightarrow$ `ACC-3` (£990) at 12:00:05.
- **Txn 3**: `ACC-3` $\rightarrow$ `ACC-4` (£980) at 12:00:10.

### Fraud Label Strategy
- Label all edges in the high-velocity chain as `fraud_label = 1`.

---

## 6. Merchant Fraud (High-Risk Cash-outs)

### Description
Transactions routed to collusive, high-risk, or fraudulent merchants designed to liquidate stolen credentials or money laundered funds.

### Actors Involved
- **Customer / Compromised Account**: The payer.
- **Collusive Merchant**: The receiver (e.g. fake retail front, high-risk crypto exchange).

### Graph Relationships
- Bipartite connectivity: Many unrelated accounts making unusually large transactions to a single merchant node, especially high-risk category merchants.
- `Account 1..N` $\rightarrow$ `PAID_TO` $\rightarrow$ `High-Risk Merchant`.

### Indicators
- High proportion of high-value transactions relative to the merchant's category baseline.
- Transactions occur outside normal operating hours (e.g. 2:00 AM to 4:00 AM).
- High merchant risk score.

### Example Transactions
- **Txn**: Current account `ACC-552` transfers £4,500 to a newly registered gaming site `M-GAMING-X` at 3:15 AM.

### Fraud Label Strategy
- Label all transactions directed to high-risk merchants that fail geographic or behavior checks as `fraud_label = 1`.

---

## 7. Insider Fraud

### Description
A bank employee abuses their internal system access to modify customer account records, approve fraudulent credit applications, or directly transfer funds.

### Actors Involved
- **Insider / Employee**: The internal actor.
- **Victim Account**: The customer account.
- **Receiver Account**: Destination controlled by the employee/accomplice.

### Graph Relationships
- The transaction originates without a device footprint (`device_id` is null or corresponds to internal server/terminal IDs).
- Unusual edits to account profiles (e.g., changes to email/phone numbers) right before a transaction occurs.

### Indicators
- Transactions executed directly via internal API endpoints.
- Transfers out of high-balance, low-activity accounts (dormant accounts).
- Missing device fingerprints.

### Example Transactions
- **Account Update**: Customer record modified (email updated) by internal operator at 14:00.
- **Txn**: £25,000 transfer from `ACC-DORMANT` to `ACC-INSIDER-RECV` at 14:02.

### Fraud Label Strategy
- Label the transaction as `fraud_label = 1`.

---

## 8. Layering and Money Laundering

### Description
The second stage of money laundering, where complex layers of financial transactions are created to obscure the audit trail and sever the link to the original crime.

### Actors Involved
- **Origin Account**: Source of illicit cash.
- **Layering Accounts**: Multi-layered accounts in different banks/jurisdictions.
- **Beneficiaries**: Final clean entities.

### Graph Relationships
- Dense cyclic loops and complex multi-hop DAGs.
- `Account A` $\rightarrow$ `Account B` $\rightarrow$ `Account C` $\rightarrow$ `Account A`.
- Fan-out followed by fan-in.

### Indicators
- Circular payment flows (money returns to where it started).
- High volume of transactions with close-to-zero net balance changes.
- Influx of foreign exchange transactions or international transfer wires.

### Example Transactions
- **Txn 1**: `ACC-A` $\rightarrow$ `ACC-B` (£2,000) at 10:00.
- **Txn 2**: `ACC-B` $\rightarrow$ `ACC-C` (£2,000) at 10:30.
- **Txn 3**: `ACC-C` $\rightarrow$ `ACC-A` (£2,000) at 11:00.

### Fraud Label Strategy
- Label all edges involved in the loop or layering structure as `fraud_label = 1`.

---

## 4. Visualizing Patterns in a Transaction Graph

The following diagrams show how these scenarios appear to GNN algorithms:

```
1. Money Mule Network:
   [Victim] --(transfer)--> [Mule A] --(transfer)--> [Master Account]
   [Victim] --(transfer)--> [Mule B] --(transfer)--> [Master Account]
   [Victim] --(transfer)--> [Mule C] --(transfer)--> [Master Account]

2. Shared Device Fraud:
   [Account 1] --(uses)--> [Shared Device] <--(uses)-- [Account 2]
   [Account 3] --(uses)--> [Shared Device] <--(uses)-- [Account 4]
   
3. Layering (Circular Flow):
   [Account A] --(transfer)--> [Account B] --(transfer)--> [Account C]
   ^                                                              |
   |__________________________(transfer)__________________________|
```

These structural motifs are captured by the GNN (GraphSAGE) as local neighborhood properties, enabling the model to learn the relational features that distinguish fraud from normal behavior.
