import os
import csv
import random
import uuid
from datetime import datetime, timedelta

# Set random seed for reproducibility
random.seed(42)

# Directory setup
OUTPUT_DIR = "d:/fraud_detection/data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Lists for realistic data generation
FIRST_NAMES = [
    "Oliver", "George", "Noah", "Arthur", "Harry", "Leo", "Muhammad", "Oscar", "Olivia", "Amelia",
    "Isla", "Ava", "Mia", "Ivy", "Lily", "Isabella", "Sophia", "Emily", "James", "John", "Robert",
    "Michael", "William", "David", "Richard", "Joseph", "Thomas", "Charles", "Christopher", "Daniel",
    "Matthew", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan", "Jessica", "Sarah",
    "Karen", "Lisa", "Nancy", "Betty", "Margaret", "Sandra", "Ashley", "Kimberly", "Donna", "Michelle"
]

LAST_NAMES = [
    "Smith", "Jones", "Taylor", "Williams", "Brown", "Davies", "Evans", "Thomas", "Wilson", "Johnson",
    "Roberts", "Robinson", "Thompson", "Wright", "Walker", "White", "Edwards", "Green", "Hall", "Wood",
    "Harris", "Martin", "Jackson", "Clarke", "Clark", "Lewis", "Hill", "Hughes", "Harrison", "Mason",
    "Patterson", "Young", "King", "Scott", "Watson", "Cooper", "Ward", "Baker", "Carter", "Mitchell"
]

OCCUPATIONS = [
    "Software Engineer", "Teacher", "Registered Nurse", "Business Owner", "Student", "Retired",
    "Administrative Assistant", "Sales Associate", "Accountant", "Project Manager", "Electrician",
    "Plumber", "Driver", "Construction Worker", "Unemployed", "Doctor", "Lawyer", "Chef", "Consultant"
]

CITIES = [
    "London", "Birmingham", "Manchester", "Leeds", "Glasgow", "Liverpool", "Newcastle", "Bristol",
    "Edinburgh", "Cardiff", "Belfast", "Sheffield", "Leicester", "Coventry", "Nottingham"
]

MERCHANT_NAMES = {
    "RETAIL": ["Tesco", "Sainsbury's", "ASDA", "Marks & Spencer", "Boots", "Next", "Argos", "Currys"],
    "TRAVEL": ["British Airways", "EasyJet", "National Express", "Trainline", "Airbnb", "Booking.com"],
    "CRYPTO_EXCHANGE": ["Binance", "Coinbase", "Kraken", "Crypto.com", "Luno"],
    "GAMING": ["Bet365", "888 Casino", "PokerStars", "PlayStation Network", "Steam"],
    "E_COMMERCE": ["Amazon", "eBay", "ASOS", "Wayfair", "Etsy", "AliExpress"]
}

DEVICE_OS = {
    "MOBILE": ["iOS 17.4", "Android 14", "iOS 16.5", "Android 13"],
    "LAPTOP": ["Windows 11", "macOS Sonoma", "Windows 10", "macOS Ventura", "Ubuntu 22.04"],
    "TABLET": ["iPadOS 17", "Android 13 Tablet", "iPadOS 16"]
}

# Date Helper Functions
def random_date(start_date, end_date):
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    random_seconds = random.randint(0, 86399)
    return start_date + timedelta(days=random_days, seconds=random_seconds)

def format_timestamp(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S")

# Generate Datasets
def generate_customers(num_customers=10000):
    customers = []
    start_date = datetime(2016, 1, 1)
    end_date = datetime(2026, 1, 1)
    
    for i in range(num_customers):
        cust_id = f"C_{100000 + i}"
        first_name = random.choice(FIRST_NAMES)
        last_name = random.choice(LAST_NAMES)
        
        # Age distribution matching normal banking distributions
        age = int(random.normalvariate(42, 14))
        age = max(18, min(85, age))
        
        occupation = random.choice(OCCUPATIONS)
        if age < 22:
            occupation = "Student" if random.random() < 0.7 else occupation
        elif age > 65:
            occupation = "Retired" if random.random() < 0.9 else occupation
            
        city = random.choice(CITIES)
        
        # Risk category allocation
        r = random.random()
        if r < 0.85:
            risk_category = "LOW"
        elif r < 0.97:
            risk_category = "MEDIUM"
        else:
            risk_category = "HIGH"
            
        created_at = random_date(start_date, end_date)
        
        customers.append({
            "customer_id": cust_id,
            "first_name": first_name,
            "last_name": last_name,
            "age": age,
            "occupation": occupation,
            "city": city,
            "risk_category": risk_category,
            "created_at": format_timestamp(created_at)
        })
        
    return customers

def generate_accounts(customers):
    accounts = []
    account_seq = 1000000
    
    for cust in customers:
        cust_id = cust["customer_id"]
        cust_created = datetime.strptime(cust["created_at"], "%Y-%m-%d %H:%M:%S")
        
        # Determine number of accounts per customer (1 to 3)
        r = random.random()
        if r < 0.60:
            num_accs = 1
        elif r < 0.90:
            num_accs = 2
        else:
            num_accs = 3
            
        for _ in range(num_accs):
            acc_id = f"ACC_{account_seq}"
            account_seq += 1
            
            # Account type distributions
            acc_type = random.choice(["CURRENT", "SAVINGS", "BUSINESS"])
            if cust["occupation"] == "Student" and acc_type == "BUSINESS":
                acc_type = "CURRENT"
            
            # Balances
            if acc_type == "CURRENT":
                balance = round(random.lognormvariate(7.5, 1.0), 2)
                balance = max(10.0, min(50000.0, balance))
            elif acc_type == "SAVINGS":
                balance = round(random.lognormvariate(9.5, 1.2), 2)
                balance = max(50.0, min(250000.0, balance))
            else:  # BUSINESS
                balance = round(random.lognormvariate(11.0, 1.5), 2)
                balance = max(1000.0, min(1500000.0, balance))
                
            status = "ACTIVE" if random.random() < 0.92 else "DORMANT"
            
            # Opening date must be after customer profile creation date
            acc_created = cust_created + timedelta(days=random.randint(0, 30), minutes=random.randint(0, 1440))
            
            accounts.append({
                "account_id": acc_id,
                "customer_id": cust_id,
                "account_type": acc_type,
                "balance": balance,
                "status": status,
                "created_at": format_timestamp(acc_created)
            })
            
    return accounts

def generate_merchants(num_merchants=500):
    merchants = []
    
    for i in range(num_merchants):
        m_id = f"M_{1000 + i}"
        category = random.choice(list(MERCHANT_NAMES.keys()))
        name = f"{random.choice(MERCHANT_NAMES[category])} #{random.randint(10, 999)}"
        
        # Location: online or a UK city
        location = "ONLINE" if random.random() < 0.6 else random.choice(CITIES)
        
        # Risk scores by category
        if category == "CRYPTO_EXCHANGE":
            risk_score = round(random.uniform(0.70, 0.95), 2)
        elif category == "GAMING":
            risk_score = round(random.uniform(0.55, 0.85), 2)
        elif category == "TRAVEL":
            risk_score = round(random.uniform(0.15, 0.45), 2)
        elif category == "E_COMMERCE":
            risk_score = round(random.uniform(0.10, 0.35), 2)
        else:  # RETAIL
            risk_score = round(random.uniform(0.01, 0.15), 2)
            
        merchants.append({
            "merchant_id": m_id,
            "merchant_name": name,
            "merchant_category": category,
            "location": location,
            "risk_score": risk_score
        })
        
    return merchants

def generate_devices(num_devices=2000):
    devices = []
    
    for i in range(num_devices):
        dev_id = f"DEV_{10000 + i}"
        dev_type = random.choice(["MOBILE", "LAPTOP", "TABLET"])
        os_ver = random.choice(DEVICE_OS[dev_type])
        
        # IP address generation
        ip = f"{random.randint(31, 195)}.{random.randint(2, 254)}.{random.randint(2, 254)}.{random.randint(2, 254)}"
        location = random.choice(CITIES)
        
        devices.append({
            "device_id": dev_id,
            "device_type": dev_type,
            "operating_system": os_ver,
            "ip_address": ip,
            "location": location
        })
        
    return devices

def generate_logins_and_beneficiaries(customers, accounts, devices, num_logins=50000):
    logins = []
    beneficiaries = []
    
    # Pre-assign a preferred device for each customer
    cust_device = {}
    for cust in customers:
        cust_device[cust["customer_id"]] = random.choice(devices)["device_id"]
        
    # Generate log activities
    start_date = datetime(2026, 5, 1)
    end_date = datetime(2026, 5, 30)
    
    for i in range(num_logins):
        login_id = f"L_{1000000 + i}"
        cust = random.choice(customers)
        cust_id = cust["customer_id"]
        
        # 90% chance to use preferred device, 10% to use a new device
        if random.random() < 0.90:
            dev_id = cust_device[cust_id]
        else:
            dev_id = random.choice(devices)["device_id"]
            
        timestamp = random_date(start_date, end_date)
        status = "SUCCESS" if random.random() < 0.96 else "FAILED"
        
        # Pick IP from device lookup or random
        matching_devices = [d for d in devices if d["device_id"] == dev_id]
        ip = matching_devices[0]["ip_address"] if matching_devices else "127.0.0.1"
        
        logins.append({
            "login_id": login_id,
            "customer_id": cust_id,
            "device_id": dev_id,
            "timestamp": format_timestamp(timestamp),
            "status": status,
            "ip_address": ip
        })
        
    # Generate beneficiaries (trusted links)
    # Give about 40% of accounts a few trusted beneficiaries
    beneficiary_seq = 100000
    for acc in accounts:
        if random.random() < 0.40:
            num_ben = random.randint(1, 3)
            # Pick other accounts as beneficiaries
            potential_bens = random.sample(accounts, num_ben)
            for ben in potential_bens:
                if ben["account_id"] != acc["account_id"]:
                    ben_id = f"B_{beneficiary_seq}"
                    beneficiary_seq += 1
                    created_at = datetime.strptime(acc["created_at"], "%Y-%m-%d %H:%M:%S") + timedelta(days=random.randint(1, 10))
                    
                    beneficiaries.append({
                        "beneficiary_id": ben_id,
                        "account_id": acc["account_id"],
                        "beneficiary_account_id": ben["account_id"],
                        "created_at": format_timestamp(created_at),
                        "status": "APPROVED" if random.random() < 0.95 else "PENDING"
                    })
                    
    return logins, beneficiaries

def generate_genuine_transactions(accounts, merchants, devices, num_transactions=95000):
    transactions = []
    txn_seq = 50000000
    
    start_date = datetime(2026, 5, 1)
    end_date = datetime(2026, 5, 30)
    
    # Map accounts to customers and customers to preferred devices
    acc_map = {a["account_id"]: a for a in accounts}
    cust_to_accs = {}
    for a in accounts:
        cust_to_accs.setdefault(a["customer_id"], []).append(a["account_id"])
        
    # Assign a preferred device to each account
    acc_device = {}
    for acc_id in acc_map:
        acc_device[acc_id] = random.choice(devices)["device_id"]
        
    for i in range(num_transactions):
        txn_id = f"TXN_{txn_seq + i}"
        timestamp = random_date(start_date, end_date)
        
        # Decide transaction type
        # SALARY (5%), BILL_PAYMENT (10%), TRANSFER (15%), PURCHASE (65%), ATM_WITHDRAWAL (5%)
        r = random.random()
        
        sender = None
        receiver = None
        merchant = None
        device = None
        amount = 0.0
        
        if r < 0.05:
            # SALARY (deposit into account)
            txn_type = "SALARY"
            receiver = random.choice(accounts)["account_id"]
            amount = round(random.uniform(1500.0, 5000.0), 2)
            # Salaries typically come from external, so sender is null
            sender = ""
        elif r < 0.15:
            # BILL_PAYMENT
            txn_type = "BILL_PAYMENT"
            sender = random.choice(accounts)["account_id"]
            amount = round(random.uniform(20.0, 250.0), 2)
            # Paid to a utility or service (can represent as merchant or null receiver)
            merchant = random.choice([m["merchant_id"] for m in merchants if m["merchant_category"] == "RETAIL"])
            device = acc_device[sender]
        elif r < 0.30:
            # TRANSFER (P2P)
            txn_type = "TRANSFER"
            sender = random.choice(accounts)["account_id"]
            
            # Find a receiver that is different and not owned by the same customer
            sender_cust = acc_map[sender]["customer_id"]
            receiver_candidates = [a["account_id"] for a in accounts if a["customer_id"] != sender_cust]
            receiver = random.choice(receiver_candidates)
            
            # Genuine transfers are usually moderate
            amount = round(random.lognormvariate(4.0, 1.2), 2)
            amount = max(5.0, min(2000.0, amount))
            device = acc_device[sender]
        elif r < 0.95:
            # PURCHASE
            txn_type = "PURCHASE"
            sender = random.choice(accounts)["account_id"]
            merchant = random.choice(merchants)["merchant_id"]
            
            # Purchase amounts follow lognormal
            amount = round(random.lognormvariate(3.2, 1.1), 2)
            amount = max(1.5, min(1500.0, amount))
            device = acc_device[sender]
        else:
            # ATM_WITHDRAWAL
            txn_type = "ATM_WITHDRAWAL"
            sender = random.choice(accounts)["account_id"]
            amount = round(random.choice([10, 20, 40, 50, 100, 200, 300]), 2)
            # Device and merchant are null for physical ATM withdrawal (in this schema)
            
        transactions.append({
            "transaction_id": txn_id,
            "sender_account_id": sender,
            "receiver_account_id": receiver,
            "merchant_id": merchant if merchant else "",
            "device_id": device if device else "",
            "amount": amount,
            "transaction_type": txn_type,
            "timestamp": format_timestamp(timestamp),
            "fraud_label": 0
        })
        
    return transactions

def inject_fraud_transactions(transactions, accounts, merchants, devices, num_fraud=5000):
    # Setup lookup variables
    acc_map = {a["account_id"]: a for a in accounts}
    cust_to_accs = {}
    for a in accounts:
        cust_to_accs.setdefault(a["customer_id"], []).append(a["account_id"])
        
    start_date = datetime(2026, 5, 1)
    end_date = datetime(2026, 5, 30)
    
    fraud_txns = []
    txn_seq_start = 60000000
    
    # Track existing device IDs to prevent duplicates in devices table
    existing_device_ids = {d["device_id"] for d in devices}
    
    # 1. Money Mule Networks (1000 txns)
    # Master accounts receive lots of transactions from mules
    num_mules = 50
    mule_accounts = random.sample([a["account_id"] for a in accounts if a["status"] == "ACTIVE"], num_mules)
    master_accounts = random.sample([a["account_id"] for a in accounts if a["status"] == "ACTIVE" and a["account_id"] not in mule_accounts], 5)
    
    for i in range(500): # 500 deposit + 500 forward = 1000 txns
        mule = random.choice(mule_accounts)
        master = random.choice(master_accounts)
        ts_dep = random_date(start_date, end_date - timedelta(hours=1))
        ts_fwd = ts_dep + timedelta(minutes=random.randint(1, 15))
        
        amount = round(random.uniform(3000.0, 9000.0), 2)
        
        # Txn 1: Large deposit into Mule (from external, sender is null)
        dep_txn = {
            "transaction_id": f"TXN_F_{txn_seq_start + len(fraud_txns)}",
            "sender_account_id": "",
            "receiver_account_id": mule,
            "merchant_id": "",
            "device_id": "",
            "amount": amount,
            "transaction_type": "TRANSFER",
            "timestamp": format_timestamp(ts_dep),
            "fraud_label": 1
        }
        fraud_txns.append(dep_txn)
        
        # Txn 2: Forward to Master (leaving 2% fee in mule account)
        fwd_amount = round(amount * 0.97, 2)
        fwd_txn = {
            "transaction_id": f"TXN_F_{txn_seq_start + len(fraud_txns)}",
            "sender_account_id": mule,
            "receiver_account_id": master,
            "merchant_id": "",
            "device_id": random.choice(devices)["device_id"],
            "amount": fwd_amount,
            "transaction_type": "TRANSFER",
            "timestamp": format_timestamp(ts_fwd),
            "fraud_label": 1
        }
        fraud_txns.append(fwd_txn)
        
    # 2. Shared Device Fraud (1000 txns)
    # Single device executes transactions across 10-15 accounts
    shared_devices = random.sample([d["device_id"] for d in devices], 5)
    for dev_id in shared_devices:
        # Pick 12 random accounts to share this device
        sharing_accounts = random.sample([a["account_id"] for a in accounts], 12)
        target_recv = random.choice([a["account_id"] for a in accounts if a["account_id"] not in sharing_accounts])
        
        # Inject 83-84 txns per shared device to reach 1000 in total
        for _ in range(200):
            if len(fraud_txns) >= 2000:
                break
            sender = random.choice(sharing_accounts)
            ts = random_date(start_date, end_date)
            amount = round(random.uniform(500.0, 3000.0), 2)
            
            fraud_txns.append({
                "transaction_id": f"TXN_F_{txn_seq_start + len(fraud_txns)}",
                "sender_account_id": sender,
                "receiver_account_id": target_recv,
                "merchant_id": "",
                "device_id": dev_id,
                "amount": amount,
                "transaction_type": "TRANSFER",
                "timestamp": format_timestamp(ts),
                "fraud_label": 1
            })

    # 3. Account Takeover (ATO) (1000 txns)
    # Dormant or active accounts compromised by new device and drained
    victim_accounts = random.sample([a["account_id"] for a in accounts], 300)
    for victim in victim_accounts:
        if len(fraud_txns) >= 3000:
            break
        # Create a unique new device for this fraudster
        while True:
            hacker_device = f"DEV_H_{random.randint(90000, 99999)}"
            if hacker_device not in existing_device_ids:
                break
        existing_device_ids.add(hacker_device)
        
        # Add to device list to maintain referential integrity
        devices.append({
            "device_id": hacker_device,
            "device_type": random.choice(["MOBILE", "LAPTOP"]),
            "operating_system": random.choice(["Android 14", "Windows 11"]),
            "ip_address": f"{random.randint(196, 223)}.{random.randint(2, 254)}.{random.randint(2, 254)}.{random.randint(2, 254)}",
            "location": random.choice(["Bucharest", "Sofia", "Kiev", "Moscow"])
        })
        ts_compromised = random_date(start_date, end_date)
        
        # Drain balance
        orig_bal = acc_map[victim]["balance"]
        drain_amount = round(orig_bal * random.uniform(0.85, 0.99), 2)
        if drain_amount < 10.0:
            drain_amount = 500.0
            
        receiver = random.choice([a["account_id"] for a in accounts if a["account_id"] != victim])
        
        fraud_txns.append({
            "transaction_id": f"TXN_F_{txn_seq_start + len(fraud_txns)}",
            "sender_account_id": victim,
            "receiver_account_id": receiver,
            "merchant_id": "",
            "device_id": hacker_device,
            "amount": drain_amount,
            "transaction_type": "TRANSFER",
            "timestamp": format_timestamp(ts_compromised),
            "fraud_label": 1
        })
        
    # 4. Synthetic Identity Fraud (500 txns)
    # Newly opened accounts with suspicious velocity
    newly_opened = sorted(accounts, key=lambda x: x["created_at"], reverse=True)[:100]
    for acc in newly_opened:
        if len(fraud_txns) >= 3500:
            break
        acc_created = datetime.strptime(acc["created_at"], "%Y-%m-%d %H:%M:%S")
        
        # Perform 5 fast transactions right after opening
        for j in range(5):
            ts = acc_created + timedelta(hours=random.randint(1, 24), minutes=random.randint(0, 59))
            amount = round(random.uniform(1000.0, 5000.0), 2)
            merchant = random.choice(merchants)["merchant_id"]
            
            fraud_txns.append({
                "transaction_id": f"TXN_F_{txn_seq_start + len(fraud_txns)}",
                "sender_account_id": acc["account_id"],
                "receiver_account_id": "",
                "merchant_id": merchant,
                "device_id": random.choice(devices)["device_id"],
                "amount": amount,
                "transaction_type": "PURCHASE",
                "timestamp": format_timestamp(ts),
                "fraud_label": 1
            })

    # 5. Rapid Transfer Chains (1000 txns)
    # A -> B -> C -> D -> E in seconds/minutes
    for _ in range(200): # 200 chains * 5 txns = 1000 txns
        chain_accs = random.sample([a["account_id"] for a in accounts], 5)
        ts_start = random_date(start_date, end_date - timedelta(minutes=30))
        amount = round(random.uniform(2000.0, 8000.0), 2)
        
        for k in range(4):
            sender = chain_accs[k]
            receiver = chain_accs[k+1]
            ts_step = ts_start + timedelta(seconds=random.randint(10, 180) * (k + 1))
            step_amount = round(amount * (0.98 ** k), 2)
            
            fraud_txns.append({
                "transaction_id": f"TXN_F_{txn_seq_start + len(fraud_txns)}",
                "sender_account_id": sender,
                "receiver_account_id": receiver,
                "merchant_id": "",
                "device_id": random.choice(devices)["device_id"],
                "amount": step_amount,
                "transaction_type": "TRANSFER",
                "timestamp": format_timestamp(ts_step),
                "fraud_label": 1
            })

    # 6. High-Risk Merchant Fraud (500 txns)
    # Large purchases at crypto/gaming in middle of night
    high_risk_merchants = [m["merchant_id"] for m in merchants if m["merchant_category"] in ["CRYPTO_EXCHANGE", "GAMING"]]
    for _ in range(500):
        acc = random.choice(accounts)
        merch = random.choice(high_risk_merchants)
        # Night hour: 01:00 to 04:59
        ts_date = random_date(start_date, end_date)
        ts_night = ts_date.replace(hour=random.choice([1, 2, 3, 4]), minute=random.randint(0, 59), second=random.randint(0, 59))
        amount = round(random.uniform(1500.0, 4500.0), 2)
        
        # Create a unique new device for this fraudster
        while True:
            hacker_device = f"DEV_H_{random.randint(80000, 89999)}"
            if hacker_device not in existing_device_ids:
                break
        existing_device_ids.add(hacker_device)
        
        # Add to device list to maintain referential integrity
        devices.append({
            "device_id": hacker_device,
            "device_type": random.choice(["MOBILE", "LAPTOP"]),
            "operating_system": random.choice(["Android 14", "Windows 11"]),
            "ip_address": f"{random.randint(196, 223)}.{random.randint(2, 254)}.{random.randint(2, 254)}.{random.randint(2, 254)}",
            "location": random.choice(["Bucharest", "Sofia", "Kiev", "Moscow"])
        })
        
        fraud_txns.append({
            "transaction_id": f"TXN_F_{txn_seq_start + len(fraud_txns)}",
            "sender_account_id": acc["account_id"],
            "receiver_account_id": "",
            "merchant_id": merch,
            "device_id": hacker_device,
            "amount": amount,
            "transaction_type": "PURCHASE",
            "timestamp": format_timestamp(ts_night),
            "fraud_label": 1
        })

    # Adjust counts to exactly 5000 if there's minor rounding differences
    if len(fraud_txns) > num_fraud:
        fraud_txns = fraud_txns[:num_fraud]
    elif len(fraud_txns) < num_fraud:
        # Pad with some additional merchant fraud
        diff = num_fraud - len(fraud_txns)
        for _ in range(diff):
            acc = random.choice(accounts)
            merch = random.choice(high_risk_merchants)
            ts = random_date(start_date, end_date)
            amount = round(random.uniform(500.0, 1500.0), 2)
            fraud_txns.append({
                "transaction_id": f"TXN_F_{txn_seq_start + len(fraud_txns)}",
                "sender_account_id": acc["account_id"],
                "receiver_account_id": "",
                "merchant_id": merch,
                "device_id": random.choice(devices)["device_id"],
                "amount": amount,
                "transaction_type": "PURCHASE",
                "timestamp": format_timestamp(ts),
                "fraud_label": 1
            })
            
    # Combine and sort chronologically
    combined = transactions + fraud_txns
    combined.sort(key=lambda x: x["timestamp"])
    
    return combined

def write_csv(filename, fieldnames, data):
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f"Exported: {filepath} ({len(data)} rows)")

def main():
    print("Starting banking dataset simulation...")
    
    # 1. Customers
    customers = generate_customers(num_customers=10000)
    write_csv(
        "customers.csv",
        ["customer_id", "first_name", "last_name", "age", "occupation", "city", "risk_category", "created_at"],
        customers
    )
    
    # 2. Accounts
    accounts = generate_accounts(customers)
    write_csv(
        "accounts.csv",
        ["account_id", "customer_id", "account_type", "balance", "status", "created_at"],
        accounts
    )
    
    # 3. Merchants
    merchants = generate_merchants(num_merchants=500)
    write_csv(
        "merchants.csv",
        ["merchant_id", "merchant_name", "merchant_category", "location", "risk_score"],
        merchants
    )
    
    # 4. Devices (Initial list)
    devices = generate_devices(num_devices=2000)
    
    # Logins and Beneficiaries (extra entities supporting schema)
    logins, beneficiaries = generate_logins_and_beneficiaries(customers, accounts, devices, num_logins=50000)
    write_csv(
        "logins.csv",
        ["login_id", "customer_id", "device_id", "timestamp", "status", "ip_address"],
        logins
    )
    write_csv(
        "beneficiaries.csv",
        ["beneficiary_id", "account_id", "beneficiary_account_id", "created_at", "status"],
        beneficiaries
    )
    
    # 5. Transactions (95,000 genuine)
    genuine_txns = generate_genuine_transactions(accounts, merchants, devices, num_transactions=95000)
    
    # 6. Inject Fraud Transactions (and append hacker devices to the devices list)
    all_transactions = inject_fraud_transactions(genuine_txns, accounts, merchants, devices, num_fraud=5000)
    
    # Now write the completed devices list (which includes hacker devices) and transactions
    write_csv(
        "devices.csv",
        ["device_id", "device_type", "operating_system", "ip_address", "location"],
        devices
    )
    
    write_csv(
        "transactions.csv",
        ["transaction_id", "sender_account_id", "receiver_account_id", "merchant_id", "device_id", "amount", "transaction_type", "timestamp", "fraud_label"],
        all_transactions
    )
    
    print("Dataset generation completed successfully.")

if __name__ == "__main__":
    main()
