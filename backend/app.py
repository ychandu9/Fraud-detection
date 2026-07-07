from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle, os, json, threading
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import os
import joblib


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

MODEL_PATH    = os.path.join(BASE_DIR, "fraud_model.pkl")
SCALER_PATH   = os.path.join(BASE_DIR, "scaler.pkl")
HISTORY_PATH  = os.path.join(BASE_DIR, "transaction_history.json")
METRICS_PATH  = os.path.join(BASE_DIR, "metrics.json")
FEATURES_PATH = os.path.join(BASE_DIR, "features.json")
RETRAIN_EVERY = 10   # auto-retrain every N new transactions

# ── Twilio SMS ─────────────────────────────────────────────
import os

TWILIO_SID   = os.getenv("TWILIO_SID")
TWILIO_TOKEN = os.getenv("TWILIO_TOKEN")
TWILIO_FROM  = os.getenv("TWILIO_FROM")
TWILIO_TO    = os.getenv("TWILIO_TO")           # ← your mobile number

def send_sms_alert(result, data):
    if not TWILIO_SID or not TWILIO_TOKEN:
        return
    try:
        from twilio.rest import Client
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        amount = float(data.get('amount', 0))
        name   = data.get('cardholder_name', 'User')
        txn_id = data.get('transaction_id', 'N/A')
        bank   = data.get('bank_name', 'N/A')
        if result['is_fraud']:
            msg = (f"FRAUD ALERT - FraudGuard AI\n"
                   f"Name: {name}\nBank: {bank}\nTXN: {txn_id}\n"
                   f"Amount: Rs.{amount:,.0f}\nRisk: {result['risk_level']}\n"
                   f"Score: {result['fraud_probability']}%\n"
                   f"ACTION: Call bank immediately!\nHelpline: 1930")
        else:
            msg = (f"Transaction Safe - FraudGuard AI\n"
                   f"Name: {name}\nTXN: {txn_id}\n"
                   f"Amount: Rs.{amount:,.0f}\nScore: {result['safe_probability']}% safe")
        client.messages.create(body=msg, from_=TWILIO_FROM, to=TWILIO_TO)
        print(f"✅ SMS sent!")
    except Exception as e:
        print(f"⚠️  SMS failed: {e}")


# ══════════════════════════════════════════════════════════════
#  KAGGLE DATA LOADER
# ══════════════════════════════════════════════════════════════
def load_kaggle_data():
    csv_path = os.path.join(BASE_DIR, "creditcard.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "\n\n❌  creditcard.csv NOT FOUND!\n"
            "📥  Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "📂  Place creditcard.csv in the backend/ folder next to app.py\n")
    print("▶  Loading Kaggle creditcard.csv ...")
    df = pd.read_csv(csv_path)
    print(f"   Rows: {len(df):,}  |  Fraud: {int((df.Class==1).sum()):,}  |  Rate: {df.Class.mean()*100:.4f}%")
    return df


# ══════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame, scaler=None, fit=False):
    from sklearn.preprocessing import RobustScaler
    df = df.copy()
    if fit:
        scaler = RobustScaler()
        df[['Amount', 'Time']] = scaler.fit_transform(df[['Amount', 'Time']])
    else:
        df[['Amount', 'Time']] = scaler.transform(df[['Amount', 'Time']])
    df['V1_V2']    = df['V1']  * df['V2']
    df['V3_V4']    = df['V3']  * df['V4']
    df['V5_V6']    = df['V5']  * df['V6']
    df['V14_sq']   = df['V14'] ** 2
    df['V4_sq']    = df['V4']  ** 2
    df['V11_sq']   = df['V11'] ** 2
    df['V17_sq']   = df['V17'] ** 2
    df['amt_V14']  = df['Amount'] * df['V14']
    df['amt_V4']   = df['Amount'] * df['V4']
    df['key_risk'] = (df['V14'].abs() + df['V4'].abs() +
                      df['V11'].abs() + df['V12'].abs()) / 4
    return df, scaler


# ══════════════════════════════════════════════════════════════
#  SMOTE
# ══════════════════════════════════════════════════════════════
def smote_numpy(X, y, k=5, random_state=42):
    rng        = np.random.default_rng(random_state)
    minority_X = X[y == 1]
    n_needed   = int((y == 0).sum()) - int((y == 1).sum())
    if n_needed <= 0:
        return X, y
    print(f"▶  SMOTE: generating {n_needed:,} synthetic fraud samples ...")
    synthetic = []
    for _ in range(n_needed):
        idx       = rng.integers(0, len(minority_X))
        sample    = minority_X[idx]
        nn_idxs   = rng.choice(len(minority_X), size=min(k, len(minority_X)), replace=False)
        neighbour = minority_X[rng.choice(nn_idxs)]
        synthetic.append(sample + rng.random() * (neighbour - sample))
    X_bal = np.vstack([X, np.array(synthetic)])
    y_bal = np.concatenate([y, np.ones(n_needed, dtype=int)])
    perm  = rng.permutation(len(y_bal))
    return X_bal[perm], y_bal[perm]


# ══════════════════════════════════════════════════════════════
#  TRAIN MODEL
# ══════════════════════════════════════════════════════════════
def train_model(silent=False):
    from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                                 ExtraTreesClassifier, VotingClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                                precision_score, recall_score, accuracy_score,
                                confusion_matrix, classification_report)

    if not silent:
        print("\n" + "═"*58)
        print("  FraudGuard AI  –  Training on Kaggle Dataset")
        print("═"*58)

    df = load_kaggle_data()

    # ✅ Feature engineering
    X_df, scaler = engineer_features(df.drop('Class', axis=1), fit=True)
    y = df['Class'].values.astype(int)

    # ✅ SAVE FEATURE NAMES (CRITICAL FIX)
    feature_columns = X_df.columns.tolist()

    X = np.array(X_df, dtype=np.float32)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    if not silent:
        print(f"▶  Train: {len(X_tr):,}  |  Test: {len(X_te):,}")

    # SMOTE
    X_bal, y_bal = smote_numpy(X_tr, y_tr, k=5)

    if not silent:
        print("▶  Training Ensemble (RF + ET + GB + LR) ...")

    rf = RandomForestClassifier(n_estimators=50, max_depth=10, max_features='sqrt',
                                class_weight='balanced', random_state=42, n_jobs=-1)

    et = ExtraTreesClassifier(n_estimators=30, max_depth=10, max_features='sqrt',
                              class_weight='balanced', random_state=42, n_jobs=-1)

    gb = GradientBoostingClassifier(n_estimators=40, learning_rate=0.1,
                                    max_depth=4, subsample=0.8,
                                    min_samples_leaf=20, random_state=42)

    lr = LogisticRegression(C=0.1, max_iter=1000,
                            class_weight='balanced', random_state=42)

    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('et', et), ('gb', gb), ('lr', lr)],
        voting='soft', weights=[3, 2, 3, 1]
    )

    ensemble.fit(X_bal, y_bal)

    # Threshold tuning
    y_proba = ensemble.predict_proba(X_te)[:, 1]
    best_f1, best_thr = 0, 0.5

    for thr in np.arange(0.20, 0.80, 0.01):
        preds = (y_proba >= thr).astype(int)
        f = f1_score(y_te, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_thr = f, thr

    y_pred = (y_proba >= best_thr).astype(int)
    cm = confusion_matrix(y_te, y_pred).tolist()
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    metrics = {
        "accuracy": round(accuracy_score(y_te, y_pred) * 100, 4),
        "auc_roc": round(roc_auc_score(y_te, y_proba) * 100, 4),
        "f1_fraud": round(f1_score(y_te, y_pred, zero_division=0) * 100, 4),
        "threshold": round(best_thr, 2),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "last_trained": datetime.now().isoformat(),
    }

    # ✅ SAVE EVERYTHING
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump((ensemble, best_thr), f)

    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    with open(FEATURES_PATH, "w") as f:
        json.dump(feature_columns, f)

    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f)

    print("✅ Model + Features saved")

    return metrics


# ══════════════════════════════════════════════════════════════
#  FEATURE 7 — AUTO RETRAIN (background thread)
# ══════════════════════════════════════════════════════════════
retrain_lock = threading.Lock()

def auto_retrain_if_needed():
    """Retrain in background every RETRAIN_EVERY transactions."""
    global model, threshold, scaler, model_metrics
    h = load_history()
    if len(h) > 0 and len(h) % RETRAIN_EVERY == 0:
        print(f"\n🔄 AUTO-RETRAIN triggered at {len(h)} transactions...")
        def retrain_bg():
            global model, threshold, scaler, model_metrics
            with retrain_lock:
                try:
                    m = train_model(silent=True)
                    with open(MODEL_PATH,  'rb') as f: model, threshold = pickle.load(f)
                    with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
                    model_metrics = m
                    print(f"✅ Auto-retrain complete. New AUC: {m['auc_roc']}%")
                except Exception as e:
                    print(f"⚠️  Auto-retrain failed: {e}")
        threading.Thread(target=retrain_bg, daemon=True).start()


# ══════════════════════════════════════════════════════════════
#  STARTUP
# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
#  STARTUP (FORCE TRAIN — TEMP FIX)
# ══════════════════════════════════════════════════════════════
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    print("✅ Loading saved model...")
    with open(MODEL_PATH, 'rb') as f:
        model, threshold = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    model_metrics = json.load(open(METRICS_PATH)) if os.path.exists(METRICS_PATH) else {}
else:
    print("🔥 FIRST TIME TRAINING...")
    try:
        model_metrics = train_model()
        with open(MODEL_PATH, 'rb') as f:
            model, threshold = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError as e:
        print(f"⚠️ Could not train model on startup: {e}")
        print("⚠️ Running in DEGRADED mode without ML model. Fallback rule-based estimation will be used.")
        model = None
        threshold = 0.5
        scaler = None
        model_metrics = {"error": "Model not trained. Dataset missing."}



# ══════════════════════════════════════════════════════════════
#  TRANSACTION ID PATTERN ANALYSER
#  Extracts real risk signals from the transaction reference ID
# ══════════════════════════════════════════════════════════════
def analyse_transaction_id(txn_id: str, payment_mode: str, sender_bank: str = "", receiver_bank: str = "") -> dict:
    """
    Parse and validate transaction ID format against real Indian banking patterns.
    Returns a dict with risk_score (0-1), flags list, and validity info.
    """
    import re

    if not txn_id:
        return {"risk": 0.0, "flags": [], "valid": False, "reason": "No transaction ID"}

    txn  = txn_id.strip().upper()
    flags = []
    risk  = 0.0

    # ── 1. LENGTH CHECK ────────────────────────────────────────
    if len(txn) < 6:
        flags.append("Transaction ID too short — likely fake or test")
        risk += 0.25
    elif len(txn) > 35:
        flags.append("Transaction ID unusually long")
        risk += 0.10

    # ── 2. SUSPICIOUS PATTERNS ────────────────────────────────
    digits_only = re.sub(r"\D", "", txn)

    # All same digits: 111111, 999999
    if len(digits_only) >= 4 and len(set(digits_only)) == 1:
        flags.append("Transaction ID contains all identical digits — suspicious pattern")
        risk += 0.35

    # Sequential ascending: 123456
    if len(digits_only) >= 4:
        is_seq = all(int(digits_only[i])+1 == int(digits_only[i+1]) for i in range(len(digits_only)-1))
        if is_seq:
            flags.append("Transaction ID is sequential (e.g. 123456) — test/fake transaction")
            risk += 0.30

    # Sequential descending: 987654
    if len(digits_only) >= 4:
        is_desc = all(int(digits_only[i])-1 == int(digits_only[i+1]) for i in range(len(digits_only)-1))
        if is_desc:
            flags.append("Transaction ID is reverse sequential — suspicious pattern")
            risk += 0.30

    # All zeros
    if digits_only and all(c == "0" for c in digits_only):
        flags.append("Transaction ID contains all zeros — invalid")
        risk += 0.40

    # Common fake IDs
    fake_patterns = ["TEST", "FAKE", "FRAUD", "DUMMY", "SAMPLE", "DEMO", "ABCD", "XXXX", "NULL", "NONE"]
    for fp in fake_patterns:
        if fp in txn:
            flags.append(f"Transaction ID contains suspicious keyword: '{fp}'")
            risk += 0.45
            break

    # ── 3. FORMAT VALIDATION BY PAYMENT MODE ──────────────────
    mode = payment_mode.upper() if payment_mode else ""

    if "UPI" in mode:
        # Valid UPI ref: 12-16 digits OR bank/UPI ref format
        upi_valid = bool(re.match(r"^\d{12,16}$", txn)) or                     bool(re.match(r"^[A-Z]{2,6}\d{8,20}$", txn)) or                     bool(re.match(r"^\d{3,6}[A-Z]{2,6}\d{6,15}$", txn))
        if not upi_valid:
            flags.append("UPI reference ID format is invalid — does not match standard UPI pattern")
            risk += 0.20

    elif "NEFT" in mode:
        # Valid NEFT: Bank IFSC prefix (4 letters) + alphanumeric
        neft_valid = bool(re.match(r"^[A-Z]{4}\d{7,20}$", txn))
        if not neft_valid:
            flags.append("NEFT reference format invalid — expected BANKCODE + digits (e.g. SBIN0261234567)")
            risk += 0.15
        else:
            # Extract bank code and cross-check with sender bank
            bank_code = txn[:4]
            known_banks = {
                "SBIN": "State Bank of India", "HDFC": "HDFC Bank",
                "ICIC": "ICICI Bank",          "UTIB": "Axis Bank",
                "KKBK": "Kotak Mahindra Bank", "PUNB": "Punjab National Bank",
                "BARB": "Bank of Baroda",      "CNRB": "Canara Bank",
                "UBIN": "Union Bank of India",  "INDB": "IndusInd Bank",
                "YESB": "Yes Bank",            "IDFB": "IDFC First Bank",
            }
            if bank_code not in known_banks:
                flags.append(f"Unknown bank code '{bank_code}' in NEFT reference")
                risk += 0.12

    elif "IMPS" in mode:
        # Valid IMPS RRN: 12 digits
        imps_valid = bool(re.match(r"^\d{12}$", txn))
        if not imps_valid:
            flags.append("IMPS RRN format invalid — must be exactly 12 digits")
            risk += 0.15

    elif "RTGS" in mode:
        # Valid RTGS: Bank code + date + sequence
        rtgs_valid = bool(re.match(r"^[A-Z]{4}\d{14,20}$", txn))
        if not rtgs_valid:
            flags.append("RTGS UTR format invalid — expected BANKCODE + 14+ digits")
            risk += 0.15

    elif "CREDIT" in mode or "DEBIT" in mode:
        # Card transactions — should not have a user-provided TXN ID normally
        # Check if it looks like a card number (16 digits)
        if re.match(r"^\d{16}$", txn):
            flags.append("Card number entered as Transaction ID — security risk!")
            risk += 0.30

    # ── 4. ROUND AMOUNT TRICK ─────────────────────────────────
    # Fraudsters often use amounts just under alert thresholds
    # We check the TXN ID timestamp if encoded
    if re.search(r"(9999|49999|99999|199999)", txn):
        flags.append("Transaction ID contains suspicious amount pattern (just-under threshold)")
        risk += 0.20

    # ── 5. REPEATED BLOCK PATTERNS ────────────────────────────
    # e.g. 123123123, ABCABC
    if len(txn) >= 6:
        half = len(txn) // 2
        if txn[:half] == txn[half:half*2]:
            flags.append("Transaction ID has repeated block pattern — unusual")
            risk += 0.20

    # ── 6. SENDER / RECEIVER BANK MISMATCH IN ID ──────────────
    bank_code_map = {
        "State Bank of India": "SBIN", "HDFC Bank": "HDFC",
        "ICICI Bank": "ICIC",          "Axis Bank": "UTIB",
        "Kotak Mahindra Bank": "KKBK", "Punjab National Bank": "PUNB",
    }
    if "NEFT" in mode and sender_bank in bank_code_map:
        expected = bank_code_map[sender_bank]
        if not txn.startswith(expected):
            flags.append(f"NEFT reference bank code does not match sender bank ({sender_bank})")
            risk += 0.18

    return {
        "risk"  : min(risk, 1.0),
        "flags" : flags,
        "valid" : risk < 0.2,
        "txn_id": txn_id,
    }

# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════
def load_history():
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH) as f: return json.load(f)
    return []

def save_history(h):
    with open(HISTORY_PATH, 'w') as f: json.dump(h[-500:], f)

def build_feature_vector(data):
    amount = float(data.get('amount', 100))
    hour   = float(data.get('hour', 12))
    time_s = hour * 3600

    loc_mm         = int(data.get('location_mismatch', False))
    new_m          = int(data.get('new_merchant', False))
    hi_freq        = int(data.get('high_frequency', False))
    intl           = int(data.get('international', False))
    no_card        = int(not data.get('card_present', True))
    online         = int(data.get('online_transaction', False))
    otp_shared     = int(data.get('otp_shared', False))
    suspicious_lnk = int(data.get('suspicious_link', False))
    first_large    = int(data.get('first_time_large_amt', False))
    no_beneficiary = int(not data.get('saved_beneficiary', True))
    unknown_device = int(data.get('device_type', '') == 'unknown')

    cat = data.get('merchant_category', 'retail')
    cat_risk = {
        'gambling': 4.5, 'crypto': 4.0, 'travel': 1.0,
        'electronics': 1.5, 'food': 0.2, 'retail': 0.1, 'other': 1.0
    }.get(cat, 0.5)

    risk = (
        loc_mm*3.5 + new_m*2.5 + hi_freq*3.0 + intl*2.5 +
        no_card*3.0 + online*1.5 + cat_risk +
        otp_shared*6.0 + suspicious_lnk*5.5 +
        first_large*2.5 + no_beneficiary*2.0 + unknown_device*2.5
    )

    if amount > 50000:
        risk += 3.0
    elif amount > 10000:
        risk += 1.5
    elif amount > 2000:
        risk += 0.8

    if hour < 5 or hour > 22:
        risk += 2.0

    np.random.seed(int(amount * 100) % 99991)
    noise_scale = max(0.05, 0.3 - risk * 0.01)

    base = np.array([
        -1.36 - risk*0.8,  -0.07 + risk*0.5,   2.54 - risk*0.7,
         1.38 - risk*0.4,  -0.34 - risk*0.6,   0.46 + risk*0.4,
         0.24 - risk*0.7,   0.10 + risk*0.5,   0.36 - risk*0.4,
         0.09 + risk*0.2,  -0.55 - risk*0.6,  -0.62 - risk*0.5,
        -0.99 + risk*0.2,  -0.31 - risk*0.9,   1.47 - risk*0.4,
        -0.47 + risk*0.2,   0.21 - risk*0.5,   0.03 + risk*0.3,
         0.40 - risk*0.2,   0.25 + risk*0.2,  -0.02 - risk*0.4,
         0.28 + risk*0.2,  -0.11 - risk*0.2,   0.07 + risk*0.2,
         0.13 - risk*0.2,  -0.19 + risk*0.2,   0.13 - risk*0.2,
        -0.02 + risk*0.1
    ]) + np.random.randn(28) * noise_scale

    # ✅ Step 1: create base dataframe
    row = pd.DataFrame(
        [[time_s] + list(base) + [amount]],
        columns=['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    )

    # ✅ Step 2: feature engineering
    eng, _ = engineer_features(row, scaler=scaler, fit=False)

    # ✅ Step 3: LOAD TRAIN FEATURE ORDER (CRITICAL FIX)
    import json
    with open(FEATURES_PATH) as f:
        feature_columns = json.load(f)

    # ✅ Step 4: FORCE SAME STRUCTURE (MOST IMPORTANT LINE)
    eng = eng.reindex(columns=feature_columns, fill_value=0)

    return np.array(eng, dtype=np.float32)


# ══════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        # ── Analyse Transaction ID ────────────────────────────
        txn_analysis = analyse_transaction_id(
            txn_id       = data.get('transaction_id', ''),
            payment_mode = data.get('payment_mode', ''),
            sender_bank  = data.get('bank_name', ''),
            receiver_bank= data.get('receiver_bank', ''),
        )

        if model is not None:
            features = build_feature_vector(data)
            proba    = model.predict_proba(features)[0]
            ml_score = float(proba[1])
        else:
            # Fallback when ML model is not available
            amount = float(data.get('amount', 0))
            score = 0.05
            if amount > 50000: score += 0.20
            elif amount > 10000: score += 0.10
            if data.get('otp_shared'): score += 0.35
            if data.get('suspicious_link'): score += 0.30
            if data.get('location_mismatch'): score += 0.15
            ml_score = min(score, 0.95)

        # Rule-based boost
        rule_boost = 0.0
        if data.get('otp_shared'):                                                   rule_boost += 0.12
        if data.get('suspicious_link'):                                              rule_boost += 0.10
        if not data.get('saved_beneficiary', True) and data.get('new_merchant'):     rule_boost += 0.06
        if data.get('high_frequency') and data.get('online_transaction'):            rule_boost += 0.05
        if data.get('international') and not data.get('card_present', True):         rule_boost += 0.06
        if float(data.get('amount', 0)) > 50000:                                     rule_boost += 0.05
        if int(data.get('hour', 12)) < 5:                                            rule_boost += 0.04
        if data.get('device_type','') == 'unknown' and data.get('otp_shared'):       rule_boost += 0.05
        cat = data.get('merchant_category','')
        if cat in ['gambling','crypto'] and data.get('online_transaction'):          rule_boost += 0.05

        # ── Transaction ID risk boost ─────────────────────────
        txn_risk_boost = txn_analysis['risk'] * 0.20   # max +20% from TXN ID
        rule_boost += txn_risk_boost

        fraud_p  = min(ml_score + rule_boost, 0.99)
        is_fraud = fraud_p >= 0.50  # lower threshold

        risk_level = ("CRITICAL" if fraud_p >= 0.80 else
              "HIGH"     if fraud_p >= 0.60 else
              "MEDIUM"   if fraud_p >= 0.40 else "LOW")

        risk_factors = []
        if data.get('otp_shared'):                          risk_factors.append("🚨 CRITICAL: OTP/PIN shared with third party")
        if data.get('suspicious_link'):                     risk_factors.append("🚨 CRITICAL: Initiated via suspicious link/QR code")
        if data.get('location_mismatch'):                   risk_factors.append("Transaction from unusual/mismatched location")
        if data.get('new_merchant'):                        risk_factors.append("Payment to new / unknown merchant")
        if data.get('high_frequency'):                      risk_factors.append("Abnormally high transaction frequency")
        if data.get('international'):                       risk_factors.append("Cross-border international transaction")
        if not data.get('card_present', True):              risk_factors.append("Card not physically present (CNP transaction)")
        if data.get('first_time_large_amt'):                risk_factors.append("First-time large amount transfer detected")
        if not data.get('saved_beneficiary', True):         risk_factors.append("Receiver is not a saved beneficiary")
        if float(data.get('amount', 0)) > 2000:             risk_factors.append(f"High amount: ₹{float(data.get('amount',0)):,.0f}")
        hour = int(data.get('hour', 12))
        if hour < 5 or hour > 22:                           risk_factors.append(f"Unusual transaction hour: {hour}:00")
        if cat in ['gambling', 'crypto']:                   risk_factors.append(f"High-risk merchant category: {cat.title()}")
        if data.get('device_type','') == 'unknown':         risk_factors.append("Transaction from unknown/unrecognised device")

        # ── Add TXN ID flags to risk factors ──────────────────
        for flag in txn_analysis['flags']:
            risk_factors.append(f"🔍 TXN ID: {flag}")

        result = {
            "is_fraud"         : bool(is_fraud),
            "fraud_probability": round(fraud_p * 100, 2),
            "safe_probability" : round((1 - fraud_p) * 100, 2),
            "risk_level"       : risk_level,
            "risk_factors"     : risk_factors,
            "transaction_id"   : data.get("transaction_id") or f"TXN{datetime.now().strftime('%Y%m%d%H%M%S%f')[:18]}",
            "timestamp"        : datetime.now().isoformat(),
            "threshold_used"   : round(threshold, 2),
            "ml_score"         : round(ml_score * 100, 2),
            "rule_boost"       : round(rule_boost * 100, 2),
            "txn_id_risk"      : round(txn_analysis['risk'] * 100, 2),
            "txn_id_valid"     : txn_analysis['valid'],
            "txn_id_flags"     : txn_analysis['flags'],
        }

        history = load_history()
        history.append({**result,
            "amount"          : data.get('amount'),
            "transaction_type": data.get('transaction_type'),
            "merchant_category":data.get('merchant_category'),
            "bank_name"       : data.get('bank_name'),
            "sender_account"  : data.get('sender_account'),
            "receiver_account": data.get('receiver_account'),
            "receiver_name"   : data.get('receiver_name'),
            "receiver_bank"   : data.get('receiver_bank'),
            "cardholder_name" : data.get('cardholder_name'),
            "transaction_id"  : data.get('transaction_id'),
            "payment_mode"    : data.get('payment_mode'),
            "merchant_name"   : data.get('merchant_name'),
            "device_type"     : data.get('device_type'),
            "transaction_note": data.get('transaction_note'),
        })
        save_history(history)

        # Feature 7: Auto-retrain check (background)
        auto_retrain_if_needed()

        # SMS alert
        send_sms_alert(result, data)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    """Feature 1: AI Chatbot proxy — called from frontend with user message."""
    data    = request.json
    message = data.get('message', '')
    history = data.get('history', [])
    lang    = data.get('language', 'en')

    lang_instruction = {
        'hi': 'Always respond in Hindi (Devanagari script).',
        'te': 'Always respond in Telugu script.',
        'ta': 'Always respond in Tamil script.',
        'en': 'Always respond in English.',
    }.get(lang, 'Always respond in English.')

    system_prompt = f"""You are FraudGuard AI Assistant — an expert on bank fraud, 
cybercrime, and financial security in India. You help users understand:
- Types of bank fraud (UPI scams, phishing, card skimming, OTP fraud, etc.)
- How to protect themselves from fraud
- What to do if they are scammed
- How the FraudGuard AI system works
- Indian banking regulations and helplines (1930, cybercrime.gov.in)
- RBI guidelines on fraud

{lang_instruction}

Keep responses concise, helpful, and practical. Use bullet points when listing steps.
Always mention the cybercrime helpline 1930 when relevant."""

    messages = []
    for h in history[-10:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": message})

    try:
        import urllib.request, json as jsonlib
        payload = jsonlib.dumps({
            "model"     : "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "system"    : system_prompt,
            "messages"  : messages,
        }).encode()

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "Content-Type"     : "application/json",
                "anthropic-version": "2023-06-01",
            }
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = jsonlib.loads(resp.read())
            reply  = result['content'][0]['text']
            return jsonify({"reply": reply})

    except Exception as e:
        # Fallback responses if API unavailable
        fallback = {
            "upi"      : "UPI fraud is common. Never share OTP/PIN. Verify receiver before sending. Report to 1930.",
            "otp"      : "🚨 OTP fraud: Never share OTP with anyone. Banks NEVER ask for OTP. Call 1930 immediately.",
            "phishing" : "Phishing uses fake links/emails. Never click suspicious links. Verify website URL carefully.",
            "default"  : "I'm your fraud prevention assistant! Ask me about UPI scams, phishing, OTP fraud, or what to do if scammed. Helpline: 1930"
        }
        key = next((k for k in fallback if k in message.lower()), 'default')
        return jsonify({"reply": fallback[key]})


@app.route('/history', methods=['GET'])
def history():
    return jsonify(load_history())


@app.route('/stats', methods=['GET'])
def stats():
    h         = load_history()
    total     = len(h)
    frauds    = sum(1 for t in h if t['is_fraud'])
    legit     = total - frauds
    total_amt = sum(float(t.get('amount') or 0) for t in h)
    fraud_amt = sum(float(t.get('amount') or 0) for t in h if t['is_fraud'])
    risk_dist = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    for t in h: risk_dist[t.get('risk_level', 'LOW')] += 1

    # Payment mode breakdown
    mode_dist = {}
    for t in h:
        m = t.get('payment_mode', 'Unknown') or 'Unknown'
        mode_dist[m] = mode_dist.get(m, 0) + 1

    # Hourly distribution
    hour_dist = {str(i): 0 for i in range(24)}
    for t in h:
        ts = t.get('timestamp','')
        try:
            hr = str(datetime.fromisoformat(ts).hour)
            hour_dist[hr] = hour_dist.get(hr, 0) + 1
        except: pass

    return jsonify({
        "total"            : total,
        "frauds"           : frauds,
        "legitimate"       : legit,
        "fraud_rate"       : round(frauds / total * 100, 2) if total else 0,
        "total_amount"     : round(total_amt, 2),
        "fraud_amount"     : round(fraud_amt, 2),
        "risk_distribution": risk_dist,
        "mode_distribution": mode_dist,
        "hour_distribution": hour_dist,
        "model_metrics"    : model_metrics,
        "next_retrain_in"  : RETRAIN_EVERY - (total % RETRAIN_EVERY) if total > 0 else RETRAIN_EVERY,
    })


@app.route('/metrics', methods=['GET'])
def metrics_route():
    return jsonify(model_metrics)


@app.route('/retrain', methods=['POST'])
def retrain():
    global model, threshold, scaler, model_metrics
    try:
        model_metrics = train_model()
        with open(MODEL_PATH,  'rb') as f: model, threshold = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
        return jsonify({"message": "Model retrained on Kaggle dataset", "metrics": model_metrics})
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404


@app.route('/health', methods=['GET'])
def health():
    h = load_history()
    return jsonify({
        "status"         : "ok",
        "model"          : "Ensemble (RF + ExtraTrees + GBM + LR)",
        "dataset"        : "Kaggle Credit Card Fraud Detection",
        "threshold"      : round(threshold, 2),
        "auc_roc"        : model_metrics.get('auc_roc', 'N/A'),
        "f1_fraud"       : model_metrics.get('f1_fraud', 'N/A'),
        "total_analyzed" : len(h),
        "last_trained"   : model_metrics.get('last_trained', 'N/A'),
    })



# ══════════════════════════════════════════════════════════════
#  FEATURE 1: PDF REPORT
# ══════════════════════════════════════════════════════════════
@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
        import io

        data   = request.json
        result = data.get('result', {})
        form   = data.get('form',   {})

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                leftMargin=2*cm, rightMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)

        styles = getSampleStyleSheet()
        fraud  = result.get('is_fraud', False)
        color  = colors.HexColor('#ff3860') if fraud else colors.HexColor('#00f5a0')

        title_style = ParagraphStyle('title', fontSize=22, fontName='Helvetica-Bold',
                                     alignment=TA_CENTER, textColor=colors.HexColor('#e2e8f0'),
                                     spaceAfter=4)
        sub_style   = ParagraphStyle('sub',   fontSize=11, fontName='Helvetica',
                                     alignment=TA_CENTER, textColor=colors.HexColor('#64748b'),
                                     spaceAfter=16)
        section_style = ParagraphStyle('sec', fontSize=13, fontName='Helvetica-Bold',
                                       textColor=colors.HexColor('#00f5a0'), spaceAfter=8, spaceBefore=14)
        label_style   = ParagraphStyle('lbl', fontSize=10, fontName='Helvetica',
                                       textColor=colors.HexColor('#64748b'))
        value_style   = ParagraphStyle('val', fontSize=10, fontName='Helvetica-Bold',
                                       textColor=colors.HexColor('#e2e8f0'))
        verdict_style = ParagraphStyle('vrd', fontSize=18, fontName='Helvetica-Bold',
                                       alignment=TA_CENTER, textColor=color, spaceAfter=6)
        risk_style    = ParagraphStyle('rsk', fontSize=10, fontName='Helvetica',
                                       textColor=colors.HexColor('#ff6b35'), spaceAfter=4)
        tip_style     = ParagraphStyle('tip', fontSize=9,  fontName='Helvetica',
                                       textColor=colors.HexColor('#94a3b8'), spaceAfter=3)

        story = []

        # ── Header ──────────────────────────────────────────
        story.append(Paragraph("FraudGuard AI", title_style))
        story.append(Paragraph("Bank Transaction Fraud Analysis Report", sub_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1f2937')))
        story.append(Spacer(1, 12))

        # ── Verdict Box ──────────────────────────────────────
        verdict_text = "FRAUD DETECTED" if fraud else "TRANSACTION SAFE"
        story.append(Paragraph(f"{'🚨' if fraud else '✅'}  {verdict_text}", verdict_style))

        verdict_table = Table([[
            Paragraph(f"Fraud Probability\n{result.get('fraud_probability','?')}%", value_style),
            Paragraph(f"Safe Probability\n{result.get('safe_probability','?')}%",  value_style),
            Paragraph(f"Risk Level\n{result.get('risk_level','?')}",               value_style),
            Paragraph(f"ML Score\n{result.get('ml_score','?')}%",                  value_style),
        ]], colWidths=[4.2*cm]*4)
        verdict_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#0f172a')),
            ('BOX',        (0,0), (-1,-1), 1, color),
            ('INNERGRID',  (0,0), (-1,-1), 0.5, colors.HexColor('#1f2937')),
            ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
            ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING',(0,0),(-1,-1),10),
        ]))
        story.append(verdict_table)
        story.append(Spacer(1, 14))

        # ── Transaction Details ──────────────────────────────
        story.append(Paragraph("Transaction Details", section_style))
        tx_data = [
            ["Reference ID",    result.get('transaction_id', form.get('transaction_id','N/A'))],
            ["Sender Name",     form.get('cardholder_name','N/A')],
            ["Sender Bank",     form.get('bank_name','N/A')],
            ["Sender Account",  "XXXX " + str(form.get('sender_account',''))[-4:] if form.get('sender_account') else 'N/A'],
            ["Receiver Name",   form.get('receiver_name','N/A')],
            ["Receiver Bank",   form.get('receiver_bank','N/A')],
            ["Receiver Account","XXXX " + str(form.get('receiver_account',''))[-4:] if form.get('receiver_account') else 'N/A'],
            ["Amount",          f"Rs. {float(form.get('amount',0)):,.2f}"],
            ["Payment Mode",    form.get('payment_mode','N/A')],
            ["Transaction Type",form.get('transaction_type','N/A')],
            ["Merchant",        form.get('merchant_name','N/A')],
            ["Device Used",     form.get('device_type','N/A')],
            ["Timestamp",       result.get('timestamp', datetime.now().isoformat())[:19]],
            ["Decision Threshold", str(result.get('threshold_used','N/A'))],
        ]
        tbl = Table([[Paragraph(r[0], label_style), Paragraph(str(r[1]), value_style)] for r in tx_data],
                    colWidths=[5*cm, 11.7*cm])
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#0f172a')),
            ('ROWBACKGROUNDS',(0,0),(-1,-1),[colors.HexColor('#0f172a'),colors.HexColor('#111827')]),
            ('BOX',        (0,0), (-1,-1), 0.5, colors.HexColor('#1f2937')),
            ('INNERGRID',  (0,0), (-1,-1), 0.3, colors.HexColor('#1f2937')),
            ('TOPPADDING', (0,0), (-1,-1), 7),
            ('BOTTOMPADDING',(0,0),(-1,-1),7),
            ('LEFTPADDING',(0,0),(-1,-1),10),
        ]))
        story.append(tbl)

        # ── Risk Factors ─────────────────────────────────────
        risk_factors = result.get('risk_factors', [])
        if risk_factors:
            story.append(Paragraph("Risk Factors Detected", section_style))
            for rf in risk_factors:
                story.append(Paragraph(f"  ▶  {rf}", risk_style))

        # ── Action Steps ────────────────────────────────────
        if fraud:
            story.append(Spacer(1, 8))
            story.append(Paragraph("Immediate Action Required", section_style))
            tips = [
                "1. Call your bank helpline immediately to block the transaction",
                "2. Do NOT share OTP, PIN, or CVV with anyone",
                "3. Report to Cyber Crime: cybercrime.gov.in or call 1930",
                "4. Change your banking password and MPIN right now",
                "5. Check all recent transactions for suspicious activity",
                "6. File a complaint at your nearest police station",
            ]
            for tip in tips:
                story.append(Paragraph(tip, tip_style))

        # ── Footer ───────────────────────────────────────────
        story.append(Spacer(1, 20))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#1f2937')))
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            f"Generated by FraudGuard AI  |  {datetime.now().strftime('%d %b %Y %I:%M %p')}  |  For security purposes only",
            ParagraphStyle('footer', fontSize=8, fontName='Helvetica',
                           alignment=TA_CENTER, textColor=colors.HexColor('#475569'))))

        doc.build(story)
        buf.seek(0)

        from flask import send_file
        return send_file(buf, mimetype='application/pdf',
                         as_attachment=True,
                         download_name=f"FraudReport_{result.get('transaction_id','TXN')}.pdf")

    except ImportError:
        return jsonify({"error": "reportlab not installed. Run: pip install reportlab"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════
#  FEATURE 4: LOGIN / REGISTER  (simple JWT-free token system)
# ══════════════════════════════════════════════════════════════
USERS_PATH = os.path.join(BASE_DIR, "users.json")

def load_users():
    if os.path.exists(USERS_PATH):
        with open(USERS_PATH) as f: return json.load(f)
    return {}

def save_users(u):
    with open(USERS_PATH, 'w') as f: json.dump(u, f)

def hash_password(pw):
    import hashlib
    return hashlib.sha256(pw.encode()).hexdigest()

def make_token(username):
    import hashlib, time
    return hashlib.sha256(f"{username}{time.time()}secret_key_fraudguard".encode()).hexdigest()

# In-memory token store {token: username}
active_tokens = {}

@app.route('/register', methods=['POST'])
def register():
    data     = request.json
    username = data.get('username','').strip().lower()
    password = data.get('password','')
    name     = data.get('name','').strip()
    if not username or not password or not name:
        return jsonify({"error": "Username, password and name are required"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400
    users = load_users()
    if username in users:
        return jsonify({"error": "Username already exists"}), 409
    users[username] = {
        "name"    : name,
        "password": hash_password(password),
        "created" : datetime.now().isoformat(),
    }
    save_users(users)
    token = make_token(username)
    active_tokens[token] = username
    return jsonify({"token": token, "name": name, "username": username})

@app.route('/login', methods=['POST'])
def login():
    data     = request.json
    username = data.get('username','').strip().lower()
    password = data.get('password','')
    users    = load_users()
    if username not in users:
        return jsonify({"error": "User not found"}), 404
    if users[username]['password'] != hash_password(password):
        return jsonify({"error": "Incorrect password"}), 401
    token = make_token(username)
    active_tokens[token] = username
    return jsonify({"token": token, "name": users[username]['name'], "username": username})

@app.route('/logout', methods=['POST'])
def logout():
    token = request.json.get('token','')
    active_tokens.pop(token, None)
    return jsonify({"message": "Logged out"})

@app.route('/verify_token', methods=['POST'])
def verify_token():
    token = request.json.get('token','')
    if token in active_tokens:
        username = active_tokens[token]
        users    = load_users()
        name     = users.get(username, {}).get('name', username)
        return jsonify({"valid": True, "username": username, "name": name})
    return jsonify({"valid": False}), 401


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

