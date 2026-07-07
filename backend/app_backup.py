from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle, os, json, threading
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

MODEL_PATH    = "fraud_model.pkl"
SCALER_PATH   = "scaler.pkl"
HISTORY_PATH  = "transaction_history.json"
METRICS_PATH  = "metrics.json"
RETRAIN_EVERY = 10   # auto-retrain every N new transactions

# ── Twilio SMS ─────────────────────────────────────────────
TWILIO_SID   = os.environ.get("TWILIO_SID",   "")
TWILIO_TOKEN = os.environ.get("TWILIO_TOKEN", "")
TWILIO_FROM  = os.environ.get("TWILIO_FROM",  "")
TWILIO_TO    = os.environ.get("TWILIO_TO",    "")

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
    csv_path = "creditcard.csv"
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
    from sklearn.linear_model  import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                                  precision_score, recall_score, accuracy_score,
                                  confusion_matrix, classification_report)

    if not silent:
        print("\n" + "═"*58)
        print("  FraudGuard AI  –  Training on Kaggle Dataset")
        print("═"*58)

    df = load_kaggle_data()
    X_df, scaler = engineer_features(df.drop('Class', axis=1), fit=True)
    X = X_df.values.astype(np.float32)
    y = df['Class'].values.astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    if not silent: print(f"▶  Train: {len(X_tr):,}  |  Test: {len(X_te):,}")

    X_bal, y_bal = smote_numpy(X_tr, y_tr, k=5)

    if not silent: print("▶  Training Ensemble (RF + ET + GB + LR) ...")
    rf = RandomForestClassifier(n_estimators=300, max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1)
    et = ExtraTreesClassifier(n_estimators=300, max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8, min_samples_leaf=20, random_state=42)
    lr = LogisticRegression(C=0.1, max_iter=2000, class_weight='balanced', random_state=42)

    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('et', et), ('gb', gb), ('lr', lr)],
        voting='soft', weights=[3, 2, 3, 1])
    ensemble.fit(X_bal, y_bal)

    y_proba = ensemble.predict_proba(X_te)[:, 1]
    best_f1, best_thr = 0, 0.5
    for thr in np.arange(0.20, 0.80, 0.01):
        preds = (y_proba >= thr).astype(int)
        f = f1_score(y_te, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_thr = f, thr

    y_pred = (y_proba >= best_thr).astype(int)
    cm     = confusion_matrix(y_te, y_pred).tolist()
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    metrics = {
        "accuracy"        : round(accuracy_score(y_te, y_pred) * 100, 4),
        "auc_roc"         : round(roc_auc_score(y_te, y_proba) * 100, 4),
        "avg_precision"   : round(average_precision_score(y_te, y_proba) * 100, 4),
        "precision_fraud" : round(precision_score(y_te, y_pred, zero_division=0) * 100, 4),
        "recall_fraud"    : round(recall_score(y_te, y_pred, zero_division=0) * 100, 4),
        "f1_fraud"        : round(f1_score(y_te, y_pred, zero_division=0) * 100, 4),
        "threshold"       : round(best_thr, 2),
        "true_positives"  : int(tp), "false_positives": int(fp),
        "false_negatives" : int(fn), "true_negatives" : int(tn),
        "confusion_matrix": cm,
        "train_samples"   : int(len(X_bal)),
        "test_samples"    : int(len(X_te)),
        "dataset"         : "Kaggle Credit Card Fraud Detection (284,807 transactions)",
        "last_trained"    : datetime.now().isoformat(),
    }

    if not silent:
        print(f"\n  ┌─────────────────────────────────────────┐")
        print(f"  │  Accuracy      : {metrics['accuracy']:>8.4f} %          │")
        print(f"  │  AUC-ROC       : {metrics['auc_roc']:>8.4f} %          │")
        print(f"  │  F1-Score      : {metrics['f1_fraud']:>8.4f} %          │")
        print(f"  │  Precision     : {metrics['precision_fraud']:>8.4f} %          │")
        print(f"  │  Recall        : {metrics['recall_fraud']:>8.4f} %          │")
        print(f"  └─────────────────────────────────────────┘\n")
        print(classification_report(y_te, y_pred, target_names=['Legitimate','Fraud']))

    with open(MODEL_PATH,   'wb') as f: pickle.dump((ensemble, best_thr), f)
    with open(SCALER_PATH,  'wb') as f: pickle.dump(scaler, f)
    with open(METRICS_PATH, 'w')  as f: json.dump(metrics, f)
    if not silent: print("✅  Model saved → fraud_model.pkl\n")
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
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    print("✅  Loading saved model ...")
    with open(MODEL_PATH,  'rb') as f: model, threshold = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
    model_metrics = json.load(open(METRICS_PATH)) if os.path.exists(METRICS_PATH) else {}
    print(f"   AUC-ROC  : {model_metrics.get('auc_roc','?')} %")
    print(f"   F1-Score : {model_metrics.get('f1_fraud','?')} %")
else:
    model_metrics = train_model()
    with open(MODEL_PATH,  'rb') as f: model, threshold = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)


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
    amount         = float(data.get('amount', 100))
    hour           = float(data.get('hour',   12))
    time_s         = hour * 3600
    loc_mm         = int(data.get('location_mismatch',    False))
    new_m          = int(data.get('new_merchant',         False))
    hi_freq        = int(data.get('high_frequency',       False))
    intl           = int(data.get('international',        False))
    no_card        = int(not data.get('card_present',     True))
    online         = int(data.get('online_transaction',   False))
    otp_shared     = int(data.get('otp_shared',           False))
    suspicious_lnk = int(data.get('suspicious_link',      False))
    first_large    = int(data.get('first_time_large_amt', False))
    no_beneficiary = int(not data.get('saved_beneficiary',True))
    unknown_device = int(data.get('device_type','') == 'unknown')
    cat            = data.get('merchant_category', 'retail')
    cat_risk       = {'gambling':4.5,'crypto':4.0,'travel':1.0,
                      'electronics':1.5,'food':0.2,'retail':0.1,'other':1.0}.get(cat, 0.5)

    risk = (loc_mm*3.5 + new_m*2.5 + hi_freq*3.0 + intl*2.5 +
            no_card*3.0 + online*1.5 + cat_risk +
            otp_shared*6.0 + suspicious_lnk*5.5 +
            first_large*2.5 + no_beneficiary*2.0 + unknown_device*2.5)

    if amount > 50000: risk += 3.0
    elif amount > 10000: risk += 1.5
    elif amount > 2000:  risk += 0.8
    if hour < 5 or hour > 22: risk += 2.0

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

    row = pd.DataFrame([[time_s] + list(base) + [amount]],
                       columns=['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount'])
    eng, _ = engineer_features(row, scaler=scaler, fit=False)
    return eng.values


# ══════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        features = build_feature_vector(data)
        proba    = model.predict_proba(features)[0]
        ml_score = float(proba[1])

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

        fraud_p  = min(ml_score + rule_boost, 0.99)
        is_fraud = fraud_p >= threshold

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
    model_metrics = train_model()
    with open(MODEL_PATH,  'rb') as f: model, threshold = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
    return jsonify({"message": "Model retrained on Kaggle dataset", "metrics": model_metrics})


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


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render gives PORT
    app.run(host="0.0.0.0", port=port)
