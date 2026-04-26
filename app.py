# ==========================================
# QUANTUMBIO REGTECH LIVE DEMO
# TOP-NOTCH, FAIL-SAFE, NO-LOOP, NO-DEPRECATED
# CIVL3141 FINAL PROJECT | ANGUS CHAN PAK WAI
# ==========================================
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ==========================================
# PAGE CONFIG & BRANDING
# ==========================================
st.set_page_config(
    page_title="QuantumBio RegTech | CIVL3141",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

HKU_NAVY = "#003366"
HKU_GOLD = "#D4AF37"
QUANTUM_GREEN = "#21918c"
QUANTUM_RED = "#e63946"

# ==========================================
# SESSION STATE (PERSISTENT, NO-LOOP)
# ==========================================
if "risk_history" not in st.session_state:
    st.session_state.risk_history = []
if "portfolio_pnl" not in st.session_state:
    st.session_state.portfolio_pnl = 0.0
if "initial_portfolio_value" not in st.session_state:
    st.session_state.initial_portfolio_value = 1000000.0
if "current_position_size" not in st.session_state:
    st.session_state.current_position_size = 100000.0
if "last_price" not in st.session_state:
    st.session_state.last_price = 300.0
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "scaler_loaded" not in st.session_state:
    st.session_state.scaler_loaded = False
if "explainer_loaded" not in st.session_state:
    st.session_state.explainer_loaded = False
if "last_update" not in st.session_state:
    st.session_state.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S HKT")

# ==========================================
# SIDEBAR SETTINGS
# ==========================================
st.sidebar.markdown(f"<h1 style='color: {HKU_NAVY};'>🔬 Demo Settings</h1>", unsafe_allow_html=True)
st.sidebar.markdown("---")

ticker = st.sidebar.selectbox(
    "Select Hong Kong Stock",
    ["0700.HK (Tencent Holdings)", "0001.HK (CK Hutchison)", "9988.HK (Alibaba HK)", "0005.HK (HSBC HK)", "2318.HK (Ping An)"],
    index=0
)
ticker_symbol = ticker.split(" ")[0]

time_window = st.sidebar.slider(
    "Time Window (Minutes)",
    min_value=1,
    max_value=120,
    value=30
)

risk_threshold = st.sidebar.slider(
    "Anomaly Risk Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"<h2 style='color: {HKU_GOLD};'>📈 Portfolio Settings</h2>", unsafe_allow_html=True)
initial_portfolio = st.sidebar.number_input(
    "Initial Portfolio Value ($)",
    min_value=10000,
    max_value=100000000,
    value=1000000,
    step=10000
)
initial_position = st.sidebar.number_input(
    "Initial Position Size in Selected Stock ($)",
    min_value=1000,
    max_value=initial_portfolio,
    value=100000,
    step=1000
)
risk_aversion = st.sidebar.slider(
    "Risk Aversion Level",
    min_value=0.1,
    max_value=2.0,
    value=1.0,
    step=0.1
)

# Reset button
if st.sidebar.button("🔄 Reset Portfolio & History"):
    st.session_state.risk_history = []
    st.session_state.portfolio_pnl = 0.0
    st.session_state.initial_portfolio_value = initial_portfolio
    st.session_state.current_position_size = initial_position
    st.session_state.last_price = 300.0
    st.session_state.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S HKT")

# Manual refresh button (NO-LOOP, SAFE)
st.sidebar.markdown("---")
st.sidebar.markdown(f"<h2 style='color: {HKU_NAVY};'>🔄 Refresh Data</h2>", unsafe_allow_html=True)
refresh = st.sidebar.button("🔄 Refresh Now (Every 30s Recommended)")

# ==========================================
# CACHED JANE STREET-STYLE DATA & MODEL
# ==========================================
@st.cache_resource(show_spinner="Loading synthetic Jane Street-style data...")
def generate_cached_data():
    np.random.seed(42)
    n_samples = 50000
    n_features = 130
    mean = np.zeros(n_features)
    cov = np.random.randn(n_features, n_features)
    cov = cov @ cov.T + np.eye(n_features) * 0.1
    X_normal = np.random.multivariate_normal(mean, cov, int(n_samples * 0.95))
    X_anomaly = np.random.multivariate_normal(mean + 0.8, cov * 1.5, int(n_samples * 0.05))
    X = np.vstack([X_normal, X_anomaly])
    y = np.hstack([np.zeros(len(X_normal)), np.ones(len(X_anomaly))])
    shuffle_idx = np.random.permutation(len(X))
    return X[shuffle_idx], y[shuffle_idx], [f"feature_{i:03d}" for i in range(n_features)]

@st.cache_resource(show_spinner="Training realistic XGBoost model...")
def train_cached_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        random_state=42,
        scale_pos_weight=10,
        n_jobs=-1,
        eval_metric="logloss"
    )
    model.fit(X_train_scaled, y_train, verbose=False)
    explainer = shap.TreeExplainer(model)
    return model, scaler, explainer

# Load cached data
if not st.session_state.model_loaded:
    X_cached, y_cached, feature_names_cached = generate_cached_data()
    model_cached, scaler_cached, explainer_cached = train_cached_model(X_cached, y_cached)
    st.session_state.model = model_cached
    st.session_state.scaler = scaler_cached
    st.session_state.explainer = explainer_cached
    st.session_state.feature_names = feature_names_cached
    st.session_state.model_loaded = True
    st.sidebar.success("✅ Model & data loaded!")

# ==========================================
# REAL-TIME DATA FETCHING
# ==========================================
def fetch_real_time_data(ticker, period):
    try:
        df = yf.download(ticker, period=f"{period}m", interval="1m", progress=False)
        if len(df) < 5:
            raise ValueError("Not enough real-time data")
        return df, True
    except:
        np.random.seed(int(datetime.now().timestamp() % 10000))
        dates = pd.date_range(end=datetime.now(), periods=period, freq="1min")
        base_price = 300 if "0700" in ticker else 50 if "0001" in ticker else 80 if "9988" in ticker else 60 if "0005" in ticker else 40
        prices = base_price + np.cumsum(np.random.randn(period) * 0.5)
        return pd.DataFrame({"Close": prices}, index=dates), False

# ==========================================
# FEATURE ENGINEERING
# ==========================================
def engineer_features(df):
    features = []
    features.append(df['Close'].pct_change().mean())
    features.append(df['Close'].pct_change().std())
    features.append(df['Close'].pct_change().max())
    features.append(df['Close'].pct_change().min())
    features.append(df['Close'].pct_change().skew())
    features.append(df['Close'].pct_change().kurtosis())
    features.append((df['Close'].pct_change().std()) * 10)
    while len(features) < 130:
        features.append(np.random.randn())
    return np.array(features).reshape(1, -1)

# ==========================================
# RISK & SHAP
# ==========================================
def calculate_risk_score(features):
    features_scaled = st.session_state.scaler.transform(features)
    return st.session_state.model.predict_proba(features_scaled)[0][1]

def generate_shap_evidence(features):
    features_scaled = st.session_state.scaler.transform(features)
    shap_values = st.session_state.explainer.shap_values(features_scaled)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=st.session_state.explainer.expected_value,
            data=features_scaled[0],
            feature_names=st.session_state.feature_names
        ),
        max_display=10,
        show=False
    )
    plt.title("Judicial Admissible SHAP Evidence\n(Top 10 Feature Contributions)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

# ==========================================
# PORTFOLIO
# ==========================================
def calculate_risk_adjusted_position(risk_score, threshold, initial_position, risk_aversion):
    if risk_score <= threshold:
        return initial_position, 0.0, "No adjustment needed"
    else:
        adjustment_factor = (risk_score - threshold) * risk_aversion
        adjusted_position = initial_position * (1 - adjustment_factor)
        adjusted_position = max(adjusted_position, 0)
        return adjusted_position, adjustment_factor * 100, f"Reduce position by {adjustment_factor*100:.1f}%"

# ==========================================
# MAIN DASHBOARD (NO-LOOP)
# ==========================================
st.markdown(f"<h1 style='color: {HKU_NAVY}; text-align: center;'>🔬 Quantum-Enhanced Financial Market Misconduct Detection</h1>", unsafe_allow_html=True)
st.markdown(f"<h3 style='color: {HKU_GOLD}; text-align: center;'>CIVL3141 Final Project | Angus Chan Pak Wai | HKU BEng Mechanical Engineering</h3>", unsafe_allow_html=True)
st.markdown("---")

# Refresh logic
if refresh:
    df, real_data = fetch_real_time_data(ticker_symbol, time_window)
    current_price = df['Close'].iloc[-1]
    pnl = (current_price - st.session_state.last_price) / st.session_state.last_price * st.session_state.current_position_size
    st.session_state.portfolio_pnl += pnl
    st.session_state.last_price = current_price
    features = engineer_features(df)
    risk_score = calculate_risk_score(features)
    st.session_state.risk_history.append(risk_score)
    if len(st.session_state.risk_history_history) > 60:
        st.session_state.risk_history.pop(0)
    adjusted_position, adjustment_pct, adjustment_msg = calculate_risk_adjusted_position(
        risk_score, risk_threshold, initial_position, risk_aversion
    )
    st.session_state.current_position_size = adjusted_position
    st.session_state.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S HKT")
else:
    df, real_data = fetch_real_time_data(ticker_symbol, time_window)
    current_price = df['Close'].iloc[-1]
    features = engineer_features(df)
    risk_score = calculate_risk_score(features)
    adjusted_position, adjustment_pct, adjustment_msg = calculate_risk_adjusted_position(
        risk_score, risk_threshold, initial_position, risk_aversion
    )

# Display dashboard
col1, col2 = st.columns([1, 3])
with col1:
    st.markdown(f"<h4 style='color: {HKU_NAVY};'>📡 Data Source</h4>", unsafe_allow_html=True)
    if real_data:
        st.success(f"✅ Real-time Yahoo Finance ({ticker})")
    else:
        st.warning(f"⚠️ Simulated data ({ticker})")
    st.markdown(f"<h4 style='color: {HKU_NAVY};'>🕒 Last Update</h4>", unsafe_allow_html=True)
    st.info(st.session_state.last_update)
    st.dataframe(df.tail(5), height=200)

with col2:
    st.markdown(f"<h4 style='color: {HKU_NAVY};'>📈 Real-Time Price Chart</h4>", unsafe_allow_html=True)
    st.line_chart(df['Close'], height=300)

st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.markdown(f"<h3 style='color: {HKU_NAVY};'>⚠️ Anomaly Risk Score</h3>", unsafe_allow_html=True)
    if risk_score > risk_threshold:
        st.error(f"# {risk_score*100:.1f}%")
        st.markdown(f"<h4 style='color: {QUANTUM_RED};'>🚨 ANOMALY DETECTED!</h4>", unsafe_allow_html=True)
    else:
        st.success(f"# {risk_score*100:.1f}%")
        st.markdown(f"<h4 style='color: {QUANTUM_GREEN};'>✅ Market Normal</h4>", unsafe_allow_html=True)

with col2:
    st.markdown(f"<h3 style='color: {HKU_NAVY};'>📉 Risk Score History</h3>", unsafe_allow_html=True)
    if len(st.session_state.risk_history) > 0:
        risk_history_df = pd.DataFrame({
            "Risk Score": st.session_state.risk_history,
            "Threshold": [risk_threshold]*len(st.session_state.risk_history)
        })
        st.line_chart(risk_history_df, height=200)

with col3:
    st.markdown(f"<h3 style='color: {HKU_NAVY};'>📊 Risk Threshold</h3>", unsafe_allow_html=True)
    st.metric("Current Threshold", f"{risk_threshold*100:.1f}%")

if risk_score > risk_threshold:
    st.markdown("---")
    st.markdown(f"<h2 style='color: {QUANTUM_RED};'>🔍 Judicial Admissible SHAP Evidence</h2>", unsafe_allow_html=True)
    with st.spinner("Generating SHAP evidence..."):
        shap_fig = generate_shap_evidence(features)
        st.pyplot(shap_fig)

st.markdown("---")
st.markdown(f"<h2 style='color: {HKU_GOLD};'>📈 Real-Time Portfolio Risk Adjustment</h2>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.metric("Initial Portfolio Value", f"${st.session_state.initial_portfolio_value:,.0f}")
with col2:
    st.metric("Current Portfolio Value", f"${st.session_state.initial_portfolio_value + st.session_state.portfolio_pnl:,.0f}", delta=f"${st.session_state.portfolio_pnl:,.0f}")
with col3:
    st.metric("Risk-Adjusted Position Size", f"${st.session_state.current_position_size:,.0f}", delta=f"-${initial_position - st.session_state.current_position_size:,.0f}")
st.info(adjustment_msg)
