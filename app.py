import streamlit as st
import pandas as pd
import numpy as np
from pymongo import MongoClient
import os
import datetime
import altair as alt
from src.pandas_pipeline.data_provider import DataProvider
from src.model_trainer import ModelTrainer

# Set up page styling and config
st.set_page_config(
    page_title="Apple Stock Predictor — AAPL Stock Forecast",
    page_icon="🍏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# PREMIUM CSS INJECTION
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

    /* ── Global Reset ── */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    .block-container { padding-top: 1.5rem; }

    /* ── Animated Gradient Title ── */
    .hero-title {
        background: linear-gradient(135deg, #00E676 0%, #00BFA5 40%, #2979FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 0;
        animation: shimmer 3s ease-in-out infinite;
    }
    @keyframes shimmer {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.85; }
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #90A4AE;
        margin-top: 2px;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }

    /* ── Glassmorphism Card ── */
    .glass-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        padding: 24px 28px;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.18);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        margin-bottom: 16px;
    }
    .glass-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.28);
    }

    /* ── KPI Styles ── */
    .kpi-label {
        font-size: 0.8rem;
        color: #78909C;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 6px;
    }
    .kpi-price {
        font-size: 2.4rem;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1.1;
    }
    .kpi-price-pred {
        font-size: 2.4rem;
        font-weight: 800;
        color: #FFB300;
        line-height: 1.1;
    }
    .kpi-change-up {
        font-size: 0.95rem;
        font-weight: 600;
        color: #00E676;
        margin-top: 6px;
    }
    .kpi-change-down {
        font-size: 0.95rem;
        font-weight: 600;
        color: #FF1744;
        margin-top: 6px;
    }
    .kpi-change-flat {
        font-size: 0.95rem;
        font-weight: 600;
        color: #FFD600;
        margin-top: 6px;
    }

    /* ── Signal Badge ── */
    .signal-badge {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        padding: 16px 28px;
        border-radius: 14px;
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin: 8px 0 12px 0;
    }
    .signal-buy {
        background: linear-gradient(135deg, rgba(0,230,118,0.15), rgba(0,191,165,0.10));
        border: 2px solid #00E676;
        color: #00E676;
    }
    .signal-sell {
        background: linear-gradient(135deg, rgba(255,23,68,0.15), rgba(213,0,0,0.10));
        border: 2px solid #FF1744;
        color: #FF1744;
    }
    .signal-hold {
        background: linear-gradient(135deg, rgba(255,214,0,0.12), rgba(255,171,0,0.08));
        border: 2px solid #FFD600;
        color: #FFD600;
    }

    /* ── Market Health Indicator ── */
    .health-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 4px 4px 4px 0;
    }
    .pill-green { background: rgba(0,230,118,0.12); color: #00E676; border: 1px solid rgba(0,230,118,0.25); }
    .pill-red { background: rgba(255,23,68,0.12); color: #FF1744; border: 1px solid rgba(255,23,68,0.25); }
    .pill-blue { background: rgba(41,121,255,0.12); color: #29B6F6; border: 1px solid rgba(41,121,255,0.25); }
    .pill-yellow { background: rgba(255,214,0,0.12); color: #FFD600; border: 1px solid rgba(255,214,0,0.25); }

    /* ── Stat Row ── */
    .stat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .stat-label { color: #90A4AE; font-size: 0.9rem; font-weight: 500; }
    .stat-value { color: #ECEFF1; font-size: 0.95rem; font-weight: 600; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: rgba(15,20,30,0.95);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    .sidebar-brand {
        text-align: center;
        padding: 12px 0 8px 0;
    }
    .sidebar-brand h2 {
        background: linear-gradient(135deg, #00E676, #2979FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
    }
    .sidebar-footer {
        text-align: center;
        color: #546E7A;
        font-size: 0.78rem;
        padding-top: 12px;
    }

    /* ── Divider ── */
    .section-divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.06);
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING & MODEL TRAINING (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_and_train_models_cached():
    provider = DataProvider()
    df = provider.load_data("aapl_features")

    if df.empty:
        with st.spinner("🔄 Data is empty. Automatically running data collection pipeline..."):
            import subprocess
            subprocess.run(["venv/bin/python3", "main_spark.py"])
            df = provider.load_data("aapl_features")

    df_for_training = df.copy()
    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test = trainer.prepare_data(df_for_training, target_col='close')
    trainer.train_all(X_train, y_train)
    return trainer

try:
    trainer = load_and_train_models_cached()
except Exception as e:
    st.error(f"❌ Failed to initialize models. Error: {e}")
    st.stop()

# ─────────────────────────────────────────────
# SPARK DATA LOADING
# ─────────────────────────────────────────────
from src.spark_pipeline.spark_processor import SparkProcessor
spark_proc = SparkProcessor()

import glob
historical_files = glob.glob("data/data_lake/part_historical_*.json")
if not historical_files:
    with st.spinner("🔄 Data Lake is empty. Automatically loading historical data..."):
        spark_proc.initialize_data_lake()

with st.spinner("⚡ Processing real-time data..."):
    df = spark_proc.process_data_lake()

if df.empty:
    st.error("❌ No data found. Please run `python scratch/kafka_demo.py` first.")
    st.stop()

if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

# Aggregate hourly data to daily for charting
# (Kafka/Spark pipeline produces hourly rows, charts need daily data)
df_daily = df.copy()
df_daily['trade_date'] = df_daily['date'].dt.date
df_daily = df_daily.groupby('trade_date').agg({
    'close': 'last',
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'volume': 'sum',
    'sma_20': 'last',
    'sma_50': 'last',
    'rsi': 'last',
    'macd': 'last'
}).reset_index()
df_daily['trade_date'] = pd.to_datetime(df_daily['trade_date'])
df_daily = df_daily.sort_values('trade_date').reset_index(drop=True)

# Prepare training data and evaluate
df_for_training = df.copy()
X_train, X_test, y_train, y_test = trainer.prepare_data(df_for_training, target_col='close')
df_metrics = trainer.evaluate_all(X_test, y_test)
trainer.save_metrics_report(df_metrics)

split_idx = len(df) - len(y_test)
test_dates = df['date'].iloc[split_idx:].values

# ─────────────────────────────────────────────
# ENSEMBLE PREDICTION (Average of all 5 models)
# ─────────────────────────────────────────────
latest_row = df.iloc[-1]
latest_close = float(latest_row['close'])

# Compare with the close of the previous trading day from df_daily
if len(df_daily) >= 2:
    prev_day_close = float(df_daily.iloc[-2]['close'])
else:
    prev_day_close = latest_close

price_change_today = latest_close - prev_day_close
price_change_today_pct = (price_change_today / prev_day_close) * 100

latest_features = X_test.iloc[-1:]

# Compute each model's prediction
ensemble_preds = []
model_predictions = {}
for model_name, model in trainer.models.items():
    try:
        pred_diff = float(model.predict(latest_features)[0])
    except Exception:
        pred_diff = float(model.predict(latest_features.values)[0])
    pred_abs = latest_close + pred_diff
    model_predictions[model_name] = pred_abs
    ensemble_preds.append(pred_abs)

ensemble_price = np.mean(ensemble_preds)
ensemble_change = ensemble_price - latest_close
ensemble_change_pct = (ensemble_change / latest_close) * 100

# Market health indicators
latest_rsi = float(latest_row['rsi'])
latest_macd = float(latest_row['macd'])
latest_sma20 = float(latest_row['sma_20'])
latest_sma50 = float(latest_row['sma_50'])
latest_volume = float(latest_row['volume'])

# Signal logic
def compute_signal(change, rsi, macd):
    score = 0
    if change > 0.3: score += 1
    elif change < -0.3: score -= 1
    if rsi < 30: score += 1  # Oversold = buy signal
    elif rsi > 70: score -= 1  # Overbought = sell signal
    if macd > 0: score += 1
    elif macd < 0: score -= 1
    if score >= 2: return "BUY"
    elif score <= -2: return "SELL"
    else: return "HOLD"

signal = compute_signal(ensemble_change, latest_rsi, latest_macd)

# ─────────────────────────────────────────────
# SIDEBAR (Simplified)
# ─────────────────────────────────────────────
st.sidebar.markdown("<div class='sidebar-brand'><h2>🍏 Apple Stock</h2></div>", unsafe_allow_html=True)
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/f/fa/Apple_logo_black.svg", width=55)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
📅 **Data from:** {df['date'].min().strftime('%d/%m/%Y')}  
📅 **Updated to:** {df['date'].max().strftime('%d/%m/%Y')}  
📊 **Total observations:** {len(df):,} trading days
""")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Update Latest Data", help="Fetch latest data from market (yfinance/Alpaca), run Spark ETL, and retrain models."):
    with st.spinner("⏳ Loading new data from market (Kafka)..."):
        import subprocess
        # Run kafka_demo.py to pull new data and write to data lake
        subprocess.run(["venv/bin/python3", "scratch/kafka_demo.py"])
        
    with st.spinner("⚡ Running Spark ETL & Feature Engineering..."):
        spark_proc_new = SparkProcessor()
        df_spark = spark_proc_new.process_data_lake()
        if not df_spark.empty:
            provider = DataProvider()
            provider.save_data(df_spark, "aapl_features")
            
    st.cache_resource.clear()
    st.sidebar.success("✅ Data updated and models retrained successfully!")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("<div class='sidebar-footer'>Big Data Pipeline System<br>Apache Kafka · Apache Spark · ML/DL<br>© 2026</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("<h1 class='hero-title'>Apple Stock Predictor</h1>", unsafe_allow_html=True)
st.markdown(f"<p class='hero-subtitle'>Apple Stock (AAPL) price forecasting using AI — Updated: {df['date'].max().strftime('%d/%m/%Y')}</p>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TAB NAVIGATION (4 Pages)
# ─────────────────────────────────────────────
tab_overview, tab_trend, tab_predict, tab_tech = st.tabs([
    "🏠 Overview",
    "📈 Trend",
    "🔮 Live Prediction",
    "🔬 Technical"
])

# ═══════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════════
with tab_overview:
    # ── KPI Row ──
    col_k1, col_k2, col_k3 = st.columns(3)

    with col_k1:
        change_class = "kpi-change-up" if price_change_today >= 0 else "kpi-change-down"
        change_icon = "▲" if price_change_today >= 0 else "▼"
        st.markdown(f"""
        <div class="glass-card">
            <div class="kpi-label">💵 Latest Close Price</div>
            <div class="kpi-price">${latest_close:,.2f}</div>
            <div class="{change_class}">{change_icon} ${abs(price_change_today):,.2f} ({price_change_today_pct:+,.2f}%) compared to last session</div>
        </div>
        """, unsafe_allow_html=True)

    with col_k2:
        pred_class = "kpi-change-up" if ensemble_change >= 0 else "kpi-change-down"
        pred_icon = "▲ Up" if ensemble_change >= 0 else "▼ Down"
        st.markdown(f"""
        <div class="glass-card">
            <div class="kpi-label">🔮 Next Session Forecast</div>
            <div class="kpi-price-pred">${ensemble_price:,.2f}</div>
            <div class="{pred_class}">Trend: {pred_icon} ${abs(ensemble_change):,.2f} ({ensemble_change_pct:+,.2f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    with col_k3:
        if signal == "BUY":
            sig_class = "signal-buy"
            sig_text = "📈 BUY"
            sig_explain = "Technical indicators suggest a bullish momentum"
        elif signal == "SELL":
            sig_class = "signal-sell"
            sig_text = "📉 SELL"
            sig_explain = "Signs of downward trend or consolidation"
        else:
            sig_class = "signal-hold"
            sig_text = "⚖️ HOLD"
            sig_explain = "Market is sideways, wait for clearer signals"

        st.markdown(f"""
        <div class="glass-card" style="text-align:center;">
            <div class="kpi-label">🎯 Action Recommendation</div>
            <div class="signal-badge {sig_class}">{sig_text}</div>
            <div style="color:#90A4AE; font-size:0.88rem; margin-top:4px;">{sig_explain}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Market Health Pills ──
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("#### 🩺 Market Health")

    if latest_rsi > 70:
        rsi_pill = "pill-red"
        rsi_text = f"RSI {latest_rsi:.0f} — Overbought ⚠️"
    elif latest_rsi < 30:
        rsi_pill = "pill-green"
        rsi_text = f"RSI {latest_rsi:.0f} — Oversold 💚"
    else:
        rsi_pill = "pill-blue"
        rsi_text = f"RSI {latest_rsi:.0f} — Neutral"

    if latest_macd > 0:
        macd_pill = "pill-green"
        macd_text = f"MACD +{latest_macd:.2f} — Bullish"
    else:
        macd_pill = "pill-red"
        macd_text = f"MACD {latest_macd:.2f} — Bearish"

    if latest_close > latest_sma20:
        sma_pill = "pill-green"
        sma_text = "Price above SMA(20) — Bullish"
    else:
        sma_pill = "pill-red"
        sma_text = "Price below SMA(20) — Bearish"

    st.markdown(f"""
    <span class="health-pill {rsi_pill}">{rsi_text}</span>
    <span class="health-pill {macd_pill}">{macd_text}</span>
    <span class="health-pill {sma_pill}">{sma_text}</span>
    """, unsafe_allow_html=True)

    # ── 30-Day Chart ──
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("#### 📊 Price Fluctuation (Last 30 Sessions)")

    df_30 = df_daily.tail(30)[['trade_date', 'close']].copy()
    df_30 = df_30.rename(columns={'close': 'Price (USD)', 'trade_date': 'Date'})
    y_min_30 = float(df_30['Price (USD)'].min()) - 3
    y_max_30 = float(df_30['Price (USD)'].max()) + 3

    # Line chart (main)
    line_30 = alt.Chart(df_30).mark_line(
        color='#00E676', strokeWidth=2.5, point=alt.OverlayMarkDef(color='#00E676', size=40)
    ).encode(
        x=alt.X('Date:T', title='', axis=alt.Axis(format='%d/%m', labelAngle=-30)),
        y=alt.Y('Price (USD):Q', title='USD', scale=alt.Scale(domain=[y_min_30, y_max_30])),
        tooltip=[alt.Tooltip('Date:T', format='%d/%m/%Y'), alt.Tooltip('Price (USD):Q', format='$,.2f')]
    )

    # Area gradient fill (from y_min baseline)
    area_30 = alt.Chart(df_30).mark_area(
        color=alt.Gradient(
            gradient='linear',
            stops=[
                alt.GradientStop(color='rgba(0,230,118,0.30)', offset=0),
                alt.GradientStop(color='rgba(0,230,118,0.02)', offset=1)
            ],
            x1=1, x2=1, y1=1, y2=0
        )
    ).encode(
        x='Date:T',
        y=alt.Y('Price (USD):Q', scale=alt.Scale(domain=[y_min_30, y_max_30]))
    )

    chart_30 = (area_30 + line_30).properties(height=320).interactive()
    st.altair_chart(chart_30, width='stretch')

    # ── Model Predictions Comparison (compact) ──
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("#### 🧠 AI Model Forecast Details")

    pred_cols = st.columns(len(model_predictions))
    for i, (name, price) in enumerate(model_predictions.items()):
        diff = price - latest_close
        with pred_cols[i]:
            diff_class = "kpi-change-up" if diff >= 0 else "kpi-change-down"
            diff_icon = "▲" if diff >= 0 else "▼"
            # Display friendly name
            display_name = name.replace("_", " + ").replace("ARIMA LSTM", "ARIMA+LSTM")
            st.markdown(f"""
            <div class="glass-card" style="text-align:center; padding:16px 12px;">
                <div style="font-size:0.75rem; color:#78909C; font-weight:600; text-transform:uppercase;">{display_name}</div>
                <div style="font-size:1.5rem; font-weight:700; color:#ECEFF1; margin:6px 0;">${price:,.2f}</div>
                <div class="{diff_class}" style="font-size:0.8rem;">{diff_icon} ${abs(diff):,.2f}</div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# TAB 2: TREND ANALYSIS
# ═══════════════════════════════════════════════
with tab_trend:
    st.markdown("#### 📈 Apple Historical Price Chart (Full)")
    st.caption("Drag to zoom, scroll to scale. Green line = actual closing price.")

    # Full history chart with SMA overlays (aggregated daily)
    df_chart = df_daily[['trade_date', 'close', 'sma_20', 'sma_50']].copy()
    df_chart = df_chart.rename(columns={'trade_date': 'Date', 'close': 'Price', 'sma_20': 'SMA 20', 'sma_50': 'SMA 50'})

    base = alt.Chart(df_chart).encode(x=alt.X('Date:T', title=''))

    line_price = base.mark_line(color='#00E676', strokeWidth=1.8).encode(
        y=alt.Y('Price:Q', title='USD', scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip('Date:T', format='%d/%m/%Y'), alt.Tooltip('Price:Q', format='$,.2f')]
    )
    line_sma20 = base.mark_line(color='#29B6F6', strokeWidth=1, strokeDash=[4, 4]).encode(
        y='SMA 20:Q'
    )
    line_sma50 = base.mark_line(color='#FFB300', strokeWidth=1, strokeDash=[6, 3]).encode(
        y='SMA 50:Q'
    )

    full_chart = (line_price + line_sma20 + line_sma50).properties(height=400).interactive()
    st.altair_chart(full_chart, width='stretch')

    st.markdown("""
    <div style="display:flex; gap:20px; font-size:0.85rem; color:#90A4AE; margin-bottom:16px;">
        <span>🟢 Close Price</span>
        <span>🔵 SMA 20 Days (Short-term trend)</span>
        <span>🟡 SMA 50 Days (Medium-term trend)</span>
    </div>
    """, unsafe_allow_html=True)

    # Volume chart (aggregated daily)
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("#### 📊 Trading Volume (Last 60 Sessions)")

    df_vol = df_daily.tail(60)[['trade_date', 'volume']].copy()
    df_vol = df_vol.rename(columns={'trade_date': 'Date', 'volume': 'Volume'})

    vol_chart = alt.Chart(df_vol).mark_bar(
        cornerRadiusTopLeft=4,
        cornerRadiusTopRight=4
    ).encode(
        x=alt.X('Date:T', title='', axis=alt.Axis(format='%d/%m', labelAngle=-30)),
        y=alt.Y('Volume:Q', title='Volume', axis=alt.Axis(format='~s')),
        color=alt.condition(
            alt.datum['Volume'] > df_vol['Volume'].median(),
            alt.value('#29B6F6'),
            alt.value('rgba(41,182,246,0.35)')
        ),
        tooltip=[alt.Tooltip('Date:T', format='%d/%m/%Y'), alt.Tooltip('Volume:Q', format=',.0f')]
    ).properties(height=220).interactive()

    st.altair_chart(vol_chart, width='stretch')

    # Quick Stats
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("#### 📋 Quick Statistics")

    df_recent = df_daily.tail(5)
    df_month = df_daily.tail(22)
    df_year = df_daily.tail(252)

    col_s1, col_s2, col_s3 = st.columns(3)

    with col_s1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="kpi-label">📅 Last Week (5 sessions)</div>
            <div class="stat-row"><span class="stat-label">Highest</span><span class="stat-value">${df_recent['close'].max():,.2f}</span></div>
            <div class="stat-row"><span class="stat-label">Lowest</span><span class="stat-value">${df_recent['close'].min():,.2f}</span></div>
            <div class="stat-row"><span class="stat-label">Average</span><span class="stat-value">${df_recent['close'].mean():,.2f}</span></div>
        </div>
        """, unsafe_allow_html=True)

    with col_s2:
        st.markdown(f"""
        <div class="glass-card">
            <div class="kpi-label">📅 Last Month (~22 sessions)</div>
            <div class="stat-row"><span class="stat-label">Highest</span><span class="stat-value">${df_month['close'].max():,.2f}</span></div>
            <div class="stat-row"><span class="stat-label">Lowest</span><span class="stat-value">${df_month['close'].min():,.2f}</span></div>
            <div class="stat-row"><span class="stat-label">Average</span><span class="stat-value">${df_month['close'].mean():,.2f}</span></div>
        </div>
        """, unsafe_allow_html=True)

    with col_s3:
        year_high = df_year['close'].max()
        year_low = df_year['close'].min()
        st.markdown(f"""
        <div class="glass-card">
            <div class="kpi-label">📅 Last Year (~252 sessions)</div>
            <div class="stat-row"><span class="stat-label">52-Week High</span><span class="stat-value">${year_high:,.2f}</span></div>
            <div class="stat-row"><span class="stat-label">52-Week Low</span><span class="stat-value">${year_low:,.2f}</span></div>
            <div class="stat-row"><span class="stat-label">Average</span><span class="stat-value">${df_year['close'].mean():,.2f}</span></div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# TAB 3: LIVE PREDICTION
# ═══════════════════════════════════════════════
with tab_predict:
    st.markdown("#### 🔮 Forecast Sandbox / Live Predictor")
    st.markdown("Adjust the sliders below to change market assumptions and see the predicted Apple stock price for tomorrow immediately.")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.markdown("##### 💵 Price Assumptions")
        sim_price = st.slider("Today's Close Price (USD)", min_value=200.0, max_value=400.0, value=float(latest_close), step=0.5)
        sim_high = st.slider("Today's High Price (USD)", min_value=200.0, max_value=400.0, value=float(latest_row['high']), step=0.5)

    with col_p2:
        st.markdown("##### 📊 Indicator Assumptions")
        sim_rsi = st.slider("Relative Strength Index (RSI) (0–100)", min_value=10.0, max_value=90.0, value=float(latest_rsi), step=1.0)
        sim_macd = st.slider("MACD Trend Index", min_value=-5.0, max_value=5.0, value=float(latest_macd), step=0.1)

    # Build feature vector
    feature_cols = list(X_train.columns)
    sim_data = {}
    sim_data['lag_1'] = sim_price
    sim_data['lag_5'] = float(latest_row['lag_5'])
    sim_data['lag_7'] = float(latest_row['lag_7'])
    sim_data['rsi'] = sim_rsi
    sim_data['macd'] = sim_macd
    sim_data['volume'] = float(latest_row['volume'])
    sim_data['sma_20'] = float(latest_row['sma_20'])
    sim_data['sma_50'] = float(latest_row['sma_50'])
    sim_data['high'] = sim_high
    for col in feature_cols:
        if col not in sim_data:
            sim_data[col] = float(latest_row[col])

    sim_df = pd.DataFrame([sim_data])[feature_cols]

    # Ensemble prediction from simulation
    sim_preds = []
    for model_name, model in trainer.models.items():
        try:
            p = float(model.predict(sim_df)[0])
        except Exception:
            p = float(model.predict(sim_df.values)[0])
        sim_preds.append(sim_price + p)

    sim_ensemble = np.mean(sim_preds)
    sim_change = sim_ensemble - sim_price
    sim_change_pct = (sim_change / sim_price) * 100
    sim_signal = compute_signal(sim_change, sim_rsi, sim_macd)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    col_r1, col_r2 = st.columns(2)

    with col_r1:
        pred_color = "#00E676" if sim_change >= 0 else "#FF1744"
        pred_icon = "▲ UP" if sim_change >= 0 else "▼ DOWN"
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;">
            <div class="kpi-label">Predicted Price for Next Session</div>
            <div style="font-size:3rem; font-weight:800; color:{pred_color}; margin:8px 0;">${sim_ensemble:,.2f}</div>
            <div style="font-size:1rem; font-weight:600; color:{pred_color};">{pred_icon} ${abs(sim_change):,.2f} ({sim_change_pct:+,.2f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    with col_r2:
        if sim_signal == "BUY":
            st.success("📈 **Signal: BUY**\n\nBased on your assumptions, technical indicators suggest a bullish momentum. The price is likely to continue rising in the next session.")
        elif sim_signal == "SELL":
            st.error("📉 **Signal: SELL**\n\nTechnical indicators show increasing selling pressure. The market might be entering a correction phase.")
        else:
            st.warning("⚖️ **Signal: HOLD**\n\nPrice is consolidating, market is going sideways. No clear signal yet, continue observing.")


# ═══════════════════════════════════════════════
# TAB 4: TECHNICAL (For Advanced Users)
# ═══════════════════════════════════════════════
with tab_tech:
    st.markdown("#### 🔬 Model Performance Comparison")
    st.caption("The table below shows evaluation results on the test set. A smaller MAE and an R² closer to 1.0 indicate a better model.")

    # Style the metrics table
    df_m = df_metrics.copy()
    df_m['MSE'] = df_m['MSE'].map('{:,.4f}'.format)
    df_m['MAE'] = df_m['MAE'].map('{:,.4f}'.format)
    df_m['R2 Score'] = df_m['R2 Score'].map('{:,.4f}'.format)

    st.dataframe(df_m, width='stretch', hide_index=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("#### 📈 Actual vs Predicted Price Chart")

    selected_model = st.selectbox("Select model:", list(trainer.models.keys()))
    plot_path = os.path.join(trainer.report_path, f"{selected_model.lower()}_prediction.png")
    if os.path.exists(plot_path):
        st.image(plot_path, caption=f"Prediction results — {selected_model}", width='stretch')
    else:
        st.warning(f"⚠️ No prediction plot available for model {selected_model}.")

    # Algorithm descriptions
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    with st.expander("📖 Algorithm Explanations", expanded=False):
        st.markdown("""
| Model | How it works | Pros | Cons |
|---|---|---|---|
| **Linear Regression** | Assumes price moves in a linear fashion based on prior days | Simple, fast, lowest error | Cannot capture non-linear movements |
| **XGBoost** | Builds hundreds of small decision trees, each learning from the errors of the previous ones | Powerful, strong overfitting prevention | Cannot predict prices outside of historical bounds |
| **LightGBM** | Similar to XGBoost but uses leaf-wise splitting to accelerate training | Extremely fast, efficient with large datasets | Similar limitations to XGBoost |
| **CatBoost** | Handles categorical features well and reduces overfitting | Stable, requires minimal tuning | Slower than LightGBM |
| **ARIMA + LSTM** | ARIMA handles short-term trends, LSTM neural network learns long-term dependencies | Combines traditional statistics and deep learning | Accumulates errors over time, resource-intensive |
        """)
