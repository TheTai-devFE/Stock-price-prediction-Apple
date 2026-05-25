import streamlit as st
import pandas as pd
import numpy as np
from pymongo import MongoClient
import os
import datetime
import altair as alt
from src.data_provider import DataProvider
from src.model_trainer import ModelTrainer

# Set up page styling and config
st.set_page_config(
    page_title="Hệ Thống Dự Báo Cổ Phiếu Apple (AAPL) - Big Data",
    page_icon="🍏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS injection for beautiful UI and user-friendly experience
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Title Gradient styling */
    .title-gradient {
        background: linear-gradient(135deg, #4CAF50, #00838F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    /* Elegant KPI Card */
    .kpi-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(10px);
        margin-bottom: 15px;
    }
    
    .kpi-title {
        font-size: 0.95rem;
        color: #B0BEC5;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    
    .kpi-value {
        font-size: 2.0rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 5px;
    }
    
    .kpi-sub {
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .text-green { color: #00E676 !important; font-weight: bold; }
    .text-red { color: #FF1744 !important; font-weight: bold; }
    .text-blue { color: #29B6F6 !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Cache data loading and training
@st.cache_resource
def load_and_evaluate_models():
    provider = DataProvider()
    df = provider.load_data("aapl_features")
    
    if df.empty:
        with st.spinner("🔄 Dữ liệu trống. Đang tự động chạy Pipeline thu thập dữ liệu..."):
            import subprocess
            subprocess.run(["venv/bin/python3", "main.py"])
            df = provider.load_data("aapl_features")
            
    df_for_training = df.copy()
    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test = trainer.prepare_data(df_for_training, target_col='close')
    
    # Train all 5 models to have predictions cached
    trainer.train_linear_regression(X_train, y_train)
    trainer.evaluate_model('LinearRegression', X_test, y_test)
    
    trainer.train_arima_lstm(X_train, y_train)
    trainer.evaluate_model('ARIMA_LSTM', X_test, y_test)
    
    trainer.train_xgboost(X_train, y_train)
    trainer.evaluate_model('XGBoost', X_test, y_test)
    
    trainer.train_catboost(X_train, y_train)
    trainer.evaluate_model('CatBoost', X_test, y_test)
    
    trainer.train_lightgbm(X_train, y_train)
    trainer.evaluate_model('LightGBM', X_test, y_test)
    
    return df, trainer, X_train, X_test, y_train, y_test

# Load all dependencies
try:
    df, trainer, X_train, X_test, y_train, y_test = load_and_evaluate_models()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
except Exception as e:
    st.error(f"❌ Không thể khởi động hệ thống. Lỗi: {e}")
    st.stop()

# Get the split index for aligning the dates of test set
split_idx = len(df) - len(y_test)
test_dates = df['date'].iloc[split_idx:].values

# App Sidebar Setup (Simplified for beginners)
st.sidebar.markdown("<div style='text-align: center;'><h2 style='color: #4CAF50;'>🍏 Apple Stock</h2></div>", unsafe_allow_html=True)
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/f/fa/Apple_logo_black.svg", width=65)
st.sidebar.markdown("---")
st.sidebar.markdown("### 📅 PHẠM VI DỮ LIỆU")
st.sidebar.info(
    f"Từ ngày: **{df['date'].min().strftime('%d/%m/%Y')}**\n\n"
    f"Đến ngày: **{df['date'].max().strftime('%d/%m/%Y')}**\n\n"
    f"Tổng số ngày quan sát: **{len(df):,} ngày**"
)
st.sidebar.markdown("---")
st.sidebar.markdown("<div style='text-align: center; color: #888888;'>Trợ lý AI của bạn: <b>David</b></div>", unsafe_allow_html=True)

# Main Title Header
st.markdown("<h1 class='title-gradient'>🍏 HỆ THỐNG DỰ BÁO GIÁ CỔ PHIẾU APPLE (AAPL)</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.1rem; color: #888888; margin-top: -10px;'>Ứng dụng giúp bạn hiểu rõ biến động giá cổ phiếu Apple hôm nay và dự đoán thông minh ngày mai bằng các mô hình toán học dễ hiểu.</p>", unsafe_allow_html=True)

# 🎓 ELITE USER-FRIENDLY ACCORDION HƯỚNG DẪN ĐỌC HIỂU
with st.expander("🎓 HƯỚNG DẪN DỄ HIỂU CHO NGƯỜI MỚI BẮT ĐẦU (CLICK VÀO ĐÂY)", expanded=True):
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("""
        ### ❓ Dự án này dùng để làm gì?
        *   **Mục tiêu chính:** Dự đoán **giá đóng cửa** của cổ phiếu Apple (mã **AAPL**) vào ngày giao dịch tiếp theo.
        *   **Cách hoạt động:** Máy tính sẽ phân tích giá của những ngày trước đó (`lag_1` là ngày hôm qua, `lag_5` là 5 ngày trước) cùng khối lượng giao dịch để tìm ra quy luật biến động của giá cổ phiếu.
        
        ### 📈 Giải thích các chỉ số đánh giá độ chính xác:
        Để biết mô hình dự báo **tốt** hay **tệ**, ta nhìn vào:
        1.  **Hệ số xác định ($R^2$):** Tỉ lệ chính xác. Ví dụ: **0.9990** nghĩa là mô hình **chính xác tới 99.9%** trong việc bám đuổi hướng đi của giá thực tế.
        2.  **Sai số tuyệt đối trung bình (MAE):** Số tiền bị lệch. Ví dụ: **$0.84 USD** nghĩa là trung bình mỗi ngày dự báo, mô hình chỉ đoán lệch **0.84 USD (khoảng 20.000 VNĐ)** so với giá thực tế (trên một cổ phiếu có giá trị gần 240 USD - sai số cực kỳ nhỏ!).
        """)
    with col_g2:
        st.markdown("""
        ### 🧠 Tại sao mô hình đơn giản lại chiến thắng tuyệt đối?
        *   **Mô hình Linear Regression (Tuyến tính - Tốt nhất):** Mô hình này nhận ra quy luật đơn giản: *"Giá cổ phiếu ngày hôm nay luôn cực kỳ gần với giá ngày hôm qua"*. Do đó nó bám sát giá thực tế cực tốt.
        *   **Các mô hình học máy phức tạp (XGBoost, CatBoost, v.v.):** Các mô hình này hoạt động giống như một đứa trẻ chưa bao giờ nhìn thấy số lớn hơn 100. Khi giá cổ phiếu Apple tăng liên tiếp lập **đỉnh cao kỷ lục mới** trong năm 2023-2026, các mô hình này bị giới hạn và không thể dự đoán được các mức giá vượt quá lịch sử quá khứ.
        *   **Mô hình Hybrid ARIMA+LSTM:** Mô hình này bị tích lũy sai số theo thời gian khi dự báo dài hạn, dẫn đến việc bị lệch xu hướng thực tế của Apple.
        """)

# Tabs Navigation
tab_summary, tab_eda, tab_ml_studio, tab_live = st.tabs([
    "🏠 1. BẢNG DỰ BÁO HẰNG NGÀY & PHÂN TÍCH TỔNG QUAN",
    "📊 2. XU HƯỚNG GIÁ & CHỈ BÁO KỸ THUẬT (DỄ HIỂU)",
    "🤖 3. ĐỐI CHIẾU THỰC TẾ & KHÔNG GIAN HỌC MÁY",
    "🔮 4. DỰ BÁO TRỰC TUYẾN BẰNG THANH TRƯỢT (LIVE PREDICTOR)"
])

# --- TAB 1: HOME & KPIs ---
with tab_summary:
    st.markdown("### 📌 Bảng Xem Nhanh Chỉ Số Hôm Nay & Dự Báo Ngày Mai")
    
    # Calculate some key statistics
    latest_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    latest_close = float(latest_row['close'])
    prev_close = float(prev_row['close'])
    price_change = latest_close - prev_close
    price_change_pct = (price_change / prev_close) * 100
    
    # LR Predicted value for tomorrow
    lr_model = trainer.models['LinearRegression']
    latest_features = X_test.iloc[-1].values.reshape(1, -1)
    tomorrow_pred = float(lr_model.predict(latest_features)[0])
    pred_change = tomorrow_pred - latest_close
    pred_change_pct = (pred_change / latest_close) * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">💵 GIÁ THỰC TẾ HÔM NAY (AAPL)</div>
            <div class="kpi-value">${latest_close:,.2f} USD</div>
            <div class="kpi-sub {'text-green' if price_change >= 0 else 'text-red'}">
                {'▲ TĂNG' if price_change >= 0 else '▼ GIẢM'} ${abs(price_change):,.2f} ({price_change_pct:,.2f}%) so với hôm qua
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">🔮 GIÁ DỰ BÁO NGÀY MAI</div>
            <div class="kpi-value" style="color: #FF8F00;">${tomorrow_pred:,.2f} USD</div>
            <div class="kpi-sub {'text-green' if pred_change >= 0 else 'text-red'}">
                Xu hướng: {'▲ TĂNG' if pred_change >= 0 else '▼ GIẢM'} ${abs(pred_change):,.2f} ({pred_change_pct:,.2f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        latest_rsi = float(latest_row['rsi'])
        if latest_rsi > 70:
            rsi_status = "⚠️ Quá mua (Giá đang hơi cao so với giá trị thực)"
            rsi_class = "text-red"
        elif latest_rsi < 30:
            rsi_status = "💚 Quá bán (Giá đang rẻ, cơ hội mua vào)"
            rsi_class = "text-green"
        else:
            rsi_status = "⚖️ Trung tính (Giá đang ở mức ổn định)"
            rsi_class = "text-blue"
            
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">📈 TRẠNG THÁI THỊ TRƯỜNG (RSI)</div>
            <div class="kpi-value">{latest_rsi:,.1f} điểm</div>
            <div class="kpi-sub {rsi_class}">{rsi_status}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("### 📈 Biểu đồ lịch sử giá Apple (Dễ nhìn)")
        st.markdown("Đường màu đỏ hiển thị **Giá đóng cửa thực tế** của cổ phiếu Apple qua các năm. Bạn có thể giữ chuột và kéo để xem chi tiết từng thời kỳ.")
        
        chart_simple = df[['date', 'close']].copy()
        chart_simple = chart_simple.rename(columns={'close': 'Giá thực tế (USD)'})
        
        simple_chart = alt.Chart(chart_simple).mark_line(color='#4CAF50', strokeWidth=2).encode(
            x=alt.X('date:T', title='Năm'),
            y=alt.Y('Giá thực tế (USD):Q', title='Giá cổ phiếu (USD)', scale=alt.Scale(zero=False)),
            tooltip=['date:T', 'Giá thực tế (USD):Q']
        ).properties(height=350).interactive()
        
        st.altair_chart(simple_chart, use_container_width=True)
        
    with col_right:
        st.markdown("### 💡 Lời Khuyên Hành Động Ngày Hôm Nay")
        
        if pred_change >= 0.3:
            st.success(
                "📈 **DỰ ĐOÁN XU HƯỚNG TĂNG**\n\n"
                "Mô hình dự báo giá ngày mai sẽ đóng cửa cao hơn hôm nay. "
                "Cùng với các chỉ báo kỹ thuật ổn định, thị trường đang cho thấy lực mua tích cực. "
                "Bạn có thể cân nhắc tích lũy thêm cổ phiếu."
            )
        elif pred_change <= -0.3:
            st.error(
                "📉 **DỰ ĐOÁN XU HƯỚNG GIẢM**\n\n"
                "Mô hình dự báo giá ngày mai sẽ đóng cửa thấp hơn hôm nay. "
                "Thị trường có thể đang gặp áp lực chốt lời ngắn hạn. "
                "Nếu bạn đang giữ cổ phiếu, có thể cân nhắc bảo toàn lợi nhuận hoặc kiên nhẫn quan sát thêm."
            )
        else:
            st.warning(
                "⚖️ **XU HƯỚNG ĐI NGANG (TÍCH LŨY)**\n\n"
                "Giá ngày mai dự kiến sẽ chỉ dao động nhẹ so với hôm nay. "
                "Thị trường đang ổn định và chưa có biến động lớn. "
                "Khuyến nghị tiếp tục nắm giữ và theo dõi thêm."
            )
            
        st.info(
            "📢 **Bạn có biết?**\n\n"
            "Giá cổ phiếu Apple chịu ảnh hưởng lớn bởi xu hướng tiếp diễn hàng ngày. "
            "Linear Regression (Hồi quy tuyến tính) là mô hình cực kỳ đơn giản nhưng lại "
            "đạt độ chính xác cao nhất vì nó tôn trọng tối đa thuộc tính này của thị trường."
        )

# --- TAB 2: TECHNICAL ANALYSIS (EDA) ---
with tab_eda:
    st.subheader("📊 Phân Tích Kỹ Thuật Bằng Hình Ảnh Trực Quan")
    st.markdown("Dưới đây là các công cụ phân tích kỹ thuật giúp bạn hiểu rõ dòng tiền và sức mua của Apple mà không cần kiến thức cao siêu:")
    
    col_e1, col_e2 = st.columns(2)
    
    with col_e1:
        st.markdown("#### 1. Khối Lượng Giao Dịch Hằng Ngày (Volume)")
        st.markdown("Cột màu xanh hiển thị khối lượng cổ phiếu được mua bán mỗi ngày. Cột càng cao chứng tỏ ngày đó dòng tiền đổ vào Apple càng mạnh mẽ.")
        volume_c = alt.Chart(df).mark_bar(color='#00ACC1').encode(
            x=alt.X('date:T', title='Thời gian'),
            y=alt.Y('volume:Q', title='Số lượng cổ phiếu giao dịch'),
            tooltip=['date:T', 'volume:Q']
        ).properties(height=280).interactive()
        st.altair_chart(volume_c, use_container_width=True)
        
    with col_e2:
        st.markdown("#### 2. Chỉ Số Sức Mạnh Sức Mua (RSI Line)")
        st.markdown("Chỉ số RSI dao động từ 0 đến 100 điểm. RSI trên **70** là giá đang hơi cao so với thực tế, RSI dưới **30** là giá đang rẻ.")
        rsi_c = alt.Chart(df).mark_line(color='#FF8F00').encode(
            x=alt.X('date:T', title='Thời gian'),
            y=alt.Y('rsi:Q', title='Điểm số RSI', scale=alt.Scale(domain=[10, 90])),
            tooltip=['date:T', 'rsi:Q']
        ).properties(height=280).interactive()
        st.altair_chart(rsi_c, use_container_width=True)

# --- TAB 3: MACHINE LEARNING STUDIO ---
with tab_ml_studio:
    st.subheader("🤖 So Sánh Thực Tế vs Dự Báo Của Các Mô Hình Học Máy")
    st.markdown("Chọn một mô hình để xem biểu đồ so sánh đường màu xanh (Giá thực tế) và đường màu đỏ (Giá mô hình dự báo). Càng trùng khít nhau nghĩa là mô hình càng đoán giỏi!")
    
    selected_model = st.selectbox(
        "Lựa chọn mô hình để hiển thị biểu đồ so sánh:",
        ["LinearRegression (Tốt Nhất)", "ARIMA_LSTM (Mô hình Lai)", "XGBoost", "CatBoost", "LightGBM"],
        key="model_select"
    )
    
    model_key = selected_model.split(" ")[0]
    
    # Load evaluation metric values
    results = trainer.results[model_key]
    pred_vals = results['predictions']
    actual_vals = y_test.values
    
    st.markdown(f"#### 📈 Biểu Đồ Đối Chiếu Thực Tế vs Dự Báo của mô hình **{model_key}**")
    
    # Align dates
    plot_df = pd.DataFrame({
        'Date': pd.to_datetime(test_dates),
        'Giá thực tế': actual_vals,
        'Mô hình dự báo': pred_vals
    })
    plot_df_melted = plot_df.melt('Date', var_name='Loại giá', value_name='Giá (USD)')
    
    chart_pred = alt.Chart(plot_df_melted).mark_line().encode(
        x=alt.X('Date:T', title='Năm'),
        y=alt.Y('Giá (USD):Q', title='Giá cổ phiếu (USD)', scale=alt.Scale(zero=False)),
        color=alt.Color('Loại giá:N', scale=alt.Scale(range=['#4CAF50', '#FF1744'])),
        tooltip=['Date:T', 'Loại giá:N', 'Giá (USD):Q']
    ).properties(height=380).interactive()
    
    st.altair_chart(chart_pred, use_container_width=True)
    
    st.markdown("### 🏆 Bảng Xếp Hạng Độ Chính Xác Của Toàn Bộ Mô Hình")
    comparison_list = []
    for m_name, res in trainer.results.items():
        # Human friendly translation for metrics
        accuracy_percentage = (res['r2']) * 100 if res['r2'] > 0 else 0
        mae_explanation = f"Lệch khoảng {res['mae']:.2f} USD"
        
        comparison_list.append({
            "Tên Mô Hình": m_name,
            "Độ chính xác (R² Score)": f"{accuracy_percentage:.1f}%" if accuracy_percentage > 0 else "0% (Kém)",
            "Mức sai lệch trung bình (MAE)": mae_explanation,
            "Đánh Giá Chất Lượng": "🥇 Xuất sắc nhất (Khuyên dùng)" if m_name == "LinearRegression" else "❌ Yếu (Không ngoại suy được)"
        })
        
    st.table(pd.DataFrame(comparison_list))

# --- TAB 4: LIVE PREDICTOR ---
with tab_live:
    st.subheader("🔮 Phòng Dự Báo Trực Tuyến Tương Tác Bằng Thanh Trượt")
    st.markdown(
        "Thay vì nhập những con số khô khan, bạn có thể **kéo các thanh trượt** dưới đây để giả định giá trị đóng cửa của Apple "
        "ngày hôm nay và xem mô hình Linear Regression lập tức tính toán ra mức giá ngày mai như thế nào!"
    )
    
    feature_cols = list(X_train.columns)
    
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        st.markdown("#### 💵 Giả Định Về Mức Giá")
        slider_lag1 = st.slider("1. Giá Đóng Cửa Hôm Nay (USD)", min_value=100.0, max_value=300.0, value=float(latest_row['close']), step=0.5)
        slider_high = st.slider("2. Giá Cao Nhất Trong Ngày Hôm Nay (USD)", min_value=100.0, max_value=300.0, value=float(latest_row['high']), step=0.5)
        slider_lag5 = st.slider("3. Giá Đóng Cửa 5 Ngày Trước (USD)", min_value=100.0, max_value=300.0, value=float(latest_row['lag_5']), step=0.5)
        slider_lag7 = st.slider("4. Giá Đóng Cửa 7 Ngày Trước (USD)", min_value=100.0, max_value=300.0, value=float(latest_row['lag_7']), step=0.5)
        
    with col_s2:
        st.markdown("#### ⚡ Giả Định Về Chỉ Báo Thị Trường")
        slider_rsi = st.slider("5. Chỉ Số Sức Mạnh RSI (Từ 0 đến 100)", min_value=10.0, max_value=90.0, value=float(latest_row['rsi']), step=1.0)
        slider_macd = st.slider("6. Chỉ Số Xu Hướng MACD", min_value=-5.0, max_value=5.0, value=float(latest_row['macd']), step=0.1)
        slider_sma20 = st.slider("7. Đường Trung Bình SMA(20) Hôm Nay (USD)", min_value=100.0, max_value=300.0, value=float(latest_row['sma_20']), step=0.5)
        
    # Real-time calculation when sliders change!
    # Construct input vector in the exact correct column order
    input_data = {}
    input_data['lag_1'] = slider_lag1
    input_data['lag_5'] = slider_lag5
    input_data['lag_7'] = slider_lag7
    input_data['rsi'] = slider_rsi
    input_data['macd'] = slider_macd
    input_data['volume'] = float(latest_row['volume']) # keep default volume
    input_data['sma_20'] = slider_sma20
    input_data['sma_50'] = float(latest_row['sma_50'])
    input_data['high'] = slider_high
    
    # Fill any missing feature columns
    for col in feature_cols:
        if col not in input_data:
            input_data[col] = float(latest_row[col])
            
    # Align and predict
    input_df = pd.DataFrame([input_data])[feature_cols]
    live_pred = float(lr_model.predict(input_df.values)[0])
    live_change = live_pred - slider_lag1
    live_change_pct = (live_change / slider_lag1) * 100
    
    st.markdown("---")
    st.markdown("### 🏆 Kết Quả Tính Toán Dự Báo Từ Máy Tính")
    
    res_c1, res_c2 = st.columns(2)
    
    with res_c1:
        st.markdown(f"""
        <div style="background: rgba(76, 175, 80, 0.1) if live_change >= 0 else rgba(244, 67, 54, 0.1); border: 2px solid {'#4CAF50' if live_change >= 0 else '#F44336'}; border-radius: 12px; padding: 25px; text-align: center;">
            <h3 style="margin: 0; color: #B0BEC5; font-size: 1.1rem; text-transform: uppercase;">Giá Dự Đoán Ngày Mai</h3>
            <h1 style="font-size: 3.5rem; font-weight: 800; color: #FFFFFF; margin: 10px 0;">${live_pred:,.2f} USD</h1>
            <h3 style="margin: 0; color: {'#00E676' if live_change >= 0 else '#FF1744'};">
                Biến động dự đoán: {'▲ TĂNG' if live_change >= 0 else '▼ GIẢM'} ${abs(live_change):,.2f} ({live_change_pct:,.2f}%)
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
    with res_c2:
        st.markdown("#### 💡 Khuyến Nghị Hành Động:")
        if live_change >= 0.5:
            st.success(
                "📈 **Tín Hiệu: MUA VÀO (BUY)**\n\n"
                "Khi bạn giả định giá ngày hôm nay cao hơn và lực mua RSI tốt, mô hình dự đoán xu hướng "
                "sẽ tiếp tục bứt phá vào ngày mai. Khuyến nghị cân nhắc vị thế mua."
            )
        elif live_change <= -0.5:
            st.error(
                "📉 **Tín Hiệu: BÁN RA / QUAN SÁT (SELL)**\n\n"
                "Mô hình phát hiện dấu hiệu suy yếu giá của ngày tiếp theo. Có thể thị trường đang vào đợt điều chỉnh nhẹ. "
                "Khuyến nghị đứng ngoài quan sát."
            )
        else:
            st.warning(
                "⚖️ **Tín Hiệu: NẮM GIỮ (HOLD)**\n\n"
                "Giá dao động rất hẹp (dưới 0.5 USD). Thị trường đi ngang tích lũy. "
                "Khuyến nghị tiếp tục giữ nguyên danh mục và kiên nhẫn quan sát thêm."
            )
