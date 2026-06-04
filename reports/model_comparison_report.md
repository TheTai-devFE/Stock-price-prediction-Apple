# 🍎 BÁO CÁO SO SÁNH HIỆU NĂNG MÔ HÌNH DỰ BÁO GIÁ CỔ PHIẾU AAPL

Báo cáo kỹ thuật phân tích và so sánh hiệu năng của 5 mô hình: **Linear Regression**, **ARIMA + LSTM (Hybrid)**, **XGBoost**, **CatBoost**, và **LightGBM** trong Giai đoạn 6 & 7 của hệ thống Big Data Pipeline dự báo giá cổ phiếu Apple (AAPL).

---

## 📊 1. BẢNG SO SÁNH CHỈ SỐ ĐÁNH GIÁ (EVALUATION METRICS)

Các mô hình được kiểm thử trên tập dữ liệu kiểm thử (Test Data) chiếm 20% thời gian cuối cùng của chuỗi dữ liệu (không xáo trộn dữ liệu - Time Series Split để tránh rò rỉ dữ liệu tương lai).

| Mô hình | MSE (Mean Squared Error) ↓ | MAE (Mean Absolute Error) ↓ | $R^2$ Score (Coefficient of Determination) ↑ | Đánh giá tổng quan |
| :--- | :---: | :---: | :---: | :--- |
| **Linear Regression** | **1.2649** | **0.8369 USD** | **0.9990** | **Xuất sắc** (Bám sát hoàn hảo xu hướng thực tế) |
| **XGBoost** | 2707.8117 | 39.7473 USD | -1.0798 | **Yếu** (Bị giới hạn lá mô hình cây, không ngoại suy được) |
| **CatBoost** | 2796.1740 | 40.7226 USD | -1.1477 | **Yếu** (Bị giới hạn lá mô hình cây, không ngoại suy được) |
| **LightGBM** | 2815.0861 | 40.8496 USD | -1.1622 | **Yếu** (Bị giới hạn lá mô hình cây, không ngoại suy được) |
| **ARIMA + LSTM (Hybrid)** | 4705.3167 | 58.5824 USD | -2.6140 | **Yếu** (Bị lệch xu hướng dài hạn & tích lũy sai số) |

> [!IMPORTANT]
> - **MSE & MAE:** Càng thấp càng tốt. MAE của Linear Regression chỉ là **0.83 USD** trên mỗi cổ phiếu, trong khi các mô hình Boosting có sai số lên tới **39 - 41 USD**, và mô hình Hybrid là **58.58 USD**.
> - **$R^2$ Score:** Điểm tối đa là 1.0. Điểm âm của các mô hình Boosting và Hybrid cho thấy hoạt động của chúng kém hơn việc sử dụng giá trị trung bình lịch sử để dự báo xu hướng dài hạn.

---

## 📉 2. BIỂU ĐỒ KẾT QUẢ DỰ ĐOÁN (ACTUAL VS PREDICTED PLOTS)

Các biểu đồ dự đoán đã được xuất bản tự động vào thư mục dự án `reports/models/`:

1.  **Linear Regression Prediction:** `reports/models/linearregression_prediction.png`
2.  **ARIMA + LSTM Prediction:** `reports/models/arima_lstm_prediction.png`
3.  **XGBoost Prediction:** `reports/models/xgboost_prediction.png`
4.  **CatBoost Prediction:** `reports/models/catboost_prediction.png`
5.  **LightGBM Prediction:** `reports/models/lightgbm_prediction.png`

---

## 🧠 3. PHÂN TÍCH CHUYÊN SÂU (DEEP SCIENTIFIC ANALYSIS)

Một hiện tượng cực kỳ thú vị và mang tính bài bản cao trong Machine Learning tài chính đã xuất hiện trong bài toán này: **Linear Regression chiến thắng tuyệt đối trước các mô hình Boosting và mô hình Hybrid ARIMA+LSTM.** Dưới đây là lý do khoa học:

### A. Tại sao Linear Regression lại đạt độ chính xác gần như hoàn hảo?
1.  **Mối tương quan cực cao của Lag Features:** Trong thị trường tài chính, giá ngày hôm nay ($P_t$) phụ thuộc cực kỳ lớn vào giá ngày hôm qua ($P_{t-1}$). Feature Engineering của chúng ta đã tạo ra các biến trễ `lag_1`, `lag_5`, `lag_7`. 
2.  **Tính chất của Mô hình Tuyến tính:** Linear Regression dễ dàng học được trọng số tối ưu $\beta \approx 1$ cho `lag_1` và $\beta \approx 0$ cho các biến khác. Phương trình dự báo tuyến tính đơn giản này hoạt động cực kỳ hiệu quả trong việc bám đuổi giá cổ phiếu theo ngày.

### B. Tại sao các mô hình Tree-based Boosting (XGBoost, LightGBM, CatBoost) lại thất bại nặng nề?
1.  **Giới hạn Ngoại suy (Extrapolation Limits of Decision Trees):**
    *   Các mô hình cây quyết định (Decision Trees) phân chia không gian đặc trưng bằng các ngưỡng nhị phân (ví dụ: `if lag_1 > 150 then...`).
    *   Giá trị dự đoán tại các lá (leaves) là một **hằng số** (trung bình của các mẫu trong lá đó).
    *   Do đó, mô hình cây **không thể dự đoán bất kỳ giá trị nào lớn hơn giá trị lớn nhất có trong tập huấn luyện (Training Data)**.
2.  **Xu hướng tăng trưởng dài hạn của AAPL:** 
    *   Giá cổ phiếu Apple có xu hướng tăng mạnh theo thời gian. Tập Test nằm ở giai đoạn cuối (giá cổ phiếu đạt đỉnh cao mới từ năm 2023 - 2026 chưa từng xuất hiện trong tập Train).
    *   Khi đối mặt với dữ liệu Test có mức giá cao kỷ lục này, cả ba mô hình XGBoost, LightGBM và CatBoost chỉ có thể trả về giá trị dự đoán cao nhất mà chúng từng thấy trong tập Train (tạo thành một đường nằm ngang phẳng lỳ trên đồ thị dự đoán). Điều này dẫn đến sai số khổng lồ (MAE ~ 40 USD) và $R^2$ bị âm.

### C. Tại sao mô hình ARIMA + LSTM (Hybrid) lại gặp sai số lớn trong dự báo dài hạn (Out-of-sample)?
1.  **Sự suy giảm xu hướng dài hạn của ARIMA:** 
    *   ARIMA(1, 1, 1) là một mô hình chuỗi thời gian tuyến tính ngắn hạn xuất sắc. Tuy nhiên, khi dự báo tĩnh trên một tập Test dài (814 ngày, tức hơn 3 năm từ 2023 đến 2026), ARIMA sẽ nhanh chóng triệt tiêu xu hướng động ngắn hạn và hội tụ về một đường xu hướng tuyến tính phẳng (do tác động của sai phân bậc 1).
    *   Trong khi đó, AAPL tăng trưởng lũy kế cực kỳ mạnh mẽ vượt xa dự báo xu hướng tuyến tính phẳng này.
2.  **Sai số lũy kế phá hủy LSTM:**
    *   LSTM được huấn luyện trên chuỗi **sai số (residuals)** của ARIMA trên tập Train (trung bình xoay quanh 0).
    *   On tập Test, do ARIMA bị lệch xu hướng thực tế ngày càng xa, chuỗi sai số thực tế tăng vọt lên mức khổng lồ (~50 đến 60 USD).
    *   Mạng LSTM chưa từng được học các mức sai số khổng lồ như vậy trong tập Train, dẫn đến việc dự đoán sai số của nó bị mất phương hướng hoàn toàn, làm cộng dồn sai số và đẩy MAE lên mức **58.58 USD**.

---

## 💻 4. ĐIỂM NHẤN HỆ THỐNG: GIẢI QUYẾT XUNG ĐỘT LUỒNG (OPENMP DEADLOCK)
Trong quá trình huấn luyện mô hình lai ARIMA + LSTM trên macOS, hệ thống đã gặp hiện tượng **khóa chết tài nguyên (Deadlock)** ở cấp độ thư viện giữa `statsmodels` (chạy trên nền OpenBLAS/gfortran) và `TensorFlow` (chạy trên luồng OpenMP). 

Để giải quyết triệt để lỗi này, tôi đã thiết kế và triển khai thành công kiến trúc **Cô lập Tiến trình (Process Isolation)**:
*   **Tiến trình chính (Main Process):** Huấn luyện ARIMA, trích xuất residuals và lưu xuống file tạm `scratch/residuals.npy`.
*   **Tiến trình con cô lập (Subprocess):** Khởi chạy script độc lập `src/train_lstm_helper.py` và `src/predict_lstm_helper.py` trong một không gian tiến trình Python sạch. Giúp TensorFlow huấn luyện LSTM chỉ trong **0.65 giây** và dự đoán không bị treo luồng.
*   **Tiến trình chính:** Đọc kết quả từ tiến trình con để hoàn thành pipeline đánh giá một cách liền mạch.

---

## 🛠️ 5. ĐỀ XUẤT TRIỂN KHAI LÊN WEB APP (STAGE 8 & 9)
Dựa trên kết quả thực tế trên đồ thị và bảng so sánh chỉ số:
1.  **Linear Regression** là mô hình xuất sắc nhất để dự báo xu hướng tiếp diễn ngắn hạn và trung hạn của cổ phiếu nhờ khả năng tận dụng hoàn hảo Lag Features.
2.  **Đề xuất Web App:** 
    *   Sử dụng **Linear Regression** làm mô hình cốt lõi chạy realtime trên Dashboard.
    *   Thiết kế giao diện Streamlit kết nối MongoDB để hiển thị trực quan các đặc trưng trễ (Lag), các chỉ báo kỹ thuật (RSI, MACD) thu thập tự động từ MongoDB.
    *   Cho phép người dùng tương tác chọn ngày và xem biểu đồ trực quan dự báo giá trị thực tế của ngày hôm sau.

---
*Báo cáo được chuẩn bị bởi **David** - Senior Data Scientist & Big Data Engineer.*
