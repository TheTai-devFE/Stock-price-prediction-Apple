# 🍎 THÔNG SỐ CHI TIẾT CÁC THUẬT TOÁN VÀ KẾT QUẢ THỰC NGHIỆM DỰ ĐOÁN GIÁ CỔ PHIẾU APPLE (AAPL)

Tài liệu này mô tả chi tiết các thuật toán, thư viện triển khai, cấu hình tham số thực tế và kết quả đánh giá thực nghiệm mới nhất thu được sau khi khắc phục lỗi rò rỉ dữ liệu (Data Leakage) và áp dụng các cải tiến kỹ thuật trong dự án dự báo giá cổ phiếu Apple (AAPL).

---

## 📈 Phần 1: Thông số chi tiết các thuật toán

Nhằm đảm bảo tính tái lập (Reproducibility) của nghiên cứu và tối ưu hóa hiệu năng dự báo giá cổ phiếu Apple (AAPL), hệ thống tiến hành thiết lập cấu hình tham số chi tiết cho từng lớp mô hình. Quy trình cấu hình được phân tách từ các mô hình học máy truyền thống, nhóm thuật toán Boosting, cho đến hệ thống mô hình lai (Hybrid) học sâu.

### 1. Linear Regression (Hồi quy tuyến tính)
Mô hình Hồi quy tuyến tính chuẩn tối thiểu hóa tổng bình phương sai số (Ordinary Least Squares - OLS) được triển khai thông qua thư viện `scikit-learn` nhằm thiết lập đường nền hiệu năng tuyến tính đơn giản.

**Bảng cấu hình tham số:**

| Tham số | Cấu hình thực tế | Mô tả / Ý nghĩa khoa học |
| :--- | :---: | :--- |
| `fit_intercept` | `True` | Chỉ định toán học cho phép mô hình tự động tính toán hệ số chặn (bias constant $\beta_0$), giúp đường hồi quy không bị ép buộc phải đi qua gốc tọa độ. |
| `copy_X` | `True` | Cơ chế quản lý bộ nhớ cho phép sao chép ma trận đặc trưng đầu vào $X$, ngăn chặn hiện tượng ghi đè dữ liệu gốc trong các tiến trình tính toán song song. |
| `n_jobs` | `None` | Thiết lập mặc định sử dụng một luồng tính toán lõi CPU duy nhất cho tiến trình tối ưu hóa OLS. |
| `positive` | `False` | Không ràng buộc các hệ số hồi quy ($\beta_j$), cho phép chúng nhận cả giá trị âm và dương để phản ánh chính xác các mối tương quan thuận nghịch của các chỉ báo tài chính. |

---

### 2. Extreme Gradient Boosting (XGBoost)
Mô hình XGBoost được tinh chỉnh tham số để tối ưu hóa khả năng học phi tuyến tính từ không gian đặc trưng dạng bảng đã được xử lý qua Apache Spark.

**Bảng cấu hình tham số:**

| Tham số | Cấu hình thực tế | Mô tả / Ý nghĩa khoa học |
| :--- | :---: | :--- |
| `n_estimators` | `1000` | Quy định số lượng cây quyết định tuần tự (boosting rounds) tối đa được khởi tạo để cực tiểu hóa hàm mất mát phần dư. |
| `learning_rate` | `0.05` | Tốc độ học (shrinkage factor), đóng vai trò thu nhỏ độ đóng góp của các cây mới, kiểm soát tốc độ hội tụ và chống lại hiện tượng quá khớp (Overfitting). |
| `max_depth` | `5` | Giới hạn độ sâu tối đa của cấu trúc cây quyết định bằng 5 tầng, cô lập mô hình khỏi việc học các tương tác quá chi tiết gây nhiễu tài chính. |
| `n_jobs` | `-1` | Kích hoạt tối đa tất cả các nhân xử lý logic của CPU (All Available Cores) để tối ưu hóa tiến trình xây dựng phân tách các nút cây song song. |

---

### 3. Light Gradient Boosting Machine (LightGBM)
Framework LightGBM vận hành dựa trên cơ chế phát triển cây dốc đứng theo lá (Leaf-wise), được cấu hình để kiểm soát chặt chẽ cấu trúc hình học của cây.

**Bảng cấu hình tham số:**

| Tham số | Cấu hình thực tế | Mô tả / Ý nghĩa khoa học |
| :--- | :---: | :--- |
| `n_estimators` | `1000` | Đồng bộ hóa hoàn toàn với XGBoost để phục vụ mục đích so sánh hiệu năng trực giao. |
| `learning_rate` | `0.05` | Đồng bộ hóa hoàn toàn với XGBoost để phục vụ mục đích so sánh hiệu năng trực giao. |
| `max_depth` | `5` | Đồng bộ hóa hoàn toàn với XGBoost để phục vụ mục đích so sánh hiệu năng trực giao. |
| `n_jobs` | `-1` | Khai thác toàn bộ hạ tầng đa luồng của hệ thống phần cứng. |
| `random_state` | `42` | Thiết lập hạt giống ngẫu nhiên cố định, đảm bảo ma trận phân tách không đổi qua các phiên chạy thực nghiệm khác nhau. |
| `verbose` | `-1` | Vô hiệu hóa các thông tin log và cảnh báo hệ thống không cần thiết, tối ưu hóa giao diện Terminal xuất đầu ra. |

---

### 4. Categorical Boosting (CatBoost)
Thuật toán CatBoost khai thác kiến trúc cây quyết định đối xứng (Symmetric Trees) để tối ưu hóa tốc độ dự báo trên hệ thống thực tế.

**Bảng cấu hình tham số:**

| Tham số | Cấu hình thực tế | Mô tả / Ý nghĩa khoa học |
| :--- | :---: | :--- |
| `iterations` | `1000` | Tương đương tham số `n_estimators`, thiết lập giới hạn 1000 vòng lặp Boosting. |
| `learning_rate` | `0.05` | Định nghĩa tốc độ học nhằm duy trì tính tương đồng thực nghiệm. |
| `depth` | `5` | Định nghĩa độ sâu tối đa của các cây đối xứng nhằm duy trì tính tương đồng thực nghiệm. |
| `random_seed` | `42` | Đồng bộ hạt giống ngẫu nhiên với hệ thống để đảm bảo tính tái lập. |
| `verbose` | `0` | Tắt tiến trình hiển thị log lặp chi tiết để giảm thiểu chi phí ghi tệp tốn tài nguyên. |

---

### 5. Hybrid Model (ARIMA + LSTM)
Mô hình lai đề xuất ARIMA + LSTM vận hành phân cấp qua hai giai đoạn, đòi hỏi việc cấu hình tham số toán học cho cả phân lớp thống kê tuyến tính và phân lớp mạng nơ-ron hồi quy.

#### A. Cấu hình phân lớp thống kê tuyến tính (ARIMA)

| Tham số | Cấu hình thực tế | Ý nghĩa toán học / Vai trò |
| :--- | :---: | :--- |
| Tự hồi quy (`p`) | `1` | Mô hình sử dụng chính xác giá trị của một phiên giao dịch liền trước ($t-1$) để tính toán trọng số dự báo tuyến tính cho phiên $t$. |
| Bậc sai phân (`d`) | `1` | Áp dụng sai phân bậc một ($\Delta P_t = P_t - P_{t-1}$) nhằm triệt tiêu tính xu hướng vĩ mô, biến đổi chuỗi giá cổ phiếu Apple từ trạng thái phi dừng (Non-stationary) về trạng thái dừng (Stationary) để thỏa mãn điều kiện ràng buộc toán học. |
| Trung bình trượt (`q`) | `1` | Sử dụng một sai số dự báo ngẫu nhiên trong quá khứ để điều chỉnh động lượng ngắn hạn cho đường giá. |

#### B. Kiến trúc tầng của mạng nơ-ron hồi quy (LSTM Layer Architecture)

| Tầng | Cấu hình / Thông số | Vai trò kỹ thuật |
| :--- | :--- | :--- |
| **Tầng đầu vào (Input Layer)** | Kích thước ma trận tensor: `(seq_length, 1)` với `seq_length = 10` | Chỉ định mạng LSTM áp dụng kỹ thuật cửa sổ trượt (Sliding Window), nhìn lại chuỗi 10 ngày phần dư liên tiếp trong quá khứ để làm cơ sở dự báo cho ngày kế tiếp. |
| **Tầng nơ-ron ẩn (LSTM Layer)** | `16 units` (đơn vị ô nhớ LSTM) | Sử dụng hàm kích hoạt phi tuyến tính Hyperbolic Tangent ($\tanh$) làm hàm kích hoạt lõi mặc định, giúp kiểm soát luồng thông tin qua các cổng và bắt được các phụ thuộc dài hạn tồn tại trong chuỗi phần dư tài chính. |
| **Tầng đầu ra (Dense Layer)** | `1 unit` kết nối đầy đủ (Fully Connected) | Thực hiện phép ánh xạ tuyến tính để trả ra duy nhất một giá trị dự báo phần dư phi tuyến cho phiên tiếp theo. |

#### C. Tham số cấu hình tiến trình huấn luyện (Training Hyperparameters)

| Tham số huấn luyện | Cấu hình thực tế | Mô tả / Ý nghĩa thực nghiệm |
| :--- | :---: | :--- |
| Bộ tối ưu hóa (`optimizer`) | `'adam'` | Sử dụng bộ tối ưu hóa Adam (Adaptive Moment Estimation) với cơ chế điều chỉnh tốc độ học động theo từng trọng số, giúp đẩy nhanh tốc độ hội tụ đạo hàm và tránh bẫy cực tiểu cục bộ. |
| Hàm mất mát (`loss`) | `'mse'` | Hàm mất mát Bình phương sai số trung bình đóng vai trò là hàm mục tiêu tối ưu, trực tiếp phạt nặng các lỗi sai phân phối lớn. |
| Số chu kỳ huấn luyện (`epochs`) | `8` | Giới hạn số chu kỳ huấn luyện lặp toàn bộ tập dữ liệu bằng 8 nhằm ngăn chặn hiện tượng mạng nơ-ron học vẹt các nhiễu ngẫu nhiên của phần dư (Overfitting). |
| Kích thước lô (`batch_size`) | `512` | Kích thước lô dữ liệu truyền vào kiến trúc mạng trong mỗi bước cập nhật trọng số ma trận, giúp tối ưu hóa bộ nhớ VRAM của hạ tầng GPU kép trên Kaggle. |

---

## 📊 Phần 2: Đánh giá thực nghiệm (Sau khi Cải tiến & Khắc phục Rò rỉ)

### 1. Kết quả thực nghiệm thực tế

Dưới đây là bảng tổng hợp các chỉ số đánh giá độ chính xác thu được từ lượt huấn luyện mới nhất sau khi loại bỏ hoàn toàn các cột rò rỉ dữ liệu (`open`, `high`, `low`, `adj_close`), đồng thời áp dụng **Target Transformation (Price Difference)** và **Early Stopping (Dừng sớm)**:

| Mô hình | RMSE (USD) | MAE (USD) | R2 Score | Trạng thái |
| :--- | :---: | :---: | :---: | :--- |
| **Linear Regression** | **3.15** | **2.13** | **0.9943** | Hoạt động tốt (Baseline) |
| **LightGBM** | **3.20** | **2.20** | **0.9941** | **Cải tiến vượt bậc** |
| **XGBoost** | **3.21** | **2.19** | **0.9941** | **Cải tiến vượt bậc** |
| **CatBoost** | **3.21** | **2.17** | **0.9941** | **Cải tiến vượt bậc** |
| **ARIMA + LSTM (Hybrid)** | **3.31** | **2.22** | **0.9937** | **Cải tiến vượt bậc** |

*Bảng 1: Kết quả thực nghiệm thực tế trên tập kiểm thử giá cổ phiếu Apple (AAPL)*

---

### 2. Nhận xét & Đánh giá chi tiết

#### Đánh giá tổng quan
Sau khi áp dụng các cải tiến kỹ thuật, hệ thống mô hình đã đạt được **sự cân bằng và chính xác thực tế vô cùng ấn tượng**:
- Các mô hình phi tuyến tính nâng cao (LightGBM, XGBoost, CatBoost) và mô hình học sâu lai (ARIMA-LSTM) đã hoàn toàn khắc phục được hiện tượng lệch pha và quá khớp. Chỉ số hệ số xác định $R^2$ đã chuyển từ mức âm sâu ($-1.20$ đến $-2.40$) lên **mức dương rất cao ($\approx 99.4\%$)** trên tập kiểm thử tương lai.
- Sai số dự đoán tuyệt đối trung bình (MAE) của tất cả các mô hình đã được kéo giảm sâu, chỉ còn dao động trong khoảng **2.13 đến 2.22 USD** cho mỗi cổ phiếu Apple (trong khi sai số trước đó khi bị rò rỉ hoặc chưa tối ưu lên tới 45.26 - 62.13 USD). Điều này chứng minh các mô hình cải tiến có khả năng đưa vào vận hành thực tế rất cao.

#### Đánh giá theo từng độ đo
- **RMSE (Root Mean Squared Error) & MAE (Mean Absolute Error):**
  - **Linear Regression** vẫn duy trì vị trí dẫn đầu nhẹ với RMSE là $3.15$ USD và MAE là $2.13$ USD nhờ học được quy luật xu hướng tuyến tính thuần túy từ giá ngày hôm trước (`lag_1`).
  - Các mô hình Boosting đã tiệm cận rất sát với Baseline. **LightGBM** đạt kết quả tốt nhất trong nhóm phi tuyến với RMSE là $3.20$ USD và MAE là $2.20$ USD, theo sau là **XGBoost** ($3.21$ USD) và **CatBoost** ($3.21$ USD với MAE đạt $2.17$ USD).
  - Mô hình lai **ARIMA + LSTM** cũng ghi nhận hiệu suất vượt trội (RMSE = $3.31$ USD, MAE = $2.22$ USD), cải thiện khổng lồ so với mức sai số cũ ($73.95$ USD).
- **$R^2$ (Hệ số xác định):**
  - Khoảng cách $R^2$ giữa các mô hình cực kỳ nhỏ (chênh lệch dưới $0.001$). Tất cả các mô hình đều giải thích được trên $99.3\%$ phương sai giá cổ phiếu ở tương lai nhờ học được cách dự đoán chênh lệch giá biến động thay vì dự đoán giá tuyệt đối trực tiếp.

---

### 3. Phân tích nguyên nhân sâu xa của sự cải tiến (Success Factors)

#### A. Hiệu quả từ Target Transformation (Dự báo chênh lệch giá)
- Bằng cách chuyển đổi biến mục tiêu thành sai khác giá hàng ngày: $\Delta Close_t = Close_t - Close_{t-1}$, chúng ta đã triệt tiêu xu hướng phi dừng (trend tăng trưởng dài hạn) của cổ phiếu Apple. Dữ liệu huấn luyện trở nên ổn định quanh mức trung bình 0 (stationarity).
- Điều này giúp các mô hình Boosting và LSTM không còn bị mất phương hướng khi gặp các vùng giá mới ở tương lai (test set) chưa từng xuất hiện ở quá khứ (train set). 
- Khi đánh giá, chúng ta khôi phục lại giá tuyệt đối bằng phương trình: $\widehat{Close}_t = Close_{t-1} + \widehat{\Delta Close}_t$. Phép cộng này giúp giữ dự báo bám sát theo chuyển động thực tế của thị trường một cách hợp lệ, không gây rò rỉ dữ liệu.

#### B. Hiệu quả từ Dừng sớm (Early Stopping) và Chuẩn hóa (Scaling)
- **Early Stopping:** Việc chia tập validation nội bộ (inner validation) và kích hoạt dừng sớm sau 30 vòng lặp đã giúp XGBoost, LightGBM, CatBoost tự động ngừng sinh thêm cây quyết định khi phát hiện bắt đầu quá khớp vào các nhiễu kỹ thuật của chỉ báo (RSI, MACD, v.v.), giúp mô hình giữ được tính tổng quát hóa cao.
- **ARIMA-LSTM Scaling:** Sử dụng `MinMaxScaler` đưa phần dư ARIMA về khoảng $[-1, 1]$ giúp mạng LSTM hội tụ rất nhanh và mượt mà hơn. Lớp `Dropout(0.1)` kết hợp giới hạn epochs ở mức tối ưu bằng 8 đã loại bỏ nhiễu ngẫu nhiên của phần dư, giúp LSTM bổ trợ hoàn hảo cho phần tuyến tính của ARIMA.
