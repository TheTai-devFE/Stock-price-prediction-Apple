# Bảng Theo Dõi Lỗi Hệ Thống (ERRORS.md)

## [2026-05-30 22:55] - Lỗi Thiếu Thư Viện tabulate Khi Gọi to_markdown()

- **Type**: Runtime / Integration
- **Severity**: Medium
- **File**: `src/model_trainer.py:181`
- **Agent**: David
- **Root Cause**: Hàm `df.to_markdown()` của thư viện Pandas yêu cầu gói phụ thuộc `tabulate` được cài đặt trong môi trường Python, nhưng môi trường ảo `venv` hiện tại của dự án chưa được cài đặt thư viện này.
- **Error Message**:
  ```
  ImportError: `Import tabulate` failed. Use pip or conda to install the tabulate package.
  ```
- **Fix Applied**: Thay đổi phương thức tạo bảng Markdown trong `save_metrics_report` tại `src/model_trainer.py` để sử dụng một đoạn mã chuyển đổi thủ công đơn giản nhưng hiệu quả, loại bỏ hoàn toàn việc phụ thuộc vào thư viện `tabulate`.
- **Prevention**: Tránh sử dụng các tính năng nâng cao của thư viện bên thứ ba mà yêu cầu cài thêm gói phụ thuộc chưa có trong `requirements.txt`.
- **Status**: Fixed
