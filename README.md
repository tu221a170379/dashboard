# CEO Dashboard - VIKODA (VKD)

Dashboard phân tích doanh thu và dự báo nhu cầu sử dụng ARIMA forecasting cho công ty VIKODA.

## 📋 Mô tả

Ứng dụng web dashboard được xây dựng bằng Streamlit để:
- Phân tích doanh thu theo nhiều chiều: thời gian, sản phẩm, khu vực, kênh bán hàng, khách hàng
- Dự báo doanh thu tổng thể và theo sản phẩm sử dụng mô hình ARIMA
- Phân tích và dự báo tháng hiện tại với nhiều phương pháp ensemble

## 🏗️ Cấu trúc dự án

```
dashboard/
├── calculation_engine.py      # Module tính toán và dự báo
├── ceo_dashboard_webapp.py    # Ứng dụng Streamlit web
├── requirements.txt           # Các thư viện cần thiết
├── 2023.xlsx                 # Dữ liệu năm 2023
├── 2024.xlsx                 # Dữ liệu năm 2024
├── 2025.xlsx                 # Dữ liệu năm 2025
├── 2026.xlsx                 # Dữ liệu năm 2026
└── README.md                 # Tài liệu này
```

## 📦 Cài đặt

### Yêu cầu hệ thống
- Python 3.8 trở lên
- pip (Python package manager)

### Các bước cài đặt

1. **Clone repository**
```bash
git clone https://github.com/tu221a170379/dashboard.git
cd dashboard
```

2. **Cài đặt thư viện**
```bash
pip install -r requirements.txt
```

## 🚀 Chạy ứng dụng

```bash
streamlit run ceo_dashboard_webapp.py
```

Sau khi chạy lệnh trên, truy cập vào trình duyệt tại:
- **Local URL**: http://localhost:8501
- **Network URL**: http://<your-ip>:8501

## 📊 Tính năng

### 1. **Tổng Quan** (Overview)
- Tổng doanh thu tất cả các năm
- So sánh doanh thu theo năm
- Biểu đồ cột doanh thu theo năm

### 2. **Doanh Thu Tháng** (Monthly Revenue)
- Phân tích doanh thu theo từng tháng
- So sánh tháng hiện tại với tháng trước và cùng kỳ năm ngoái
- Biểu đồ đường xu hướng theo tháng
- Bảng chi tiết doanh thu từng tháng

### 3. **Doanh Thu Quý** (Quarterly Revenue)
- Tổng hợp và phân tích theo quý
- Biểu đồ cột doanh thu theo quý

### 4. **Top Sản Phẩm** (Top Products)
- Top 20 sản phẩm bán chạy nhất
- Biểu đồ ngang hiển thị doanh thu theo sản phẩm
- Bảng chi tiết với doanh thu, số lượng, số hóa đơn

### 5. **Vùng Miền** (Regional Analysis)
- Phân tích doanh thu theo vùng địa lý
- Biểu đồ tròn phân bố theo vùng
- Chi tiết theo tỉnh thành trong từng vùng

### 6. **Kênh Bán Hàng** (Sales Channels)
- Phân tích theo kênh phân phối
- Biểu đồ so sánh các kênh bán hàng

### 7. **Khách Hàng** (Customers)
- Top 20 khách hàng lớn nhất
- Phân tích chi tiết theo khách hàng

### 8. **ARIMA Dự Báo Tổng** (ARIMA Total Forecast)
- Dự báo doanh thu tổng thể 6 tháng tiếp theo
- Sử dụng mô hình ARIMA tự động tìm tham số tối ưu
- Hiển thị khoảng tin cậy (confidence interval)
- Các chỉ số thống kê: ADF test, AIC

### 9. **ARIMA Dự Báo Sản Phẩm** (ARIMA Products Forecast)
- Dự báo cho top 10 sản phẩm
- Mỗi sản phẩm có mô hình ARIMA riêng
- Bảng chi tiết dự báo từng sản phẩm

### 10. **Tháng Hiện Tại** (Current Month)
- Phân tích chi tiết tháng đang diễn ra
- Dự báo các ngày còn lại của tháng
- **Phương pháp Ensemble** kết hợp:
  - Historical weighted average (trung bình có trọng số từ các năm trước)
  - Recent trend (xu hướng 7 ngày gần nhất)
  - ARIMA daily forecast (dự báo ARIMA theo ngày)
- Dự báo tổng doanh thu cả tháng
- Top sản phẩm, khách hàng, kênh bán trong tháng

## 🧮 Calculation Engine Module

Module `calculation_engine.py` chứa tất cả logic tính toán và dự báo, bao gồm:

### Các hàm chính:

#### 1. **Utility Functions**
- `format_vnd(n)`: Format số thành chuỗi VND (Tỷ, Tr)
- `calculate_growth_percentage(current, previous)`: Tính % tăng trưởng

#### 2. **ARIMA Forecasting Functions**
- `find_best_arima_order(time_series, p_range, d_range, q_range)`: Tìm tham số ARIMA tối ưu
- `forecast_total_revenue(df, periods)`: Dự báo tổng doanh thu
- `forecast_products_revenue(df, top_n, periods)`: Dự báo theo sản phẩm

#### 3. **Current Month Analysis**
- `analyze_current_month(df)`: Phân tích toàn diện tháng hiện tại
- `_compute_daily_forecast(...)`: Tính dự báo hàng ngày (ensemble method)
- `_analyze_top_products(df_cur)`: Phân tích top sản phẩm
- `_analyze_top_customers(df_cur)`: Phân tích top khách hàng
- `_analyze_channels(df_cur)`: Phân tích kênh bán hàng

### Đặc điểm thiết kế:

1. **Type Hints**: Tất cả hàm đều có type hints rõ ràng
2. **Docstrings**: Mỗi hàm có docstring chi tiết với Args, Returns, Examples
3. **Modular**: Logic tách biệt khỏi UI, dễ test và maintain
4. **Error Handling**: Try-except để xử lý các trường hợp dữ liệu không đủ

## 📈 Phương pháp Dự báo

### ARIMA (AutoRegressive Integrated Moving Average)

Mô hình ARIMA được sử dụng với tham số (p, d, q):
- **p**: Order của AR (AutoRegressive)
- **d**: Degree của differencing (Integration)
- **q**: Order của MA (Moving Average)

**Tự động tìm tham số tối ưu**:
- Grid search trên không gian tham số
- Chọn mô hình có AIC (Akaike Information Criterion) thấp nhất
- Kiểm tra tính dừng với ADF test (Augmented Dickey-Fuller)

### Ensemble Method (Tháng Hiện Tại)

Kết hợp 3 phương pháp với trọng số:
1. **Historical Average** (30%): Trung bình có trọng số từ cùng tháng các năm trước
2. **Recent Trend** (30%): Trung bình động 7 ngày gần nhất
3. **ARIMA Daily** (40%): Dự báo ARIMA trên chuỗi theo ngày

## 🔧 Cấu hình

### Streamlit Configuration

File `.streamlit/config.toml` (nếu cần tùy chỉnh):

```toml
[theme]
primaryColor = "#2E86C1"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
port = 8501
headless = true
```

### Cache TTL

- **ARIMA Forecasts**: 3600 giây (1 giờ)
- **Current Month Analysis**: 1800 giây (30 phút)

Để refresh cache, nhấn **⟳ Rerun** trên Streamlit UI.

## 📊 Cấu trúc dữ liệu

Mỗi file Excel (2023.xlsx, 2024.xlsx, ...) cần có các cột:

- `NgayHoaDon`: Ngày hóa đơn (datetime)
- `SoHoaDon`: Số hóa đơn
- `TenSanPham`: Tên sản phẩm
- `TenKhachHang`: Tên khách hàng
- `SoLuong`: Số lượng
- `DonGia`: Đơn giá
- `ThanhTien`: Thành tiền (trước VAT)
- `ThanhTienSauVAT`: Thành tiền sau VAT
- `ChietKhau`: Chiết khấu
- `ThueVAT`: Thuế VAT
- `KenhBanHang`: Kênh bán hàng (optional)
- Các cột địa lý: Vùng, Tỉnh/Thành phố (optional)

## 🛠️ Development

### Thêm tính năng mới

1. **Tính toán/Dự báo**: Thêm vào `calculation_engine.py`
2. **UI/Rendering**: Thêm vào `ceo_dashboard_webapp.py`
3. **Test**: Viết test đơn giản để verify logic

### Best Practices

- Giữ logic tính toán trong `calculation_engine.py`
- Giữ logic hiển thị trong `ceo_dashboard_webapp.py`
- Sử dụng `@st.cache_data` cho các hàm tính toán nặng
- Thêm docstrings cho tất cả hàm mới

## 📝 License

Copyright © 2026 VIKODA

## 👥 Contributors

- Dashboard Development Team
- Data Analytics Team

## 📞 Liên hệ

Nếu có vấn đề hoặc câu hỏi, vui lòng liên hệ team phát triển.

---

**Powered by**: Streamlit + Plotly + ARIMA (statsmodels)
