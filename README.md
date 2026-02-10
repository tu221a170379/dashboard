# CEO Executive Cockpit + Early Warning Dashboard

Dashboard phân tích doanh thu & sản lượng dành cho CEO ngành FMCG/Beverage.
Tự động đọc file Excel, tính KPI, cảnh báo sớm khi doanh thu/sản lượng giảm bất thường.

## Tính năng chính

| Tab | Nội dung |
|-----|----------|
| **Overview** | KPI cards (Today/WTD/MTD/QTD/YTD), trend charts, YoY comparison, executive summary |
| **Drill-Down** | Bộ lọc kênh/vùng/sản phẩm/khách hàng, Top/Bottom analysis, Waterfall chart |
| **Alerts & DQ** | Bảng cảnh báo RED/YELLOW/GREEN, Data Quality report, Export Excel |

## Cấu trúc thư mục

```
Data/                     ← Workspace root
├── 2023.xlsx             ← File dữ liệu (sheet "Data")
├── 2024.xlsx
├── 2025.xlsx
├── 2026.xlsx
├── app.py                ← Streamlit entry point
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── config.py         ← Cấu hình, ngưỡng, đường dẫn
│   ├── etl.py            ← Extract-Transform-Load + Data Quality
│   ├── metrics.py        ← Tính KPI doanh thu & sản lượng
│   ├── alerts.py         ← Early warning alerts
│   └── viz.py            ← Plotly chart builders
├── tests/
│   ├── __init__.py
│   └── test_metrics.py   ← Unit tests (pytest)
└── data_cache/           ← Auto-generated parquet cache
```

## Hướng dẫn chạy (VS Code)

### 1. Tạo môi trường ảo

```bash
cd "D:\Công việc\BC doanh thu\Data"
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 3. Chạy Dashboard

```bash
streamlit run app.py
```

Mở trình duyệt tại `http://localhost:8501`.

## Cấu hình

| Biến môi trường | Mặc định | Mô tả |
|---|---|---|
| `DATA_DIR` | Thư mục chứa app.py | Đường dẫn thư mục chứa file .xlsx |
| `YELLOW_THRESHOLD` | 0.10 | Ngưỡng cảnh báo vàng (10%) |
| `RED_THRESHOLD` | 0.20 | Ngưỡng cảnh báo đỏ (20%) |
| `AUTO_REFRESH_MINUTES` | 5 | Tần suất auto-refresh (phút) |

## Data Schema

File Excel cần có sheet tên **"Data"** với 46 cột chuẩn:
`Vung, KhuVuc, TinhThanh, ..., TongKhoi, TongTan, DienThoaiLienHe, SoDonHang, CTKMSauThang11`

## KPI Definitions

- **Gross Sales**: `sum(ThanhTien)` cho dòng Hàng hóa
- **Net Revenue**: `sum(ThanhTienSauVAT)` toàn bộ dòng (bao gồm chiết khấu)
- **Paid Units**: `SoLuong` where `DonGia>0, ThanhTienSauVAT>0, LoaiSanPham='Hàng hóa'`
- **Promo Units**: `SoLuong` where `DonGia=0, ThanhTienSauVAT=0, Scheme not null`
- **ASP**: `Net Revenue / Paid Units`
- **Discount Rate**: `|sum(ChietKhau)| / sum(ThanhTien)`

## Alert Rules

- Daily revenue vs rolling 28-day average & D-7
- Weekly WoW & YoY comparison
- MTD vs same MTD last year
- Channel/Region driver decomposition (top-5 drops)
- Customer/Product concentration (top-5 share > 50%)
- Abnormal discount rate vs rolling 12-week average
- Promo "burn volume" detection

## Tests

```bash
pytest tests/ -v
```

## License

Internal use only.
