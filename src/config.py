"""
Configuration for CEO Executive Dashboard.
All thresholds, paths, and constants in one place.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Final

# ──────────────────────────── Paths ────────────────────────────
DATA_DIR: Path = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parent.parent))
CACHE_DIR: Path = DATA_DIR / "data_cache"
EXCEL_GLOB: str = "*.xlsx"
SHEET_NAME: str = "Data"

# ──────────────────────────── Schema ───────────────────────────
EXPECTED_COLUMNS: list[str] = [
    "Vung", "KhuVuc", "TinhThanh", "QuanHuyen", "LoaiHoaDon", "SoHoaDon",
    "NgayHoaDon", "DienGiaiHD", "Kho", "MaKhachHangCu", "MaKhachHangMoi",
    "TenKhachHang", "TrinhDuocVien", "MaSanPhamCu", "MaSanPhamMoi", "TenSanPham",
    "SoLuong", "DonGia", "ThanhTien", "ChietKhau", "ThueVAT", "ThanhTienSauVAT",
    "Scheme", "TrangThai", "KenhBanHang", "LoaiDonHang", "NgayGhiSo", "IDChungTu",
    "IDXuatKho", "ChiNhanh", "LoaiSanPham", "GhiChu", "NgayDeNghiGiao",
    "NgayXuatKho", "SoPhieuXuatKho", "SoChungTu", "DiaChiGiaoHang", "LoaiGia",
    "DiaChiGiaoHangImport", "NgayDuyetDH", "NgayChotPXK", "TongKhoi", "TongTan",
    "DienThoaiLienHe", "SoDonHang", "CTKMSauThang11",
]

NUMERIC_COLUMNS: list[str] = [
    "SoLuong", "DonGia", "ThanhTien", "ChietKhau", "ThueVAT",
    "ThanhTienSauVAT", "TongKhoi", "TongTan",
]

DATE_COLUMNS: list[str] = [
    "NgayHoaDon", "NgayGhiSo", "NgayDeNghiGiao", "NgayXuatKho",
    "NgayDuyetDH", "NgayChotPXK",
]

# ──────────────────────────── Business Rules ───────────────────
# Product types to include in main revenue/volume view
LOAI_SP_HANG_HOA: Final[str] = "Hàng hóa"
LOAI_SP_BAO_BI: Final[str] = "Bao bì"   # partial match
LOAI_SP_VTQC: Final[str] = "Vật tư"     # partial match

# Sales channels
CHANNELS: list[str] = ["GT", "MT", "KA", "B2C", "OTH", "INTER"]

# ──────────────────────────── Dimension Labels (Vietnamese) ────
DIMENSION_LABELS: dict[str, str] = {
    "KenhBanHang": "Kênh bán hàng",
    "Vung": "Vùng",
    "KhuVuc": "Khu vực",
    "TinhThanh": "Tỉnh/Thành",
    "TenKhachHang": "Khách hàng",
    "TenSanPham": "Sản phẩm",
    "LoaiSanPham": "Loại sản phẩm",
    "ChiNhanh": "Chi nhánh",
    "TrinhDuocVien": "Trình dược viên",
    "Kho": "Kho",
}

# Drillable dimensions (must exist in real data columns)
DRILL_DIMENSIONS: list[str] = [
    "KenhBanHang", "Vung", "KhuVuc", "TinhThanh",
    "TenKhachHang", "TenSanPham", "ChiNhanh", "TrinhDuocVien",
]

# ──────────────────────────── Alert Thresholds ─────────────────
# % drop thresholds (decimal, e.g. 0.10 = 10%)
YELLOW_THRESHOLD: float = float(os.environ.get("YELLOW_THRESHOLD", "0.10"))
RED_THRESHOLD: float = float(os.environ.get("RED_THRESHOLD", "0.20"))

# Rolling window sizes (days / weeks)
ROLLING_DAYS: int = 28
ROLLING_WEEKS: int = 12

# Concentration alert: top-N customers/products share > threshold
CONCENTRATION_TOP_N: int = 5
CONCENTRATION_THRESHOLD: float = 0.50  # 50%

# Baseline comparison window (weeks)
BASELINE_WEEKS: int = 8

# ──────────────────────────── Data Quality ─────────────────────
# Tolerance for formula check: ThanhTienSauVAT ≈ ThanhTien + ChietKhau + ThueVAT
FORMULA_TOLERANCE: float = 1.0  # VND

# Max allowed null rate before alert (%)
NULL_RATE_ALERT: float = 0.05

# ──────────────────────────── App Settings ─────────────────────
AUTO_REFRESH_MINUTES: int = int(os.environ.get("AUTO_REFRESH_MINUTES", "5"))
PAGE_TITLE: str = "CEO Executive Cockpit"
PAGE_ICON: str = "📊"

# Currency formatting
CURRENCY_UNIT: str = "VND"
REVENUE_DIVISOR: float = 1_000_000  # Display in triệu VND
REVENUE_LABEL: str = "Triệu VND"

# ──────────────────────────── Custom CSS ───────────────────────
CUSTOM_CSS: str = """
<style>
    /* ── KPI metric cards ── */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.82rem !important;
        color: #495057 !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        color: #212529 !important;
    }
    [data-testid="stMetricDelta"] > div {
        font-size: 0.78rem !important;
    }

    /* ── Sidebar styling ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stDateInput label {
        color: #adb5bd !important;
        font-weight: 500 !important;
    }

    /* ── Header improvements ── */
    h1, h2, h3 {
        color: #1a1a2e !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }

    /* ── DataFrame styling ── */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        font-size: 1rem !important;
    }

    /* ── Alert badges ── */
    .alert-red { background: #d62728; color: white; padding: 2px 10px; border-radius: 4px; font-weight: bold; }
    .alert-yellow { background: #ff9800; color: white; padding: 2px 10px; border-radius: 4px; font-weight: bold; }
    .alert-green { background: #2ca02c; color: white; padding: 2px 10px; border-radius: 4px; font-weight: bold; }

    /* ── Footer ── */
    .footer-text {
        text-align: center;
        color: #6c757d;
        font-size: 0.8rem;
        padding: 12px 0;
    }
</style>
"""
