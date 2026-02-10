"""
ETL module – Extract, Transform, Load for CEO Dashboard.

Responsibilities:
  - Discover & fingerprint Excel files in DATA_DIR
  - Read all sheets named "Data", concat into one DataFrame
  - Clean types, add computed columns (Year, Month, Week, Quarter, etc.)
  - Cache to parquet for fast reload
  - Data quality checks
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import (
    CACHE_DIR,
    DATA_DIR,
    DATE_COLUMNS,
    EXCEL_GLOB,
    EXPECTED_COLUMNS,
    FORMULA_TOLERANCE,
    LOAI_SP_HANG_HOA,
    NULL_RATE_ALERT,
    NUMERIC_COLUMNS,
    SHEET_NAME,
)

logger = logging.getLogger(__name__)


# ──────────────────────── File Fingerprinting ──────────────────
def _file_fingerprint(path: Path) -> str:
    """Return a lightweight fingerprint based on path + mtime + size."""
    stat = path.stat()
    raw = f"{path.name}|{stat.st_mtime}|{stat.st_size}"
    return hashlib.md5(raw.encode()).hexdigest()


def get_data_fingerprint(data_dir: Optional[Path] = None) -> str:
    """Combined fingerprint for all Excel files in data_dir."""
    data_dir = data_dir or DATA_DIR
    files = sorted(data_dir.glob(EXCEL_GLOB))
    if not files:
        return "EMPTY"
    parts = [_file_fingerprint(f) for f in files]
    combined = "|".join(parts)
    return hashlib.md5(combined.encode()).hexdigest()


# ──────────────────────── Data Loading ─────────────────────────
def discover_files(data_dir: Optional[Path] = None) -> list[Path]:
    """Return sorted list of xlsx files in data_dir (exclude temp files)."""
    data_dir = data_dir or DATA_DIR
    files = [
        f for f in sorted(data_dir.glob(EXCEL_GLOB))
        if not f.name.startswith("~$") and not f.name.startswith("_")
    ]
    logger.info("Discovered %d Excel files in %s", len(files), data_dir)
    return files


def read_single_file(path: Path) -> pd.DataFrame:
    """Read one Excel file, sheet 'Data', with basic validation."""
    logger.info("Reading %s …", path.name)
    df = pd.read_excel(path, sheet_name=SHEET_NAME, engine="openpyxl")
    # Validate columns
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        logger.warning("File %s missing columns: %s", path.name, missing)
    df["_source_file"] = path.name
    return df


def load_all_files(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Read & concat all Excel files into one DataFrame."""
    files = discover_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No Excel files found in {data_dir or DATA_DIR}")
    dfs = [read_single_file(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    logger.info("Loaded %d rows from %d files", len(df), len(files))
    return df


# ──────────────────────── Cleaning / Transform ─────────────────
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean types, fill defaults, add computed time columns."""
    df = df.copy()

    # ─── Numeric columns: coerce to float ───
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # ─── Date columns: coerce to datetime ───
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # ─── String columns: strip whitespace, normalise empty ───
    str_cols = ["TenSanPham", "LoaiSanPham", "KenhBanHang", "Scheme",
                "Vung", "KhuVuc", "TinhThanh", "TenKhachHang",
                "MaKhachHangMoi", "MaSanPhamMoi", "LoaiDonHang"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"": None, "None": None, "nan": None, "NaN": None})

    # ─── Force all remaining object columns to string (fixes mixed types for parquet) ───
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str)
            df[col] = df[col].replace({"nan": None, "None": None, "NaN": None, "": None})

    # ─── Computed time columns from NgayHoaDon ───
    if "NgayHoaDon" in df.columns:
        dt = df["NgayHoaDon"]
        df["Year"] = dt.dt.year
        df["Month"] = dt.dt.month
        df["Quarter"] = dt.dt.quarter
        df["Week"] = dt.dt.isocalendar().week.astype(int)
        df["DayOfWeek"] = dt.dt.dayofweek  # 0=Mon
        df["Date"] = dt.dt.normalize()  # datetime64 for parquet compat
        df["YearMonth"] = dt.dt.to_period("M").astype(str)
        df["YearWeek"] = dt.dt.strftime("%G-W%V")

    # ─── Row type flags ───
    df["is_hang_hoa"] = df["LoaiSanPham"].str.contains(LOAI_SP_HANG_HOA, case=False, na=False)
    df["is_bao_bi"] = df["LoaiSanPham"].str.contains("Bao bì|bao bi", case=False, na=False)
    df["is_vtqc"] = df["LoaiSanPham"].str.contains("Vật tư|vat tu", case=False, na=False)

    # has product name (not null, not empty)
    df["has_product"] = df["TenSanPham"].notna()

    # Discount / adjustment row: no product, SoLuong=0
    df["is_discount_row"] = (~df["has_product"]) & (df["SoLuong"] == 0)

    # Paid units: SoLuong>0, DonGia>0, ThanhTienSauVAT>0, Hàng hóa
    df["is_paid"] = (
        (df["SoLuong"] > 0)
        & (df["DonGia"] > 0)
        & (df["ThanhTienSauVAT"] > 0)
        & df["is_hang_hoa"]
    )

    # Promo units: SoLuong>0, DonGia==0, ThanhTienSauVAT==0, Scheme not null
    df["is_promo"] = (
        (df["SoLuong"] > 0)
        & (df["DonGia"] == 0)
        & (df["ThanhTienSauVAT"] == 0)
        & df["Scheme"].notna()
    )

    # Volume-eligible rows (for TongTan, TongKhoi, SoLuong aggregation)
    df["is_volume_eligible"] = df["has_product"] & (df["SoLuong"] > 0) & df["is_hang_hoa"]

    return df


# ──────────────────────── Caching ──────────────────────────────
def _cache_path(fingerprint: str) -> Path:
    return CACHE_DIR / f"data_{fingerprint}.parquet"


def load_with_cache(data_dir: Optional[Path] = None, force_reload: bool = False) -> pd.DataFrame:
    """Load data using parquet cache; rebuild if fingerprint changed."""
    data_dir = data_dir or DATA_DIR
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    fp = get_data_fingerprint(data_dir)
    cache = _cache_path(fp)

    if cache.exists() and not force_reload:
        logger.info("Loading from cache %s", cache.name)
        df = pd.read_parquet(cache)
        # Restore date columns
        for col in ["NgayHoaDon"] + DATE_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        # Ensure boolean columns are actual bool dtype after parquet round-trip
        bool_cols = [
            "is_hang_hoa", "is_bao_bi", "is_vtqc", "has_product",
            "is_discount_row", "is_paid", "is_promo", "is_volume_eligible",
        ]
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        return df

    # Fresh load
    df = load_all_files(data_dir)
    df = clean_dataframe(df)
    df.to_parquet(cache, index=False)
    logger.info("Cached %d rows to %s", len(df), cache.name)

    # Cleanup old caches
    for old in CACHE_DIR.glob("data_*.parquet"):
        if old != cache:
            old.unlink(missing_ok=True)

    return df


# ──────────────────────── Data Quality ─────────────────────────
def run_data_quality_checks(df: pd.DataFrame) -> list[dict]:
    """Return a list of data quality findings."""
    findings: list[dict] = []
    total_rows = len(df)
    if total_rows == 0:
        findings.append({"check": "empty_data", "severity": "RED", "detail": "Dataset is empty"})
        return findings

    # 1) Null rates for key columns
    key_cols = ["NgayHoaDon", "TenKhachHang", "KenhBanHang", "ThanhTienSauVAT", "SoLuong"]
    for col in key_cols:
        if col in df.columns:
            null_rate = df[col].isna().mean()
            if null_rate > NULL_RATE_ALERT:
                findings.append({
                    "check": "high_null_rate",
                    "severity": "YELLOW" if null_rate < 0.15 else "RED",
                    "detail": f"Column '{col}' null rate: {null_rate:.1%}",
                })

    # 2) Duplicate invoices: IDChungTu + MaSanPhamMoi + NgayHoaDon
    dup_cols = ["IDChungTu", "MaSanPhamMoi", "NgayHoaDon"]
    if all(c in df.columns for c in dup_cols):
        dup_count = df.duplicated(subset=dup_cols, keep=False).sum()
        dup_rate = dup_count / total_rows
        if dup_rate > 0.01:
            findings.append({
                "check": "duplicate_rows",
                "severity": "YELLOW" if dup_rate < 0.05 else "RED",
                "detail": f"{dup_count:,} potential duplicate rows ({dup_rate:.1%})",
            })

    # 3) Formula check: ThanhTienSauVAT ≈ ThanhTien + ChietKhau + ThueVAT
    if all(c in df.columns for c in ["ThanhTien", "ChietKhau", "ThueVAT", "ThanhTienSauVAT"]):
        expected = df["ThanhTien"] + df["ChietKhau"] + df["ThueVAT"]
        delta = (df["ThanhTienSauVAT"] - expected).abs()
        bad_formula = (delta > FORMULA_TOLERANCE).sum()
        if bad_formula > 0:
            bad_pct = bad_formula / total_rows
            findings.append({
                "check": "formula_mismatch",
                "severity": "YELLOW" if bad_pct < 0.01 else "RED",
                "detail": (
                    f"{bad_formula:,} rows ({bad_pct:.1%}) where "
                    f"ThanhTienSauVAT ≠ ThanhTien+ChietKhau+ThueVAT (tolerance={FORMULA_TOLERANCE})"
                ),
            })

    # 4) Abnormal negatives
    for col in ["SoLuong", "DonGia"]:
        if col in df.columns:
            neg = (df[col] < 0).sum()
            if neg > 0:
                findings.append({
                    "check": "negative_values",
                    "severity": "YELLOW",
                    "detail": f"{neg:,} negative values in '{col}'",
                })

    # 5) Missing date coverage (gaps in NgayHoaDon)
    if "Date" in df.columns:
        dates = pd.to_datetime(pd.Series(df["Date"].dropna().unique()))
        if len(dates) > 1:
            full_range = pd.date_range(dates.min(), dates.max(), freq="D")
            existing = set(pd.to_datetime(df["Date"].dropna().unique()).normalize())
            missing_dates = set(full_range) - existing
            # Only flag weekdays as missing (Mon-Fri)
            missing_weekdays = [d for d in missing_dates if d.weekday() < 5]
            if len(missing_weekdays) > 5:
                findings.append({
                    "check": "missing_dates",
                    "severity": "YELLOW",
                    "detail": f"{len(missing_weekdays)} weekday dates with no data in range",
                })

    # 6) Row count by file (detect import spikes)
    if "_source_file" in df.columns:
        file_counts = df["_source_file"].value_counts()
        findings.append({
            "check": "row_counts",
            "severity": "GREEN",
            "detail": "Row counts by file: " + ", ".join(
                f"{k}: {v:,}" for k, v in file_counts.items()
            ),
        })

    return findings
