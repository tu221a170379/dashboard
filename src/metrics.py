"""
Metrics module – KPI calculations for CEO Dashboard.

All metric functions accept a pre-cleaned DataFrame (output of etl.clean_dataframe)
and return aggregated results.  Rules follow FMCG revenue & volume accounting.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import numpy as np

from src.config import REVENUE_DIVISOR


# ═══════════════════════════════════════════════════════════════
#  LOW-LEVEL BUILDING BLOCKS
# ═══════════════════════════════════════════════════════════════

def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division guarded against zero / nan."""
    if denominator == 0 or pd.isna(denominator):
        return default
    return numerator / denominator


def _pct_change(current: float, previous: float) -> Optional[float]:
    """Percentage change: (current - previous) / |previous|.  None if base=0."""
    if previous == 0 or pd.isna(previous):
        return None
    return (current - previous) / abs(previous)


# ═══════════════════════════════════════════════════════════════
#  REVENUE KPIs
# ═══════════════════════════════════════════════════════════════

def gross_sales(df: pd.DataFrame) -> float:
    """Gross Sales = sum(ThanhTien) for Hàng hóa rows."""
    mask = df["is_hang_hoa"]
    return float(df.loc[mask, "ThanhTien"].sum())


def total_discount(df: pd.DataFrame) -> float:
    """Total Discount = sum(ChietKhau). Usually negative."""
    return float(df["ChietKhau"].sum())


def net_before_vat(df: pd.DataFrame) -> float:
    """Net before VAT = sum(ThanhTien + ChietKhau).
    Includes *all* rows (discount/adjustment rows have ThanhTien=0, ChietKhau<0)
    so the net correctly accounts for discounts."""
    return float((df["ThanhTien"] + df["ChietKhau"]).sum())


def total_vat(df: pd.DataFrame) -> float:
    """VAT = sum(ThueVAT)."""
    return float(df["ThueVAT"].sum())


def net_revenue(df: pd.DataFrame) -> float:
    """Net Revenue = sum(ThanhTienSauVAT).
    Includes all rows (revenue + discount + promo) to avoid double-counting."""
    return float(df["ThanhTienSauVAT"].sum())


# ═══════════════════════════════════════════════════════════════
#  VOLUME KPIs
# ═══════════════════════════════════════════════════════════════

def paid_units(df: pd.DataFrame) -> float:
    """Paid Units = sum(SoLuong) where is_paid==True."""
    return float(df.loc[df["is_paid"], "SoLuong"].sum())


def promo_units(df: pd.DataFrame) -> float:
    """Promo / Free Units = sum(SoLuong) where is_promo==True."""
    return float(df.loc[df["is_promo"], "SoLuong"].sum())


def total_units(df: pd.DataFrame) -> float:
    """All volume-eligible units (paid + promo)."""
    return float(df.loc[df["is_volume_eligible"], "SoLuong"].sum())


def total_tons(df: pd.DataFrame) -> float:
    """Total weight in tons for volume-eligible rows."""
    return float(df.loc[df["is_volume_eligible"], "TongTan"].sum())


def total_cbm(df: pd.DataFrame) -> float:
    """Total cubic meters for volume-eligible rows."""
    return float(df.loc[df["is_volume_eligible"], "TongKhoi"].sum())


# ═══════════════════════════════════════════════════════════════
#  DERIVED KPIs
# ═══════════════════════════════════════════════════════════════

def asp(df: pd.DataFrame) -> float:
    """Average Selling Price = Net Revenue / Paid Units."""
    pu = paid_units(df)
    nr = net_revenue(df)
    return _safe_div(nr, pu)


def discount_rate(df: pd.DataFrame) -> float:
    """Discount Rate = |sum(ChietKhau)| / sum(ThanhTien).  Guard div-by-0."""
    gs = gross_sales(df)
    td = abs(total_discount(df))
    return _safe_div(td, gs)


# ═══════════════════════════════════════════════════════════════
#  AGGREGATED KPI DICT  (for KPI cards)
# ═══════════════════════════════════════════════════════════════

def compute_kpis(df: pd.DataFrame) -> dict:
    """Return dict with all core KPIs for given slice of data."""
    nr = net_revenue(df)
    pu = paid_units(df)
    return {
        "gross_sales": gross_sales(df),
        "discount": total_discount(df),
        "net_before_vat": net_before_vat(df),
        "vat": total_vat(df),
        "net_revenue": nr,
        "net_revenue_m": nr / REVENUE_DIVISOR,
        "paid_units": pu,
        "promo_units": promo_units(df),
        "total_units": total_units(df),
        "total_tons": total_tons(df),
        "total_cbm": total_cbm(df),
        "asp": _safe_div(nr, pu),
        "discount_rate": discount_rate(df),
    }


# ═══════════════════════════════════════════════════════════════
#  TIME-SERIES AGGREGATION
# ═══════════════════════════════════════════════════════════════

def daily_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Daily Net Revenue and Paid Units."""
    if df.empty or "Date" not in df.columns:
        return pd.DataFrame(columns=["Date", "net_revenue", "paid_units", "promo_units"])
    grp = df.groupby("Date", dropna=False).apply(
        lambda g: pd.Series({
            "net_revenue": net_revenue(g),
            "paid_units": paid_units(g),
            "promo_units": promo_units(g),
            "gross_sales": gross_sales(g),
            "discount": total_discount(g),
        }),
        include_groups=False,
    ).reset_index()
    grp["Date"] = pd.to_datetime(grp["Date"])
    return grp.sort_values("Date").reset_index(drop=True)


def weekly_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Weekly aggregation."""
    if df.empty:
        return pd.DataFrame()
    grp = df.groupby(["Year", "Week"], dropna=False).apply(
        lambda g: pd.Series({
            "net_revenue": net_revenue(g),
            "paid_units": paid_units(g),
            "promo_units": promo_units(g),
        }),
        include_groups=False,
    ).reset_index()
    return grp.sort_values(["Year", "Week"]).reset_index(drop=True)


def monthly_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly aggregation."""
    if df.empty:
        return pd.DataFrame()
    grp = df.groupby(["Year", "Month"], dropna=False).apply(
        lambda g: pd.Series({
            "net_revenue": net_revenue(g),
            "paid_units": paid_units(g),
            "promo_units": promo_units(g),
            "gross_sales": gross_sales(g),
            "discount": total_discount(g),
            "total_tons": total_tons(g),
            "asp": asp(g),
            "discount_rate": discount_rate(g),
        }),
        include_groups=False,
    ).reset_index()
    return grp.sort_values(["Year", "Month"]).reset_index(drop=True)


def revenue_by_channel(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue breakdown by KenhBanHang."""
    if df.empty:
        return pd.DataFrame()
    grp = df.groupby("KenhBanHang", dropna=False).apply(
        lambda g: pd.Series({
            "net_revenue": net_revenue(g),
            "paid_units": paid_units(g),
            "pct_revenue": 0.0,  # placeholder, filled below
        }),
        include_groups=False,
    ).reset_index()
    total = grp["net_revenue"].sum()
    grp["pct_revenue"] = grp["net_revenue"] / total if total else 0
    return grp.sort_values("net_revenue", ascending=False).reset_index(drop=True)


def revenue_by_dimension(df: pd.DataFrame, dim: str) -> pd.DataFrame:
    """Generic revenue breakdown by a dimension column."""
    if df.empty or dim not in df.columns:
        return pd.DataFrame()
    grp = df.groupby(dim, dropna=False).apply(
        lambda g: pd.Series({
            "net_revenue": net_revenue(g),
            "paid_units": paid_units(g),
            "total_tons": total_tons(g),
        }),
        include_groups=False,
    ).reset_index()
    total = grp["net_revenue"].sum()
    grp["pct_revenue"] = grp["net_revenue"] / total if total else 0
    return grp.sort_values("net_revenue", ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
#  PERIOD MATCHING HELPERS  (for WoW, MoM, YoY)
# ═══════════════════════════════════════════════════════════════

def filter_period(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Filter DataFrame to rows where NgayHoaDon is in [start, end]."""
    mask = (df["NgayHoaDon"] >= start) & (df["NgayHoaDon"] <= end)
    return df.loc[mask]


def mtd_data(df: pd.DataFrame, ref_date: pd.Timestamp) -> pd.DataFrame:
    """Month-to-date: from 1st of month to ref_date."""
    start = ref_date.replace(day=1)
    return filter_period(df, start, ref_date)


def ytd_data(df: pd.DataFrame, ref_date: pd.Timestamp) -> pd.DataFrame:
    """Year-to-date: from Jan 1 to ref_date."""
    start = ref_date.replace(month=1, day=1)
    return filter_period(df, start, ref_date)


def qtd_data(df: pd.DataFrame, ref_date: pd.Timestamp) -> pd.DataFrame:
    """Quarter-to-date."""
    q = (ref_date.month - 1) // 3
    q_start_month = q * 3 + 1
    start = ref_date.replace(month=q_start_month, day=1)
    return filter_period(df, start, ref_date)


def wtd_data(df: pd.DataFrame, ref_date: pd.Timestamp) -> pd.DataFrame:
    """Week-to-date (ISO: Monday = start)."""
    start = ref_date - pd.Timedelta(days=ref_date.weekday())
    return filter_period(df, start, ref_date)


def compute_period_kpis(df: pd.DataFrame, ref_date: pd.Timestamp) -> dict:
    """Compute KPIs for Today, WTD, MTD, QTD, YTD."""
    today = filter_period(df, ref_date, ref_date)
    periods = {
        "Today": today,
        "WTD": wtd_data(df, ref_date),
        "MTD": mtd_data(df, ref_date),
        "QTD": qtd_data(df, ref_date),
        "YTD": ytd_data(df, ref_date),
    }
    result = {}
    for label, sub in periods.items():
        result[label] = compute_kpis(sub)
    return result


def compute_period_kpis_with_yoy(df: pd.DataFrame, ref_date: pd.Timestamp) -> dict:
    """Compute KPIs for each period, plus YoY delta for st.metric delta display.

    Returns dict[period] = {kpi_name: value, ..., 'delta_net_revenue_m': float, 'delta_paid_units': float, ...}
    """
    ly_ref = ref_date - pd.DateOffset(years=1)

    cur_periods = {
        "Today": filter_period(df, ref_date, ref_date),
        "WTD": wtd_data(df, ref_date),
        "MTD": mtd_data(df, ref_date),
        "QTD": qtd_data(df, ref_date),
        "YTD": ytd_data(df, ref_date),
    }
    ly_periods = {
        "Today": filter_period(df, ly_ref, ly_ref),
        "WTD": wtd_data(df, ly_ref),
        "MTD": mtd_data(df, ly_ref),
        "QTD": qtd_data(df, ly_ref),
        "YTD": ytd_data(df, ly_ref),
    }

    result = {}
    for label in cur_periods:
        cur_kpi = compute_kpis(cur_periods[label])
        ly_kpi = compute_kpis(ly_periods[label])
        cur_kpi["delta_net_revenue_m"] = cur_kpi["net_revenue_m"] - ly_kpi["net_revenue_m"]
        cur_kpi["delta_paid_units"] = cur_kpi["paid_units"] - ly_kpi["paid_units"]
        cur_kpi["delta_asp"] = cur_kpi["asp"] - ly_kpi["asp"]
        cur_kpi["delta_discount_rate"] = cur_kpi["discount_rate"] - ly_kpi["discount_rate"]
        cur_kpi["pct_revenue"] = _pct_change(cur_kpi["net_revenue"], ly_kpi["net_revenue"])
        cur_kpi["pct_units"] = _pct_change(cur_kpi["paid_units"], ly_kpi["paid_units"])
        cur_kpi["ly_kpi"] = ly_kpi
        result[label] = cur_kpi
    return result


def monthly_growth_table(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly summary with MoM and YoY growth rates."""
    mr = monthly_revenue(df)
    if mr.empty:
        return pd.DataFrame()

    mr = mr.sort_values(["Year", "Month"]).reset_index(drop=True)

    # YoY: compare same month previous year
    yoy_map = {}
    for _, row in mr.iterrows():
        yoy_map[(int(row["Year"]), int(row["Month"]))] = row["net_revenue"]

    mr["YoY %"] = mr.apply(
        lambda r: _pct_change(
            r["net_revenue"],
            yoy_map.get((int(r["Year"]) - 1, int(r["Month"])), 0)
        ), axis=1
    )

    # MoM: compare to previous row (sequential month)
    mr["MoM %"] = mr["net_revenue"].pct_change()
    mr.loc[mr.index[0], "MoM %"] = None

    # Clean display columns
    mr["Tháng"] = mr.apply(lambda r: f"{int(r['Month']):02d}/{int(r['Year'])}", axis=1)
    mr["Revenue (M)"] = mr["net_revenue"] / REVENUE_DIVISOR
    mr["Units"] = mr["paid_units"]
    mr["Tons"] = mr["total_tons"]
    mr["ASP"] = mr["asp"]
    mr["Discount %"] = mr["discount_rate"]

    return mr[["Tháng", "Year", "Month", "Revenue (M)", "Units", "Tons",
              "ASP", "Discount %", "YoY %", "MoM %"]]


def top_bottom_table(
    df: pd.DataFrame,
    dim: str,
    current_start: pd.Timestamp,
    current_end: pd.Timestamp,
    prev_start: pd.Timestamp,
    prev_end: pd.Timestamp,
    top_n: int = 10,
) -> pd.DataFrame:
    """Build a top/bottom table showing delta for a dimension."""
    cur = filter_period(df, current_start, current_end)
    prev = filter_period(df, prev_start, prev_end)

    cur_agg = revenue_by_dimension(cur, dim).rename(columns={
        "net_revenue": "cur_revenue", "paid_units": "cur_units"
    })
    prev_agg = revenue_by_dimension(prev, dim).rename(columns={
        "net_revenue": "prev_revenue", "paid_units": "prev_units"
    })

    merged = cur_agg[[dim, "cur_revenue", "cur_units"]].merge(
        prev_agg[[dim, "prev_revenue", "prev_units"]],
        on=dim, how="outer"
    ).fillna(0)

    merged["delta_revenue"] = merged["cur_revenue"] - merged["prev_revenue"]
    merged["delta_pct"] = merged.apply(
        lambda r: _pct_change(r["cur_revenue"], r["prev_revenue"]), axis=1
    )
    merged = merged.sort_values("delta_revenue")

    bottom = merged.head(top_n).copy()
    bottom["rank_type"] = "Bottom"
    top = merged.tail(top_n).sort_values("delta_revenue", ascending=False).copy()
    top["rank_type"] = "Top"

    return pd.concat([top, bottom], ignore_index=True)


# ═══════════════════════════════════════════════════════════════
#  YoY MONTH COMPARISON
# ═══════════════════════════════════════════════════════════════

def _yoy_month_ranges(ref_date: pd.Timestamp):
    """Return (cur_start, cur_end, prev_start, prev_end) for current month vs same month last year."""
    cur_start = ref_date.replace(day=1)
    cur_end = ref_date  # MTD
    prev_start = cur_start - pd.DateOffset(years=1)
    prev_end = cur_end - pd.DateOffset(years=1)
    return cur_start, cur_end, prev_start, prev_end


def yoy_month_comparison(df: pd.DataFrame, ref_date: pd.Timestamp) -> pd.DataFrame:
    """Overall KPI comparison: current month MTD vs same period last year.

    Returns a DataFrame with columns: KPI, Tháng này, Cùng kì năm trước, Chênh lệch, %Thay đổi.
    """
    cur_start, cur_end, prev_start, prev_end = _yoy_month_ranges(ref_date)
    cur = filter_period(df, cur_start, cur_end)
    prev = filter_period(df, prev_start, prev_end)

    cur_kpi = compute_kpis(cur)
    prev_kpi = compute_kpis(prev)

    rows = []
    kpi_defs = [
        ("Net Revenue (M)", "net_revenue_m", ",.1f"),
        ("Gross Sales", "gross_sales", ",.0f"),
        ("Discount", "discount", ",.0f"),
        ("Paid Units", "paid_units", ",.0f"),
        ("Promo Units", "promo_units", ",.0f"),
        ("Total Tons", "total_tons", ",.2f"),
        ("ASP", "asp", ",.0f"),
        ("Discount Rate", "discount_rate", ".2%"),
    ]
    for label, key, fmt in kpi_defs:
        cv = cur_kpi.get(key, 0)
        pv = prev_kpi.get(key, 0)
        delta = cv - pv
        pct = _pct_change(cv, pv)
        rows.append({
            "KPI": label,
            "Tháng này (MTD)": cv,
            "Cùng kì năm trước": pv,
            "Chênh lệch": delta,
            "%Thay đổi": pct,
            "_fmt": fmt,
        })
    return pd.DataFrame(rows)


def yoy_month_by_dimension(
    df: pd.DataFrame,
    ref_date: pd.Timestamp,
    dim: str,
) -> pd.DataFrame:
    """YoY month comparison broken down by a dimension (e.g. KenhBanHang, Vung, TenKhachHang).

    Returns a DataFrame with columns: <dim>, Revenue (MTD), Revenue (CK), Delta, %Delta,
    Units (MTD), Units (CK), Delta Units, %Delta Units.
    """
    cur_start, cur_end, prev_start, prev_end = _yoy_month_ranges(ref_date)
    cur = filter_period(df, cur_start, cur_end)
    prev = filter_period(df, prev_start, prev_end)

    if dim not in df.columns:
        return pd.DataFrame()

    cur_agg = revenue_by_dimension(cur, dim).rename(columns={
        "net_revenue": "Revenue (MTD)",
        "paid_units": "Units (MTD)",
        "total_tons": "Tons (MTD)",
    })
    prev_agg = revenue_by_dimension(prev, dim).rename(columns={
        "net_revenue": "Revenue (CK)",
        "paid_units": "Units (CK)",
        "total_tons": "Tons (CK)",
    })

    keep_cur = [dim, "Revenue (MTD)", "Units (MTD)", "Tons (MTD)"]
    keep_prev = [dim, "Revenue (CK)", "Units (CK)", "Tons (CK)"]
    cur_cols = [c for c in keep_cur if c in cur_agg.columns]
    prev_cols = [c for c in keep_prev if c in prev_agg.columns]

    merged = cur_agg[cur_cols].merge(prev_agg[prev_cols], on=dim, how="outer").fillna(0)

    merged["Δ Revenue"] = merged["Revenue (MTD)"] - merged["Revenue (CK)"]
    merged["%Δ Revenue"] = merged.apply(
        lambda r: _pct_change(r["Revenue (MTD)"], r["Revenue (CK)"]), axis=1
    )
    merged["Δ Units"] = merged["Units (MTD)"] - merged["Units (CK)"]
    merged["%Δ Units"] = merged.apply(
        lambda r: _pct_change(r["Units (MTD)"], r["Units (CK)"]), axis=1
    )

    return merged.sort_values("Revenue (MTD)", ascending=False).reset_index(drop=True)
