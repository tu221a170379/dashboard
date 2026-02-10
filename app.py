"""
CEO Executive Cockpit + Early Warning Dashboard
================================================
Main Streamlit application.

Run:  streamlit run app.py
"""
from __future__ import annotations

import sys
import logging
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

# ─── Ensure project root is on sys.path ───
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    AUTO_REFRESH_MINUTES,
    CUSTOM_CSS,
    DATA_DIR,
    DIMENSION_LABELS,
    DRILL_DIMENSIONS,
    PAGE_ICON,
    PAGE_TITLE,
    REVENUE_DIVISOR,
    REVENUE_LABEL,
)
from src.etl import get_data_fingerprint, load_with_cache, run_data_quality_checks
from src.metrics import (
    compute_kpis,
    compute_period_kpis,
    compute_period_kpis_with_yoy,
    daily_revenue,
    monthly_revenue,
    monthly_growth_table,
    revenue_by_channel,
    revenue_by_dimension,
    top_bottom_table,
    filter_period,
    net_revenue,
    paid_units,
    yoy_month_comparison,
    yoy_month_by_dimension,
)
from src.alerts import generate_all_alerts, alerts_to_dataframe, executive_summary
from src.viz import (
    daily_trend_chart,
    daily_revenue_units_chart,
    yoy_comparison_chart,
    channel_revenue_chart,
    revenue_stacked_by_channel,
    dimension_pie_chart,
    waterfall_chart,
    monthly_heatmap,
    monthly_growth_chart,
    kpi_summary_bar,
    severity_badge,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Inject custom CSS ───
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  DATA LOADING  (cached by file fingerprint)
# ═══════════════════════════════════════════════════════════════

@st.cache_data(ttl=AUTO_REFRESH_MINUTES * 60, show_spinner="Đang tải dữ liệu…")
def load_data(fingerprint: str) -> pd.DataFrame:
    """Load & cache data keyed by file fingerprint."""
    return load_with_cache(DATA_DIR, force_reload=True)


def get_df() -> pd.DataFrame:
    fp = get_data_fingerprint(DATA_DIR)
    return load_data(fp)


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════

def render_sidebar(df: pd.DataFrame) -> dict:
    """Render sidebar filters, return filter config dict."""
    st.sidebar.title(f"{PAGE_ICON} CEO Dashboard")

    # Data summary
    total_rows = len(df)
    max_date = df["NgayHoaDon"].max()
    min_date = df["NgayHoaDon"].min()
    if pd.isna(max_date):
        max_date = pd.Timestamp.now()
    if pd.isna(min_date):
        min_date = max_date - pd.Timedelta(days=365)

    st.sidebar.caption(
        f"📁 {total_rows:,} dòng | "
        f"{min_date.strftime('%d/%m/%Y')} → {max_date.strftime('%d/%m/%Y')}"
    )

    # Refresh button
    if st.sidebar.button("🔄 Làm mới dữ liệu", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("---")

    # Reference date
    ref_date = st.sidebar.date_input(
        "📅 Ngày tham chiếu",
        value=max_date.date() if hasattr(max_date, 'date') else max_date,
        min_value=min_date.date() if hasattr(min_date, 'date') else min_date,
        max_value=max_date.date() if hasattr(max_date, 'date') else max_date,
    )

    # Product type toggle
    include_baobi = st.sidebar.checkbox("Bao gồm Bao bì luân chuyển", value=False)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Bộ lọc Drill-down")

    # Channels
    channels = sorted(df["KenhBanHang"].dropna().unique().tolist())
    sel_channels = st.sidebar.multiselect("Kênh bán hàng", channels, default=channels)

    # Regions
    regions = sorted(df["Vung"].dropna().unique().tolist())
    sel_regions = st.sidebar.multiselect("Vùng", regions, default=regions)

    # Date range for drill-down
    st.sidebar.markdown("**Khoảng thời gian (drill-down)**")
    col1, col2 = st.sidebar.columns(2)
    drill_start = col1.date_input("Từ ngày", value=(max_date - pd.Timedelta(days=30)).date())
    drill_end = col2.date_input("Đến ngày", value=max_date.date())

    st.sidebar.markdown("---")
    st.sidebar.caption(f"🔄 Tự động cập nhật mỗi {AUTO_REFRESH_MINUTES} phút")

    return {
        "ref_date": pd.Timestamp(ref_date),
        "include_baobi": include_baobi,
        "channels": sel_channels,
        "regions": sel_regions,
        "drill_start": pd.Timestamp(drill_start),
        "drill_end": pd.Timestamp(drill_end),
    }


def apply_base_filters(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply product-type filter (Hàng hóa + optionally Bao bì)."""
    if cfg["include_baobi"]:
        mask = df["is_hang_hoa"] | df["is_bao_bi"] | df["is_discount_row"]
    else:
        mask = df["is_hang_hoa"] | df["is_discount_row"]
    return df[mask].copy()


def apply_drill_filters(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply channel, region, date range drill-down filters."""
    out = df.copy()
    if cfg["channels"]:
        out = out[out["KenhBanHang"].isin(cfg["channels"])]
    if cfg["regions"]:
        out = out[out["Vung"].isin(cfg["regions"])]
    out = filter_period(out, cfg["drill_start"], cfg["drill_end"])
    return out


# ═══════════════════════════════════════════════════════════════
#  KPI CARD HELPER
# ═══════════════════════════════════════════════════════════════

def kpi_card(label: str, value, fmt: str = ",.0f", prefix: str = "", suffix: str = ""):
    """Render a metric card."""
    if isinstance(value, float):
        display = f"{prefix}{value:{fmt}}{suffix}"
    else:
        display = str(value)
    st.metric(label=label, value=display)


# ═══════════════════════════════════════════════════════════════
#  YoY COMPARISON TABLE RENDERER
# ═══════════════════════════════════════════════════════════════

def _format_yoy_summary(row):
    """Format a YoY summary row based on its _fmt."""
    fmt = row.get("_fmt", ",.0f")
    vals = {}
    for col in ["Tháng này (MTD)", "Cùng kì năm trước", "Chênh lệch"]:
        v = row[col]
        if fmt.endswith("%"):
            vals[col] = f"{v:{fmt}}"
        else:
            vals[col] = f"{v:{fmt}}"
    pct = row["%Thay đổi"]
    vals["%Thay đổi"] = f"{pct:+.1%}" if pct is not None else "N/A"
    return pd.Series(vals)


def _color_delta(val):
    """Color green for positive, red for negative."""
    if isinstance(val, str):
        val_clean = val.replace(",", "").replace("%", "").replace("+", "").strip()
        if val_clean == "N/A":
            return ""
        try:
            num = float(val_clean)
        except ValueError:
            return ""
        if num > 0:
            return "color: #27ae60; font-weight: bold"
        elif num < 0:
            return "color: #e74c3c; font-weight: bold"
    return ""


def _color_pct_delta(val):
    """Color for %Thay đổi column."""
    return _color_delta(val)


def _render_yoy_comparison(df: pd.DataFrame, ref_date: pd.Timestamp):
    """Render the complete YoY month comparison section."""
    cur_month = ref_date.strftime("%m/%Y")
    prev_month = (ref_date - pd.DateOffset(years=1)).strftime("%m/%Y")
    st.caption(f"So sánh MTD tháng {cur_month} với cùng kì {prev_month}")

    # --- Overall KPI Table ---
    yoy_df = yoy_month_comparison(df, ref_date)
    if yoy_df.empty:
        st.info("Không đủ dữ liệu để so sánh.")
        return

    display_df = yoy_df.apply(_format_yoy_summary, axis=1)
    display_df.insert(0, "KPI", yoy_df["KPI"])

    styled = display_df.style.map(
        _color_delta, subset=["Chênh lệch"]
    ).map(
        _color_pct_delta, subset=["%Thay đổi"]
    ).set_properties(**{
        "text-align": "right",
    }, subset=["Tháng này (MTD)", "Cùng kì năm trước", "Chênh lệch", "%Thay đổi"]).set_properties(**{
        "text-align": "left", "font-weight": "bold",
    }, subset=["KPI"])

    st.dataframe(styled, hide_index=True, height=330)

    # --- Breakdown by dimension ---
    dim_options = ["KenhBanHang", "Vung", "KhuVuc", "TinhThanh",
                   "TenKhachHang", "TenSanPham", "ChiNhanh", "TrinhDuocVien"]
    dim_labels = [DIMENSION_LABELS.get(d, d) for d in dim_options]
    dim_choice_idx = st.selectbox(
        "📋 Chi tiết theo chiều phân tích",
        range(len(dim_options)),
        format_func=lambda i: dim_labels[i],
        index=0,
        key="yoy_dim_select",
    )
    dim_option = dim_options[dim_choice_idx]
    dim_df = yoy_month_by_dimension(df, ref_date, dim_option)
    if dim_df.empty:
        st.info(f"Không có dữ liệu theo {dim_option}.")
        return

    # Format numeric columns
    fmt_cols = {
        "Revenue (MTD)": "{:,.0f}",
        "Revenue (CK)": "{:,.0f}",
        "Δ Revenue": "{:,.0f}",
        "%Δ Revenue": "{:+.1%}",
        "Units (MTD)": "{:,.0f}",
        "Units (CK)": "{:,.0f}",
        "Δ Units": "{:,.0f}",
        "%Δ Units": "{:+.1%}",
        "Tons (MTD)": "{:,.2f}",
        "Tons (CK)": "{:,.2f}",
    }

    # Build display DataFrame with formatted strings
    dim_display = dim_df.copy()
    for col, fmt in fmt_cols.items():
        if col in dim_display.columns:
            dim_display[col] = dim_display[col].apply(
                lambda v: fmt.format(v) if pd.notna(v) else "N/A"
            )

    def _color_dim_delta(val):
        return _color_delta(val)

    delta_cols = [c for c in ["Δ Revenue", "%Δ Revenue", "Δ Units", "%Δ Units"] if c in dim_display.columns]
    styled_dim = dim_display.style
    for dc in delta_cols:
        styled_dim = styled_dim.map(_color_dim_delta, subset=[dc])

    st.dataframe(styled_dim, hide_index=True, height=400)


# ═══════════════════════════════════════════════════════════════
#  TAB 1: OVERVIEW (CEO COCKPIT)
# ═══════════════════════════════════════════════════════════════

def render_overview(df: pd.DataFrame, cfg: dict):
    ref = cfg["ref_date"]
    st.header("📊 CEO Cockpit — Tổng quan")

    # ─── Executive Summary ───
    alerts = generate_all_alerts(df, ref)
    bullets = executive_summary(alerts, df, ref)
    with st.expander("📝 Tóm tắt điều hành", expanded=True):
        for b in bullets:
            st.markdown(f"- {b}")

    # ─── KPI Cards by period WITH YoY deltas ───
    period_kpis = compute_period_kpis_with_yoy(df, ref)
    st.subheader("📌 KPI theo kì")

    periods = ["Today", "WTD", "MTD", "QTD", "YTD"]
    period_vn = {"Today": "Hôm nay", "WTD": "Tuần", "MTD": "Tháng", "QTD": "Quý", "YTD": "Năm"}
    cols = st.columns(len(periods))
    for i, p in enumerate(periods):
        kpi = period_kpis.get(p, {})
        pct_rev = kpi.get("pct_revenue")
        delta_rev_str = f"{pct_rev:+.1%}" if pct_rev is not None else None
        pct_units = kpi.get("pct_units")
        delta_units_str = f"{pct_units:+.1%}" if pct_units is not None else None
        with cols[i]:
            st.markdown(f"**{period_vn.get(p, p)}**")
            st.metric("Doanh thu (M)", f"{kpi.get('net_revenue_m', 0):,.0f}",
                       delta=delta_rev_str, delta_color="normal")
            st.metric("SL bán", f"{kpi.get('paid_units', 0):,.0f}",
                       delta=delta_units_str, delta_color="normal")
            st.metric("ASP", f"{kpi.get('asp', 0):,.0f}")
            st.metric("CK %", f"{kpi.get('discount_rate', 0):.1%}")

    # ─── Period comparison bar chart ───
    st.plotly_chart(kpi_summary_bar(period_kpis), key="kpi_summary_bar")

    st.markdown("---")

    # ─── YoY Month Comparison Table ───
    st.subheader("📊 So sánh tháng này vs cùng kì năm trước")
    _render_yoy_comparison(df, ref)

    st.markdown("---")

    # ─── Monthly Growth Table ───
    with st.expander("📈 Bảng tăng trưởng theo tháng", expanded=False):
        growth = monthly_growth_table(df)
        if not growth.empty:
            col_gt1, col_gt2 = st.columns([3, 2])
            with col_gt1:
                display_growth = growth.copy()
                for c in ["YoY %", "MoM %"]:
                    display_growth[c] = display_growth[c].apply(
                        lambda v: f"{v:+.1%}" if pd.notna(v) else "—"
                    )
                display_growth["Revenue (M)"] = display_growth["Revenue (M)"].apply(lambda v: f"{v:,.0f}")
                display_growth["Units"] = display_growth["Units"].apply(lambda v: f"{v:,.0f}")
                display_growth["ASP"] = display_growth["ASP"].apply(lambda v: f"{v:,.0f}")
                display_growth["Discount %"] = display_growth["Discount %"].apply(lambda v: f"{v:.1%}")
                display_growth["Tons"] = display_growth["Tons"].apply(lambda v: f"{v:,.1f}")
                show_cols = ["Tháng", "Revenue (M)", "Units", "Tons", "ASP", "Discount %", "YoY %", "MoM %"]
                st.dataframe(display_growth[show_cols], hide_index=True, height=400)
            with col_gt2:
                st.plotly_chart(monthly_growth_chart(growth), key="monthly_growth_chart")

    st.markdown("---")

    # ─── Trend Charts ───
    st.subheader("📈 Xu hướng")
    dr = daily_revenue(df)

    col_a, col_b = st.columns(2)
    with col_a:
        # Filter to recent 90 days for readability
        recent = dr[dr["Date"] >= pd.Timestamp(ref - pd.Timedelta(days=90))]
        st.plotly_chart(daily_revenue_units_chart(recent), key="daily_rev_units")
    with col_b:
        current_year = ref.year
        st.plotly_chart(yoy_comparison_chart(dr, current_year), key="yoy_chart")

    col_c, col_d = st.columns(2)
    with col_c:
        st.plotly_chart(
            revenue_stacked_by_channel(dr, df),
            key="stacked_channel",
        )
    with col_d:
        mr = monthly_revenue(df)
        st.plotly_chart(monthly_heatmap(mr), key="monthly_heatmap")


# ═══════════════════════════════════════════════════════════════
#  TAB 2: DRILL-DOWN
# ═══════════════════════════════════════════════════════════════

def render_drilldown(df: pd.DataFrame, cfg: dict):
    st.header("🔍 Phân tích chi tiết")

    drill_df = apply_drill_filters(df, cfg)
    st.caption(
        f"Khoảng thời gian: {cfg['drill_start'].strftime('%d/%m/%Y')} → {cfg['drill_end'].strftime('%d/%m/%Y')} "
        f"| Số dòng: {len(drill_df):,}"
    )

    if drill_df.empty:
        st.warning("Không có dữ liệu cho bộ lọc đã chọn.")
        return

    # ─── Summary KPIs ───
    kpi = compute_kpis(drill_df)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Doanh thu (M)", f"{kpi['net_revenue_m']:,.0f}")
    c2.metric("Gross Sales (M)", f"{kpi['gross_sales'] / REVENUE_DIVISOR:,.0f}")
    c3.metric("SL bán", f"{kpi['paid_units']:,.0f}")
    c4.metric("Tấn", f"{kpi['total_tons']:,.1f}")
    c5.metric("CK %", f"{kpi['discount_rate']:.1%}")

    st.markdown("---")

    # ─── Breakdown charts ───
    col1, col2 = st.columns(2)
    with col1:
        ch_df = revenue_by_channel(drill_df)
        st.plotly_chart(channel_revenue_chart(ch_df), key="channel_bar")
    with col2:
        vung_df = revenue_by_dimension(drill_df, "Vung")
        st.plotly_chart(dimension_pie_chart(vung_df, "Vung", "Doanh thu theo Vùng"), key="vung_pie")

    st.markdown("---")

    # ─── Top / Bottom tables ───
    st.subheader("📊 Top / Bottom Analysis")
    dim_options = ["TenKhachHang", "TenSanPham", "KhuVuc", "TinhThanh", "KenhBanHang", "ChiNhanh", "TrinhDuocVien"]
    dim_labels = [DIMENSION_LABELS.get(d, d) for d in dim_options]
    dim_idx = st.selectbox(
        "Chiều phân tích",
        range(len(dim_options)),
        format_func=lambda i: dim_labels[i],
        index=0,
        key="drilldown_dim_select",
    )
    dim_choice = dim_options[dim_idx]

    # Compare drill period vs same-length previous period
    period_len = (cfg["drill_end"] - cfg["drill_start"]).days
    prev_end = cfg["drill_start"] - pd.Timedelta(days=1)
    prev_start = prev_end - pd.Timedelta(days=period_len)

    tb = top_bottom_table(df, dim_choice, cfg["drill_start"], cfg["drill_end"], prev_start, prev_end)
    if not tb.empty:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Top Gainers**")
            top_part = tb[tb["rank_type"] == "Top"]
            if not top_part.empty:
                display_cols = [dim_choice, "cur_revenue", "prev_revenue", "delta_revenue", "delta_pct"]
                show = top_part[display_cols].copy()
                show["cur_revenue"] = (show["cur_revenue"] / REVENUE_DIVISOR).map("{:,.0f}".format)
                show["prev_revenue"] = (show["prev_revenue"] / REVENUE_DIVISOR).map("{:,.0f}".format)
                show["delta_revenue"] = (show["delta_revenue"] / REVENUE_DIVISOR).map("{:+,.0f}".format)
                show["delta_pct"] = show["delta_pct"].map(lambda x: f"{x:+.1%}" if pd.notna(x) else "N/A")
                st.dataframe(show, hide_index=True)

        with col_b:
            st.markdown("**Bottom Losers**")
            bottom_part = tb[tb["rank_type"] == "Bottom"]
            if not bottom_part.empty:
                display_cols = [dim_choice, "cur_revenue", "prev_revenue", "delta_revenue", "delta_pct"]
                show = bottom_part[display_cols].copy()
                show["cur_revenue"] = (show["cur_revenue"] / REVENUE_DIVISOR).map("{:,.0f}".format)
                show["prev_revenue"] = (show["prev_revenue"] / REVENUE_DIVISOR).map("{:,.0f}".format)
                show["delta_revenue"] = (show["delta_revenue"] / REVENUE_DIVISOR).map("{:+,.0f}".format)
                show["delta_pct"] = show["delta_pct"].map(lambda x: f"{x:+.1%}" if pd.notna(x) else "N/A")
                st.dataframe(show, hide_index=True)

        # Waterfall
        st.plotly_chart(
            waterfall_chart(tb, dim_choice, f"Revenue Delta by {dim_choice}"),
            key="waterfall",
        )
    else:
        st.info("Không đủ dữ liệu để so sánh.")


# ═══════════════════════════════════════════════════════════════
#  TAB 3: ALERTS & DATA QUALITY
# ═══════════════════════════════════════════════════════════════

def render_alerts(df: pd.DataFrame, cfg: dict):
    st.header("⚠️ Cảnh báo & Chất lượng dữ liệu")

    ref = cfg["ref_date"]

    # ─── Business Alerts ───
    st.subheader("🚨 Cảnh báo kinh doanh")
    alerts = generate_all_alerts(df, ref)
    alert_df = alerts_to_dataframe(alerts)

    if alert_df.empty:
        st.success("✅ Không có cảnh báo bất thường — hoạt động ổn định.")
    else:
        # Summary counts
        reds = sum(1 for a in alerts if a.severity == "RED")
        yellows = sum(1 for a in alerts if a.severity == "YELLOW")
        c1, c2, c3 = st.columns(3)
        c1.metric("🔴 Nghiêm trọng", reds)
        c2.metric("🟡 Cần theo dõi", yellows)
        c3.metric("📊 Tổng cảnh báo", len(alerts))

        # Color-code severity
        def color_severity(val):
            colors_map = {
                "RED": "background-color: #ffcdd2; color: #b71c1c; font-weight: bold",
                "YELLOW": "background-color: #fff9c4; color: #f57f17; font-weight: bold",
                "GREEN": "background-color: #c8e6c9; color: #1b5e20; font-weight: bold",
            }
            return colors_map.get(val, "")

        display_cols = ["severity", "category", "metric", "period", "segment", "pct_change_str", "detail"]
        show_df = alert_df[[c for c in display_cols if c in alert_df.columns]].copy()
        show_df.columns = ["Mức độ", "Loại", "Chỉ số", "Kì", "Phân khúc", "% Thay đổi", "Chi tiết"][:len(show_df.columns)]
        styled = show_df.style.map(color_severity, subset=["Mức độ"])
        st.dataframe(styled, hide_index=True, height=400)

    st.markdown("---")

    # ─── Data Quality ───
    st.subheader("🔬 Báo cáo chất lượng dữ liệu")
    dq_findings = run_data_quality_checks(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Tổng số dòng", f"{len(df):,}")
    unique_invoices = df["IDChungTu"].nunique() if "IDChungTu" in df.columns else 0
    col2.metric("Số hóa đơn", f"{unique_invoices:,}")
    date_range = ""
    if "NgayHoaDon" in df.columns:
        mn = df["NgayHoaDon"].min()
        mx = df["NgayHoaDon"].max()
        if pd.notna(mn) and pd.notna(mx):
            date_range = f"{mn.strftime('%d/%m/%Y')} → {mx.strftime('%d/%m/%Y')}"
    col3.metric("Khoảng ngày", date_range)

    # Null rates in a clean table
    with st.expander("📊 Tỷ lệ null các cột quan trọng", expanded=False):
        null_rows = []
        for c in ["NgayHoaDon", "TenKhachHang", "TenSanPham", "KenhBanHang",
                   "ThanhTienSauVAT", "LoaiSanPham", "Vung", "KhuVuc"]:
            if c in df.columns:
                rate = df[c].isna().mean()
                null_rows.append({"Cột": c, "Tỷ lệ null": f"{rate:.2%}",
                                  "Trạng thái": "✅" if rate < 0.05 else "⚠️"})
        st.dataframe(pd.DataFrame(null_rows), hide_index=True)

    # Findings table
    if dq_findings:
        dq_df = pd.DataFrame(dq_findings)
        # Translate severity column
        severity_vn = {"RED": "🔴 Nghiêm trọng", "YELLOW": "🟡 Cảnh báo", "GREEN": "✅ Bình thường"}
        dq_df["severity"] = dq_df["severity"].map(lambda x: severity_vn.get(x, x))
        dq_df.columns = ["Kiểm tra", "Mức độ", "Chi tiết"]
        st.dataframe(dq_df, hide_index=True)

    st.markdown("---")

    # ─── Export ───
    st.subheader("📥 Xuất báo cáo")
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        if st.button("📤 Xuất báo cáo đầy đủ (Excel)", use_container_width=True):
            buf = BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                # Alerts sheet
                if not alert_df.empty:
                    alert_df.to_excel(w, sheet_name="Cảnh báo", index=False)
                # Data quality
                if dq_findings:
                    pd.DataFrame(dq_findings).to_excel(w, sheet_name="Chất lượng DL", index=False)
                # Period KPIs
                pkpis = compute_period_kpis(df, ref)
                rows = []
                for period, kdict in pkpis.items():
                    row = {"Kì": period}
                    row.update(kdict)
                    rows.append(row)
                pd.DataFrame(rows).to_excel(w, sheet_name="KPIs", index=False)
                # YoY comparison
                yoy_df = yoy_month_comparison(df, ref)
                if not yoy_df.empty:
                    yoy_df.drop(columns=["_fmt"], errors="ignore").to_excel(
                        w, sheet_name="So sánh YoY", index=False
                    )
                # Monthly growth
                growth = monthly_growth_table(df)
                if not growth.empty:
                    growth.to_excel(w, sheet_name="Tăng trưởng tháng", index=False)
            buf.seek(0)
            st.download_button(
                "⬇️ Tải Excel",
                data=buf,
                file_name=f"ceo_report_{ref.strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    try:
        df_raw = get_df()
    except FileNotFoundError as e:
        st.error(f"❌ {e}")
        st.info(f"Vui lòng đặt file Excel (*.xlsx) vào thư mục: `{DATA_DIR}`")
        st.stop()
    except Exception as e:
        st.error(f"❌ Lỗi đọc dữ liệu: {e}")
        logger.exception("Data loading error")
        st.stop()

    # Sidebar filters
    cfg = render_sidebar(df_raw)

    # Apply base filter (Hàng hóa + optional Bao bì)
    df = apply_base_filters(df_raw, cfg)

    # Auto-refresh
    if AUTO_REFRESH_MINUTES > 0:
        st.empty()  # placeholder for auto-refresh ticker

    # ─── Tabs ───
    tab1, tab2, tab3 = st.tabs([
        "📊 Tổng quan",
        "🔍 Phân tích chi tiết",
        "⚠️ Cảnh báo & Chất lượng DL",
    ])

    with tab1:
        render_overview(df, cfg)

    with tab2:
        render_drilldown(df, cfg)

    with tab3:
        render_alerts(df_raw, cfg)  # use raw for full DQ check

    # Footer
    st.markdown("---")
    fp = get_data_fingerprint(DATA_DIR)
    st.markdown(
        f'<div class="footer-text">'
        f'CEO Executive Cockpit v2.0 | '
        f'Fingerprint: <code>{fp[:12]}…</code> | '
        f'Cập nhật lúc: {pd.Timestamp.now().strftime("%H:%M:%S %d/%m/%Y")}'
        f'</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
