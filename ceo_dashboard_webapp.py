# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════╗
║   CEO DASHBOARD - WEB APP (Streamlit)                              ║
║   Chạy: streamlit run ceo_dashboard_webapp.py                      ║
║   Truy cập: http://localhost:8501                                  ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import calendar
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YEAR_FILES = [2023, 2024, 2025, 2026]

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CEO Dashboard - VKD",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

.stApp {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* Header */
.dashboard-header {
    background: linear-gradient(135deg, #1B4F72, #2E86C1);
    color: white;
    padding: 20px 30px;
    border-radius: 12px;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.dashboard-header h1 {
    font-size: 24px;
    margin: 0;
    font-weight: 700;
}
.dashboard-header .info {
    font-size: 13px;
    opacity: 0.85;
}

/* KPI Metric Cards */
.kpi-card {
    background: white;
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.08);
    border-left: 4px solid #2E86C1;
    transition: transform 0.2s;
    height: 100%;
}
.kpi-card:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.12); }
.kpi-card .label { font-size: 12px; color: #7F8C8D; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 5px; }
.kpi-card .value { font-size: 22px; font-weight: 700; color: #1B4F72; }
.kpi-card .sub { font-size: 11px; color: #7F8C8D; margin-top: 3px; }
.kpi-card.green { border-left-color: #27AE60; }
.kpi-card.gold { border-left-color: #F39C12; }
.kpi-card.red { border-left-color: #E74C3C; }
.kpi-card.purple { border-left-color: #8E44AD; }

.growth-up { color: #27AE60; font-weight: 700; }
.growth-down { color: #E74C3C; font-weight: 700; }

/* Hide Streamlit default elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: white;
    padding: 6px 10px;
    border-radius: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 18px;
    font-weight: 600;
    font-size: 13px;
}
.stTabs [aria-selected="true"] {
    background: #1B4F72 !important;
    color: white !important;
}

/* Dataframe */
.stDataFrame { border-radius: 10px; overflow: hidden; }

div[data-testid="stMetric"] {
    background: white;
    border-radius: 12px;
    padding: 15px 18px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.08);
    border-left: 4px solid #2E86C1;
}
</style>
""", unsafe_allow_html=True)

COLORS = ['#2E86C1', '#27AE60', '#F39C12', '#E74C3C', '#8E44AD',
          '#1ABC9C', '#E67E22', '#3498DB', '#9B59B6', '#2ECC71',
          '#E74C3C', '#F1C40F', '#1B4F72', '#D35400', '#16A085',
          '#C0392B', '#7D3C98', '#2874A6', '#138D75', '#B7950B']


# ═══════════════════════════════════════════════════════════════
# DATA LOADING (cached)
# ═══════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner="📊 Đang đọc dữ liệu...")
def load_all_data():
    frames = []
    for yr in YEAR_FILES:
        fp = os.path.join(BASE_DIR, f"{yr}.xlsx")
        if os.path.exists(fp):
            frames.append(pd.read_excel(fp))
    if not frames:
        st.error("❌ Không có file dữ liệu!")
        st.stop()
    df = pd.concat(frames, ignore_index=True)
    df['NgayHoaDon'] = pd.to_datetime(df['NgayHoaDon'], errors='coerce')
    df['Nam'] = df['NgayHoaDon'].dt.year
    df['Thang_Num'] = df['NgayHoaDon'].dt.month
    df['Quy'] = df['NgayHoaDon'].dt.quarter
    for col in ['SoLuong', 'ThanhTien', 'ThanhTienSauVAT', 'ChietKhau', 'ThueVAT', 'DonGia']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df


# ═══════════════════════════════════════════════════════════════
# ANALYTICS FUNCTIONS
# ═══════════════════════════════════════════════════════════════
def fmt_vnd(n):
    """Format VND number to readable string."""
    if n is None or pd.isna(n):
        return "-"
    if abs(n) >= 1e9:
        return f"{n/1e9:,.1f} Tỷ"
    if abs(n) >= 1e6:
        return f"{n/1e6:,.0f} Tr"
    return f"{n:,.0f}"


def growth_pct(cur, prev):
    if prev and prev != 0:
        return round((cur - prev) / prev * 100, 2)
    return None


@st.cache_data(ttl=3600, show_spinner="🔮 Đang chạy ARIMA tổng doanh thu...")
def run_arima_total(df_serialized, periods=6):
    df = pd.DataFrame(df_serialized)
    df['NgayHoaDon'] = pd.to_datetime(df['NgayHoaDon'])
    ts = df.groupby(df['NgayHoaDon'].dt.to_period('M'))['ThanhTienSauVAT'].sum()
    ts.index = ts.index.to_timestamp()
    ts = ts.sort_index().asfreq('MS', fill_value=0)

    adf = adfuller(ts.dropna())
    best_aic, best_order = np.inf, (1, 1, 1)
    for p in range(0, 4):
        for d in range(0, 3):
            for q in range(0, 4):
                try:
                    r = ARIMA(ts, order=(p, d, q)).fit()
                    if r.aic < best_aic:
                        best_aic, best_order = r.aic, (p, d, q)
                except:
                    pass

    model = ARIMA(ts, order=best_order).fit()
    fc = model.get_forecast(steps=periods)
    ci = fc.conf_int()

    return {
        'hist_dates': [d.strftime('%m/%Y') for d in ts.index],
        'hist_values': [float(v) for v in ts.values],
        'fc_dates': [d.strftime('%m/%Y') for d in fc.predicted_mean.index],
        'fc_values': [float(v) for v in fc.predicted_mean.values],
        'fc_lower': [float(v) for v in ci.iloc[:, 0].values],
        'fc_upper': [float(v) for v in ci.iloc[:, 1].values],
        'order': best_order,
        'aic': round(best_aic, 2),
        'adf_stat': round(float(adf[0]), 4),
        'adf_pvalue': round(float(adf[1]), 4),
    }


@st.cache_data(ttl=3600, show_spinner="🔮 Đang chạy ARIMA sản phẩm...")
def run_arima_products(df_serialized, top_n=10, periods=6):
    df = pd.DataFrame(df_serialized)
    df['NgayHoaDon'] = pd.to_datetime(df['NgayHoaDon'])
    top_names = df.groupby('TenSanPham')['ThanhTienSauVAT'].sum().nlargest(top_n).index.tolist()
    results = []
    for name in top_names:
        try:
            sub = df[df['TenSanPham'] == name]
            ts = sub.groupby(sub['NgayHoaDon'].dt.to_period('M'))['ThanhTienSauVAT'].sum()
            ts.index = ts.index.to_timestamp()
            ts = ts.sort_index().asfreq('MS', fill_value=0)
            if len(ts) < 12:
                continue
            best_aic, best_order = np.inf, (1, 1, 1)
            for p in range(0, 3):
                for d in range(0, 2):
                    for q in range(0, 3):
                        try:
                            r = ARIMA(ts, order=(p, d, q)).fit()
                            if r.aic < best_aic:
                                best_aic, best_order = r.aic, (p, d, q)
                        except:
                            pass
            model = ARIMA(ts, order=best_order).fit()
            fc = model.get_forecast(steps=periods)
            ci = fc.conf_int()
            results.append({
                'name': name,
                'order': best_order,
                'aic': round(best_aic, 2),
                'hist_dates': [d.strftime('%m/%Y') for d in ts.index],
                'hist_values': [float(v) for v in ts.values],
                'fc_dates': [d.strftime('%m/%Y') for d in fc.predicted_mean.index],
                'fc_values': [float(v) for v in fc.predicted_mean.values],
                'fc_lower': [float(v) for v in ci.iloc[:, 0].values],
                'fc_upper': [float(v) for v in ci.iloc[:, 1].values],
            })
        except:
            pass
    return results


@st.cache_data(ttl=1800, show_spinner="📅 Phân tích tháng hiện tại...")
def compute_current_month(df_serialized):
    df = pd.DataFrame(df_serialized)
    df['NgayHoaDon'] = pd.to_datetime(df['NgayHoaDon'])
    df['Nam'] = df['NgayHoaDon'].dt.year
    df['Thang_Num'] = df['NgayHoaDon'].dt.month

    today = datetime.now()
    cur_year, cur_month = today.year, today.month
    days_in_month = calendar.monthrange(cur_year, cur_month)[1]

    df_cur = df[(df['Nam'] == cur_year) & (df['Thang_Num'] == cur_month)].copy()

    prev_month, prev_year = (12, cur_year - 1) if cur_month == 1 else (cur_month - 1, cur_year)
    df_prev = df[(df['Nam'] == prev_year) & (df['Thang_Num'] == prev_month)]
    df_yoy = df[(df['Nam'] == cur_year - 1) & (df['Thang_Num'] == cur_month)]

    def sum_rev(d): return float(d['ThanhTienSauVAT'].sum()) if not d.empty else 0
    def sum_qty(d): return int(d['SoLuong'].sum()) if not d.empty else 0
    def cnt_inv(d): return int(d['SoHoaDon'].nunique()) if not d.empty and 'SoHoaDon' in d.columns else 0
    def cnt_cust(d): return int(d['TenKhachHang'].nunique()) if not d.empty and 'TenKhachHang' in d.columns else 0

    rev_cur, rev_prev, rev_yoy = sum_rev(df_cur), sum_rev(df_prev), sum_rev(df_yoy)
    qty_cur, qty_prev = sum_qty(df_cur), sum_qty(df_prev)

    # Daily actual
    if not df_cur.empty:
        df_cur['Ngay'] = df_cur['NgayHoaDon'].dt.day
        daily_agg = df_cur.groupby('Ngay').agg(
            revenue=('ThanhTienSauVAT', 'sum'),
            quantity=('SoLuong', 'sum'),
            invoices=('SoHoaDon', 'nunique'),
            customers=('TenKhachHang', 'nunique')
        ).reset_index().sort_values('Ngay')
    else:
        daily_agg = pd.DataFrame(columns=['Ngay', 'revenue', 'quantity', 'invoices', 'customers'])

    last_actual_day = int(daily_agg['Ngay'].max()) if not daily_agg.empty else 0
    forecast_days = list(range(last_actual_day + 1, days_in_month + 1))

    # === DAILY FORECAST ===
    daily_forecast = {}
    if forecast_days and not daily_agg.empty:
        # Historical weighted average
        hist_daily = {}
        df_hist = df[(df['Thang_Num'] == cur_month) & (df['Nam'] < cur_year)].copy()
        if not df_hist.empty:
            df_hist['Ngay'] = df_hist['NgayHoaDon'].dt.day
            past_years = sorted(df_hist['Nam'].unique())
            hbd = df_hist.groupby(['Nam', 'Ngay'])['ThanhTienSauVAT'].sum().reset_index()
            for day in forecast_days:
                dd = hbd[hbd['Ngay'] == day]
                if not dd.empty:
                    ws = [1 + (y - past_years[0]) for y in dd['Nam']]
                    hist_daily[day] = float(np.average(dd['ThanhTienSauVAT'], weights=ws))

        recent_avg = float(daily_agg['revenue'].tail(7).mean()) if len(daily_agg) >= 3 else float(daily_agg['revenue'].mean())

        # ARIMA on daily series
        arima_fc = {}
        if len(daily_agg) >= 7:
            try:
                ts = daily_agg.set_index('Ngay')['revenue']
                ts.index = pd.RangeIndex(start=1, stop=len(ts) + 1)
                best_aic, best_order = np.inf, (1, 0, 0)
                for p in range(0, 3):
                    for d in range(0, 2):
                        for q in range(0, 3):
                            try:
                                r = ARIMA(ts, order=(p, d, q)).fit()
                                if r.aic < best_aic:
                                    best_aic, best_order = r.aic, (p, d, q)
                            except:
                                pass
                model = ARIMA(ts, order=best_order).fit()
                fc = model.get_forecast(steps=len(forecast_days))
                fc_ci = fc.conf_int()
                for i, day in enumerate(forecast_days):
                    arima_fc[day] = {
                        'mean': max(float(fc.predicted_mean.iloc[i]), 0),
                        'lower': max(float(fc_ci.iloc[i, 0]), 0),
                        'upper': float(fc_ci.iloc[i, 1])
                    }
            except:
                pass

        # Combine methods
        for day in forecast_days:
            estimates, weights = [], []
            if day in hist_daily and hist_daily[day] > 0:
                estimates.append(hist_daily[day]); weights.append(0.3)
            estimates.append(recent_avg); weights.append(0.3)
            if day in arima_fc:
                estimates.append(arima_fc[day]['mean']); weights.append(0.4)
            val = float(np.average(estimates, weights=weights[:len(estimates)]))
            lo = arima_fc[day]['lower'] if day in arima_fc else val * 0.75
            hi = arima_fc[day]['upper'] if day in arima_fc else val * 1.25
            daily_forecast[day] = {'value': round(val), 'lower': round(max(lo, 0)), 'upper': round(hi)}

    # Build full daily arrays
    days_labels = list(range(1, days_in_month + 1))
    actual_values, forecast_values, forecast_lower, forecast_upper = [], [], [], []
    cumulative = []
    running = 0
    actual_map = {int(r['Ngay']): float(r['revenue']) for _, r in daily_agg.iterrows()} if not daily_agg.empty else {}

    for day in days_labels:
        if day in actual_map:
            actual_values.append(actual_map[day])
            forecast_values.append(None)
            forecast_lower.append(None)
            forecast_upper.append(None)
            running += actual_map[day]
        elif day in daily_forecast:
            actual_values.append(None)
            forecast_values.append(daily_forecast[day]['value'])
            forecast_lower.append(daily_forecast[day]['lower'])
            forecast_upper.append(daily_forecast[day]['upper'])
            running += daily_forecast[day]['value']
        else:
            actual_values.append(None)
            forecast_values.append(None)
            forecast_lower.append(None)
            forecast_upper.append(None)
        cumulative.append(running)

    forecast_total = sum(v['value'] for v in daily_forecast.values())
    projected_month = rev_cur + forecast_total

    # Top SP, KH, Channels
    top_sp, top_kh, month_channels = [], [], []
    if not df_cur.empty:
        sp_agg = df_cur.groupby('TenSanPham').agg(
            revenue=('ThanhTienSauVAT', 'sum'), quantity=('SoLuong', 'sum'),
            invoices=('SoHoaDon', 'nunique')
        ).reset_index().sort_values('revenue', ascending=False).head(10)
        total_sp = sp_agg['revenue'].sum()
        for _, r in sp_agg.iterrows():
            top_sp.append({'name': r['TenSanPham'], 'revenue': float(r['revenue']),
                           'quantity': int(r['quantity']), 'invoices': int(r['invoices']),
                           'pct': round(float(r['revenue']) / total_sp * 100, 1) if total_sp > 0 else 0})

        if 'TenKhachHang' in df_cur.columns:
            kh_agg = df_cur.groupby('TenKhachHang').agg(
                revenue=('ThanhTienSauVAT', 'sum'), quantity=('SoLuong', 'sum'),
                invoices=('SoHoaDon', 'nunique')
            ).reset_index().sort_values('revenue', ascending=False).head(10)
            for _, r in kh_agg.iterrows():
                top_kh.append({'name': r['TenKhachHang'], 'revenue': float(r['revenue']),
                               'quantity': int(r['quantity']), 'invoices': int(r['invoices'])})

        if 'KenhBanHang' in df_cur.columns:
            ch_agg = df_cur.groupby('KenhBanHang')['ThanhTienSauVAT'].sum().reset_index().sort_values('ThanhTienSauVAT', ascending=False)
            ch_total = ch_agg['ThanhTienSauVAT'].sum()
            for _, r in ch_agg.iterrows():
                month_channels.append({'name': r['KenhBanHang'], 'revenue': float(r['ThanhTienSauVAT']),
                                       'pct': round(float(r['ThanhTienSauVAT']) / ch_total * 100, 1) if ch_total > 0 else 0})

    return {
        'month_label': f"{cur_month:02d}/{cur_year}",
        'days_in_month': days_in_month,
        'last_actual_day': last_actual_day,
        'forecast_days_count': len(forecast_days),
        'rev_cur': rev_cur, 'rev_prev': rev_prev, 'rev_yoy': rev_yoy,
        'qty_cur': qty_cur, 'qty_prev': qty_prev,
        'growth_prev': growth_pct(rev_cur, rev_prev),
        'growth_yoy': growth_pct(rev_cur, rev_yoy),
        'invoices': cnt_inv(df_cur),
        'customers': cnt_cust(df_cur),
        'avg_daily': round(rev_cur / max(last_actual_day, 1)),
        'forecast_total': forecast_total,
        'projected_month': projected_month,
        'daily': {
            'labels': days_labels,
            'actual': actual_values,
            'forecast': forecast_values,
            'forecast_lower': forecast_lower,
            'forecast_upper': forecast_upper,
            'cumulative': cumulative,
        },
        'top_products': top_sp,
        'top_customers': top_kh,
        'channels': month_channels,
    }


# ═══════════════════════════════════════════════════════════════
# RENDERING FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def render_overview(df):
    """Tab 1: Tổng Quan KPI."""
    total_rev = float(df['ThanhTienSauVAT'].sum())
    total_qty = int(df['SoLuong'].sum())
    total_cust = int(df['TenKhachHang'].nunique())
    total_prod = int(df['TenSanPham'].nunique())
    total_inv = int(df['SoHoaDon'].nunique()) if 'SoHoaDon' in df.columns else 0
    total_prov = int(df['TinhThanh'].nunique()) if 'TinhThanh' in df.columns else 0

    # KPI row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("💰 Tổng Doanh Thu", fmt_vnd(total_rev))
    c2.metric("📦 Tổng Số Lượng", f"{total_qty:,}")
    c3.metric("👥 Tổng Khách Hàng", f"{total_cust:,}")
    c4.metric("🏭 Tổng Sản Phẩm", f"{total_prod:,}")
    c5.metric("🧾 Tổng Hóa Đơn", f"{total_inv:,}")
    c6.metric("🗺️ Số Tỉnh/Thành", f"{total_prov:,}")

    # Yearly KPIs
    yearly = df.groupby('Nam')['ThanhTienSauVAT'].sum().sort_index()
    yrs = sorted(yearly.index)

    st.markdown("---")
    yr_cols = st.columns(len(yrs))
    for i, yr in enumerate(yrs):
        delta = None
        if i > 0 and yearly[yrs[i - 1]] > 0:
            delta = f"{(yearly[yr] - yearly[yrs[i-1]]) / yearly[yrs[i-1]] * 100:.1f}%"
        yr_cols[i].metric(f"📅 Doanh Thu {int(yr)}", fmt_vnd(yearly[yr]), delta=delta)

    # Charts
    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        fig = go.Figure(go.Bar(
            x=[str(int(y)) for y in yrs],
            y=[yearly[y] for y in yrs],
            marker_color=COLORS[:len(yrs)],
            text=[fmt_vnd(yearly[y]) for y in yrs],
            textposition='outside',
        ))
        fig.update_layout(title="📊 Doanh Thu Theo Năm", yaxis_title="VNĐ",
                          template="plotly_white", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        growths_labels, growths_values, g_colors = [], [], []
        for i in range(1, len(yrs)):
            if yearly[yrs[i - 1]] > 0:
                g = (yearly[yrs[i]] - yearly[yrs[i - 1]]) / yearly[yrs[i - 1]] * 100
                growths_labels.append(f"{int(yrs[i])} vs {int(yrs[i-1])}")
                growths_values.append(round(g, 1))
                g_colors.append('#27AE60' if g >= 0 else '#E74C3C')

        fig = go.Figure(go.Bar(
            y=growths_labels, x=growths_values,
            orientation='h',
            marker_color=g_colors,
            text=[f"{v:+.1f}%" for v in growths_values],
            textposition='outside',
        ))
        fig.update_layout(title="📈 Tăng Trưởng YoY (%)", xaxis_title="%",
                          template="plotly_white", height=400)
        st.plotly_chart(fig, use_container_width=True)


def render_monthly(df):
    """Tab 2: Doanh Thu Tháng."""
    monthly = df.groupby(['Nam', 'Thang_Num']).agg(
        revenue=('ThanhTienSauVAT', 'sum'),
        quantity=('SoLuong', 'sum'),
        invoices=('SoHoaDon', 'nunique'),
        customers=('TenKhachHang', 'nunique'),
    ).reset_index().sort_values(['Nam', 'Thang_Num'])
    monthly['label'] = monthly.apply(lambda r: f"{int(r['Thang_Num']):02d}/{int(r['Nam'])}", axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly['label'], y=monthly['revenue'],
        mode='lines+markers', name='Doanh thu',
        line=dict(color='#2E86C1', width=2.5),
        fill='tozeroy', fillcolor='rgba(46,134,193,0.1)',
        hovertemplate='%{x}: %{customdata}<extra></extra>',
        customdata=[fmt_vnd(v) for v in monthly['revenue']],
    ))
    fig.update_layout(title="📅 Xu Hướng Doanh Thu Theo Tháng", yaxis_title="VNĐ",
                      template="plotly_white", height=450)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Bar(
            x=monthly['label'], y=monthly['quantity'],
            marker_color='#27AE60', name='Số lượng',
        ))
        fig.update_layout(title="📦 Số Lượng Bán Theo Tháng", template="plotly_white", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure(go.Scatter(
            x=monthly['label'], y=monthly['customers'],
            mode='lines+markers', name='Khách hàng',
            line=dict(color='#F39C12', width=2),
        ))
        fig.update_layout(title="👥 Số Khách Hàng Theo Tháng", template="plotly_white", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Bảng Chi Tiết Doanh Thu Tháng", expanded=False):
        display_df = monthly[['label', 'revenue', 'quantity', 'invoices', 'customers']].copy()
        display_df.columns = ['Tháng', 'Doanh Thu', 'Số Lượng', 'Số HĐ', 'Số KH']
        display_df['Doanh Thu'] = display_df['Doanh Thu'].apply(fmt_vnd)
        st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_quarterly(df):
    """Tab 3: Doanh Thu Quý."""
    quarterly = df.groupby(['Nam', 'Quy']).agg(
        revenue=('ThanhTienSauVAT', 'sum'),
        quantity=('SoLuong', 'sum'),
    ).reset_index().sort_values(['Nam', 'Quy'])
    quarterly['label'] = quarterly.apply(lambda r: f"Q{int(r['Quy'])}/{int(r['Nam'])}", axis=1)

    fig = go.Figure(go.Bar(
        x=quarterly['label'], y=quarterly['revenue'],
        marker_color=[COLORS[i % len(COLORS)] for i in range(len(quarterly))],
        text=[fmt_vnd(v) for v in quarterly['revenue']],
        textposition='outside',
    ))
    fig.update_layout(title="📊 Doanh Thu Theo Quý", yaxis_title="VNĐ",
                      template="plotly_white", height=500)
    st.plotly_chart(fig, use_container_width=True)


def render_products(df):
    """Tab 4: Top Sản Phẩm."""
    prod = df.groupby('TenSanPham').agg(
        revenue=('ThanhTienSauVAT', 'sum'),
        quantity=('SoLuong', 'sum'),
        invoices=('SoHoaDon', 'nunique'),
        customers=('TenKhachHang', 'nunique'),
        avg_price=('DonGia', 'mean'),
    ).reset_index().sort_values('revenue', ascending=False).head(20)
    total_r = prod['revenue'].sum()
    prod['pct'] = (prod['revenue'] / total_r * 100).round(2)

    # Top 20 bar chart
    fig = go.Figure(go.Bar(
        y=prod['TenSanPham'].tolist()[::-1],
        x=prod['revenue'].tolist()[::-1],
        orientation='h',
        marker_color='#F39C12',
        text=[fmt_vnd(v) for v in prod['revenue'].tolist()[::-1]],
        textposition='outside',
    ))
    fig.update_layout(title="🏆 Top 20 Sản Phẩm Doanh Thu Cao Nhất", xaxis_title="VNĐ",
                      template="plotly_white", height=600, margin=dict(l=250))
    st.plotly_chart(fig, use_container_width=True)

    # Top 10 yearly trend
    top10 = prod['TenSanPham'].head(10).tolist()
    df_t10 = df[df['TenSanPham'].isin(top10)]
    pivot = df_t10.pivot_table(values='ThanhTienSauVAT', index='TenSanPham',
                               columns='Nam', aggfunc='sum', fill_value=0)

    fig = go.Figure()
    for i, name in enumerate(pivot.index):
        fig.add_trace(go.Scatter(
            x=[str(int(y)) for y in pivot.columns],
            y=pivot.loc[name].values,
            name=name[:35], mode='lines+markers',
            line=dict(color=COLORS[i], width=2),
        ))
    fig.update_layout(title="📈 Xu Hướng Top 10 Sản Phẩm", template="plotly_white",
                      height=450, legend=dict(font=dict(size=10)))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Chi Tiết Top 20 Sản Phẩm", expanded=False):
        tbl = prod[['TenSanPham', 'revenue', 'quantity', 'invoices', 'customers', 'avg_price', 'pct']].copy()
        tbl.columns = ['Sản Phẩm', 'Doanh Thu', 'Số Lượng', 'Số HĐ', 'Số KH', 'Giá TB', 'Tỷ Lệ %']
        tbl['Doanh Thu'] = tbl['Doanh Thu'].apply(fmt_vnd)
        tbl['Giá TB'] = tbl['Giá TB'].apply(lambda x: f"{x:,.0f}")
        tbl.index = range(1, len(tbl) + 1)
        st.dataframe(tbl, use_container_width=True)


def render_regional(df):
    """Tab 5: Vùng Miền."""
    if 'Vung' not in df.columns:
        st.warning("Không có dữ liệu Vùng.")
        return

    col1, col2 = st.columns(2)

    with col1:
        reg = df.groupby('Vung').agg(
            revenue=('ThanhTienSauVAT', 'sum'),
            customers=('TenKhachHang', 'nunique'),
        ).reset_index().sort_values('revenue', ascending=False)
        reg['pct'] = (reg['revenue'] / reg['revenue'].sum() * 100).round(1)

        fig = go.Figure(go.Pie(
            labels=reg['Vung'], values=reg['revenue'],
            hole=0.45, marker_colors=COLORS,
            textinfo='label+percent',
            hovertemplate='%{label}: %{customdata}<extra></extra>',
            customdata=[fmt_vnd(v) for v in reg['revenue']],
        ))
        fig.update_layout(title="🗺️ Doanh Thu Theo Vùng", template="plotly_white", height=450)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if 'TinhThanh' in df.columns:
            prov = df.groupby('TinhThanh').agg(
                revenue=('ThanhTienSauVAT', 'sum'),
            ).reset_index().sort_values('revenue', ascending=False).head(15)

            fig = go.Figure(go.Bar(
                y=prov['TinhThanh'].tolist()[::-1],
                x=prov['revenue'].tolist()[::-1],
                orientation='h', marker_color='#E74C3C',
                text=[fmt_vnd(v) for v in prov['revenue'].tolist()[::-1]],
                textposition='outside',
            ))
            fig.update_layout(title="🏙️ Top 15 Tỉnh Thành", xaxis_title="VNĐ",
                              template="plotly_white", height=450, margin=dict(l=180))
            st.plotly_chart(fig, use_container_width=True)


def render_channel(df):
    """Tab 6: Kênh Bán Hàng."""
    if 'KenhBanHang' not in df.columns:
        st.warning("Không có dữ liệu Kênh Bán Hàng.")
        return

    col1, col2 = st.columns(2)

    with col1:
        ch = df.groupby('KenhBanHang').agg(
            revenue=('ThanhTienSauVAT', 'sum'),
            customers=('TenKhachHang', 'nunique'),
        ).reset_index().sort_values('revenue', ascending=False)
        ch['pct'] = (ch['revenue'] / ch['revenue'].sum() * 100).round(1)

        fig = go.Figure(go.Pie(
            labels=ch['KenhBanHang'], values=ch['revenue'],
            hole=0.45, marker_colors=COLORS,
            textinfo='label+percent',
        ))
        fig.update_layout(title="🏪 Tỷ Lệ Kênh Bán Hàng", template="plotly_white", height=450)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        ch_yr = df.pivot_table(values='ThanhTienSauVAT', index='KenhBanHang',
                               columns='Nam', aggfunc='sum', fill_value=0).reset_index()
        fig = go.Figure()
        for i, row in ch_yr.iterrows():
            fig.add_trace(go.Bar(
                name=row['KenhBanHang'],
                x=[str(int(y)) for y in ch_yr.columns[1:]],
                y=[row[y] for y in ch_yr.columns[1:]],
                marker_color=COLORS[i % len(COLORS)],
            ))
        fig.update_layout(title="📈 Xu Hướng Kênh Theo Năm", barmode='group',
                          template="plotly_white", height=450)
        st.plotly_chart(fig, use_container_width=True)


def render_customers(df):
    """Tab 7: Top Khách Hàng."""
    if 'TenKhachHang' not in df.columns:
        st.warning("Không có dữ liệu Khách Hàng.")
        return

    cust = df.groupby('TenKhachHang').agg(
        revenue=('ThanhTienSauVAT', 'sum'),
        quantity=('SoLuong', 'sum'),
        invoices=('SoHoaDon', 'nunique'),
    ).reset_index().sort_values('revenue', ascending=False).head(15)

    fig = go.Figure(go.Bar(
        y=cust['TenKhachHang'].tolist()[::-1],
        x=cust['revenue'].tolist()[::-1],
        orientation='h', marker_color='#D35400',
        text=[fmt_vnd(v) for v in cust['revenue'].tolist()[::-1]],
        textposition='outside',
    ))
    fig.update_layout(title="👥 Top 15 Khách Hàng Doanh Thu Cao Nhất", xaxis_title="VNĐ",
                      template="plotly_white", height=500, margin=dict(l=250))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Chi Tiết Top Khách Hàng", expanded=False):
        tbl = cust[['TenKhachHang', 'revenue', 'quantity', 'invoices']].copy()
        tbl.columns = ['Khách Hàng', 'Doanh Thu', 'Số Lượng', 'Số HĐ']
        tbl['Doanh Thu'] = tbl['Doanh Thu'].apply(fmt_vnd)
        tbl.index = range(1, len(tbl) + 1)
        st.dataframe(tbl, use_container_width=True)


def render_arima_total(df):
    """Tab 8: ARIMA Total."""
    df_dict = df[['NgayHoaDon', 'ThanhTienSauVAT']].to_dict('list')
    arima = run_arima_total(df_dict)

    # Info box
    st.info(f"""
    🔮 **Mô hình ARIMA** | Tham số: **{arima['order']}** | AIC: **{arima['aic']}** |
    ADF Statistic: **{arima['adf_stat']}** | p-value: **{arima['adf_pvalue']}** |
    Dự báo: **6 tháng tiếp theo**
    """)

    # Chart
    all_labels = arima['hist_dates'] + arima['fc_dates']
    hist_vals = arima['hist_values'] + [None] * len(arima['fc_dates'])
    fc_vals = [None] * (len(arima['hist_dates']) - 1) + [arima['hist_values'][-1]] + arima['fc_values']
    lower = [None] * (len(arima['hist_dates']) - 1) + [arima['hist_values'][-1]] + arima['fc_lower']
    upper = [None] * (len(arima['hist_dates']) - 1) + [arima['hist_values'][-1]] + arima['fc_upper']

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=all_labels, y=hist_vals, name='Thực tế',
                             line=dict(color='#2E86C1', width=2.5), mode='lines'))
    fig.add_trace(go.Scatter(x=all_labels, y=fc_vals, name='Dự báo ARIMA',
                             line=dict(color='#8E44AD', width=2.5, dash='dash'),
                             mode='lines+markers', marker=dict(size=7, symbol='diamond')))
    fig.add_trace(go.Scatter(x=all_labels, y=upper, name='Giới hạn trên (95%)',
                             line=dict(color='rgba(189,195,199,0.6)', width=1, dash='dot'),
                             mode='lines', showlegend=True))
    fig.add_trace(go.Scatter(x=all_labels, y=lower, name='Giới hạn dưới (95%)',
                             line=dict(color='rgba(189,195,199,0.6)', width=1, dash='dot'),
                             mode='lines', fill='tonexty', fillcolor='rgba(142,68,173,0.08)'))
    fig.update_layout(title="🔮 Dự Báo Doanh Thu ARIMA (Khoảng Tin Cậy 95%)",
                      yaxis_title="VNĐ", template="plotly_white", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Forecast table
    with st.expander("📋 Bảng Số Liệu Dự Báo", expanded=True):
        fc_df = pd.DataFrame({
            'Tháng': arima['fc_dates'],
            'Dự Báo': [fmt_vnd(v) for v in arima['fc_values']],
            'Giới Hạn Dưới': [fmt_vnd(v) for v in arima['fc_lower']],
            'Giới Hạn Trên': [fmt_vnd(v) for v in arima['fc_upper']],
        })
        st.dataframe(fc_df, use_container_width=True, hide_index=True)


def render_arima_products(df):
    """Tab 9: ARIMA per product."""
    df_dict = df[['NgayHoaDon', 'TenSanPham', 'ThanhTienSauVAT']].to_dict('list')
    products = run_arima_products(df_dict)

    if not products:
        st.warning("Không đủ dữ liệu để chạy ARIMA sản phẩm.")
        return

    st.info(f"🔮 **ARIMA Forecast Theo Sản Phẩm** — Dự báo ARIMA riêng cho top {len(products)} sản phẩm, mỗi SP tự động tìm (p,d,q) tối ưu.")

    for i, prod in enumerate(products):
        with st.expander(f"🔮 {prod['name']} — ARIMA{prod['order']} (AIC: {prod['aic']})", expanded=(i < 3)):
            all_labels = prod['hist_dates'] + prod['fc_dates']
            h = prod['hist_values'] + [None] * len(prod['fc_dates'])
            f_v = [None] * (len(prod['hist_dates']) - 1) + [prod['hist_values'][-1]] + prod['fc_values']
            lo = [None] * (len(prod['hist_dates']) - 1) + [prod['hist_values'][-1]] + prod['fc_lower']
            up = [None] * (len(prod['hist_dates']) - 1) + [prod['hist_values'][-1]] + prod['fc_upper']

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=all_labels, y=h, name='Thực tế',
                                     line=dict(color=COLORS[i], width=2), mode='lines'))
            fig.add_trace(go.Scatter(x=all_labels, y=f_v, name='Dự báo',
                                     line=dict(color='#8E44AD', width=2, dash='dash'),
                                     mode='lines+markers', marker=dict(size=5)))
            fig.add_trace(go.Scatter(x=all_labels, y=up, name='CI trên',
                                     line=dict(color='rgba(189,195,199,0.5)', width=1, dash='dot'), mode='lines'))
            fig.add_trace(go.Scatter(x=all_labels, y=lo, name='CI dưới',
                                     line=dict(color='rgba(189,195,199,0.5)', width=1, dash='dot'),
                                     mode='lines', fill='tonexty', fillcolor='rgba(142,68,173,0.06)'))
            fig.update_layout(template="plotly_white", height=350,
                              margin=dict(t=30), legend=dict(font=dict(size=10)))
            st.plotly_chart(fig, use_container_width=True)

    # Summary table
    with st.expander("📋 Bảng Dự Báo Chi Tiết Tất Cả SP"):
        rows = []
        for prod in products:
            for j, lbl in enumerate(prod['fc_dates']):
                rows.append({
                    'Sản Phẩm': prod['name'],
                    'ARIMA': str(prod['order']),
                    'AIC': prod['aic'],
                    'Tháng': lbl,
                    'Dự Báo': fmt_vnd(prod['fc_values'][j]),
                    'CI Dưới': fmt_vnd(prod['fc_lower'][j]),
                    'CI Trên': fmt_vnd(prod['fc_upper'][j]),
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_current_month(df):
    """Tab 10: Tháng Hiện Tại."""
    df_dict = df.to_dict('list')
    cm = compute_current_month(df_dict)

    # KPI Cards
    st.markdown(f"### 📅 Báo Cáo Tháng {cm['month_label']} (Dữ liệu đến ngày {cm['last_actual_day']})")

    c1, c2, c3 = st.columns(3)
    c1.metric(f"💰 DT Tháng {cm['month_label']}", fmt_vnd(cm['rev_cur']),
              delta=f"{cm['growth_prev']:+.1f}% vs tháng trước" if cm['growth_prev'] is not None else None)
    c2.metric("📊 DT Tháng Trước", fmt_vnd(cm['rev_prev']))
    c3.metric("📊 DT Cùng Kỳ Năm Trước", fmt_vnd(cm['rev_yoy']),
              delta=f"{cm['growth_yoy']:+.1f}% vs cùng kỳ" if cm['growth_yoy'] is not None else None)

    c4, c5, c6 = st.columns(3)
    c4.metric("📦 Số Lượng Bán", f"{cm['qty_cur']:,}")
    c5.metric("🧾 Số Hóa Đơn", f"{cm['invoices']:,}")
    c6.metric("👥 Số Khách Hàng", f"{cm['customers']:,}")

    c7, c8, c9 = st.columns(3)
    c7.metric("📊 DT Trung Bình/Ngày", fmt_vnd(cm['avg_daily']))
    c8.metric("🔮 DT Dự Báo Còn Lại", fmt_vnd(cm['forecast_total']),
              delta=f"{cm['forecast_days_count']} ngày còn lại", delta_color="off")
    c9.metric("🎯 DT Dự Báo Cả Tháng", fmt_vnd(cm['projected_month']),
              delta="Thực tế + Dự báo", delta_color="off")

    st.markdown("---")

    d = cm['daily']

    # Chart 1: Daily Actual + Forecast
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=d['labels'], y=d['actual'], name='Thực tế',
        marker_color='#27AE60', marker_line_color='#1E8449', marker_line_width=1,
    ))
    fig.add_trace(go.Bar(
        x=d['labels'], y=d['forecast'], name='Dự báo',
        marker_color='rgba(142,68,173,0.65)', marker_line_color='#8E44AD', marker_line_width=1,
    ))
    fig.add_trace(go.Scatter(
        x=d['labels'], y=d['forecast_upper'], name='Giới hạn trên',
        line=dict(color='rgba(210,180,222,0.7)', width=1.5, dash='dash'),
        mode='lines',
    ))
    fig.add_trace(go.Scatter(
        x=d['labels'], y=d['forecast_lower'], name='Giới hạn dưới',
        line=dict(color='rgba(210,180,222,0.7)', width=1.5, dash='dash'),
        mode='lines', fill='tonexty', fillcolor='rgba(142,68,173,0.06)',
    ))
    fig.update_layout(
        title="📊 Doanh Thu Theo Ngày — Thực Tế & Dự Báo",
        xaxis_title="Ngày trong tháng", yaxis_title="VNĐ",
        template="plotly_white", height=450, barmode='overlay',
    )
    st.plotly_chart(fig, use_container_width=True)

    # Chart 2: Cumulative
    fig2 = go.Figure()
    # Split cumulative into actual part and forecast part
    actual_cumul = []
    forecast_cumul = []
    for i, day in enumerate(d['labels']):
        if d['actual'][i] is not None:
            actual_cumul.append(d['cumulative'][i])
            forecast_cumul.append(None)
        else:
            if not actual_cumul or actual_cumul[-1] is None:
                actual_cumul.append(None)
            else:
                actual_cumul.append(d['cumulative'][i - 1] if i > 0 else None)  # bridge point
            forecast_cumul.append(d['cumulative'][i])

    fig2.add_trace(go.Scatter(
        x=d['labels'], y=actual_cumul, name='Lũy kế thực tế',
        line=dict(color='#27AE60', width=3), mode='lines',
        fill='tozeroy', fillcolor='rgba(39,174,96,0.1)',
    ))
    fig2.add_trace(go.Scatter(
        x=d['labels'], y=d['cumulative'], name='Lũy kế (Thực tế + Dự báo)',
        line=dict(color='#D35400', width=2.5, dash='dash'), mode='lines',
        fill='tozeroy', fillcolor='rgba(250,215,160,0.2)',
    ))
    fig2.update_layout(
        title="📈 Doanh Thu Lũy Kế Toàn Tháng", xaxis_title="Ngày", yaxis_title="VNĐ",
        template="plotly_white", height=400,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Top SP + Channel
    col_l, col_r = st.columns(2)

    with col_l:
        if cm['top_products']:
            sp_names = [p['name'][:35] for p in cm['top_products']]
            sp_rev = [p['revenue'] for p in cm['top_products']]
            fig = go.Figure(go.Bar(
                y=sp_names[::-1], x=sp_rev[::-1], orientation='h',
                marker_color='#E67E22',
                text=[fmt_vnd(v) for v in sp_rev[::-1]], textposition='outside',
            ))
            fig.update_layout(title="🏆 Top 10 Sản Phẩm Trong Tháng",
                              template="plotly_white", height=400, margin=dict(l=220))
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        if cm['channels']:
            fig = go.Figure(go.Pie(
                labels=[c['name'] for c in cm['channels']],
                values=[c['revenue'] for c in cm['channels']],
                hole=0.45, marker_colors=COLORS,
                textinfo='label+percent',
            ))
            fig.update_layout(title="🏪 Kênh Bán Hàng Trong Tháng",
                              template="plotly_white", height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Tables
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        if cm['top_products']:
            with st.expander("📋 Chi Tiết Top SP", expanded=True):
                sp_df = pd.DataFrame(cm['top_products'])
                sp_df.columns = ['Sản Phẩm', 'Doanh Thu', 'SL', 'HĐ', '%']
                sp_df['Doanh Thu'] = sp_df['Doanh Thu'].apply(fmt_vnd)
                sp_df.index = range(1, len(sp_df) + 1)
                st.dataframe(sp_df, use_container_width=True)

    with col_t2:
        if cm['top_customers']:
            with st.expander("📋 Chi Tiết Top KH", expanded=True):
                kh_df = pd.DataFrame(cm['top_customers'])
                kh_df.columns = ['Khách Hàng', 'Doanh Thu', 'SL', 'HĐ']
                kh_df['Doanh Thu'] = kh_df['Doanh Thu'].apply(fmt_vnd)
                kh_df.index = range(1, len(kh_df) + 1)
                st.dataframe(kh_df, use_container_width=True)

    # Daily full table
    with st.expander("📋 Bảng Doanh Thu Theo Ngày (Chi Tiết)", expanded=False):
        daily_rows = []
        for i, day in enumerate(d['labels']):
            daily_rows.append({
                'Ngày': day,
                'DT Thực Tế': fmt_vnd(d['actual'][i]) if d['actual'][i] is not None else '-',
                'DT Dự Báo': fmt_vnd(d['forecast'][i]) if d['forecast'][i] is not None else '-',
                'CI Dưới': fmt_vnd(d['forecast_lower'][i]) if d['forecast_lower'][i] is not None else '-',
                'CI Trên': fmt_vnd(d['forecast_upper'][i]) if d['forecast_upper'][i] is not None else '-',
                'Lũy Kế': fmt_vnd(d['cumulative'][i]),
                'Loại': '🔮 Dự báo' if d['forecast'][i] is not None else '✅ Thực tế',
            })
        st.dataframe(pd.DataFrame(daily_rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════

def main():
    # Header
    gen_date = datetime.now().strftime('%d/%m/%Y %H:%M')
    st.markdown(f"""
    <div class="dashboard-header">
        <div>
            <h1>📊 CEO DASHBOARD - VIKODA (VKD)</h1>
            <div class="info">Báo cáo phân tích doanh thu & Dự báo ARIMA | Dữ liệu 2023 - 2026</div>
        </div>
        <div style="text-align:right">
            <div style="font-size:12px;opacity:0.8">Ngày tạo</div>
            <div style="font-size:16px;font-weight:700">{gen_date}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    df = load_all_data()
    st.success(f"✅ Đã tải {len(df):,} dòng dữ liệu từ {df['Nam'].nunique()} năm")

    # Tabs
    tabs = st.tabs([
        "📊 Tổng Quan",
        "📅 Doanh Thu Tháng",
        "📈 Doanh Thu Quý",
        "🏆 Top Sản Phẩm",
        "🗺️ Vùng Miền",
        "🏪 Kênh Bán Hàng",
        "👥 Khách Hàng",
        "🔮 ARIMA Dự Báo",
        "🔮 ARIMA Sản Phẩm",
        "📅 Tháng Hiện Tại",
    ])

    with tabs[0]:
        render_overview(df)
    with tabs[1]:
        render_monthly(df)
    with tabs[2]:
        render_quarterly(df)
    with tabs[3]:
        render_products(df)
    with tabs[4]:
        render_regional(df)
    with tabs[5]:
        render_channel(df)
    with tabs[6]:
        render_customers(df)
    with tabs[7]:
        render_arima_total(df)
    with tabs[8]:
        render_arima_products(df)
    with tabs[9]:
        render_current_month(df)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#7F8C8D; font-size:12px; padding:10px;">
        📊 CEO Dashboard - VIKODA | Powered by Streamlit + Plotly + ARIMA |
        Dữ liệu được cache 1 giờ — Nhấn <b>⟳ Rerun</b> để cập nhật
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
