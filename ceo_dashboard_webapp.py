# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════╗
║   CEO DASHBOARD - WEB APP (Streamlit)                              ║
║   Chạy: streamlit run ceo_dashboard_webapp.py                      ║
║   Truy cập: http://localhost:8501                                  ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime

# Import calculation engine functions
from calculation_engine import (
    format_vnd,
    calculate_growth_percentage,
    forecast_total_revenue,
    forecast_products_revenue,
    analyze_current_month,
)

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
# Note: Core calculation functions moved to calculation_engine.py
# Keeping wrapper functions here for Streamlit caching

@st.cache_data(ttl=3600, show_spinner="🔮 Đang chạy ARIMA tổng doanh thu...")
def run_arima_total(df_serialized, periods=6):
    """Wrapper for ARIMA total revenue forecasting with Streamlit caching."""
    df = pd.DataFrame(df_serialized)
    return forecast_total_revenue(df, periods=periods)


@st.cache_data(ttl=3600, show_spinner="🔮 Đang chạy ARIMA sản phẩm...")
def run_arima_products(df_serialized, top_n=10, periods=6):
    """Wrapper for ARIMA products forecasting with Streamlit caching."""
    df = pd.DataFrame(df_serialized)
    return forecast_products_revenue(df, top_n=top_n, periods=periods)


@st.cache_data(ttl=1800, show_spinner="📅 Phân tích tháng hiện tại...")
def compute_current_month(df_serialized):
    """Wrapper for current month analysis with Streamlit caching."""
    df = pd.DataFrame(df_serialized)
    return analyze_current_month(df)


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
    c1.metric("💰 Tổng Doanh Thu", format_vnd(total_rev))
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
        yr_cols[i].metric(f"📅 Doanh Thu {int(yr)}", format_vnd(yearly[yr]), delta=delta)

    # Charts
    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        fig = go.Figure(go.Bar(
            x=[str(int(y)) for y in yrs],
            y=[yearly[y] for y in yrs],
            marker_color=COLORS[:len(yrs)],
            text=[format_vnd(yearly[y]) for y in yrs],
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
        customdata=[format_vnd(v) for v in monthly['revenue']],
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
        display_df['Doanh Thu'] = display_df['Doanh Thu'].apply(format_vnd)
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
        text=[format_vnd(v) for v in quarterly['revenue']],
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
        text=[format_vnd(v) for v in prod['revenue'].tolist()[::-1]],
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
        tbl['Doanh Thu'] = tbl['Doanh Thu'].apply(format_vnd)
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
            customdata=[format_vnd(v) for v in reg['revenue']],
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
                text=[format_vnd(v) for v in prov['revenue'].tolist()[::-1]],
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
        text=[format_vnd(v) for v in cust['revenue'].tolist()[::-1]],
        textposition='outside',
    ))
    fig.update_layout(title="👥 Top 15 Khách Hàng Doanh Thu Cao Nhất", xaxis_title="VNĐ",
                      template="plotly_white", height=500, margin=dict(l=250))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Chi Tiết Top Khách Hàng", expanded=False):
        tbl = cust[['TenKhachHang', 'revenue', 'quantity', 'invoices']].copy()
        tbl.columns = ['Khách Hàng', 'Doanh Thu', 'Số Lượng', 'Số HĐ']
        tbl['Doanh Thu'] = tbl['Doanh Thu'].apply(format_vnd)
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
            'Dự Báo': [format_vnd(v) for v in arima['fc_values']],
            'Giới Hạn Dưới': [format_vnd(v) for v in arima['fc_lower']],
            'Giới Hạn Trên': [format_vnd(v) for v in arima['fc_upper']],
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
                    'Dự Báo': format_vnd(prod['fc_values'][j]),
                    'CI Dưới': format_vnd(prod['fc_lower'][j]),
                    'CI Trên': format_vnd(prod['fc_upper'][j]),
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_current_month(df):
    """Tab 10: Tháng Hiện Tại."""
    df_dict = df.to_dict('list')
    cm = compute_current_month(df_dict)

    # KPI Cards
    st.markdown(f"### 📅 Báo Cáo Tháng {cm['month_label']} (Dữ liệu đến ngày {cm['last_actual_day']})")

    c1, c2, c3 = st.columns(3)
    c1.metric(f"💰 DT Tháng {cm['month_label']}", format_vnd(cm['rev_cur']),
              delta=f"{cm['growth_prev']:+.1f}% vs tháng trước" if cm['growth_prev'] is not None else None)
    c2.metric("📊 DT Tháng Trước", format_vnd(cm['rev_prev']))
    c3.metric("📊 DT Cùng Kỳ Năm Trước", format_vnd(cm['rev_yoy']),
              delta=f"{cm['growth_yoy']:+.1f}% vs cùng kỳ" if cm['growth_yoy'] is not None else None)

    c4, c5, c6 = st.columns(3)
    c4.metric("📦 Số Lượng Bán", f"{cm['qty_cur']:,}")
    c5.metric("🧾 Số Hóa Đơn", f"{cm['invoices']:,}")
    c6.metric("👥 Số Khách Hàng", f"{cm['customers']:,}")

    c7, c8, c9 = st.columns(3)
    c7.metric("📊 DT Trung Bình/Ngày", format_vnd(cm['avg_daily']))
    c8.metric("🔮 DT Dự Báo Còn Lại", format_vnd(cm['forecast_total']),
              delta=f"{cm['forecast_days_count']} ngày còn lại", delta_color="off")
    c9.metric("🎯 DT Dự Báo Cả Tháng", format_vnd(cm['projected_month']),
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
                text=[format_vnd(v) for v in sp_rev[::-1]], textposition='outside',
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
                sp_df['Doanh Thu'] = sp_df['Doanh Thu'].apply(format_vnd)
                sp_df.index = range(1, len(sp_df) + 1)
                st.dataframe(sp_df, use_container_width=True)

    with col_t2:
        if cm['top_customers']:
            with st.expander("📋 Chi Tiết Top KH", expanded=True):
                kh_df = pd.DataFrame(cm['top_customers'])
                kh_df.columns = ['Khách Hàng', 'Doanh Thu', 'SL', 'HĐ']
                kh_df['Doanh Thu'] = kh_df['Doanh Thu'].apply(format_vnd)
                kh_df.index = range(1, len(kh_df) + 1)
                st.dataframe(kh_df, use_container_width=True)

    # Daily full table
    with st.expander("📋 Bảng Doanh Thu Theo Ngày (Chi Tiết)", expanded=False):
        daily_rows = []
        for i, day in enumerate(d['labels']):
            daily_rows.append({
                'Ngày': day,
                'DT Thực Tế': format_vnd(d['actual'][i]) if d['actual'][i] is not None else '-',
                'DT Dự Báo': format_vnd(d['forecast'][i]) if d['forecast'][i] is not None else '-',
                'CI Dưới': format_vnd(d['forecast_lower'][i]) if d['forecast_lower'][i] is not None else '-',
                'CI Trên': format_vnd(d['forecast_upper'][i]) if d['forecast_upper'][i] is not None else '-',
                'Lũy Kế': format_vnd(d['cumulative'][i]),
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
