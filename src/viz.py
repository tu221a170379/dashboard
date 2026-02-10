"""
Visualization module – Plotly chart builders for CEO Dashboard.

Each function returns a plotly Figure ready for st.plotly_chart().
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import REVENUE_DIVISOR, REVENUE_LABEL


# ──────────────────────── Color palette ────────────────────────
COLORS = {
    "primary": "#2563eb",
    "secondary": "#f59e0b",
    "green": "#10b981",
    "red": "#ef4444",
    "purple": "#8b5cf6",
    "brown": "#92400e",
    "pink": "#ec4899",
    "gray": "#6b7280",
    "light_blue": "#93c5fd",
    "dark_blue": "#1e3a5f",
}
SEVERITY_COLORS = {"RED": "#ef4444", "YELLOW": "#f59e0b", "GREEN": "#10b981"}

CHART_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Segoe UI, Inter, Arial", size=12, color="#374151"),
    margin=dict(l=40, r=20, t=45, b=40),
    height=380,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)


def _apply_layout(fig: go.Figure, title: str = "", **kwargs) -> go.Figure:
    merged = {**CHART_LAYOUT, **kwargs}  # kwargs override defaults
    fig.update_layout(title=title, **merged)
    return fig


# ═══════════════════════════════════════════════════════════════
#  TREND CHARTS
# ═══════════════════════════════════════════════════════════════

def daily_trend_chart(
    daily_df: pd.DataFrame,
    metric: str = "net_revenue",
    title: str = "Daily Net Revenue",
    show_avg: bool = True,
) -> go.Figure:
    """Line chart of daily metric with optional rolling average."""
    if daily_df.empty:
        return go.Figure().update_layout(title="No data")

    df = daily_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    y_vals = df[metric] / REVENUE_DIVISOR if "revenue" in metric or "sales" in metric else df[metric]
    y_label = REVENUE_LABEL if "revenue" in metric or "sales" in metric else metric

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=y_vals,
        mode="lines", name=title,
        line=dict(color=COLORS["primary"], width=1.5),
    ))

    if show_avg and len(df) > 7:
        rolling = y_vals.rolling(7, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df["Date"], y=rolling,
            mode="lines", name="7-day MA",
            line=dict(color=COLORS["secondary"], dash="dash", width=2),
        ))

    return _apply_layout(fig, title=title, yaxis_title=y_label)


def daily_revenue_units_chart(daily_df: pd.DataFrame) -> go.Figure:
    """Dual-axis: Revenue (bar) + Paid Units (line)."""
    if daily_df.empty:
        return go.Figure().update_layout(title="No data")

    df = daily_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    rev = df["net_revenue"] / REVENUE_DIVISOR

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=df["Date"], y=rev, name="Net Revenue", marker_color=COLORS["primary"], opacity=0.6),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=df["Date"], y=df["paid_units"], name="Paid Units",
                   line=dict(color=COLORS["secondary"], width=2)),
        secondary_y=True,
    )
    fig.update_yaxes(title_text=REVENUE_LABEL, secondary_y=False)
    fig.update_yaxes(title_text="Units", secondary_y=True)
    return _apply_layout(fig, title="Daily Revenue & Units")


def yoy_comparison_chart(daily_df: pd.DataFrame, current_year: int) -> go.Figure:
    """Overlay current year vs previous year daily revenue."""
    if daily_df.empty:
        return go.Figure().update_layout(title="No data")

    df = daily_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["year"] = df["Date"].dt.year
    df["day_of_year"] = df["Date"].dt.dayofyear

    fig = go.Figure()
    for yr in sorted(df["year"].unique()):
        sub = df[df["year"] == yr].copy()
        rev = sub["net_revenue"] / REVENUE_DIVISOR
        width = 2.5 if yr == current_year else 1.2
        dash = None if yr == current_year else "dot"
        fig.add_trace(go.Scatter(
            x=sub["day_of_year"], y=rev.cumsum(),
            mode="lines", name=str(yr),
            line=dict(width=width, dash=dash),
        ))

    return _apply_layout(
        fig, title=f"Cumulative Revenue YoY",
        xaxis_title="Day of Year", yaxis_title=f"Cumulative {REVENUE_LABEL}",
    )


# ═══════════════════════════════════════════════════════════════
#  BREAKDOWN CHARTS
# ═══════════════════════════════════════════════════════════════

def channel_revenue_chart(channel_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of revenue by channel."""
    if channel_df.empty:
        return go.Figure().update_layout(title="No data")

    df = channel_df.sort_values("net_revenue", ascending=True).copy()
    rev = df["net_revenue"] / REVENUE_DIVISOR

    fig = go.Figure(go.Bar(
        x=rev, y=df["KenhBanHang"],
        orientation="h",
        marker_color=COLORS["primary"],
        text=[f"{v:,.0f}" for v in rev],
        textposition="auto",
    ))
    return _apply_layout(fig, title="Revenue by Channel", xaxis_title=REVENUE_LABEL)


def revenue_stacked_by_channel(daily_df: pd.DataFrame, df_full: pd.DataFrame) -> go.Figure:
    """Stacked area of monthly revenue by channel."""
    if df_full.empty:
        return go.Figure().update_layout(title="No data")

    grp = df_full.groupby(["YearMonth", "KenhBanHang"])["ThanhTienSauVAT"].sum().reset_index()
    grp["rev_m"] = grp["ThanhTienSauVAT"] / REVENUE_DIVISOR

    fig = px.area(
        grp, x="YearMonth", y="rev_m", color="KenhBanHang",
        title="Monthly Revenue by Channel",
        labels={"rev_m": REVENUE_LABEL, "YearMonth": "Month", "KenhBanHang": "Channel"},
    )
    return _apply_layout(fig, title="Monthly Revenue by Channel")


def dimension_pie_chart(dim_df: pd.DataFrame, dim_col: str, title: str = "") -> go.Figure:
    """Pie/donut chart for a dimension breakdown."""
    if dim_df.empty:
        return go.Figure().update_layout(title="No data")

    fig = go.Figure(go.Pie(
        labels=dim_df[dim_col],
        values=dim_df["net_revenue"],
        hole=0.4,
        textinfo="label+percent",
    ))
    return _apply_layout(fig, title=title or f"Revenue by {dim_col}", height=400)


# ═══════════════════════════════════════════════════════════════
#  WATERFALL / CONTRIBUTION
# ═══════════════════════════════════════════════════════════════

def waterfall_chart(top_bottom_df: pd.DataFrame, dim: str, title: str = "Revenue Delta Waterfall") -> go.Figure:
    """Waterfall chart showing top gainers and losers."""
    if top_bottom_df.empty:
        return go.Figure().update_layout(title="No data")

    df = top_bottom_df.copy()
    # Sort by delta (positive to negative)
    df = df.sort_values("delta_revenue", ascending=False)
    df["delta_m"] = df["delta_revenue"] / REVENUE_DIVISOR
    labels = df[dim].astype(str).str[:25]

    colors = ["green" if v >= 0 else "red" for v in df["delta_m"]]

    fig = go.Figure(go.Bar(
        x=labels, y=df["delta_m"],
        marker_color=colors,
        text=[f"{v:+,.0f}" for v in df["delta_m"]],
        textposition="outside",
    ))
    return _apply_layout(fig, title=title, yaxis_title=f"Delta ({REVENUE_LABEL})")


# ═══════════════════════════════════════════════════════════════
#  MONTHLY COMPARISON TABLE-CHART
# ═══════════════════════════════════════════════════════════════

def monthly_heatmap(monthly_df: pd.DataFrame) -> go.Figure:
    """Heatmap: Month × Year with net revenue."""
    if monthly_df.empty:
        return go.Figure().update_layout(title="No data")

    df = monthly_df.copy()
    df["rev_m"] = df["net_revenue"] / REVENUE_DIVISOR
    pivot = df.pivot_table(index="Month", columns="Year", values="rev_m", aggfunc="sum")
    pivot = pivot.reindex(range(1, 13))

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Safely handle NaN for text display
    text_matrix = np.where(
        pd.isna(pivot.values),
        "",
        np.vectorize(lambda x: f"{x:,.0f}")(np.nan_to_num(pivot.values, nan=0.0))
    )

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=month_labels[:len(pivot)],
        colorscale="Blues",
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(size=11),
        hovertemplate="Tháng %{y} / %{x}<br>Revenue: %{text} M<extra></extra>",
    ))
    return _apply_layout(fig, title="Monthly Revenue Heatmap (Triệu VND)", height=420)


# ═══════════════════════════════════════════════════════════════
#  MONTHLY GROWTH BAR CHART
# ═══════════════════════════════════════════════════════════════

def monthly_growth_chart(growth_df: pd.DataFrame) -> go.Figure:
    """Bar chart of monthly YoY growth rates."""
    if growth_df.empty or "YoY %" not in growth_df.columns:
        return go.Figure().update_layout(title="No data")

    df = growth_df.dropna(subset=["YoY %"]).copy()
    if df.empty:
        return go.Figure().update_layout(title="No data")

    colors = [COLORS["green"] if v >= 0 else COLORS["red"] for v in df["YoY %"]]

    fig = go.Figure(go.Bar(
        x=df["Tháng"],
        y=df["YoY %"],
        marker_color=colors,
        text=[f"{v:+.1%}" for v in df["YoY %"]],
        textposition="outside",
        textfont=dict(size=10),
        hovertemplate="Tháng %{x}<br>YoY: %{text}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["gray"], line_width=1)
    return _apply_layout(fig, title="Tăng trưởng Doanh thu YoY theo tháng", yaxis_title="YoY %",
                         yaxis_tickformat=".0%")


# ═══════════════════════════════════════════════════════════════
#  ALERT SEVERITY BADGE (HTML helper)
# ═══════════════════════════════════════════════════════════════

def severity_badge(sev: str) -> str:
    """Return HTML badge for severity."""
    color = SEVERITY_COLORS.get(sev, "#999")
    return (
        f'<span style="background:{color};color:white;padding:2px 10px;'
        f'border-radius:4px;font-weight:bold;font-size:0.8rem">{sev}</span>'
    )


def kpi_summary_bar(period_kpis: dict) -> go.Figure:
    """Horizontal grouped bar comparing periods' net revenue."""
    periods = ["Today", "WTD", "MTD", "QTD", "YTD"]
    cur_vals = [period_kpis.get(p, {}).get("net_revenue_m", 0) for p in periods]
    ly_vals = [period_kpis.get(p, {}).get("ly_kpi", {}).get("net_revenue_m", 0) for p in periods]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=periods, x=cur_vals, orientation="h",
        name="Kì này", marker_color=COLORS["primary"],
        text=[f"{v:,.0f}" for v in cur_vals], textposition="auto",
    ))
    fig.add_trace(go.Bar(
        y=periods, x=ly_vals, orientation="h",
        name="Cùng kì năm trước", marker_color=COLORS["light_blue"],
        text=[f"{v:,.0f}" for v in ly_vals], textposition="auto",
    ))
    return _apply_layout(fig, title="Net Revenue: Kì này vs Cùng kì (Triệu VND)",
                         barmode="group", height=320)
