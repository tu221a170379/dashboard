"""
Alerts module – Early-warning system for CEO Dashboard.

Generates a list of alerts with severity (RED / YELLOW / GREEN),
metric name, time period, affected segment, % change, and root-cause driver.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import pandas as pd
import numpy as np

from src.config import (
    BASELINE_WEEKS,
    CONCENTRATION_THRESHOLD,
    CONCENTRATION_TOP_N,
    RED_THRESHOLD,
    ROLLING_DAYS,
    ROLLING_WEEKS,
    YELLOW_THRESHOLD,
)
from src.metrics import (
    _pct_change,
    daily_revenue,
    net_revenue,
    paid_units,
    promo_units,
    total_tons,
    discount_rate,
    revenue_by_dimension,
    filter_period,
)


# ═══════════════════════════════════════════════════════════════
#  Alert data structure
# ═══════════════════════════════════════════════════════════════

@dataclass
class Alert:
    severity: str          # RED, YELLOW, GREEN
    category: str          # e.g. "Revenue", "Volume", "Channel", "Customer", "Promo", "DataQuality"
    metric: str            # e.g. "Net Revenue WoW"
    period: str            # e.g. "Week 5 vs Week 4"
    segment: str = ""      # e.g. "MT channel" or "All"
    pct_change: Optional[float] = None
    detail: str = ""
    drivers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pct_change_str"] = f"{self.pct_change:+.1%}" if self.pct_change is not None else "N/A"
        return d


def _severity(pct: Optional[float], yellow: float = YELLOW_THRESHOLD, red: float = RED_THRESHOLD) -> str:
    """Map a drop % to severity. pct is negative for drops."""
    if pct is None:
        return "GREEN"
    drop = -pct  # convert negative % to positive drop magnitude
    if drop >= red:
        return "RED"
    elif drop >= yellow:
        return "YELLOW"
    return "GREEN"


# ═══════════════════════════════════════════════════════════════
#  A. Revenue drop alerts
# ═══════════════════════════════════════════════════════════════

def _daily_revenue_alerts(df: pd.DataFrame, ref_date: pd.Timestamp) -> list[Alert]:
    """Compare today's revenue vs rolling 28-day average and D-7."""
    alerts: list[Alert] = []
    dr = daily_revenue(df)
    if dr.empty:
        return alerts

    dr = dr.set_index("Date").sort_index()

    # Today's revenue
    today_val = dr.loc[dr.index == pd.Timestamp(ref_date), "net_revenue"]
    if today_val.empty:
        return alerts
    today_rev = float(today_val.iloc[0])

    # Rolling 28-day average (excluding today)
    window = dr.loc[dr.index < pd.Timestamp(ref_date)].tail(ROLLING_DAYS)
    if len(window) >= 7:
        avg_28 = window["net_revenue"].mean()
        pct = _pct_change(today_rev, avg_28)
        if pct is not None and pct < 0:
            sev = _severity(pct)
            if sev != "GREEN":
                alerts.append(Alert(
                    severity=sev,
                    category="Revenue",
                    metric="Daily Revenue vs Rolling 28D Avg",
                    period=f"{ref_date.strftime('%Y-%m-%d')}",
                    pct_change=pct,
                    detail=f"Today: {today_rev/1e6:,.1f}M vs Avg: {avg_28/1e6:,.1f}M",
                ))

    # D-7 comparison
    d7 = pd.Timestamp(ref_date) - pd.Timedelta(days=7)
    d7_val = dr.loc[dr.index == d7, "net_revenue"]
    if not d7_val.empty:
        pct_d7 = _pct_change(today_rev, float(d7_val.iloc[0]))
        if pct_d7 is not None and pct_d7 < 0:
            sev = _severity(pct_d7)
            if sev != "GREEN":
                alerts.append(Alert(
                    severity=sev,
                    category="Revenue",
                    metric="Daily Revenue D vs D-7",
                    period=f"{ref_date.strftime('%Y-%m-%d')} vs {d7.strftime('%Y-%m-%d')}",
                    pct_change=pct_d7,
                ))

    return alerts


def _weekly_revenue_alerts(df: pd.DataFrame, ref_date: pd.Timestamp) -> list[Alert]:
    """WoW and YoY weekly comparison."""
    alerts: list[Alert] = []

    # Current week (Mon–ref_date)
    cur_start = ref_date - pd.Timedelta(days=ref_date.weekday())
    cur_end = ref_date
    cur_df = filter_period(df, cur_start, cur_end)
    cur_rev = net_revenue(cur_df)
    cur_units = paid_units(cur_df)

    # Previous week
    prev_start = cur_start - pd.Timedelta(days=7)
    prev_end = cur_start - pd.Timedelta(days=1)
    prev_df = filter_period(df, prev_start, prev_end)
    prev_rev = net_revenue(prev_df)

    pct_wow = _pct_change(cur_rev, prev_rev)
    if pct_wow is not None and pct_wow < 0:
        sev = _severity(pct_wow)
        if sev != "GREEN":
            alerts.append(Alert(
                severity=sev,
                category="Revenue",
                metric="Weekly Revenue WoW",
                period=f"{cur_start.strftime('%m/%d')}–{cur_end.strftime('%m/%d')} vs prev week",
                pct_change=pct_wow,
                detail=f"Current: {cur_rev/1e6:,.1f}M vs Previous: {prev_rev/1e6:,.1f}M",
            ))

    # YoY same-week comparison
    yoy_start = cur_start - pd.DateOffset(years=1)
    yoy_end = cur_end - pd.DateOffset(years=1)
    yoy_df = filter_period(df, yoy_start, yoy_end)
    yoy_rev = net_revenue(yoy_df)
    pct_yoy = _pct_change(cur_rev, yoy_rev)
    if pct_yoy is not None and pct_yoy < 0:
        sev = _severity(pct_yoy)
        if sev != "GREEN":
            alerts.append(Alert(
                severity=sev,
                category="Revenue",
                metric="Weekly Revenue YoY",
                period=f"Cur week vs same week last year",
                pct_change=pct_yoy,
            ))

    # Volume WoW
    prev_units = paid_units(prev_df)
    pct_vol = _pct_change(cur_units, prev_units)
    if pct_vol is not None and pct_vol < 0:
        sev = _severity(pct_vol)
        if sev != "GREEN":
            alerts.append(Alert(
                severity=sev,
                category="Volume",
                metric="Weekly Paid Units WoW",
                period=f"{cur_start.strftime('%m/%d')}–{cur_end.strftime('%m/%d')}",
                pct_change=pct_vol,
            ))

    return alerts


def _mtd_revenue_alerts(df: pd.DataFrame, ref_date: pd.Timestamp) -> list[Alert]:
    """MTD vs same MTD last year — revenue and volume."""
    alerts: list[Alert] = []

    mtd_start = ref_date.replace(day=1)
    mtd_df = filter_period(df, mtd_start, ref_date)
    mtd_rev = net_revenue(mtd_df)
    mtd_units = paid_units(mtd_df)
    mtd_tons = total_tons(mtd_df)

    # Same MTD last year
    ly_start = mtd_start - pd.DateOffset(years=1)
    ly_end = ref_date - pd.DateOffset(years=1)
    ly_df = filter_period(df, ly_start, ly_end)
    ly_rev = net_revenue(ly_df)
    ly_units = paid_units(ly_df)

    pct = _pct_change(mtd_rev, ly_rev)
    if pct is not None and pct < 0:
        sev = _severity(pct)
        if sev != "GREEN":
            alerts.append(Alert(
                severity=sev,
                category="Revenue",
                metric="MTD Revenue YoY",
                period=f"MTD {ref_date.strftime('%b %Y')} vs {(ref_date - pd.DateOffset(years=1)).strftime('%b %Y')}",
                pct_change=pct,
                detail=f"MTD: {mtd_rev/1e6:,.1f}M vs LY MTD: {ly_rev/1e6:,.1f}M",
            ))

    # MTD Volume YoY
    pct_vol = _pct_change(mtd_units, ly_units)
    if pct_vol is not None and pct_vol < 0:
        sev = _severity(pct_vol)
        if sev != "GREEN":
            alerts.append(Alert(
                severity=sev,
                category="Volume",
                metric="MTD Paid Units YoY",
                period=f"MTD {ref_date.strftime('%b %Y')} vs LY",
                pct_change=pct_vol,
                detail=f"MTD: {mtd_units:,.0f} units vs LY: {ly_units:,.0f} units",
            ))

    return alerts


# ═══════════════════════════════════════════════════════════════
#  C. Channel / Region driver alerts
# ═══════════════════════════════════════════════════════════════

def _driver_decomposition_alerts(
    df: pd.DataFrame, ref_date: pd.Timestamp, dim: str = "KenhBanHang", top_n: int = 5
) -> list[Alert]:
    """Identify top-N segments with largest negative delta contributing to total drop."""
    alerts: list[Alert] = []

    # Current week
    cur_start = ref_date - pd.Timedelta(days=ref_date.weekday())
    cur_df = filter_period(df, cur_start, ref_date)
    prev_start = cur_start - pd.Timedelta(days=7)
    prev_end = cur_start - pd.Timedelta(days=1)
    prev_df = filter_period(df, prev_start, prev_end)

    cur_agg = revenue_by_dimension(cur_df, dim)
    prev_agg = revenue_by_dimension(prev_df, dim)

    if cur_agg.empty or prev_agg.empty:
        return alerts

    if dim not in cur_agg.columns or dim not in prev_agg.columns:
        return alerts

    merged = cur_agg[[dim, "net_revenue"]].rename(columns={"net_revenue": "cur"}).merge(
        prev_agg[[dim, "net_revenue"]].rename(columns={"net_revenue": "prev"}),
        on=dim, how="outer"
    ).fillna(0)
    merged["delta"] = merged["cur"] - merged["prev"]
    merged["pct"] = merged.apply(lambda r: _pct_change(r["cur"], r["prev"]), axis=1)

    # Top drops
    worst = merged.nsmallest(top_n, "delta")
    for _, row in worst.iterrows():
        if row["delta"] < 0 and row["pct"] is not None:
            sev = _severity(row["pct"])
            if sev != "GREEN":
                alerts.append(Alert(
                    severity=sev,
                    category="Driver",
                    metric=f"{dim} Revenue WoW",
                    period=f"Week of {cur_start.strftime('%m/%d')}",
                    segment=str(row[dim]),
                    pct_change=row["pct"],
                    detail=f"Delta: {row['delta']/1e6:,.1f}M",
                ))

    return alerts


# ═══════════════════════════════════════════════════════════════
#  D. Customer / Product concentration
# ═══════════════════════════════════════════════════════════════

def _concentration_alerts(df: pd.DataFrame, ref_date: pd.Timestamp) -> list[Alert]:
    """Alert when top-N customers / products concentrate too much revenue or drop."""
    alerts: list[Alert] = []

    # Use last 4 weeks as current
    cur_end = ref_date
    cur_start = ref_date - pd.Timedelta(weeks=4)
    cur_df = filter_period(df, cur_start, cur_end)
    baseline_start = cur_start - pd.Timedelta(weeks=BASELINE_WEEKS)
    baseline_end = cur_start - pd.Timedelta(days=1)
    baseline_df = filter_period(df, baseline_start, baseline_end)

    for dim, label in [("TenKhachHang", "Customer"), ("TenSanPham", "Product")]:
        cur_agg = revenue_by_dimension(cur_df, dim)
        if cur_agg.empty:
            continue

        total_rev = cur_agg["net_revenue"].sum()
        if total_rev <= 0:
            continue

        top = cur_agg.head(CONCENTRATION_TOP_N)
        top_share = top["net_revenue"].sum() / total_rev

        if top_share > CONCENTRATION_THRESHOLD:
            alerts.append(Alert(
                severity="YELLOW",
                category="Concentration",
                metric=f"Top-{CONCENTRATION_TOP_N} {label} Revenue Share",
                period=f"Last 4 weeks",
                pct_change=top_share,
                detail=f"Top-{CONCENTRATION_TOP_N} = {top_share:.1%} of total revenue",
                drivers=[str(x) for x in top[dim].tolist()],
            ))

        # Check if top entities dropped vs baseline
        bl_agg = revenue_by_dimension(baseline_df, dim)
        if bl_agg.empty:
            continue

        merged = top[[dim, "net_revenue"]].rename(columns={"net_revenue": "cur"}).merge(
            bl_agg[[dim, "net_revenue"]].rename(columns={"net_revenue": "prev"}),
            on=dim, how="left"
        ).fillna(0)
        # Normalize to weekly average
        weeks_cur = max((cur_end - cur_start).days / 7, 1)
        weeks_bl = max((baseline_end - baseline_start).days / 7, 1)
        merged["cur_weekly"] = merged["cur"] / weeks_cur
        merged["prev_weekly"] = merged["prev"] / weeks_bl

        for _, row in merged.iterrows():
            pct = _pct_change(row["cur_weekly"], row["prev_weekly"])
            if pct is not None and pct < -YELLOW_THRESHOLD:
                sev = _severity(pct)
                alerts.append(Alert(
                    severity=sev,
                    category="Concentration",
                    metric=f"Top {label} Revenue Drop",
                    period=f"Last 4W vs Baseline {BASELINE_WEEKS}W",
                    segment=str(row[dim])[:50],
                    pct_change=pct,
                ))

    return alerts


# ═══════════════════════════════════════════════════════════════
#  E. Discount / Promo abnormal
# ═══════════════════════════════════════════════════════════════

def _promo_discount_alerts(df: pd.DataFrame, ref_date: pd.Timestamp) -> list[Alert]:
    """Alert on abnormal discount rate or promo burns."""
    alerts: list[Alert] = []

    # Current week discount rate vs rolling 12-week average
    cur_start = ref_date - pd.Timedelta(days=ref_date.weekday())
    cur_df = filter_period(df, cur_start, ref_date)
    cur_disc = discount_rate(cur_df)

    rolling_start = cur_start - pd.Timedelta(weeks=ROLLING_WEEKS)
    rolling_end = cur_start - pd.Timedelta(days=1)
    rolling_df = filter_period(df, rolling_start, rolling_end)
    rolling_disc = discount_rate(rolling_df)

    if rolling_disc > 0:
        pct_disc_change = _pct_change(cur_disc, rolling_disc)
        if pct_disc_change is not None and pct_disc_change > YELLOW_THRESHOLD:
            # Discount rate INCREASED => bad
            alerts.append(Alert(
                severity="YELLOW" if pct_disc_change < RED_THRESHOLD else "RED",
                category="Discount",
                metric="Discount Rate vs Rolling Avg",
                period=f"This week vs {ROLLING_WEEKS}W avg",
                pct_change=pct_disc_change,
                detail=f"Current: {cur_disc:.1%} vs Avg: {rolling_disc:.1%}",
            ))

    # Promo volume up but revenue not up  ("burn volume")
    cur_promo = promo_units(cur_df)
    cur_rev = net_revenue(cur_df)
    prev_start = cur_start - pd.Timedelta(days=7)
    prev_end = cur_start - pd.Timedelta(days=1)
    prev_df = filter_period(df, prev_start, prev_end)
    prev_promo = promo_units(prev_df)
    prev_rev = net_revenue(prev_df)

    promo_pct = _pct_change(cur_promo, prev_promo)
    rev_pct = _pct_change(cur_rev, prev_rev)

    if (promo_pct is not None and promo_pct > 0.2
            and rev_pct is not None and rev_pct <= 0):
        alerts.append(Alert(
            severity="YELLOW",
            category="Promo",
            metric="Promo Volume Up / Revenue Flat-Down",
            period=f"WoW",
            detail=f"Promo units {promo_pct:+.1%}, Revenue {rev_pct:+.1%}",
        ))

    return alerts


# ═══════════════════════════════════════════════════════════════
#  MASTER ALERT GENERATOR
# ═══════════════════════════════════════════════════════════════

def generate_all_alerts(df: pd.DataFrame, ref_date: Optional[pd.Timestamp] = None) -> list[Alert]:
    """Run all alert checks and return sorted by severity."""
    if ref_date is None:
        ref_date = pd.Timestamp.now().normalize()

    all_alerts: list[Alert] = []
    all_alerts.extend(_daily_revenue_alerts(df, ref_date))
    all_alerts.extend(_weekly_revenue_alerts(df, ref_date))
    all_alerts.extend(_mtd_revenue_alerts(df, ref_date))
    all_alerts.extend(_driver_decomposition_alerts(df, ref_date, "KenhBanHang"))
    all_alerts.extend(_driver_decomposition_alerts(df, ref_date, "Vung"))
    all_alerts.extend(_driver_decomposition_alerts(df, ref_date, "KhuVuc"))
    all_alerts.extend(_concentration_alerts(df, ref_date))
    all_alerts.extend(_promo_discount_alerts(df, ref_date))

    # Sort: RED first, then YELLOW, then GREEN
    severity_order = {"RED": 0, "YELLOW": 1, "GREEN": 2}
    all_alerts.sort(key=lambda a: severity_order.get(a.severity, 3))

    return all_alerts


def alerts_to_dataframe(alerts: list[Alert]) -> pd.DataFrame:
    """Convert alerts to a DataFrame for display."""
    if not alerts:
        return pd.DataFrame(columns=[
            "severity", "category", "metric", "period", "segment",
            "pct_change_str", "detail"
        ])
    return pd.DataFrame([a.to_dict() for a in alerts])


def executive_summary(alerts: list[Alert], df: pd.DataFrame, ref_date: pd.Timestamp) -> list[str]:
    """Generate 5 bullet points of executive summary."""
    bullets: list[str] = []

    # 1) Overall revenue status with YoY indicator
    from src.metrics import compute_period_kpis, _pct_change as pct_fn, filter_period as fp, net_revenue as nr_fn
    kpis = compute_period_kpis(df, ref_date)
    mtd = kpis.get("MTD", {})
    ytd = kpis.get("YTD", {})

    # MTD YoY
    ly_ref = ref_date - pd.DateOffset(years=1)
    ly_mtd_start = ly_ref.replace(day=1)
    ly_mtd_rev = nr_fn(fp(df, ly_mtd_start, ly_ref))
    mtd_yoy = pct_fn(mtd.get("net_revenue", 0), ly_mtd_rev)
    yoy_str = f" ({mtd_yoy:+.1%} YoY)" if mtd_yoy is not None else ""

    bullets.append(
        f"📊 MTD Net Revenue: **{mtd.get('net_revenue_m', 0):,.0f}** {_unit()}{yoy_str} "
        f"| YTD: **{ytd.get('net_revenue_m', 0):,.0f}** {_unit()}"
    )

    # 2) Count of RED / YELLOW alerts
    reds = sum(1 for a in alerts if a.severity == "RED")
    yellows = sum(1 for a in alerts if a.severity == "YELLOW")
    if reds > 0:
        bullets.append(f"🔴 **{reds}** cảnh báo nghiêm trọng cần xem ngay!")
    if yellows > 0:
        bullets.append(f"🟡 **{yellows}** cảnh báo cần theo dõi.")
    if reds == 0 and yellows == 0:
        bullets.append("✅ Không có cảnh báo bất thường — hoạt động ổn định.")

    # 3) Top revenue driver alert
    rev_alerts = [a for a in alerts if a.category == "Revenue" and a.severity in ("RED", "YELLOW")]
    if rev_alerts:
        top = rev_alerts[0]
        if top.pct_change is not None:
            bullets.append(f"📉 {top.metric}: **{top.pct_change:+.1%}** ({top.period})")

    # 4) Top driver decomposition
    driver_alerts = [a for a in alerts if a.category == "Driver" and a.severity in ("RED", "YELLOW")]
    if driver_alerts:
        segs = [f"**{a.segment}** ({a.pct_change:+.1%})" for a in driver_alerts[:3] if a.pct_change]
        if segs:
            bullets.append(f"🔍 Kênh/Vùng giảm mạnh: {', '.join(segs)}")

    # 5) Promo/Discount
    promo_alerts = [a for a in alerts if a.category in ("Discount", "Promo")]
    if promo_alerts:
        bullets.append(f"⚠️ {promo_alerts[0].detail}")

    # Ensure at least 5 bullets
    while len(bullets) < 5:
        bullets.append(f"ℹ️ Dữ liệu cập nhật đến **{ref_date.strftime('%d/%m/%Y')}**.")

    return bullets[:5]


def _unit() -> str:
    from src.config import REVENUE_LABEL
    return REVENUE_LABEL
