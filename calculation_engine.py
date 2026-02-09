# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════╗
║   CALCULATION ENGINE - Demand Planning & Analytics                  ║
║   Module xử lý tính toán, phân tích và dự báo ARIMA                ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import calendar
import warnings
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def format_vnd(n: Optional[float]) -> str:
    """
    Format a number to Vietnamese Dong (VND) readable string.
    
    Args:
        n: Number to format
        
    Returns:
        Formatted string with Tỷ (billions), Tr (millions), or exact value
        
    Examples:
        >>> format_vnd(1500000000)
        '1.5 Tỷ'
        >>> format_vnd(25000000)
        '25 Tr'
    """
    if n is None or pd.isna(n):
        return "-"
    if abs(n) >= 1e9:
        return f"{n/1e9:,.1f} Tỷ"
    if abs(n) >= 1e6:
        return f"{n/1e6:,.0f} Tr"
    return f"{n:,.0f}"


def calculate_growth_percentage(current: float, previous: float) -> Optional[float]:
    """
    Calculate growth percentage between two values.
    
    Args:
        current: Current period value
        previous: Previous period value
        
    Returns:
        Growth percentage rounded to 2 decimals, or None if previous is 0
        
    Examples:
        >>> calculate_growth_percentage(120, 100)
        20.0
        >>> calculate_growth_percentage(80, 100)
        -20.0
    """
    if previous and previous != 0:
        return round((current - previous) / previous * 100, 2)
    return None


# ═══════════════════════════════════════════════════════════════
# ARIMA FORECASTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def find_best_arima_order(
    time_series: pd.Series,
    p_range: Tuple[int, int] = (0, 4),
    d_range: Tuple[int, int] = (0, 3),
    q_range: Tuple[int, int] = (0, 4)
) -> Tuple[Tuple[int, int, int], float]:
    """
    Find the best ARIMA order using AIC criterion.
    
    Args:
        time_series: Time series data to fit
        p_range: Range for AR parameter (min, max)
        d_range: Range for differencing parameter (min, max)
        q_range: Range for MA parameter (min, max)
        
    Returns:
        Tuple of (best_order, best_aic)
    """
    best_aic = np.inf
    best_order = (1, 1, 1)
    
    for p in range(p_range[0], p_range[1]):
        for d in range(d_range[0], d_range[1]):
            for q in range(q_range[0], q_range[1]):
                try:
                    model = ARIMA(time_series, order=(p, d, q)).fit()
                    if model.aic < best_aic:
                        best_aic = model.aic
                        best_order = (p, d, q)
                except:
                    pass
    
    return best_order, best_aic


def forecast_total_revenue(
    df: pd.DataFrame,
    periods: int = 6
) -> Dict[str, Any]:
    """
    Run ARIMA forecasting on total revenue time series.
    
    Args:
        df: DataFrame with 'NgayHoaDon' and 'ThanhTienSauVAT' columns
        periods: Number of periods to forecast
        
    Returns:
        Dictionary containing historical data, forecast, confidence intervals,
        and model statistics
    """
    # Prepare time series
    df = df.copy()
    df['NgayHoaDon'] = pd.to_datetime(df['NgayHoaDon'])
    ts = df.groupby(df['NgayHoaDon'].dt.to_period('M'))['ThanhTienSauVAT'].sum()
    ts.index = ts.index.to_timestamp()
    ts = ts.sort_index().asfreq('MS', fill_value=0)
    
    # ADF test for stationarity
    adf_result = adfuller(ts.dropna())
    
    # Find best ARIMA order
    best_order, best_aic = find_best_arima_order(ts)
    
    # Fit model and forecast
    model = ARIMA(ts, order=best_order).fit()
    forecast = model.get_forecast(steps=periods)
    confidence_interval = forecast.conf_int()
    
    return {
        'hist_dates': [d.strftime('%m/%Y') for d in ts.index],
        'hist_values': [float(v) for v in ts.values],
        'fc_dates': [d.strftime('%m/%Y') for d in forecast.predicted_mean.index],
        'fc_values': [float(v) for v in forecast.predicted_mean.values],
        'fc_lower': [float(v) for v in confidence_interval.iloc[:, 0].values],
        'fc_upper': [float(v) for v in confidence_interval.iloc[:, 1].values],
        'order': best_order,
        'aic': round(best_aic, 2),
        'adf_stat': round(float(adf_result[0]), 4),
        'adf_pvalue': round(float(adf_result[1]), 4),
    }


def forecast_products_revenue(
    df: pd.DataFrame,
    top_n: int = 10,
    periods: int = 6
) -> List[Dict[str, Any]]:
    """
    Run ARIMA forecasting on top N products by revenue.
    
    Args:
        df: DataFrame with product and revenue data
        top_n: Number of top products to analyze
        periods: Number of periods to forecast
        
    Returns:
        List of dictionaries containing forecast results for each product
    """
    df = df.copy()
    df['NgayHoaDon'] = pd.to_datetime(df['NgayHoaDon'])
    
    # Get top N products by revenue
    top_products = df.groupby('TenSanPham')['ThanhTienSauVAT'].sum().nlargest(top_n).index.tolist()
    
    results = []
    for product_name in top_products:
        try:
            # Filter data for this product
            product_df = df[df['TenSanPham'] == product_name]
            ts = product_df.groupby(product_df['NgayHoaDon'].dt.to_period('M'))['ThanhTienSauVAT'].sum()
            ts.index = ts.index.to_timestamp()
            ts = ts.sort_index().asfreq('MS', fill_value=0)
            
            # Need at least 12 months of data
            if len(ts) < 12:
                continue
            
            # Find best order with smaller search space for products
            best_order, best_aic = find_best_arima_order(
                ts,
                p_range=(0, 3),
                d_range=(0, 2),
                q_range=(0, 3)
            )
            
            # Fit and forecast
            model = ARIMA(ts, order=best_order).fit()
            forecast = model.get_forecast(steps=periods)
            confidence_interval = forecast.conf_int()
            
            results.append({
                'name': product_name,
                'order': best_order,
                'aic': round(best_aic, 2),
                'hist_dates': [d.strftime('%m/%Y') for d in ts.index],
                'hist_values': [float(v) for v in ts.values],
                'fc_dates': [d.strftime('%m/%Y') for d in forecast.predicted_mean.index],
                'fc_values': [float(v) for v in forecast.predicted_mean.values],
                'fc_lower': [float(v) for v in confidence_interval.iloc[:, 0].values],
                'fc_upper': [float(v) for v in confidence_interval.iloc[:, 1].values],
            })
        except:
            pass
    
    return results


# ═══════════════════════════════════════════════════════════════
# CURRENT MONTH ANALYSIS & FORECASTING
# ═══════════════════════════════════════════════════════════════

def analyze_current_month(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive analysis and forecast for the current month.
    
    Combines multiple forecasting methods:
    1. Historical weighted average from same month in previous years
    2. Recent trend (7-day moving average)
    3. ARIMA daily forecast
    
    Args:
        df: DataFrame with transaction data
        
    Returns:
        Dictionary containing current month statistics, daily actuals,
        daily forecasts, and top products/customers/channels
    """
    df = df.copy()
    df['NgayHoaDon'] = pd.to_datetime(df['NgayHoaDon'])
    df['Nam'] = df['NgayHoaDon'].dt.year
    df['Thang_Num'] = df['NgayHoaDon'].dt.month
    
    # Current period info
    today = datetime.now()
    cur_year, cur_month = today.year, today.month
    days_in_month = calendar.monthrange(cur_year, cur_month)[1]
    
    # Filter data for current, previous, and year-over-year comparison
    df_cur = df[(df['Nam'] == cur_year) & (df['Thang_Num'] == cur_month)].copy()
    
    prev_month = 12 if cur_month == 1 else cur_month - 1
    prev_year = cur_year - 1 if cur_month == 1 else cur_year
    df_prev = df[(df['Nam'] == prev_year) & (df['Thang_Num'] == prev_month)]
    df_yoy = df[(df['Nam'] == cur_year - 1) & (df['Thang_Num'] == cur_month)]
    
    # Helper functions for aggregation
    def sum_revenue(data): 
        return float(data['ThanhTienSauVAT'].sum()) if not data.empty else 0
    
    def sum_quantity(data): 
        return int(data['SoLuong'].sum()) if not data.empty else 0
    
    def count_invoices(data): 
        return int(data['SoHoaDon'].nunique()) if not data.empty and 'SoHoaDon' in data.columns else 0
    
    def count_customers(data): 
        return int(data['TenKhachHang'].nunique()) if not data.empty and 'TenKhachHang' in data.columns else 0
    
    # Calculate summary metrics
    rev_cur = sum_revenue(df_cur)
    rev_prev = sum_revenue(df_prev)
    rev_yoy = sum_revenue(df_yoy)
    qty_cur = sum_quantity(df_cur)
    qty_prev = sum_quantity(df_prev)
    
    # Get daily actuals for current month
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
    
    # === DAILY FORECAST COMPUTATION ===
    daily_forecast = _compute_daily_forecast(
        df, df_cur, daily_agg, forecast_days, cur_year, cur_month
    )
    
    # Build full daily arrays for visualization
    days_labels = list(range(1, days_in_month + 1))
    actual_values, forecast_values, forecast_lower, forecast_upper = [], [], [], []
    cumulative = []
    running_total = 0
    
    actual_map = {
        int(row['Ngay']): float(row['revenue']) 
        for _, row in daily_agg.iterrows()
    } if not daily_agg.empty else {}
    
    for day in days_labels:
        if day in actual_map:
            actual_values.append(actual_map[day])
            forecast_values.append(None)
            forecast_lower.append(None)
            forecast_upper.append(None)
            running_total += actual_map[day]
        elif day in daily_forecast:
            actual_values.append(None)
            forecast_values.append(daily_forecast[day]['value'])
            forecast_lower.append(daily_forecast[day]['lower'])
            forecast_upper.append(daily_forecast[day]['upper'])
            running_total += daily_forecast[day]['value']
        else:
            actual_values.append(None)
            forecast_values.append(None)
            forecast_lower.append(None)
            forecast_upper.append(None)
        cumulative.append(running_total)
    
    forecast_total = sum(v['value'] for v in daily_forecast.values())
    projected_month = rev_cur + forecast_total
    
    # Top products, customers, and channels analysis
    top_products = _analyze_top_products(df_cur)
    top_customers = _analyze_top_customers(df_cur)
    month_channels = _analyze_channels(df_cur)
    
    return {
        'month_label': f"{cur_month:02d}/{cur_year}",
        'days_in_month': days_in_month,
        'last_actual_day': last_actual_day,
        'forecast_days_count': len(forecast_days),
        'rev_cur': rev_cur,
        'rev_prev': rev_prev,
        'rev_yoy': rev_yoy,
        'qty_cur': qty_cur,
        'qty_prev': qty_prev,
        'growth_prev': calculate_growth_percentage(rev_cur, rev_prev),
        'growth_yoy': calculate_growth_percentage(rev_cur, rev_yoy),
        'invoices': count_invoices(df_cur),
        'customers': count_customers(df_cur),
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
        'top_products': top_products,
        'top_customers': top_customers,
        'channels': month_channels,
    }


def _compute_daily_forecast(
    df: pd.DataFrame,
    df_cur: pd.DataFrame,
    daily_agg: pd.DataFrame,
    forecast_days: List[int],
    cur_year: int,
    cur_month: int
) -> Dict[int, Dict[str, float]]:
    """
    Compute daily forecast using ensemble of methods.
    
    Internal helper function that combines:
    - Historical weighted average
    - Recent trend
    - ARIMA forecast
    """
    daily_forecast = {}
    
    if not forecast_days or daily_agg.empty:
        return daily_forecast
    
    # 1. Historical weighted average from previous years
    hist_daily = {}
    df_hist = df[(df['Thang_Num'] == cur_month) & (df['Nam'] < cur_year)].copy()
    if not df_hist.empty:
        df_hist['Ngay'] = df_hist['NgayHoaDon'].dt.day
        past_years = sorted(df_hist['Nam'].unique())
        hist_by_day = df_hist.groupby(['Nam', 'Ngay'])['ThanhTienSauVAT'].sum().reset_index()
        
        for day in forecast_days:
            day_data = hist_by_day[hist_by_day['Ngay'] == day]
            if not day_data.empty:
                # Weight more recent years higher
                weights = [1 + (year - past_years[0]) for year in day_data['Nam']]
                hist_daily[day] = float(np.average(day_data['ThanhTienSauVAT'], weights=weights))
    
    # 2. Recent trend (7-day moving average)
    recent_avg = (
        float(daily_agg['revenue'].tail(7).mean()) 
        if len(daily_agg) >= 3 
        else float(daily_agg['revenue'].mean())
    )
    
    # 3. ARIMA daily forecast
    arima_fc = {}
    if len(daily_agg) >= 7:
        try:
            ts = daily_agg.set_index('Ngay')['revenue']
            ts.index = pd.RangeIndex(start=1, stop=len(ts) + 1)
            
            best_order, _ = find_best_arima_order(
                ts,
                p_range=(0, 3),
                d_range=(0, 2),
                q_range=(0, 3)
            )
            
            model = ARIMA(ts, order=best_order).fit()
            forecast = model.get_forecast(steps=len(forecast_days))
            forecast_ci = forecast.conf_int()
            
            for i, day in enumerate(forecast_days):
                arima_fc[day] = {
                    'mean': max(float(forecast.predicted_mean.iloc[i]), 0),
                    'lower': max(float(forecast_ci.iloc[i, 0]), 0),
                    'upper': float(forecast_ci.iloc[i, 1])
                }
        except:
            pass
    
    # 4. Ensemble: Combine all methods with weights
    for day in forecast_days:
        estimates = []
        weights = []
        
        # Historical average (weight: 0.3)
        if day in hist_daily and hist_daily[day] > 0:
            estimates.append(hist_daily[day])
            weights.append(0.3)
        
        # Recent trend (weight: 0.3)
        estimates.append(recent_avg)
        weights.append(0.3)
        
        # ARIMA (weight: 0.4)
        if day in arima_fc:
            estimates.append(arima_fc[day]['mean'])
            weights.append(0.4)
        
        # Weighted average
        final_value = float(np.average(estimates, weights=weights[:len(estimates)]))
        
        # Confidence bounds
        lower_bound = arima_fc[day]['lower'] if day in arima_fc else final_value * 0.75
        upper_bound = arima_fc[day]['upper'] if day in arima_fc else final_value * 1.25
        
        daily_forecast[day] = {
            'value': round(final_value),
            'lower': round(max(lower_bound, 0)),
            'upper': round(upper_bound)
        }
    
    return daily_forecast


def _analyze_top_products(df_cur: pd.DataFrame) -> List[Dict[str, Any]]:
    """Analyze top 10 products in current month."""
    if df_cur.empty:
        return []
    
    product_agg = df_cur.groupby('TenSanPham').agg(
        revenue=('ThanhTienSauVAT', 'sum'),
        quantity=('SoLuong', 'sum'),
        invoices=('SoHoaDon', 'nunique')
    ).reset_index().sort_values('revenue', ascending=False).head(10)
    
    total_revenue = product_agg['revenue'].sum()
    
    top_products = []
    for _, row in product_agg.iterrows():
        top_products.append({
            'name': row['TenSanPham'],
            'revenue': float(row['revenue']),
            'quantity': int(row['quantity']),
            'invoices': int(row['invoices']),
            'pct': round(float(row['revenue']) / total_revenue * 100, 1) if total_revenue > 0 else 0
        })
    
    return top_products


def _analyze_top_customers(df_cur: pd.DataFrame) -> List[Dict[str, Any]]:
    """Analyze top 10 customers in current month."""
    if df_cur.empty or 'TenKhachHang' not in df_cur.columns:
        return []
    
    customer_agg = df_cur.groupby('TenKhachHang').agg(
        revenue=('ThanhTienSauVAT', 'sum'),
        quantity=('SoLuong', 'sum'),
        invoices=('SoHoaDon', 'nunique')
    ).reset_index().sort_values('revenue', ascending=False).head(10)
    
    top_customers = []
    for _, row in customer_agg.iterrows():
        top_customers.append({
            'name': row['TenKhachHang'],
            'revenue': float(row['revenue']),
            'quantity': int(row['quantity']),
            'invoices': int(row['invoices'])
        })
    
    return top_customers


def _analyze_channels(df_cur: pd.DataFrame) -> List[Dict[str, Any]]:
    """Analyze sales channels in current month."""
    if df_cur.empty or 'KenhBanHang' not in df_cur.columns:
        return []
    
    channel_agg = df_cur.groupby('KenhBanHang')['ThanhTienSauVAT'].sum().reset_index()
    channel_agg = channel_agg.sort_values('ThanhTienSauVAT', ascending=False)
    
    total_revenue = channel_agg['ThanhTienSauVAT'].sum()
    
    channels = []
    for _, row in channel_agg.iterrows():
        channels.append({
            'name': row['KenhBanHang'],
            'revenue': float(row['ThanhTienSauVAT']),
            'pct': round(float(row['ThanhTienSauVAT']) / total_revenue * 100, 1) if total_revenue > 0 else 0
        })
    
    return channels
