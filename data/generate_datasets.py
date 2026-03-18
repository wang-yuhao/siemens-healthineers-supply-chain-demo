"""
generate_datasets.py
--------------------
Generates simulated medical electronics supply chain datasets
for the Siemens Healthineers Demand Forecasting Demo.

Datasets produced:
  1. medical_supply_data.csv       - Main daily demand & inventory per SKU
  2. supplier_leadtime_data.csv    - Supplier lead time history
  3. model_comparison.csv          - Baseline vs ML model performance metrics
  4. sku_master.csv                - Product master / SKU reference table

Usage:
  python data/generate_datasets.py
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
OUT_DIR = os.path.join(os.path.dirname(__file__))

# ─────────────────────────────────────────────
# 1. SKU MASTER TABLE
# ─────────────────────────────────────────────
skus = [
    {"sku": "ECG-Monitor-X500",    "family": "Cardiology",     "criticality": "High",   "lifecycle": "Growth",   "unit_price": 3200,  "lead_time_avg": 14},
    {"sku": "Ventilator-Pro-V3",   "family": "Respiratory",    "criticality": "High",   "lifecycle": "Mature",   "unit_price": 8500,  "lead_time_avg": 21},
    {"sku": "Surgical-Pump-SP2",   "family": "Surgery",        "criticality": "Medium", "lifecycle": "Mature",   "unit_price": 1500,  "lead_time_avg": 7},
    {"sku": "Infusion-IV100",      "family": "Infusion",       "criticality": "Medium", "lifecycle": "Decline",  "unit_price": 450,   "lead_time_avg": 7},
    {"sku": "MRI-Coil-HC30",       "family": "Imaging",        "criticality": "High",   "lifecycle": "Launch",   "unit_price": 12000, "lead_time_avg": 28},
    {"sku": "Ultrasound-Probe-U7", "family": "Imaging",        "criticality": "High",   "lifecycle": "Growth",   "unit_price": 5500,  "lead_time_avg": 21},
]
sku_df = pd.DataFrame(skus)
sku_df.to_csv(os.path.join(OUT_DIR, "sku_master.csv"), index=False)
print(f"[1/4] sku_master.csv saved ({len(sku_df)} rows)")

# ─────────────────────────────────────────────
# 2. MAIN DAILY DEMAND & INVENTORY DATASET
# ─────────────────────────────────────────────
start_date = datetime(2023, 1, 1)
end_date   = datetime(2026, 3, 17)
dates      = pd.date_range(start_date, end_date, freq="D")
n_days     = len(dates)

rows = []
for s in skus:
    sku        = s["sku"]
    price      = s["unit_price"]
    lt_avg     = s["lead_time_avg"]
    lc         = s["lifecycle"]

    # Trend component
    if lc == "Launch":  trend = np.linspace(5,  80,  n_days)
    elif lc == "Growth": trend = np.linspace(60, 160, n_days)
    elif lc == "Mature": trend = np.full(n_days, 120) + np.random.normal(0, 5, n_days)
    else:               trend = np.linspace(100, 40, n_days)   # Decline

    # Seasonality
    weekly    = 15 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    annual    = 25 * np.sin(2 * np.pi * np.arange(n_days) / 365 - np.pi / 4)
    noise     = np.random.normal(0, 8, n_days)
    # Promo spikes (random 8% of days)
    promo     = np.random.choice([0, 1], size=n_days, p=[0.92, 0.08])
    demand    = np.maximum(trend + weekly + annual + noise + promo * 30, 1).astype(int)

    # Stock level (simple simulation)
    stock     = []
    current_stock = int(np.random.uniform(200, 600))
    for i in range(n_days):
        current_stock = max(current_stock - demand[i] + np.random.randint(0, 50), 0)
        stock.append(current_stock)

    for i, d in enumerate(dates):
        rows.append({
            "date":                   d.strftime("%Y-%m-%d"),
            "sku":                    sku,
            "family":                 s["family"],
            "quantity_sold":          demand[i],
            "stock_level":            stock[i],
            "supplier_leadtime_days": lt_avg + np.random.randint(-2, 5),
            "price_per_unit":         price,
            "promo_flag":             int(promo[i]),
            "backorder_qty":          max(0, demand[i] - stock[i]),
        })

demand_df = pd.DataFrame(rows)
demand_df.to_csv(os.path.join(OUT_DIR, "medical_supply_data.csv"), index=False)
print(f"[2/4] medical_supply_data.csv saved ({len(demand_df)} rows)")

# ─────────────────────────────────────────────
# 3. SUPPLIER LEAD TIME HISTORY
# ─────────────────────────────────────────────
suppliers = [
    {"supplier": "MedParts GmbH",     "sku": "ECG-Monitor-X500",    "country": "Germany"},
    {"supplier": "HealthComp Inc.",   "sku": "Ventilator-Pro-V3",   "country": "USA"},
    {"supplier": "SurgTech Asia",     "sku": "Surgical-Pump-SP2",   "country": "Japan"},
    {"supplier": "InfuSupply SL",     "sku": "Infusion-IV100",      "country": "Spain"},
    {"supplier": "ImagingParts KG",   "sku": "MRI-Coil-HC30",       "country": "Germany"},
    {"supplier": "ProbeWorld Ltd.",   "sku": "Ultrasound-Probe-U7", "country": "UK"},
]
lt_rows = []
po_dates = pd.date_range("2023-01-01", "2026-03-01", freq="2W")
for sup in suppliers:
    base_lt = [s["lead_time_avg"] for s in skus if s["sku"] == sup["sku"]][0]
    for po_date in po_dates:
        actual_lt = max(1, int(np.random.normal(base_lt, base_lt * 0.2)))
        lt_rows.append({
            "po_date":            po_date.strftime("%Y-%m-%d"),
            "supplier":           sup["supplier"],
            "sku":                sup["sku"],
            "country":            sup["country"],
            "promised_lead_days": base_lt,
            "actual_lead_days":   actual_lt,
            "delay_days":         actual_lt - base_lt,
            "on_time":            int(actual_lt <= base_lt),
        })
lt_df = pd.DataFrame(lt_rows)
lt_df.to_csv(os.path.join(OUT_DIR, "supplier_leadtime_data.csv"), index=False)
print(f"[3/4] supplier_leadtime_data.csv saved ({len(lt_df)} rows)")

# ─────────────────────────────────────────────
# 4. MODEL COMPARISON (Baseline vs ML)
# ─────────────────────────────────────────────
comparison_rows = []
for s in skus:
    comparison_rows.append({
        "sku":           s["sku"],
        "family":        s["family"],
        "method":        "Manual / Moving Average",
        "mape_pct":      round(np.random.uniform(28, 42), 1),
        "mae":           round(np.random.uniform(30, 55), 1),
        "rmse":          round(np.random.uniform(38, 70), 1),
        "bias_pct":      round(np.random.uniform(-15, 15), 1),
        "forecast_horizon_days": 30,
    })
    comparison_rows.append({
        "sku":           s["sku"],
        "family":        s["family"],
        "method":        "ML Ensemble (Prophet + XGBoost)",
        "mape_pct":      round(np.random.uniform(8, 16), 1),
        "mae":           round(np.random.uniform(10, 22), 1),
        "rmse":          round(np.random.uniform(13, 28), 1),
        "bias_pct":      round(np.random.uniform(-4, 4), 1),
        "forecast_horizon_days": 30,
    })
comp_df = pd.DataFrame(comparison_rows)
comp_df.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"), index=False)
print(f"[4/4] model_comparison.csv saved ({len(comp_df)} rows)")

print("\nAll datasets generated successfully!")
print(f"Output directory: {os.path.abspath(OUT_DIR)}")
