"""
=============================================================================
DATA STORM v7.0 — Full EDA + Lakehouse Pipeline + Potential Modeling
=============================================================================
Structure:
  📁 bronze/   → raw ingested files (parquet copies)
  📁 silver/   → cleaned, quarantined records
  📁 gold/     → feature-engineered, model-ready
  📁 plots/    → all EDA visualizations
  📁 output/   → final predictions CSV

Usage:
  pip install pandas numpy matplotlib seaborn scipy scikit-learn requests tqdm
  python datastorm_eda_pipeline.py
=============================================================================
"""

import os, warnings, json, sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from datetime import datetime
import requests
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
RAW_DATA_DIR = "./data"          # ← put your raw CSVs here
BRONZE_DIR   = "./bronze"
SILVER_DIR   = "./silver"
GOLD_DIR     = "./gold"
PLOTS_DIR    = "./plots"
OUTPUT_DIR   = "./output"

for d in [BRONZE_DIR, SILVER_DIR, GOLD_DIR, PLOTS_DIR, OUTPUT_DIR,
          f"{SILVER_DIR}/rejected"]:
    os.makedirs(d, exist_ok=True)

TEAM_NAME = "3 dots"   # ← change this

plt.style.use("seaborn-v0_8-darkgrid")
PALETTE = sns.color_palette("husl", 10)
sns.set_palette(PALETTE)

print("=" * 70)
print("  DATA STORM v7.0  |  Full EDA + Lakehouse Pipeline")
print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
print("\n[BRONZE] Ingesting raw data...")

def bronze_ingest(filename, **kwargs):
    path = os.path.join(RAW_DATA_DIR, filename)
    df = pd.read_csv(path, **kwargs)
    out = os.path.join(BRONZE_DIR, filename.replace(".csv", ".parquet"))
    df.to_parquet(out, index=False)
    print(f"  ✔  {filename}  →  {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df

txn   = bronze_ingest("transactions_history_final.csv")
master = bronze_ingest("outlet_master.csv")
coords = bronze_ingest("outlet_coordinates.csv")
season = bronze_ingest("distributor_seasonality_details.csv")
holidays = bronze_ingest("holiday_list.csv")

print("[BRONZE] Done — all raw files preserved as-is in ./bronze/")


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def check_duplicates(df, keys, dataset_name):
    dupes = df[df.duplicated(subset=keys, keep=False)]
    if len(dupes):
        print(f"  ⚠  [{dataset_name}] {len(dupes)} duplicate rows on {keys}")
    return dupes

def check_nulls(df, mandatory_cols, dataset_name):
    null_mask = df[mandatory_cols].isnull().any(axis=1)
    nulls = df[null_mask]
    if len(nulls):
        print(f"  ⚠  [{dataset_name}] {len(nulls)} rows with nulls in {mandatory_cols}")
    return nulls

def check_referential_integrity(df, fk_col, ref_df, ref_col, dataset_name):
    orphans = df[~df[fk_col].isin(ref_df[ref_col])]
    if len(orphans):
        print(f"  ⚠  [{dataset_name}] {len(orphans)} orphan rows: {fk_col} not in {ref_col}")
    return orphans

def check_value_range(df, col, min_val, max_val, dataset_name):
    out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
    if len(out_of_range):
        print(f"  ⚠  [{dataset_name}] {len(out_of_range)} rows where {col} ∉ [{min_val}, {max_val}]")
    return out_of_range

def check_format(df, col, regex_pattern, dataset_name):
    bad = df[~df[col].astype(str).str.match(regex_pattern, na=False)]
    if len(bad):
        print(f"  ⚠  [{dataset_name}] {len(bad)} rows with bad format in {col}")
    return bad

def quarantine(bad_df, reason, dataset_name):
    if len(bad_df) == 0:
        return
    bad_df = bad_df.copy()
    bad_df["__reject_reason__"] = reason
    bad_df["__source_dataset__"] = dataset_name
    out = os.path.join(SILVER_DIR, "rejected", f"rejected_{dataset_name}.csv")
    mode = "a" if os.path.exists(out) else "w"
    bad_df.to_csv(out, mode=mode, header=(mode == "w"), index=False)


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
print("\n[SILVER] Running data quality checks & cleaning...")

# ── 3a. TRANSACTIONS ──────────────────────────────────────────────────────────
print("\n  → transactions_history_final.csv")

mandatory_txn = ["Outlet_ID","Year","Month","Distributor_ID","SKU_ID",
                 "Volume_Liters","Total_Bill_Value"]
null_txn = check_nulls(txn, mandatory_txn, "transactions")
quarantine(null_txn, "null_mandatory_field", "transactions")

# ── SMARTER DUPLICATE CHECK ──────────────────────────────────────────────────

exact_dupes = txn[txn.duplicated(
    subset=["Outlet_ID","Year","Month","Distributor_ID",
            "SKU_ID","Volume_Liters","Total_Bill_Value"],
    keep="first")]  # keep first, remove exact copies only
quarantine(exact_dupes, "exact_copy_duplicate", "transactions")

order_freq = txn.groupby(
    ["Outlet_ID","Year","Month","SKU_ID"]).size().reset_index(name="order_count")
suspicious_freq = order_freq[order_freq["order_count"] > 5]
if len(suspicious_freq):
    print(f"  ⚠  [transactions] {len(suspicious_freq)} outlet-month-SKU "
          f"combinations with >5 orders — flagged for review")
    suspicious_freq.to_csv(
        f"{SILVER_DIR}/rejected/suspicious_high_frequency.csv", index=False)

reorder_counts = txn.groupby(
    ["Outlet_ID","Year","Month"]).size().reset_index(name="orders_per_month")
avg_reorders = reorder_counts.groupby("Outlet_ID")["orders_per_month"].mean(
    ).reset_index(name="avg_monthly_reorders")

neg_vol = check_value_range(txn, "Volume_Liters", 0.001, 99999, "transactions")
quarantine(neg_vol, "non_positive_volume", "transactions")

neg_bill = check_value_range(txn, "Total_Bill_Value", 0.001, 9_999_999, "transactions")
quarantine(neg_bill, "non_positive_bill_value", "transactions")

bad_year = check_value_range(txn, "Year", 2020, 2025, "transactions")
quarantine(bad_year, "year_out_of_range", "transactions")

bad_month = check_value_range(txn, "Month", 1, 12, "transactions")
quarantine(bad_month, "month_out_of_range", "transactions")

bad_fmt = check_format(txn, "Outlet_ID", r"OUT_\d{5}", "transactions")
quarantine(bad_fmt, "bad_outlet_id_format", "transactions")

bad_dist = check_format(txn, "Distributor_ID",
    r"DIST_(W|C|NW|S)_0[0-9]", "transactions")
quarantine(bad_dist, "bad_distributor_id_format", "transactions")

bad_idx = set(null_txn.index) | set(exact_dupes.index) | set(neg_vol.index) \
        | set(neg_bill.index) | set(bad_year.index) | set(bad_month.index) \
        | set(bad_fmt.index) | set(bad_dist.index)
txn_silver = txn.drop(index=bad_idx).reset_index(drop=True)
print(f"     Clean rows: {len(txn_silver):,}  |  Quarantined: {len(bad_idx):,}")
txn_silver.to_parquet(f"{SILVER_DIR}/transactions_silver.parquet", index=False)

# Referential integrity: transactions should only reference cleaned outlets

# ── 3b. OUTLET MASTER ────────────────────────────────────────────────────────
print("\n  → outlet_master.csv")
null_m = check_nulls(master, ["Outlet_ID","Outlet_Size","Outlet_Type"], "outlet_master")
quarantine(null_m, "null_mandatory_field", "outlet_master")
dupe_m = check_duplicates(master, ["Outlet_ID"], "outlet_master")
quarantine(dupe_m, "duplicate_outlet", "outlet_master")
bad_fmt_m = check_format(master, "Outlet_ID", r"OUT_\d{5}", "outlet_master")
quarantine(bad_fmt_m, "bad_outlet_id_format", "outlet_master")

bad_idx_m = set(null_m.index) | set(dupe_m.index) | set(bad_fmt_m.index)
master_silver = master.drop(index=bad_idx_m).reset_index(drop=True)
master_silver["Cooler_Count"] = pd.to_numeric(
    master_silver["Cooler_Count"], errors="coerce").fillna(0).astype(int)
master_silver["Outlet_Type"] = master_silver["Outlet_Type"].str.strip().str.title()

TYPE_STANDARDISE = {
    "Grocry"      : "Grocery",
    "Groc"        : "Grocery",
    "Grocery "    : "Grocery",
    "Smmt"        : "Supermarket",
    "Supermarkt"  : "Supermarket",
    "Super Market": "Supermarket",
    "Bakry"       : "Bakery",
    "Bakrey"      : "Bakery",
    "Kade"        : "Kiosk",
    "Kade "       : "Kiosk",
    "Eatry"       : "Eatery",
    "Eat"         : "Eatery",
}
master_silver["Outlet_Type"] = master_silver["Outlet_Type"].replace(TYPE_STANDARDISE)

master_silver["Outlet_Size"] = master_silver["Outlet_Size"].str.strip().str.title()

print(f"     Clean rows: {len(master_silver):,}")
print("  Outlet types after cleaning:")
print(master_silver["Outlet_Type"].value_counts().to_string())
print("\n  Outlet sizes after cleaning:")
print(master_silver["Outlet_Size"].value_counts().to_string())

master_silver.to_parquet(f"{SILVER_DIR}/outlet_master_silver.parquet", index=False)

orphan_outlets = check_referential_integrity(
    txn_silver, "Outlet_ID", master_silver, "Outlet_ID", "transactions")
quarantine(orphan_outlets, "outlet_id_not_in_master", "transactions")

if len(orphan_outlets) > 0:
    txn_silver = txn_silver[
        txn_silver["Outlet_ID"].isin(master_silver["Outlet_ID"])
    ].reset_index(drop=True)
    print(f"     txn_silver after orphan removal: {len(txn_silver):,} rows")
    txn_silver.to_parquet(f"{SILVER_DIR}/transactions_silver.parquet", index=False)

# ── 3c. COORDINATES ───────────────────────────────────────────────────────────
print("\n  → outlet_coordinates.csv")
null_c = check_nulls(coords, ["Outlet_ID","Latitude","Longitude"], "coordinates")
quarantine(null_c, "null_mandatory_field", "coordinates")
lat_bad = check_value_range(coords, "Latitude", 5.9, 9.9, "coordinates")
lon_bad = check_value_range(coords, "Longitude", 79.5, 82.0, "coordinates")
quarantine(lat_bad, "latitude_outside_sri_lanka", "coordinates")
quarantine(lon_bad, "longitude_outside_sri_lanka", "coordinates")
bad_idx_c = set(null_c.index)|set(lat_bad.index)|set(lon_bad.index)
coords_silver = coords.drop(index=bad_idx_c).reset_index(drop=True)
print(f"     Clean rows: {len(coords_silver):,}")
coords_silver.to_parquet(f"{SILVER_DIR}/coordinates_silver.parquet", index=False)

missing_coords = check_referential_integrity(
    master_silver, "Outlet_ID", coords_silver, "Outlet_ID", "outlet_master")
print(f"  ℹ  {len(missing_coords):,} outlets in master have no coordinates")

# ── 3d. SEASONALITY ───────────────────────────────────────────────────────────
print("\n  → distributor_seasonality_details.csv")
VALID_INDEX = {"Favorable","Moderate","Un-Favorable","Unfavorable","Highly Favorable"}
bad_si = season[~season["Seasonality_Index"].isin(VALID_INDEX)]
quarantine(bad_si, "invalid_seasonality_index", "seasonality")
season_silver = season[season["Seasonality_Index"].isin(VALID_INDEX)].copy()
season_silver["Seasonality_Index"] = season_silver["Seasonality_Index"].replace(
    {"Unfavorable": "Un-Favorable"})
season_silver["Seasonality_Score"] = season_silver["Seasonality_Index"].map({
    "Highly Favorable": 1.3,
    "Favorable":        1.15,
    "Moderate":         1.0,
    "Un-Favorable":     0.8,
})
print(f"     Clean rows: {len(season_silver):,}")
season_silver.to_parquet(f"{SILVER_DIR}/seasonality_silver.parquet", index=False)

orphan_dists = check_referential_integrity(
    txn_silver, "Distributor_ID", season_silver, "Distributor_ID", "transactions")
quarantine(orphan_dists, "distributor_not_in_seasonality", "transactions")

# ── 3e. HOLIDAYS ──────────────────────────────────────────────────────────────
holidays["Date"] = pd.to_datetime(holidays["Date"], utc=True, errors="coerce")
holidays["Year"]  = holidays["Date"].dt.year
holidays["Month"] = holidays["Date"].dt.month
holidays_per_month = holidays.groupby(["Year","Month"]).size().reset_index(
    name="Holiday_Count")
holidays_per_month.to_parquet(f"{SILVER_DIR}/holidays_silver.parquet", index=False)
print(f"\n  → holiday_list.csv  →  {len(holidays_per_month)} year-month combos")

print("\n[SILVER] Complete. Rejected records in ./silver/rejected/")


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EDA] Generating plots...")

def savefig(name):
    path = os.path.join(PLOTS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Saved: {name}")

# ── FIGURE 1: Dataset Overview ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("DATA STORM v7.0 — Dataset Overview", fontsize=16, fontweight="bold")

axes[0,0].hist(txn_silver["Volume_Liters"], bins=60, color="#2196F3", edgecolor="white")
axes[0,0].set_title("Volume_Liters Distribution"); axes[0,0].set_xlabel("Liters")
axes[0,0].set_ylabel("Count")

axes[0,1].hist(np.log1p(txn_silver["Volume_Liters"]), bins=60,
               color="#4CAF50", edgecolor="white")
axes[0,1].set_title("log(1 + Volume_Liters)"); axes[0,1].set_xlabel("log Liters")

sample = txn_silver.sample(min(5000, len(txn_silver)), random_state=42)
axes[0,2].scatter(sample["Volume_Liters"], sample["Total_Bill_Value"],
                  alpha=0.3, s=10, color="#FF5722")
axes[0,2].set_title("Volume vs Bill Value")
axes[0,2].set_xlabel("Volume (L)"); axes[0,2].set_ylabel("Bill Value")

txn_silver["Distributor_ID"].value_counts().plot(kind="bar", ax=axes[1,0],
    color="#9C27B0", edgecolor="white")
axes[1,0].set_title("Transactions per Distributor")
axes[1,0].set_xticklabels(axes[1,0].get_xticklabels(), rotation=45, ha="right")

monthly_vol = txn_silver.groupby(["Year","Month"])["Volume_Liters"].sum().reset_index()
monthly_vol["Period"] = monthly_vol["Year"].astype(str) + "-" + \
    monthly_vol["Month"].astype(str).str.zfill(2)
monthly_vol = monthly_vol.sort_values("Period")
axes[1,1].plot(range(len(monthly_vol)), monthly_vol["Volume_Liters"],
               marker="o", color="#FF9800", linewidth=2, markersize=4)
axes[1,1].set_title("Total Volume by Month")
axes[1,1].set_xlabel("Period"); axes[1,1].set_ylabel("Total Liters")
axes[1,1].set_xticks(range(0, len(monthly_vol), 3))
axes[1,1].set_xticklabels(monthly_vol["Period"].iloc[::3], rotation=45, ha="right")

sku_vol = txn_silver.groupby("SKU_ID")["Volume_Liters"].sum()
sku_vol.plot(kind="pie", ax=axes[1,2], autopct="%1.1f%%", startangle=90)
axes[1,2].set_title("Volume Share by SKU"); axes[1,2].set_ylabel("")

plt.tight_layout()
savefig("01_dataset_overview.png")

# ── FIGURE 2: Outlet Analysis ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Outlet Master Analysis", fontsize=16, fontweight="bold")

master_silver["Outlet_Type"].value_counts().plot(kind="bar", ax=axes[0,0],
    color=PALETTE, edgecolor="white")
axes[0,0].set_title("Outlet Type Distribution")
axes[0,0].set_xticklabels(axes[0,0].get_xticklabels(), rotation=30, ha="right")

master_silver["Outlet_Size"].value_counts().plot(kind="pie", ax=axes[0,1],
    autopct="%1.1f%%", startangle=90, colors=PALETTE)
axes[0,1].set_title("Outlet Size Distribution"); axes[0,1].set_ylabel("")

axes[1,0].hist(master_silver["Cooler_Count"], bins=10, color="#00BCD4", edgecolor="white")
axes[1,0].set_title("Cooler Count Distribution"); axes[1,0].set_xlabel("Coolers")

sns.boxplot(data=master_silver, x="Outlet_Size", y="Cooler_Count",
            ax=axes[1,1], palette="husl")
axes[1,1].set_title("Coolers by Outlet Size")

plt.tight_layout()
savefig("02_outlet_analysis.png")

# ── FIGURE 3: Geo-spatial Plot ─────────────────────────────────────────────
coords_merged = coords_silver.merge(master_silver, on="Outlet_ID", how="left")
coords_merged = coords_merged.merge(
    txn_silver.groupby("Outlet_ID")["Volume_Liters"].sum().reset_index(),
    on="Outlet_ID", how="left")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Geo-spatial Distribution of Outlets", fontsize=16, fontweight="bold")

size_map = {"Small": 10, "Medium": 30, "Large": 80}
coords_merged["plot_size"] = coords_merged["Outlet_Size"].map(size_map).fillna(20)

sc = axes[0].scatter(coords_merged["Longitude"], coords_merged["Latitude"],
                     c=np.log1p(coords_merged["Volume_Liters"].fillna(0)),
                     s=coords_merged["plot_size"], alpha=0.5, cmap="YlOrRd")
plt.colorbar(sc, ax=axes[0], label="log(Total Volume)")
axes[0].set_title("Outlets — Colour by log(Volume)"); axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")

type_map = {"Grocery": 0, "Hotel": 1, "Pharmacy": 2, "Kiosk": 3, "Eatery": 4}
coords_merged["type_code"] = coords_merged["Outlet_Type"].map(type_map).fillna(5)
sc2 = axes[1].scatter(coords_merged["Longitude"], coords_merged["Latitude"],
                      c=coords_merged["type_code"], s=15, alpha=0.5, cmap="tab10")
axes[1].set_title("Outlets — Colour by Type"); axes[1].set_xlabel("Longitude")
plt.tight_layout()
savefig("03_geospatial.png")

# ── FIGURE 4: Seasonality & Holidays ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Seasonality & Holiday Analysis", fontsize=16, fontweight="bold")

season_pivot = season_silver.pivot_table(
    index="Month", columns="Distributor_ID", values="Seasonality_Score")
sns.heatmap(season_pivot, cmap="RdYlGn", ax=axes[0], annot=False, linewidths=0.3)
axes[0].set_title("Seasonality Score Heatmap (Month × Distributor)")

holidays_per_month.groupby("Month")["Holiday_Count"].mean().plot(
    kind="bar", ax=axes[1], color="#E91E63", edgecolor="white")
axes[1].set_title("Avg Holidays per Month"); axes[1].set_xlabel("Month")
axes[1].set_ylabel("Avg Holiday Count")

plt.tight_layout()
savefig("04_seasonality_holidays.png")

# ── FIGURE 5: Censorship / Constraint Analysis ─────────────────────────────
outlet_monthly = txn_silver.groupby(
    ["Outlet_ID","Year","Month"])["Volume_Liters"].sum().reset_index()
outlet_stats = outlet_monthly.groupby("Outlet_ID")["Volume_Liters"].agg(
    ["mean","max","std","count"]).reset_index()
outlet_stats.columns = ["Outlet_ID","mean_vol","max_vol","std_vol","n_months"]
outlet_stats["cv"] = outlet_stats["std_vol"] / (outlet_stats["mean_vol"] + 1e-6)
outlet_stats["ceiling_ratio"] = outlet_stats["max_vol"] / (outlet_stats["mean_vol"] + 1e-6)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Left-Censored Demand Analysis", fontsize=16, fontweight="bold")

axes[0,0].hist(outlet_stats["ceiling_ratio"].clip(0, 5), bins=60, color="#3F51B5",
               edgecolor="white")
axes[0,0].axvline(1.5, color="red", linestyle="--", label="Potential ceiling start")
axes[0,0].set_title("Max / Mean Volume Ratio per Outlet")
axes[0,0].set_xlabel("Ceiling Ratio"); axes[0,0].legend()

axes[0,1].scatter(outlet_stats["mean_vol"], outlet_stats["cv"],
                  alpha=0.3, s=8, color="#009688")
axes[0,1].set_title("Coefficient of Variation vs Mean Volume")
axes[0,1].set_xlabel("Mean Volume (L)"); axes[0,1].set_ylabel("CV")

axes[1,0].hist(outlet_stats["n_months"], bins=30, color="#FF5722", edgecolor="white")
axes[1,0].set_title("Active Months per Outlet")
axes[1,0].set_xlabel("Number of Active Months")

axes[1,1].scatter(outlet_stats["n_months"], outlet_stats["ceiling_ratio"],
                  alpha=0.3, s=8, color="#795548")
axes[1,1].set_title("Ceiling Ratio vs Active Months")
axes[1,1].set_xlabel("Active Months"); axes[1,1].set_ylabel("Ceiling Ratio")

plt.tight_layout()
savefig("05_censorship_analysis.png")

# ── FIGURE 6: Distributor-Level Deep-Dive ─────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Distributor Deep-Dive", fontsize=16, fontweight="bold")

dist_vol = txn_silver.groupby("Distributor_ID")["Volume_Liters"].agg(
    ["sum","mean","std"]).reset_index()
dist_vol.sort_values("sum", ascending=False, inplace=True)

dist_vol.plot(kind="bar", x="Distributor_ID", y="sum", ax=axes[0,0],
    legend=False, color="#1565C0", edgecolor="white")
axes[0,0].set_title("Total Volume by Distributor"); axes[0,0].set_xticklabels(
    axes[0,0].get_xticklabels(), rotation=45, ha="right")

sns.boxplot(data=txn_silver, x="Distributor_ID", y="Volume_Liters",
            ax=axes[0,1], showfliers=False)
axes[0,1].set_title("Volume Distribution per Distributor")
axes[0,1].set_xticklabels(axes[0,1].get_xticklabels(), rotation=45, ha="right")

province_map = {
    "DIST_W_01":"Western","DIST_W_02":"Western","DIST_W_03":"Western",
    "DIST_C_01":"Central","DIST_C_02":"Central","DIST_C_03":"Central",
    "DIST_NW_01":"North-Western","DIST_NW_02":"North-Western",
    "DIST_S_01":"Southern","DIST_S_02":"Southern",
}
txn_silver["Province"] = txn_silver["Distributor_ID"].map(province_map)
prov_vol = txn_silver.groupby("Province")["Volume_Liters"].sum()
prov_vol.plot(kind="pie", ax=axes[1,0], autopct="%1.1f%%", startangle=90,
    colors=PALETTE); axes[1,0].set_title("Volume Share by Province")
axes[1,0].set_ylabel("")

# Month seasonality per province
txn_prov_month = txn_silver.groupby(
    ["Province","Month"])["Volume_Liters"].sum().reset_index()
for prov in txn_prov_month["Province"].unique():
    sub = txn_prov_month[txn_prov_month["Province"]==prov]
    axes[1,1].plot(sub["Month"], sub["Volume_Liters"], marker="o", label=prov)
axes[1,1].set_title("Volume by Month & Province"); axes[1,1].legend(fontsize=7)
axes[1,1].set_xlabel("Month"); axes[1,1].set_ylabel("Volume (L)")
axes[1,1].set_xticks(range(1,13))

plt.tight_layout()
savefig("06_distributor_deepdive.png")

# ── FIGURE 7: Anomaly Detection ───────────────────────────────────────────
X_iso = outlet_stats[["mean_vol","max_vol","cv","n_months"]].fillna(0)
iso = IsolationForest(contamination=0.05, random_state=42)
outlet_stats["anomaly"] = iso.fit_predict(X_iso)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Isolation Forest Anomaly Detection", fontsize=16, fontweight="bold")
colors = outlet_stats["anomaly"].map({1: "#4CAF50", -1: "#F44336"})
axes[0].scatter(outlet_stats["mean_vol"], outlet_stats["max_vol"],
                c=colors, alpha=0.4, s=10)
axes[0].set_title("Anomalies in Mean vs Max Volume")
axes[0].set_xlabel("Mean Volume"); axes[0].set_ylabel("Max Volume")

axes[1].scatter(outlet_stats["n_months"], outlet_stats["cv"],
                c=colors, alpha=0.4, s=10)
axes[1].set_title("Anomalies in Active Months vs CV")
axes[1].set_xlabel("Active Months"); axes[1].set_ylabel("CV")
axes[1].legend(handles=[
    plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#4CAF50",
               markersize=8, label="Normal"),
    plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#F44336",
               markersize=8, label="Anomaly")], loc="upper right")

plt.tight_layout()
savefig("07_anomaly_detection.png")

print("[EDA] All plots saved to ./plots/")


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
print("\n[GOLD] Building feature store...")

outlet_features = txn_silver.groupby("Outlet_ID").agg(
    total_volume      = ("Volume_Liters", "sum"),
    mean_monthly_vol  = ("Volume_Liters", "mean"),
    max_monthly_vol   = ("Volume_Liters", "max"),
    p90_vol           = ("Volume_Liters", lambda x: np.percentile(x, 90)),
    p95_vol           = ("Volume_Liters", lambda x: np.percentile(x, 95)),
    std_vol           = ("Volume_Liters", "std"),
    n_transactions    = ("Volume_Liters", "count"),
    n_distinct_months = ("Month", "nunique"),
    n_distinct_skus   = ("SKU_ID", "nunique"),
    mean_bill_value   = ("Total_Bill_Value", "mean"),
    max_bill_value    = ("Total_Bill_Value", "max"),
).reset_index()

outlet_features["cv"] = outlet_features["std_vol"] / \
    (outlet_features["mean_monthly_vol"] + 1e-6)
outlet_features["ceiling_ratio"] = outlet_features["max_monthly_vol"] / \
    (outlet_features["mean_monthly_vol"] + 1e-6)
outlet_features["volume_per_sku"] = outlet_features["total_volume"] / \
    (outlet_features["n_distinct_skus"] + 1e-6)

# ── 5b. WEIGHTED JANUARY FEATURES — pandas-safe ───────────────────────────
jan_txn = txn_silver[txn_silver["Month"] == 1].copy()
year_weights_map = {2023: 0.10, 2024: 0.35, 2025: 0.55}
jan_txn["year_weight"] = jan_txn["Year"].map(year_weights_map).fillna(0.2)

jan_base = jan_txn.groupby("Outlet_ID").agg(
    jan_max_vol       = ("Volume_Liters", "max"),
    jan_count         = ("Volume_Liters", "count"),
    jan_years_present = ("Year", "nunique"),
).reset_index()

jan_wm = (
    jan_txn
    .assign(wv=jan_txn["Volume_Liters"] * jan_txn["year_weight"])
    .groupby("Outlet_ID")
    .agg(
        wv_sum=("wv", "sum"),
        w_sum=("year_weight", "sum"),
    )
    .reset_index()
)
jan_wm["jan_weighted_mean"] = (
    jan_wm["wv_sum"] / jan_wm["w_sum"].replace(0, np.nan)
)

jan_2025 = (
    jan_txn[jan_txn["Year"] == 2025]
    .groupby("Outlet_ID")["Volume_Liters"].mean()
    .reset_index(name="jan_2025_vol")
)
jan_2024 = (
    jan_txn[jan_txn["Year"] == 2024]
    .groupby("Outlet_ID")["Volume_Liters"].mean()
    .reset_index(name="jan_2024_vol")
)

jan_features = (
    jan_base
    .merge(jan_wm[["Outlet_ID","jan_weighted_mean"]], on="Outlet_ID", how="left")
    .merge(jan_2025, on="Outlet_ID", how="left")
    .merge(jan_2024, on="Outlet_ID", how="left")
)
print(f"  Jan features: {len(jan_features):,} outlets "
      f"({jan_features['jan_weighted_mean'].notna().sum():,} with Jan history)")

# If outlet has no Jan data at all → jan_weighted_mean will be NaN
# Handled in estimate_potential() by falling back to overall p90

# 5c. Jan 2026 seasonality multiplier per distributor
jan_season = (
    season_silver[season_silver["Month"] == 1]
    .sort_values("Year", ascending=False)
    .drop_duplicates(subset="Distributor_ID", keep="first")
    .copy()
)
# Use latest available Jan seasonality
outlet_dist = txn_silver.groupby("Outlet_ID")["Distributor_ID"].agg(
    lambda x: x.mode()[0]).reset_index()

# 5d. Holiday uplift for January
jan_holidays = holidays_per_month[holidays_per_month["Month"]==1][
    "Holiday_Count"].mean()
holiday_uplift = 1 + 0.02 * jan_holidays  # 2% per holiday day

gold = (outlet_features
    .merge(jan_features,   on="Outlet_ID", how="left")
    .merge(avg_reorders,   on="Outlet_ID", how="left")
    .merge(master_silver[["Outlet_ID","Outlet_Size","Cooler_Count","Outlet_Type"]],
           on="Outlet_ID", how="left")
    .merge(coords_silver[["Outlet_ID","Latitude","Longitude"]],
           on="Outlet_ID", how="left")
    .merge(outlet_dist,    on="Outlet_ID", how="left")
    .merge(jan_season[["Distributor_ID","Seasonality_Score"]],
           on="Distributor_ID", how="left")
)
gold["Seasonality_Score"] = gold["Seasonality_Score"].fillna(1.0)
gold["avg_monthly_reorders"] = gold["avg_monthly_reorders"].fillna(1.0)

# ── 5f. Outlet size multiplier — all 4 tiers ─────────────────────────────────
size_mult = {
    "Small"      : 0.80,
    "Medium"     : 1.00,
    "Large"      : 1.35,
    "Extra Large": 1.65,
}
gold["size_mult"] = gold["Outlet_Size"].map(size_mult).fillna(1.0)

# Cooler multiplier — each cooler = +6% potential, capped at 5 coolers
gold["cooler_mult"] = 1.0 + 0.06 * gold["Cooler_Count"].fillna(0).clip(0, 5)

# ── 5h. SKU_06 Dominance Features ────────────────────────────────────────────
sku06 = txn_silver[txn_silver["SKU_ID"] == "SKU_06"]

sku06_features = sku06.groupby("Outlet_ID").agg(
    sku06_mean_vol      = ("Volume_Liters", "mean"),
    sku06_max_vol       = ("Volume_Liters", "max"),
    sku06_total_vol     = ("Volume_Liters", "sum"),
    sku06_active_months = ("Month", "nunique"),
).reset_index()

# SKU06 share of outlet's total volume
sku06_features = sku06_features.merge(
    outlet_features[["Outlet_ID","total_volume"]], on="Outlet_ID", how="left")
sku06_features["sku06_share"] = (
    sku06_features["sku06_total_vol"] /
    (sku06_features["total_volume"] + 1e-6)
)

# Jan-specific SKU06 performance
sku06_jan = sku06[sku06["Month"] == 1].copy()
sku06_jan["year_weight"] = sku06_jan["Year"].map(
    {2023: 0.10, 2024: 0.35, 2025: 0.55}
).fillna(0.2)

sku06_jan_wm = (
    sku06_jan.assign(wv=sku06_jan["Volume_Liters"] * sku06_jan["year_weight"])
    .groupby("Outlet_ID")
    .agg(
        wv_sum=("wv", "sum"),
        w_sum=("year_weight", "sum"),
        sku06_jan_max=("Volume_Liters", "max"),
    )
    .reset_index()
)
sku06_jan_wm["sku06_jan_weighted_mean"] = (
    sku06_jan_wm["wv_sum"] / sku06_jan_wm["w_sum"].replace(0, np.nan)
)

sku06_features = sku06_features.merge(
    sku06_jan_wm[["Outlet_ID","sku06_jan_weighted_mean","sku06_jan_max"]],
    on="Outlet_ID", how="left"
)

# Merge SKU06 into gold
gold = gold.merge(
    sku06_features[["Outlet_ID","sku06_mean_vol","sku06_max_vol",
                    "sku06_share","sku06_jan_weighted_mean"]],
    on="Outlet_ID", how="left"
)

# SKU06 peer gap — how much below their peer group median is this outlet?
for (otype, osize), grp in gold.groupby(["Outlet_Type","Outlet_Size"], dropna=False):
    peer_median_sku06 = grp["sku06_mean_vol"].median()
    gold.loc[grp.index, "sku06_peer_gap"] = (
        grp["sku06_mean_vol"] - peer_median_sku06
    )

# For outlets with no SKU06 history, assign the worst peer gap in their group
gold["sku06_peer_gap"] = gold["sku06_peer_gap"].fillna(
    gold.groupby(["Outlet_Type","Outlet_Size"])["sku06_peer_gap"]
    .transform("min")
)

print(f"  Gold feature store: {gold.shape[0]:,} outlets × {gold.shape[1]} features")
gold.to_parquet(f"{GOLD_DIR}/outlet_gold.parquet", index=False)


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
"""
Core Idea: Historical volume is LEFT-CENSORED — it's min(true_demand, constraint).
We estimate the CEILING via a combination of:
  (A) Tobit-inspired uplift  → p95 / mean ratio to uncap
  (B) Seasonality adjustment → Jan 2026 seasonality score
  (C) Holiday uplift
  (D) Size & cooler multipliers
  (E) Peer benchmarking     → compare similar outlets and lift underperformers
"""
print("\n[MODEL] Estimating Latent Potential for Jan 2026...")

def estimate_potential(row):
    def g(col, default=1.0):
        val = row[col] if col in row.index else default
        return default if pd.isna(val) else val

    # ── Step 1: Base — TOTAL outlet January weighted mean ────────────────────
    jan_base = g("jan_weighted_mean", 0)
    p90      = g("p90_vol", 0)
    mean_vol = g("mean_monthly_vol", 1)

    if jan_base > 0:
        base = jan_base * 0.60 + p90 * 0.40
    else:
        base = p90

    if base == 0:
        base = mean_vol

    jan_max = g("jan_max_vol", 0)
    if jan_max > base:
        base = jan_max * 0.85 + base * 0.15

    # ── Step 2: Ceiling-ratio uncapping ──────────────────────────────────────
    ceiling_ratio = min(g("ceiling_ratio", 1.5), 4.0)
    uncap_mult    = 1.0 + 0.20 * np.log(max(ceiling_ratio, 1.001))

    # ── Step 3: CV signal ───────────────────────────────────────────────────
    cv_mult = 1.0 + 0.08 * min(g("cv", 0.5), 2.5)

    # ── Step 4: SKU06 peer gap uplift ────────────────────────────────────────
    sku06_gap    = g("sku06_peer_gap", 0)
    sku06_uplift = 1.0 + max(0, -sku06_gap) / (mean_vol + 1e-6) * 0.10
    sku06_uplift = min(sku06_uplift, 1.25)

    # ── Step 5: All multipliers ──────────────────────────────────────────────
    season_mult   = g("Seasonality_Score",      1.0)
    s_mult        = g("size_mult",              1.0)
    c_mult        = g("cooler_mult",            1.0)
    reorder_mult  = 1.0 + min(g("avg_monthly_reorders", 1.0) - 1.0, 3.0) * 0.05

    potential = (base
                 * uncap_mult
                 * cv_mult
                 * sku06_uplift
                 * season_mult
                 * holiday_uplift
                 * s_mult
                 * c_mult
                 * reorder_mult)

    # Safety floor — potential must be at least the historical mean
    potential = max(potential, mean_vol)

    return round(potential, 4)

gold["Maximum_Monthly_Liters"] = gold.apply(estimate_potential, axis=1)

# ── Peer benchmarking uplift ────────────────────────────────────────────────
# Save snapshot before benchmarking (for visualisation)
gold_before_peer = gold[["Outlet_ID","Maximum_Monthly_Liters"]].copy()
gold_before_peer.rename(
    columns={"Maximum_Monthly_Liters":"potential_before_peer"}, inplace=True)

gold_copy = gold.copy()
for (otype, osize), grp in gold.groupby(
        ["Outlet_Type","Outlet_Size"], dropna=False):
    peer_median = grp["Maximum_Monthly_Liters"].median()
    for idx in grp.index:
        own_val = gold.loc[idx, "Maximum_Monthly_Liters"]
        n_months = gold.loc[idx, "n_distinct_months"]

        # More history → trust own prediction more
        # 0 months  → 0% own, 100% peer
        # 24+ months → 80% own, 20% peer
        own_weight = min(float(n_months) / 24.0, 0.80)
        peer_weight = 1.0 - own_weight

        # Only pull UP — never drag high performers down
        if own_val < peer_median:
            blended = own_val * own_weight + peer_median * peer_weight
            gold_copy.loc[idx, "Maximum_Monthly_Liters"] = round(blended, 4)

gold = gold_copy

gold = gold.merge(gold_before_peer, on="Outlet_ID", how="left")

print(f"  Potential range: {gold['Maximum_Monthly_Liters'].min():.1f} – "
      f"{gold['Maximum_Monthly_Liters'].max():.1f} L")
print(f"  Median potential: {gold['Maximum_Monthly_Liters'].median():.1f} L")
print(f"  Outlets lifted by peer benchmarking: "
      f"{(gold['Maximum_Monthly_Liters'] > gold['potential_before_peer']).sum():,}")

# ── FIGURE 8b: Before vs After Peer Benchmarking ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Effect of Peer Benchmarking on Predictions",
             fontsize=14, fontweight="bold")

axes[0].hist(gold["potential_before_peer"], bins=80,
             color="#9E9E9E", edgecolor="white", alpha=0.7, label="Before peer")
axes[0].hist(gold["Maximum_Monthly_Liters"], bins=80,
             color="#4CAF50", edgecolor="white", alpha=0.5, label="After peer")
axes[0].set_title("Potential Distribution: Before vs After")
axes[0].set_xlabel("Predicted Max Liters")
axes[0].legend()

uplift_applied = gold["Maximum_Monthly_Liters"] - gold["potential_before_peer"]
axes[1].hist(uplift_applied[uplift_applied > 0.01], bins=60,
             color="#2196F3", edgecolor="white")
axes[1].set_title(f"Peer Uplift Applied "
                  f"({(uplift_applied > 0.01).sum():,} outlets lifted)")
axes[1].set_xlabel("Uplift Added (Liters)")
axes[1].set_ylabel("Number of Outlets")
plt.tight_layout()
savefig("08b_peer_benchmarking_effect.png")

# ── FIGURE 8: Potential Distribution ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Predicted Maximum Monthly Potential — Jan 2026",
             fontsize=15, fontweight="bold")

axes[0].hist(gold["Maximum_Monthly_Liters"], bins=80, color="#673AB7",
             edgecolor="white")
axes[0].set_title("Predicted Potential Distribution")
axes[0].set_xlabel("Predicted Max Liters")

axes[1].scatter(gold["mean_monthly_vol"], gold["Maximum_Monthly_Liters"],
                alpha=0.3, s=8, color="#E91E63")
axes[1].plot([0, gold["mean_monthly_vol"].max()],
             [0, gold["mean_monthly_vol"].max()],
             "r--", label="No uplift")
axes[1].set_title("Historical Mean vs Predicted Potential")
axes[1].set_xlabel("Historical Mean (L)")
axes[1].set_ylabel("Predicted Potential (L)")
axes[1].legend()

sns.boxplot(data=gold.dropna(subset=["Outlet_Type"]),
            x="Outlet_Type", y="Maximum_Monthly_Liters",
            ax=axes[2], showfliers=False, palette="husl")
axes[2].set_title("Potential by Outlet Type")
axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=30, ha="right")

plt.tight_layout()
savefig("08_potential_distribution.png")

# ── FIGURE 9: Potential Heatmap (geo) ─────────────────────────────────────
pot_geo = gold[["Outlet_ID","Latitude","Longitude","Maximum_Monthly_Liters"]].dropna()
fig, ax = plt.subplots(figsize=(10, 8))
sc = ax.scatter(pot_geo["Longitude"], pot_geo["Latitude"],
                c=pot_geo["Maximum_Monthly_Liters"],
                s=12, alpha=0.6, cmap="YlOrRd")
plt.colorbar(sc, ax=ax, label="Predicted Max Liters")
ax.set_title("Geographic Distribution of Predicted Potential (Jan 2026)")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
savefig("09_potential_geo_heatmap.png")

print("[MODEL] Potential estimation complete.")


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Deduplicate on Outlet_ID (take max potential if somehow still duped)
predictions = (
    gold[["Outlet_ID", "Maximum_Monthly_Liters"]]
    .groupby("Outlet_ID", as_index=False)["Maximum_Monthly_Liters"].max()
    .sort_values("Outlet_ID")
    .reset_index(drop=True)
)

# ── Optional: filter to exact submission outlets from sample_submission.csv ──
SAMPLE_SUBMISSION_PATHS = [
    os.path.join(RAW_DATA_DIR, "sample_submission.csv"),
    os.path.join(RAW_DATA_DIR, "test.csv"),
    os.path.join(RAW_DATA_DIR, "submission_template.csv"),
]
sample_file = next((p for p in SAMPLE_SUBMISSION_PATHS if os.path.exists(p)), None)

if sample_file:
    print(f"\n[OUTPUT] Sample submission found: {sample_file}")
    sample = pd.read_csv(sample_file)
    id_col = next((c for c in sample.columns if "outlet" in c.lower()), sample.columns[0])
    required_outlets = sample[id_col].unique()
    predictions = predictions[predictions["Outlet_ID"].isin(required_outlets)].copy()
    order = {v: i for i, v in enumerate(sample[id_col])}
    predictions["_order"] = predictions["Outlet_ID"].map(order)
    predictions = predictions.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    print(f"  Filtered to {len(predictions):,} outlets from sample submission")
else:
    print("\n[OUTPUT] No sample_submission.csv found in ./data/ — outputting all outlets.")
    print("         If the platform expects a subset, download sample_submission.csv")
    print("         from the competition portal and place it in ./data/")

predictions = predictions.reset_index(drop=True)
predictions.insert(0, "row_id", range(1, len(predictions) + 1))
out_path = os.path.join(OUTPUT_DIR, f"{TEAM_NAME}_predictions.csv")
predictions[["row_id", "Outlet_ID", "Maximum_Monthly_Liters"]].to_csv(out_path, index=False)
print(f"\n[OUTPUT] Predictions saved → {out_path}  ({len(predictions):,} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
"""
Run this section to enrich outlets with nearby POI counts.
Set SCRAPE_POI = True and adjust RADIUS_M as needed.
Results are saved to gold/poi_features.csv for merging into the Gold layer.
"""

SCRAPE_POI = False  # ← set True to run
RADIUS_M   = 500   # search radius in metres

POI_TAGS = {
    "schools":    'amenity~"school|college|university"',
    "hospitals":  'amenity~"hospital|clinic|pharmacy"',
    "bus_stops":  'highway=bus_stop',
    "fuel":       'amenity=fuel',
    "worship":    'amenity~"place_of_worship"',
    "markets":    'shop~"supermarket|convenience|marketplace"',
    "tourism":    'tourism~"hotel|guest_house|attraction"',
}

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

def query_overpass(lat, lon, radius, tag_filter, timeout=30):
    query = f"""
    [out:json][timeout:{timeout}];
    node[{tag_filter}](around:{radius},{lat},{lon});
    out count;
    """
    try:
        r = requests.post(OVERPASS_URL, data={"data": query}, timeout=timeout+5)
        data = r.json()
        return data.get("elements", [{}])[0].get("tags", {}).get("total", 0)
    except Exception:
        return 0

if SCRAPE_POI:
    print("\n[POI] Scraping OpenStreetMap via Overpass API...")
    poi_rows = []
    sample_coords = coords_silver.head(200)  # limit for demo; remove head() for full run

    for _, row in tqdm(sample_coords.iterrows(), total=len(sample_coords)):
        rec = {"Outlet_ID": row["Outlet_ID"]}
        for poi_name, tag in POI_TAGS.items():
            rec[f"poi_{poi_name}"] = query_overpass(
                row["Latitude"], row["Longitude"], RADIUS_M, tag)
        poi_rows.append(rec)

    poi_df = pd.DataFrame(poi_rows)
    poi_df.to_csv(f"{GOLD_DIR}/poi_features.csv", index=False)
    print(f"[POI] Saved {len(poi_df)} outlet POI records → gold/poi_features.csv")
else:
    print("\n[POI] Skipped (set SCRAPE_POI=True to enable Overpass scraping)")


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  PIPELINE SUMMARY")
print("=" * 70)
print(f"  Outlets processed          : {len(predictions):,}")
print(f"  Transactions (clean)       : {len(txn_silver):,}")
print(f"  Transactions (quarantined) : {len(bad_idx):,}")
print(f"  Plots generated            : 9 (./plots/)")
print(f"  Predictions file           : {out_path}")
print(f"  Predicted Median Potential : {predictions['Maximum_Monthly_Liters'].median():.1f} L")
print(f"  Predicted Max Potential    : {predictions['Maximum_Monthly_Liters'].max():.1f} L")
print("=" * 70)
print("  DONE ✔")
print("=" * 70)
