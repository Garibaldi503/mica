import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import io

st.set_page_config(page_title="ABC/XYZ + Monthly Demand", layout="wide")
st.title("ABC/XYZ Analysis & Monthly Demand Explorer")

st.caption("Upload your file or use the built-in demo. Expected fields (auto-detected): sku_desc/item/product, date, quantity (or price+value).")

# ---------- Helpers ----------
def first_match(df, candidates):
    cols = [c.lower() for c in df.columns]
    for c in candidates:
        if c in cols:
            return c
    return None

def try_parse_date(s):
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.to_datetime(s, errors="coerce")

def load_input():
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx","xls"])
    if uploaded:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            xls = pd.ExcelFile(uploaded)
            df = xls.parse(xls.sheet_names[0])
    else:
        # fallback: try to use /mnt/data/load1.xlsx if present
        try:
            xls = pd.ExcelFile("/mnt/data/load1.xlsx")
            df = xls.parse(xls.sheet_names[0])
            st.info("Using the previously uploaded /mnt/data/load1.xlsx")
        except Exception:
            st.stop()
    # normalize columns
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def run_abc_xyz(df):
    # detect columns
    item_col = first_match(df, ["sku_desc","sku","item","product","product_id","description","part_no","code"])
    date_col = first_match(df, ["date","trans_date","invoice_date","day","period"])
    qty_col  = first_match(df, ["quantity","qty","units","sales_units","units_sold","qty_sold"])
    price_col = first_match(df, ["price","unit_price","sell_price","selling_price"])
    value_col = first_match(df, ["value","revenue","sales","amount","line_total","sale_value"])
    cost_col = first_match(df, ["cost","unit_cost","avg_cost","cost_price"])

    # validations
    miss = []
    if item_col is None: miss.append("sku/item column")
    if date_col is None: miss.append("date column")
    if qty_col is None and value_col is None and price_col is None:
        miss.append("quantity or revenue/price columns")
    if miss:
        st.error(f"Missing required columns: {', '.join(miss)}")
        st.write("Columns seen:", list(df.columns))
        st.stop()

    work = df.copy()
    work[date_col] = try_parse_date(work[date_col])

    # ensure numeric
    if qty_col is not None:
        work[qty_col] = pd.to_numeric(work[qty_col], errors="coerce")
    if price_col is not None:
        work[price_col] = pd.to_numeric(work[price_col], errors="coerce")
    if value_col is not None:
        work[value_col] = pd.to_numeric(work[value_col], errors="coerce")

    # make value if needed
    if value_col is None:
        if qty_col is not None and price_col is not None:
            work["__value__"] = work[qty_col].fillna(0) * work[price_col].fillna(0)
            value_col = "__value__"
        elif qty_col is not None and cost_col is not None:
            work["__value__"] = work[qty_col].fillna(0) * pd.to_numeric(work[cost_col], errors="coerce").fillna(0)
            value_col = "__value__"
        else:
            work["__value__"] = 0.0
            value_col = "__value__"
    work[value_col] = work[value_col].fillna(0.0)

    if qty_col is None:
        # derive qty from value/price if possible
        if value_col is not None and price_col is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                work["__qty__"] = np.where(
                    (pd.to_numeric(work[price_col], errors="coerce").fillna(0)!=0),
                    work[value_col] / pd.to_numeric(work[price_col], errors="coerce").replace(0, np.nan),
                    np.nan
                )
            qty_col = "__qty__"
        else:
            # still none -> we'll use value as proxy in XYZ
            pass

    # aggregate per item
    agg = work.groupby(item_col, dropna=False).agg(
        total_units=(qty_col, "sum") if qty_col is not None else (value_col, "size"),
        total_value=(value_col, "sum"),
        first_date=(date_col, "min"),
        last_date=(date_col, "max"),
        n_txns=(value_col, "size")
    ).reset_index()

    # ABC by value
    agg_sorted = agg.sort_values("total_value", ascending=False).reset_index(drop=True)
    agg_sorted["cum_value"] = agg_sorted["total_value"].cumsum()
    tv = agg_sorted["total_value"].sum()
    agg_sorted["cum_pct"] = (agg_sorted["cum_value"]/tv) if tv>0 else 0.0

    def abc_label(p):
        if p <= 0.80: return "A"
        elif p <= 0.95: return "B"
        else: return "C"
    agg_sorted["ABC"] = agg_sorted["cum_pct"].apply(abc_label)

    # Monthly aggregation for XYZ and for SKU trends
    work["month"] = work[date_col].dt.to_period("M").dt.to_timestamp()

    if qty_col is not None:
        measure = qty_col
    else:
        measure = value_col  # proxy

    monthly_item = work.groupby([item_col, "month"], dropna=False)[measure].sum().reset_index()

    # Coefficient of Variation per item
    cov_vals = []
    for sku, g in monthly_item.groupby(item_col):
        vals = g[measure].astype(float).values
        mu = np.mean(vals)
        sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        cov = np.inf if mu == 0 else (sd / mu)
        cov_vals.append((sku, cov))
    cov_df = pd.DataFrame(cov_vals, columns=[item_col, "cov"])

    def xyz_label(cov):
        if not np.isfinite(cov):
            return "Z"
        if cov <= 0.2: return "X"
        elif cov <= 0.5: return "Y"
        else: return "Z"

    cov_df["XYZ"] = cov_df["cov"].apply(xyz_label)

    summary = agg_sorted.merge(cov_df, on=item_col, how="left")
    summary["XYZ"] = summary["XYZ"].fillna("Z")
    summary["ABC_XYZ"] = summary["ABC"] + "-" + summary["XYZ"]

    matrix_counts = summary.groupby(["ABC","XYZ"]).size().unstack(fill_value=0)

    return {
        "work": work,
        "item_col": item_col,
        "date_col": date_col,
        "qty_col": qty_col,
        "value_col": value_col,
        "measure": measure,
        "summary": summary,
        "matrix_counts": matrix_counts,
        "monthly_item": monthly_item
    }

def export_excel(summary, matrix_counts):
    import xlsxwriter
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        summary.to_excel(writer, sheet_name="item_summary", index=False)
        matrix_counts.reset_index().to_excel(writer, sheet_name="matrix_counts", index=False)
        pv_val = summary.pivot_table(index="ABC", columns="XYZ", values="total_value", aggfunc="sum", fill_value=0.0)
        pv_val.to_excel(writer, sheet_name="pivot_value")
    return output.getvalue()

# ---------- Main ----------
df = load_input()
res = run_abc_xyz(df)

summary = res["summary"]
matrix_counts = res["matrix_counts"]
monthly_item = res["monthly_item"]
item_col = res["item_col"]
measure = res["measure"]

# Top panels
col1, col2 = st.columns([3,2], gap="large")
with col1:
    st.subheader("ABC–XYZ Matrix (Counts)")
    st.dataframe(matrix_counts)
with col2:
    st.subheader("Download ABC–XYZ Excel")
    xls_bytes = export_excel(summary, matrix_counts)
    st.download_button("Download abc_xyz_report.xlsx", data=xls_bytes, file_name="abc_xyz_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.subheader("Item Summary (first 500)")
st.dataframe(summary.head(500))

# ---------- SKU Query & 6-month Trend ----------
st.markdown("---")
st.header("SKU Monthly Demand")

# Let user pick months window
months_window = st.slider("Show last N months", min_value=3, max_value=24, value=6, step=1)

# SKU pick (searchable)
all_skus = summary[item_col].astype(str).tolist()
default_sku = all_skus[0] if all_skus else None
sku_pick = st.selectbox("Choose SKU", options=all_skus, index=0 if default_sku else None)

if sku_pick:
    # Determine last N months based on data
    max_month = monthly_item["month"].max()
    if pd.isna(max_month):
        st.warning("No valid dates to form months.")
    else:
        start_month = (max_month.to_period("M") - (months_window-1)).to_timestamp()
        sku_df = monthly_item[(monthly_item[item_col].astype(str) == str(sku_pick)) & (monthly_item["month"] >= start_month)].copy()
        sku_df = sku_df.sort_values("month")
        # rename measure for display
        disp_col = "demand"
        sku_df.rename(columns={measure: disp_col}, inplace=True)

        # Chart
        fig = plt.figure(figsize=(8,3.2))
        plt.plot(sku_df["month"], sku_df[disp_col], marker="o")
        plt.title(f"Monthly Demand for {sku_pick} (last {months_window} months)")
        plt.xlabel("Month")
        plt.ylabel("Units" if res["qty_col"] is not None else "Value (proxy)")
        plt.grid(True, alpha=0.3)
        st.pyplot(fig, clear_figure=True)

        # Table
        st.dataframe(sku_df[["month", disp_col]].reset_index(drop=True))

# ---------- Monthly Totals ----------
st.markdown("---")
st.subheader("Monthly Totals (All Items)")
monthly_total = monthly_item.groupby("month")[measure].sum().reset_index()
monthly_total.rename(columns={measure:"total_demand"}, inplace=True)
st.dataframe(monthly_total)
