# -*- coding: utf-8 -*-
"""
Chain Ladder & Bornhuetter-Ferguson IBNR Calculator
African Actuarial Consultants - Production Grade Streamlit App
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import date
import re
import logging

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
DEFAULT_ELR_PCT        = 70.0          # Expected Loss Ratio default (%)
MAX_ELR_PCT            = 200.0
DEFAULT_PREMIUM        = 1_000_000.0
PREMIUM_STEP           = 100_000.0
MAX_PREMIUM_COLS       = 6             # Max columns per row in premium entry
DEFAULT_FROM_YEAR      = 2020
DEFAULT_TO_YEAR        = 2024
ALLOWED_FILE_TYPES     = ["csv", "xlsx", "xls"]
ENCODINGS              = ["utf-8", "cp1252", "latin-1"]

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Chain Ladder & BF IBNR Calculator",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background-color: #FFFFFF;
        color: #000000;
        font-family: 'Calisto MT', serif;
        font-size: 11pt;
    }
    h1, h2, h3, p, div, span, label {
        font-family: 'Calisto MT', serif;
    }
    .header {
        background-color: #000000;
        padding: 1rem 2rem;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        border-bottom: 3px solid #D4AF37;
        margin-bottom: 2rem;
    }
    .nav-links a {
        color: #FFFFFF;
        margin-left: 2rem;
        text-decoration: none;
        font-weight: 500;
    }
    .nav-links a:hover { color: #D4AF37; }
    .hero {
        background: linear-gradient(135deg, #000000 0%, #333333 100%);
        color: #FFFFFF;
        padding: 2rem;
        text-align: center;
        border-bottom: 3px solid #D4AF37;
        margin-bottom: 2rem;
    }
    .hero h1 { color: #D4AF37; font-size: 2.5rem; margin-bottom: 0.5rem; }
    .hero p  { font-size: 1.2rem; max-width: 800px; margin: 0 auto; }
    .gold-container {
        background-color: #F9F9F9;
        border: 2px solid #D4AF37;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .gold-container h3 { color: #D4AF37; margin-top: 0; margin-bottom: 0.75rem; font-size: 1.2rem; font-weight: bold; }
    .gold-container p  { color: #666666; font-size: 0.85rem; margin-bottom: 0.5rem; }
    .stButton > button { background-color: #D4AF37 !important; color: #000000 !important; border: none !important; border-radius: 4px !important; font-weight: bold !important; }
    .stButton > button:hover { background-color: #B8960F !important; color: #FFFFFF !important; }
    .stFileUploader { border: 2px dashed #D4AF37; border-radius: 10px; padding: 1rem; background-color: #F9F9F9; }
    .card { background-color: #F9F9F9; border: 1px solid #D4AF37; border-radius: 8px; padding: 1.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 2rem; }
    .card h3 { color: #D4AF37; border-bottom: 2px solid #D4AF37; padding-bottom: 0.5rem; }
    .footer { background-color: #000000; color: #FFFFFF; text-align: center; padding: 1.5rem; border-top: 3px solid #D4AF37; margin-top: 3rem; }
    .footer a { color: #D4AF37; text-decoration: none; }
    .dq-info    { background-color: #E8F4FD; border-left: 5px solid #2196F3; padding: 15px; border-radius: 5px; margin: 10px 0; }
    .dq-warning { background-color: #FFF3E0; border-left: 5px solid #FF9800; padding: 15px; border-radius: 5px; margin: 10px 0; }
    .dq-error   { background-color: #FFEBEE; border-left: 5px solid #F44336; padding: 15px; border-radius: 5px; margin: 10px 0; }
    .dq-success { background-color: #E8F5E9; border-left: 5px solid #4CAF50; padding: 15px; border-radius: 5px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# UTILITY HELPERS
# ─────────────────────────────────────────────

def dq_box(level: str, message: str) -> None:
    """Render a styled data-quality notification box."""
    css_map = {"info": "dq-info", "warning": "dq-warning",
               "error": "dq-error", "success": "dq-success"}
    css = css_map.get(level, "dq-info")
    st.markdown(f'<div class="{css}">{message}</div>', unsafe_allow_html=True)


def safe_filename(raw: str, fallback: str = "Output") -> str:
    """Strip OS-illegal characters from a filename component."""
    cleaned = re.sub(r'[\\/*?:"<>|]', "", raw).strip()
    return cleaned or fallback


def read_uploaded_file(uploaded, label: str = "file") -> pd.DataFrame | None:
    """
    Read a CSV or Excel uploaded file with encoding fallback.

    Returns a DataFrame or None if reading fails.
    """
    ext = uploaded.name.rsplit(".", 1)[-1].lower()
    if ext == "csv":
        for enc in ENCODINGS:
            try:
                uploaded.seek(0)
                return pd.read_csv(uploaded, encoding=enc)
            except UnicodeDecodeError:
                continue
            except pd.errors.EmptyDataError:
                dq_box("error", f"The {label} appears to be empty.")
                return None
            except pd.errors.ParserError as exc:
                dq_box("error", f"CSV parse error in {label}: {exc}")
                return None
        dq_box("error", f"Could not decode {label} with any known encoding.")
        return None
    elif ext in ("xlsx", "xls"):
        try:
            uploaded.seek(0)
            return pd.read_excel(uploaded)
        except Exception as exc:
            dq_box("error", f"Excel read error in {label}: {exc}")
            return None
    else:
        dq_box("error", f"Unsupported file type: {ext}")
        return None


def format_number(val, pct: bool = False) -> str:
    """Format numeric value for display."""
    if not isinstance(val, (int, float, np.integer, np.floating)):
        return str(val)
    if np.isnan(val):
        return "—"
    if pct:
        return f"{val:.2%}"
    return f"{val:,.2f}"


def generate_period_label(ap_idx: int, from_dt: pd.Timestamp, unit: str) -> str:
    """Return a human-readable accident-period label."""
    if unit == "Y":
        return str(from_dt.year + ap_idx)
    if unit == "Q":
        base_q = from_dt.year * 4 + (from_dt.month - 1) // 3
        total_q = base_q + ap_idx
        y, q = divmod(total_q, 4)
        return f"{y}-Q{q + 1}"
    # Monthly
    total_m = from_dt.year * 12 + from_dt.month + ap_idx
    y = (total_m - 1) // 12
    m = (total_m - 1) % 12 + 1
    return f"{y}-{m:02d}"


# ─────────────────────────────────────────────
# CORE ACTUARIAL FUNCTIONS
# ─────────────────────────────────────────────

def get_accident_period(loss_date: pd.Timestamp,
                        start_date: pd.Timestamp,
                        period_unit: str) -> int:
    """
    Assign an accident-period index (0-based) given granularity.

    Parameters
    ----------
    loss_date    : Date of loss
    start_date   : First date of the study window
    period_unit  : 'Y', 'Q', or 'M'

    Returns
    -------
    Integer period index (>= 0)
    """
    if period_unit == "M":
        return (loss_date.year - start_date.year) * 12 + (loss_date.month - start_date.month)
    if period_unit == "Q":
        return (loss_date.year * 4 + (loss_date.month - 1) // 3) - \
               (start_date.year * 4 + (start_date.month - 1) // 3)
    return loss_date.year - start_date.year


def get_development_period(loss_date: pd.Timestamp,
                           report_date: pd.Timestamp,
                           period_unit: str,
                           n_dev_periods: int) -> int:
    """
    Calculate the development-period lag (capped at n_dev_periods - 1).

    Parameters
    ----------
    loss_date      : Date of loss
    report_date    : Date claim was reported
    period_unit    : 'Y', 'Q', or 'M'
    n_dev_periods  : Total number of development periods in the triangle

    Returns
    -------
    Integer lag in [0, n_dev_periods - 1]
    """
    if period_unit == "M":
        lag = (report_date.year - loss_date.year) * 12 + (report_date.month - loss_date.month)
    elif period_unit == "Q":
        lag = (report_date.year * 4 + (report_date.month - 1) // 3) - \
              (loss_date.year * 4 + (loss_date.month - 1) // 3)
    else:
        lag = report_date.year - loss_date.year
    return max(0, min(lag, n_dev_periods - 1))


@st.cache_data(show_spinner=False)
def create_incremental_triangle(data: pd.DataFrame,
                                amount_col: str,
                                n_dev_periods: int,
                                max_accident_period: int) -> pd.DataFrame:
    """
    Pivot claim data into an incremental loss development triangle.

    Parameters
    ----------
    data               : DataFrame with 'Accident_Period' and 'Development_Period'
    amount_col         : Column name for claim amounts
    n_dev_periods      : Number of development periods
    max_accident_period: Highest accident-period index

    Returns
    -------
    DataFrame (rows = accident periods, columns = development periods)
    """
    inc = data.pivot_table(
        index="Accident_Period",
        columns="Development_Period",
        values=amount_col,
        aggfunc="sum",
        fill_value=0.0,
    )
    # Ensure all expected columns and rows exist
    for dev in range(n_dev_periods):
        if dev not in inc.columns:
            inc[dev] = 0.0
    for ap in range(max_accident_period + 1):
        if ap not in inc.index:
            inc.loc[ap] = 0.0

    inc = inc.reindex(sorted(inc.columns), axis=1).sort_index().fillna(0.0)
    return inc


def calculate_development_factors(cum_triangle: pd.DataFrame) -> list[float]:
    """
    Compute weighted-average link ratios from a cumulative triangle.

    The triangle is assumed to be upper-left complete (standard actuarial layout):
    row i has observed values up to column n_ay - 1 - i.

    Parameters
    ----------
    cum_triangle : Cumulative claims triangle (DataFrame)

    Returns
    -------
    List of development factors (length = n_dev - 1)
    """
    n_ay, n_dev = cum_triangle.shape
    dev_factors: list[float] = []

    for j in range(n_dev - 1):
        sum_curr = sum_next = 0.0
        for i in range(n_ay):
            # Only include cells where the next diagonal is observed
            if i + j + 1 < n_ay:
                curr = cum_triangle.iloc[i, j]
                nxt  = cum_triangle.iloc[i, j + 1]
                if curr > 0:
                    sum_curr += curr
                    sum_next += nxt
        dev_factors.append(sum_next / sum_curr if sum_curr > 0 else 1.0)

    return dev_factors


def calculate_cdfs(dev_factors: list[float]) -> list[float]:
    """
    Convert period-to-period development factors to CDFs (tail factors).

    Returns
    -------
    List of cumulative development factors, same length as dev_factors.
    cdfs[j] = product of all factors from period j onward.
    """
    cdfs: list[float] = []
    running = 1.0
    for f in reversed(dev_factors):
        running *= f
        cdfs.insert(0, running)
    return cdfs


def project_ultimate(cum_triangle: pd.DataFrame,
                     dev_factors: list[float]) -> pd.DataFrame:
    """
    Fill in the lower-right of a cumulative triangle using Chain Ladder factors.

    Parameters
    ----------
    cum_triangle : Upper-left complete cumulative triangle
    dev_factors  : Period-to-period development factors

    Returns
    -------
    Completed cumulative triangle (DataFrame)
    """
    n_ay, n_dev = cum_triangle.shape
    completed = cum_triangle.copy().astype(float)

    for i in range(n_ay):
        # Identify last observed development period for row i
        last_obs = next(
            (j for j in range(n_dev - 1, -1, -1) if i + j < n_ay),
            -1,
        )
        if last_obs == -1:
            continue
        for j in range(last_obs, n_dev - 1):
            if j < len(dev_factors):
                prev = completed.iloc[i, j]
                completed.iloc[i, j + 1] = prev * dev_factors[j] if prev > 0 else 0.0

    return completed


def calculate_chain_ladder_ibnr(cum_triangle: pd.DataFrame,
                                start_date: pd.Timestamp,
                                period_unit: str) -> dict:
    """
    Calculate IBNR using the Chain Ladder (development) method.

    Parameters
    ----------
    cum_triangle : Cumulative claims triangle
    start_date   : First accident period start date
    period_unit  : 'Y', 'Q', or 'M'

    Returns
    -------
    Dictionary with keys:
        results_df, total_ibnr, completed_triangle, dev_factors, cdfs
    """
    n_ay, n_dev = cum_triangle.shape
    dev_factors        = calculate_development_factors(cum_triangle)
    cdfs               = calculate_cdfs(dev_factors)
    completed_triangle = project_ultimate(cum_triangle, dev_factors)

    results: list[dict] = []
    total_ibnr = 0.0

    for i in range(n_ay):
        last_obs = next(
            (j for j in range(n_dev - 1, -1, -1) if i + j < n_ay),
            -1,
        )
        if last_obs == -1:
            continue

        current  = float(cum_triangle.iloc[i, last_obs])
        ultimate = float(completed_triangle.iloc[i, n_dev - 1])
        ibnr     = max(ultimate - current, 0.0)
        total_ibnr += ibnr

        results.append({
            "Accident_Period":       i,
            "Accident_Period_Label": generate_period_label(i, start_date, period_unit),
            "Current_Claims":        current,
            "CL_Ultimate":           ultimate,
            "CL_IBNR":               ibnr,
        })

    return {
        "results_df":          pd.DataFrame(results),
        "total_ibnr":          total_ibnr,
        "completed_triangle":  completed_triangle,
        "dev_factors":         dev_factors,
        "cdfs":                cdfs,
    }


def calculate_bf_ibnr(cum_triangle: pd.DataFrame,
                      dev_factors: list[float],
                      premiums: list[float],
                      elr: float,
                      start_date: pd.Timestamp,
                      period_unit: str) -> dict:
    """
    Calculate IBNR using the Bornhuetter-Ferguson (BF) method.

    Parameters
    ----------
    cum_triangle : Cumulative claims triangle
    dev_factors  : Development factors from Chain Ladder
    premiums     : Earned premium per accident period (length = n_ay)
    elr          : Expected Loss Ratio (decimal, e.g. 0.70)
    start_date   : First accident period start date
    period_unit  : 'Y', 'Q', or 'M'

    Returns
    -------
    Dictionary with keys:
        results_df, total_bf_ibnr, cdfs, pct_unreported
    """
    n_ay, n_dev  = cum_triangle.shape
    cdfs         = calculate_cdfs(dev_factors)
    pct_reported = [1.0 / cdf if cdf > 0 else 1.0 for cdf in cdfs]
    pct_unrep    = [1.0 - p for p in pct_reported]

    results: list[dict] = []
    total_bf_ibnr = 0.0

    for i in range(n_ay):
        last_obs = next(
            (j for j in range(n_dev - 1, -1, -1) if i + j < n_ay),
            -1,
        )
        if last_obs == -1:
            continue

        current           = float(cum_triangle.iloc[i, last_obs])
        expected_ultimate = premiums[i] * elr
        unrep_frac        = pct_unrep[last_obs] if last_obs < len(pct_unrep) else 0.0
        bf_ibnr           = expected_ultimate * unrep_frac
        bf_ultimate       = current + bf_ibnr
        pct_dev           = (1.0 - unrep_frac) * 100.0
        total_bf_ibnr    += bf_ibnr

        results.append({
            "Accident_Period":       i,
            "Accident_Period_Label": generate_period_label(i, start_date, period_unit),
            "Current_Claims":        current,
            "Premium":               premiums[i],
            "ELR":                   elr,
            "Expected_Ultimate":     expected_ultimate,
            "Pct_Developed":         pct_dev,
            "CDF_to_Ultimate":       cdfs[last_obs] if last_obs < len(cdfs) else 1.0,
            "BF_IBNR":               bf_ibnr,
            "BF_Ultimate":           bf_ultimate,
        })

    return {
        "results_df":     pd.DataFrame(results),
        "total_bf_ibnr":  total_bf_ibnr,
        "cdfs":           cdfs,
        "pct_unreported": pct_unrep,
    }


def validate_dataframe_columns(df: pd.DataFrame,
                               required: list[str]) -> tuple[bool, list[str]]:
    """
    Check that all required columns exist in df.

    Returns (all_present: bool, missing_cols: list[str])
    """
    missing = [c for c in required if c not in df.columns]
    return (len(missing) == 0, missing)


def run_data_quality_checks(df: pd.DataFrame,
                            loss_col: str,
                            report_col: str,
                            amount_cols: list[str],
                            from_date: pd.Timestamp,
                            to_date: pd.Timestamp) -> pd.DataFrame | None:
    """
    Perform comprehensive data quality checks and return a cleaned DataFrame.

    Returns None if a fatal error is encountered.
    """
    st.write("### Data Quality Checks")

    # ── Date parsing ──────────────────────────────────────────────────────
    try:
        df[loss_col]   = pd.to_datetime(df[loss_col],   errors="coerce")
        df[report_col] = pd.to_datetime(df[report_col], errors="coerce")
    except Exception as exc:
        dq_box("error", f"Date conversion failed: {exc}")
        return None

    invalid = df[loss_col].isna().sum() + df[report_col].isna().sum()
    if invalid > 0:
        dq_box("error", f"{invalid} unparseable date values found. "
               "Please ensure dates are in YYYY-MM-DD format and re-upload.")
        return None
    dq_box("success", "All dates parsed successfully.")

    # ── Drop null dates ───────────────────────────────────────────────────
    df = df.dropna(subset=[loss_col, report_col])

    # ── Future-date warnings ──────────────────────────────────────────────
    now = pd.Timestamp.now()
    future_loss   = (df[loss_col]   > now).sum()
    future_report = (df[report_col] > now).sum()
    if future_loss > 0:
        dq_box("warning", f"{future_loss} row(s) have a Loss Date in the future.")
    if future_report > 0:
        dq_box("warning", f"{future_report} row(s) have a Report Date in the future.")

    # ── Report before loss ────────────────────────────────────────────────
    bad_order = (df[report_col] < df[loss_col]).sum()
    if bad_order > 0:
        dq_box("error", f"{bad_order} row(s) have Report Date before Loss Date — excluded.")
        df = df[df[report_col] >= df[loss_col]]

    # ── Amount column checks ──────────────────────────────────────────────
    for col in amount_cols:
        missing_amt  = df[col].isna().sum()
        negative_amt = (df[col] < 0).sum()
        extreme_amt  = (df[col] > df[col].quantile(0.999)).sum()

        if missing_amt > 0:
            dq_box("warning", f'"{col}": {missing_amt} missing value(s) — rows will be excluded.')
        if negative_amt > 0:
            dq_box("warning", f'"{col}": {negative_amt} negative value(s) found.')
        if extreme_amt > 0:
            dq_box("info", f'"{col}": {extreme_amt} extreme value(s) above the 99.9th percentile detected.')

    # ── Duplicates ────────────────────────────────────────────────────────
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df = df.drop_duplicates()
        dq_box("info", f"Removed {dup_count} duplicate row(s).")
    else:
        dq_box("success", "No duplicate rows found.")

    # ── Date range filter ─────────────────────────────────────────────────
    mask       = (df[loss_col] >= from_date) & (df[loss_col] <= to_date)
    df_out     = df[mask].copy()
    excluded   = len(df) - len(df_out)

    if df_out.empty:
        dq_box("error", f"No records found between {from_date.date()} and {to_date.date()}.")
        return None

    dq_box("success", f"{len(df_out):,} records in selected period "
           f"({excluded:,} excluded outside range).")
    return df_out


# ─────────────────────────────────────────────
# PREMIUM VALIDATION HELPER
# ─────────────────────────────────────────────

def validate_premiums(premiums: list[float],
                      n_accident_periods: int) -> bool:
    """
    Validate that premiums list is correct length and contains no fatal values.

    Returns True if valid, False otherwise.
    """
    if len(premiums) != n_accident_periods:
        dq_box("error", f"Number of premiums ({len(premiums)}) does not match "
               f"number of accident periods ({n_accident_periods}).")
        return False

    nan_count  = sum(1 for p in premiums if pd.isna(p))
    zero_count = sum(1 for p in premiums if not pd.isna(p) and p <= 0)

    if nan_count > 0:
        dq_box("error", f"{nan_count} missing premium value(s) found.")
        return False
    if zero_count > 0:
        dq_box("warning", f"{zero_count} accident period(s) have zero or negative premiums.")

    return True


# ─────────────────────────────────────────────
# HEADER & HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="header">
    <div class="nav-links">
        <a href="#">Home</a>
        <a href="#">Services</a>
        <a href="#">Tools</a>
        <a href="#">Contact</a>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>Chain Ladder &amp; Bornhuetter-Ferguson IBNR Calculator</h1>
    <p>Upload your claims data, map columns, configure BF parameters,
       and calculate IBNR using both Chain Ladder and Bornhuetter-Ferguson methods.</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CLIENT INFORMATION
# ─────────────────────────────────────────────
st.markdown("""
<div class="gold-container">
    <h3>Client Information</h3>
    <p>Enter a client name for file tracking purposes.</p>
</div>
""", unsafe_allow_html=True)
client_name = st.text_input(
    "Client Name",
    value="Client",
    label_visibility="collapsed",
    help="Used to label the output Excel file.",
).strip()


# ─────────────────────────────────────────────
# IBNR PERIOD
# ─────────────────────────────────────────────
st.markdown("""
<div class="gold-container">
    <h3>IBNR Period</h3>
    <p>Select the accident-date range for claims to be included in the analysis.</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    from_date = st.date_input(
        "From Date",
        value=date(DEFAULT_FROM_YEAR, 1, 1),
        label_visibility="collapsed",
        help="Start of accident period window.",
    )
    st.caption("Start of period (Accident Date)")
with col2:
    to_date = st.date_input(
        "To Date",
        value=date(DEFAULT_TO_YEAR, 12, 31),
        label_visibility="collapsed",
        help="End of accident period window.",
    )
    st.caption("End of period (Accident Date)")

from_date = pd.to_datetime(from_date)
to_date   = pd.to_datetime(to_date)

if from_date >= to_date:
    st.error("From Date must be earlier than To Date.")
    st.stop()

st.info(f"**Selected Period:** {from_date.date()} to {to_date.date()}")


# ─────────────────────────────────────────────
# GRANULARITY
# ─────────────────────────────────────────────
st.markdown("""
<div class="gold-container">
    <h3>Triangle Granularity</h3>
    <p>Select the granularity for the claims development triangle.</p>
</div>
""", unsafe_allow_html=True)

granularity = st.selectbox(
    "Granularity",
    ["Yearly", "Quarterly", "Monthly"],
    index=0,
    label_visibility="collapsed",
    help="Determines how accident periods and development periods are bucketed.",
)

PERIOD_MAP = {"Monthly": "M", "Quarterly": "Q", "Yearly": "Y"}
period_unit = PERIOD_MAP[granularity]
period_name = granularity
st.caption(f"Selected: **{period_name}**")


# ─────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────
st.markdown("""
<div class="gold-container">
    <h3>Upload Claims Data File</h3>
    <p>Upload your claims data (CSV or Excel). The file must contain a Loss Date
       and a Report Date column.</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a claims file",
    type=ALLOWED_FILE_TYPES,
    label_visibility="collapsed",
    help="Supported formats: CSV, XLSX, XLS",
)

# ─────────────────────────────────────────────
# MAIN PROCESSING BLOCK
# ─────────────────────────────────────────────
if uploaded_file is not None:
    original_filename = uploaded_file.name
    base_filename     = re.sub(r"\.[^.]*$", "", original_filename)

    df = read_uploaded_file(uploaded_file, label="claims file")
    if df is None:
        st.stop()

    # Remove unnamed index columns generated by Excel/Colab exports
    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    st.write("### Preview of Uploaded Data")
    st.dataframe(df.head(10))
    st.caption(f"File contains **{len(df):,} rows** and **{len(df.columns)} columns**.")

    # ── Column Mapping ─────────────────────────────────────────────────────
    st.write("### Map Your Columns")
    all_cols = df.columns.tolist()

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            loss_date_col = st.selectbox(
                "Loss Date column",
                ["-- Select --"] + all_cols,
                help="Column containing the date of loss / accident date.",
            )
        with col2:
            report_date_col = st.selectbox(
                "Report Date column",
                ["-- Select --"] + all_cols,
                help="Column containing the date the claim was reported.",
            )

    if "-- Select --" in (loss_date_col, report_date_col):
        st.warning("Please select both date columns to continue.")
        st.stop()

    if loss_date_col == report_date_col:
        st.error("Loss Date and Report Date must be different columns.")
        st.stop()

    # ── Grouping Columns ───────────────────────────────────────────────────
    grouping_cols = st.multiselect(
        "Grouping columns (optional)",
        [c for c in all_cols if c not in (loss_date_col, report_date_col)],
        help="E.g. Line of Business, Product. Leave blank to analyse all data together.",
    )

    # ── Amount Columns ─────────────────────────────────────────────────────
    num_cols = df.select_dtypes(include=["float64", "int64", "float32", "int32"]).columns.tolist()
    amount_candidates = [
        c for c in num_cols
        if c not in (loss_date_col, report_date_col)
        and "date" not in c.lower()
    ]

    if not amount_candidates:
        st.error("No numeric columns found for claim amounts. "
                 "Please verify that the file contains numeric data.")
        st.stop()

    amount_cols = st.multiselect(
        "Select claim amount column(s)",
        amount_candidates,
        help="Incurred losses, paid losses, case reserves, etc.",
    )
    if not amount_cols:
        st.warning("Please select at least one amount column to continue.")
        st.stop()

    # ── BF Parameters ──────────────────────────────────────────────────────
    st.markdown("""
    <div class="gold-container">
        <h3>Bornhuetter-Ferguson Parameters</h3>
        <p>Configure the BF method — Expected Loss Ratio and Earned Premiums.</p>
    </div>
    """, unsafe_allow_html=True)

    enable_bf = st.checkbox(
        "Enable Bornhuetter-Ferguson Calculation",
        value=True,
        help="When enabled, IBNR is also estimated using the BF method alongside Chain Ladder.",
    )

    elr_input             = DEFAULT_ELR_PCT / 100.0
    premium_input_method  = "Manual Entry"
    premium_file          = None

    if enable_bf:
        col1, col2 = st.columns(2)
        with col1:
            elr_input = st.number_input(
                "Expected Loss Ratio (ELR) %",
                min_value=0.0,
                max_value=MAX_ELR_PCT,
                value=DEFAULT_ELR_PCT,
                step=1.0,
                help="A-priori loss ratio used in the BF method.",
            ) / 100.0
            st.caption(f"A-priori loss ratio: **{elr_input * 100:.1f}%**")
        with col2:
            premium_input_method = st.radio(
                "Premium Input Method",
                ["Manual Entry", "Upload Premium File"],
                horizontal=True,
            )

        if premium_input_method == "Upload Premium File":
            st.markdown("""
            <div class="gold-container">
                <h3>Upload Premium Data</h3>
                <p>Upload a CSV/Excel file with one row per accident period.
                   The file should contain a column with earned premiums.</p>
            </div>
            """, unsafe_allow_html=True)
            premium_file = st.file_uploader(
                "Upload Premium File",
                type=ALLOWED_FILE_TYPES,
                key="premium_upload",
                help="Must have exactly one row per accident period in chronological order.",
            )

    # ── DATA QUALITY ───────────────────────────────────────────────────────
    df_filtered = run_data_quality_checks(
        df.copy(),
        loss_date_col,
        report_date_col,
        amount_cols,
        from_date,
        to_date,
    )
    if df_filtered is None:
        st.stop()

    # ── TRIANGLE PARAMETERS ────────────────────────────────────────────────
    if period_unit == "M":
        n_dev_periods = (to_date.year - from_date.year) * 12 + (to_date.month - from_date.month) + 1
    elif period_unit == "Q":
        sq = from_date.year * 4 + (from_date.month - 1) // 3
        eq = to_date.year   * 4 + (to_date.month   - 1) // 3
        n_dev_periods = eq - sq + 1
    else:
        n_dev_periods = to_date.year - from_date.year + 1

    df_filtered["Accident_Period"] = df_filtered[loss_date_col].apply(
        lambda x: get_accident_period(x, from_date, period_unit)
    )
    df_filtered["Development_Period"] = df_filtered.apply(
        lambda row: get_development_period(
            row[loss_date_col], row[report_date_col], period_unit, n_dev_periods
        ),
        axis=1,
    )

    accident_periods     = sorted(df_filtered["Accident_Period"].unique())
    max_accident_period  = max(accident_periods)
    n_accident_periods   = max_accident_period + 1

    st.write("### Triangle Configuration")
    m1, m2, m3 = st.columns(3)
    m1.metric("Granularity",          period_name)
    m2.metric("Development Periods",  n_dev_periods)
    m3.metric("Accident Periods",     n_accident_periods)

    # ── PREMIUM COLLECTION ─────────────────────────────────────────────────
    premiums:           list[float] | None = None
    premium_data_valid: bool               = True

    if enable_bf:
        if premium_input_method == "Manual Entry":
            st.write(f"### Enter Premiums per Accident Period ({n_accident_periods} periods)")
            st.caption("Enter the earned premium for each accident period.")
            premiums = []
            n_cols   = min(MAX_PREMIUM_COLS, n_accident_periods)

            for row_start in range(0, n_accident_periods, n_cols):
                cols = st.columns(n_cols)
                for col_idx in range(n_cols):
                    ap_idx = row_start + col_idx
                    if ap_idx < n_accident_periods:
                        label = generate_period_label(ap_idx, from_date, period_unit)
                        with cols[col_idx]:
                            prem = st.number_input(
                                f"{label}",
                                min_value=0.0,
                                value=DEFAULT_PREMIUM,
                                step=PREMIUM_STEP,
                                format="%.0f",
                                key=f"premium_{ap_idx}",
                                help=f"Earned premium for accident period {label}.",
                            )
                            premiums.append(prem)

            premium_data_valid = validate_premiums(premiums, n_accident_periods)

        else:  # Upload Premium File
            if premium_file is not None:
                prem_df = read_uploaded_file(premium_file, label="premium file")
                if prem_df is None:
                    premium_data_valid = False
                else:
                    st.write("### Premium Data Preview")
                    st.dataframe(prem_df.head())

                    prem_num_cols = prem_df.select_dtypes(
                        include=["float64", "int64", "float32", "int32"]
                    ).columns.tolist()
                    if not prem_num_cols:
                        dq_box("error", "No numeric columns found in the premium file.")
                        premium_data_valid = False
                    else:
                        premium_amount_col = st.selectbox(
                            "Select Premium Amount Column",
                            prem_num_cols,
                            help="Column containing the earned premium values.",
                        )
                        premiums_raw = prem_df[premium_amount_col].tolist()
                        premium_data_valid = validate_premiums(premiums_raw, n_accident_periods)
                        if premium_data_valid:
                            premiums = [float(p) for p in premiums_raw]
                            dq_box("success", f"Premium data loaded — {len(premiums)} periods.")
            else:
                st.warning("Please upload a premium file to use the BF method.")
                premium_data_valid = False

    # ── CALCULATIONS ───────────────────────────────────────────────────────
    group_combinations = (
        df_filtered[grouping_cols].drop_duplicates()
        if grouping_cols
        else pd.DataFrame([{"All Data": "All"}])
    )

    all_cl_results:           list[pd.DataFrame]    = []
    all_bf_results:           list[pd.DataFrame]    = []
    all_incremental_triangles: dict[str, pd.DataFrame] = {}
    all_dev_factors:           dict[str, list]      = {}
    all_cdfs:                  dict[str, list]      = {}
    all_bf_details:            dict[str, dict]      = {}

    with st.spinner("Calculating IBNR — please wait..."):
        for _, group_row in group_combinations.iterrows():
            # Build group mask
            group_mask = pd.Series(True, index=df_filtered.index)
            for col in grouping_cols:
                group_mask &= df_filtered[col] == group_row[col]

            group_name = (
                " | ".join(str(group_row[c]) for c in grouping_cols)
                if grouping_cols else "All Data"
            )
            group_data = df_filtered[group_mask].copy()
            if group_data.empty:
                continue

            for amount_col in amount_cols:
                valid_data = group_data.dropna(subset=[amount_col])
                if valid_data.empty:
                    continue

                inc_triangle = create_incremental_triangle(
                    valid_data, amount_col, n_dev_periods, max_accident_period
                )
                cum_triangle = inc_triangle.cumsum(axis=1)

                key = f"{group_name} | {amount_col}"
                all_incremental_triangles[key] = inc_triangle

                # Chain Ladder
                cl_res = calculate_chain_ladder_ibnr(cum_triangle, from_date, period_unit)
                all_dev_factors[key] = cl_res["dev_factors"]
                all_cdfs[key]        = cl_res["cdfs"]

                cl_df = cl_res["results_df"].copy()
                for col in grouping_cols:
                    cl_df[col] = group_row[col]
                cl_df["Amount_Column"] = amount_col
                cl_df["Method"]        = "Chain Ladder"
                all_cl_results.append(cl_df)

                # BF
                if enable_bf and premium_data_valid and premiums is not None:
                    bf_res = calculate_bf_ibnr(
                        cum_triangle,
                        cl_res["dev_factors"],
                        premiums,
                        elr_input,
                        from_date,
                        period_unit,
                    )
                    bf_df = bf_res["results_df"].copy()
                    for col in grouping_cols:
                        bf_df[col] = group_row[col]
                    bf_df["Amount_Column"] = amount_col
                    bf_df["Method"]        = "Bornhuetter-Ferguson"
                    all_bf_results.append(bf_df)
                    all_bf_details[key] = bf_res

    # ── DISPLAY RESULTS ────────────────────────────────────────────────────
    st.write("## IBNR Results")

    if not all_cl_results:
        st.warning("No results generated. Please verify your data and column mapping.")
        st.stop()

    cl_combined = pd.concat(all_cl_results, ignore_index=True)
    id_cols     = (grouping_cols + ["Amount_Column"]) if grouping_cols else ["Amount_Column"]

    cl_summary = (
        cl_combined.groupby(id_cols)
        .agg(Current_Claims=("Current_Claims", "sum"),
             CL_Ultimate=("CL_Ultimate", "sum"),
             CL_IBNR=("CL_IBNR", "sum"))
        .reset_index()
    )
    cl_summary["CL_IBNR_Ratio"] = (
        cl_summary["CL_IBNR"] / cl_summary["CL_Ultimate"].replace(0, np.nan)
    ).fillna(0)

    if all_bf_results:
        bf_combined = pd.concat(all_bf_results, ignore_index=True)
        bf_summary = (
            bf_combined.groupby(id_cols)
            .agg(Premium=("Premium", "sum"),
                 Expected_Ultimate=("Expected_Ultimate", "sum"),
                 BF_IBNR=("BF_IBNR", "sum"),
                 BF_Ultimate=("BF_Ultimate", "sum"))
            .reset_index()
        )
        bf_summary["BF_IBNR_Ratio"] = (
            bf_summary["BF_IBNR"] / bf_summary["BF_Ultimate"].replace(0, np.nan)
        ).fillna(0)
        combined_summary = cl_summary.merge(bf_summary, on=id_cols, how="outer").fillna(0)
    else:
        combined_summary = cl_summary.copy()

    # Totals row
    total_row: dict = {c: "TOTAL" for c in id_cols}
    for num_col in ["Current_Claims", "CL_Ultimate", "CL_IBNR"]:
        if num_col in combined_summary.columns:
            total_row[num_col] = combined_summary[num_col].sum()
    total_row["CL_IBNR_Ratio"] = (
        total_row["CL_IBNR"] / total_row["CL_Ultimate"]
        if total_row.get("CL_Ultimate", 0) > 0 else 0
    )
    if all_bf_results:
        for num_col in ["Premium", "Expected_Ultimate", "BF_IBNR", "BF_Ultimate"]:
            if num_col in combined_summary.columns:
                total_row[num_col] = combined_summary[num_col].sum()
        total_row["BF_IBNR_Ratio"] = (
            total_row["BF_IBNR"] / total_row["BF_Ultimate"]
            if total_row.get("BF_Ultimate", 0) > 0 else 0
        )

    combined_summary = pd.concat(
        [combined_summary, pd.DataFrame([total_row])],
        ignore_index=True,
    )

    # Formatted display copy
    display_summary = combined_summary.copy()
    for col in display_summary.columns:
        if col in id_cols:
            continue
        if "Ratio" in col:
            display_summary[col] = display_summary[col].apply(
                lambda x: format_number(x, pct=True)
            )
        else:
            display_summary[col] = display_summary[col].apply(format_number)

    st.subheader("IBNR Summary — Chain Ladder vs Bornhuetter-Ferguson")
    st.dataframe(display_summary, use_container_width=True)

    # BF parameter summary
    if enable_bf and premium_data_valid and premiums:
        st.subheader("BF Parameters Used")
        b1, b2, b3 = st.columns(3)
        b1.metric("Expected Loss Ratio",      f"{elr_input * 100:.1f}%")
        b2.metric("Total Earned Premium",     f"{sum(premiums):,.0f}")
        b3.metric("Expected Ultimate (Total)", f"{sum(premiums) * elr_input:,.0f}")

    # Development factors expander
    with st.expander("Development Factors & CDFs"):
        for key, factors in all_dev_factors.items():
            st.write(f"**{key}**")
            c1, c2 = st.columns(2)
            with c1:
                st.write("Development Factors:")
                st.write([round(x, 4) for x in factors])
            with c2:
                if key in all_cdfs:
                    st.write("CDFs to Ultimate:")
                    st.write([round(x, 4) for x in all_cdfs[key]])
            if key in all_bf_details:
                pct = all_bf_details[key].get("pct_unreported", [])
                st.write("BF % Unreported:")
                st.write([f"{x * 100:.1f}%" for x in pct])
            st.write("---")

    # Detailed accident-period breakdown expander
    with st.expander("Detailed IBNR by Accident Period"):
        for key in all_incremental_triangles:
            st.write(f"**{key}**")
            amount_val = key.split(" | ")[-1]

            cl_detail = cl_combined[cl_combined["Amount_Column"] == amount_val].copy()
            if grouping_cols:
                group_val = key.split(" | ")[0]
                if group_val != "All Data":
                    for col in grouping_cols:
                        cl_detail = cl_detail[cl_detail[col].astype(str) == group_val]

            if cl_detail.empty:
                st.write("No detail available.")
                continue

            if all_bf_results:
                bf_detail = bf_combined[bf_combined["Amount_Column"] == amount_val].copy()
                if grouping_cols:
                    group_val = key.split(" | ")[0]
                    if group_val != "All Data":
                        for col in grouping_cols:
                            bf_detail = bf_detail[bf_detail[col].astype(str) == group_val]

                merged = cl_detail[
                    ["Accident_Period_Label", "Current_Claims", "CL_IBNR", "CL_Ultimate"]
                ].merge(
                    bf_detail[
                        ["Accident_Period_Label", "Premium", "Expected_Ultimate",
                         "Pct_Developed", "BF_IBNR", "BF_Ultimate"]
                    ],
                    on="Accident_Period_Label",
                    how="left",
                ).fillna(0)
                disp_detail = merged.copy()
            else:
                disp_detail = cl_detail[
                    ["Accident_Period_Label", "Current_Claims", "CL_IBNR", "CL_Ultimate"]
                ].copy()

            for col in disp_detail.columns:
                if col == "Accident_Period_Label":
                    continue
                if "Pct_" in col:
                    disp_detail[col] = disp_detail[col].apply(
                        lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else x
                    )
                else:
                    disp_detail[col] = disp_detail[col].apply(format_number)

            st.dataframe(disp_detail, use_container_width=True)
            st.write("---")

    # ── EXCEL DOWNLOAD ─────────────────────────────────────────────────────
    st.write("### Download Results")
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        combined_summary.to_excel(writer, index=False, sheet_name="IBNR_Summary")
        cl_combined.to_excel(writer,      index=False, sheet_name="CL_IBNR_Detail")

        if all_bf_results:
            bf_combined.to_excel(writer,  index=False, sheet_name="BF_IBNR_Detail")

        # Development factors sheet
        dev_rows = []
        for key, factors in all_dev_factors.items():
            for i, f in enumerate(factors):
                dev_rows.append({"Group_Amount": key, "Development_Period": i, "Development_Factor": f})
        pd.DataFrame(dev_rows).to_excel(writer, index=False, sheet_name="Development_Factors")

        # CDFs sheet
        cdf_rows = []
        for key, cdf_list in all_cdfs.items():
            for i, c in enumerate(cdf_list):
                cdf_rows.append({"Group_Amount": key, "Development_Period": i, "CDF_to_Ultimate": c})
        pd.DataFrame(cdf_rows).to_excel(writer, index=False, sheet_name="CDFs_to_Ultimate")

        # Incremental triangles sheet
        inc_parts = []
        for key, inc_tri in all_incremental_triangles.items():
            tmp = inc_tri.copy()
            tmp["Group_Amount"] = key
            inc_parts.append(tmp)
        if inc_parts:
            pd.concat(inc_parts).to_excel(writer, sheet_name="Incremental_Triangles")

        # Parameters sheet
        param_rows = [
            ("Start Date",        str(from_date.date())),
            ("End Date",          str(to_date.date())),
            ("Granularity",       period_name),
            ("Dev Periods",       n_dev_periods),
            ("Grouping Columns",  ", ".join(grouping_cols) if grouping_cols else "None"),
            ("Amount Columns",    ", ".join(amount_cols)),
        ]
        if enable_bf:
            param_rows += [
                ("BF Method",     "Bornhuetter-Ferguson"),
                ("ELR",           f"{elr_input * 100:.1f}%"),
                ("Premium Input", premium_input_method),
            ]
        pd.DataFrame(param_rows, columns=["Parameter", "Value"]).to_excel(
            writer, index=False, sheet_name="Parameters"
        )

    safe_client   = safe_filename(client_name,  "Client")
    safe_original = safe_filename(base_filename, "Data")
    filename      = f"{safe_client}_{safe_original}_CL_BF_IBNR.xlsx"

    st.download_button(
        "Download Complete IBNR Results (Excel)",
        data=output.getvalue(),
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <p>&copy; 2026 African Actuarial Consultants. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
