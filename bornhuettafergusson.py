# -*- coding: utf-8 -*-
"""
Bornhuetter-Ferguson IBNR Calculator
With per-group A-priori Loss Ratio
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import date
import re

st.set_page_config(page_title="Bornhuetter-Ferguson IBNR Calculator", layout="wide")

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
    .stApp { background-color: #FFFFFF; color: #000000; font-family: 'Calisto MT', serif; font-size: 11pt; }
    body, p, h1, h2, h3, h4, h5, h6, div, span, label, .stMarkdown, 
    .stTextInput label, .stDateInput label, .stSelectbox label, .stMultiSelect label,
    .stButton button, .stDownloadButton button, .stFileUploader label,
    .stAlert, .stInfo, .stWarning, .stError, .stSuccess, .stSpinner, 
    .stProgress, .stToast, .stSidebar, .stMetric {
        font-family: 'Calisto MT', serif !important;
    }
    .header {
        background-color: #000000;
        padding: 1rem 2rem;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        border-bottom: 3px solid #D4AF37;
    }
    .nav-links a {
        color: #FFFFFF;
        margin-left: 2rem;
        text-decoration: none;
        font-weight: 500;
        transition: color 0.3s;
    }
    .nav-links a:hover { color: #D4AF37; }
    .hero {
        background: linear-gradient(135deg, #000000 0%, #333333 100%);
        color: #FFFFFF;
        padding: 2rem 2rem;
        text-align: center;
        border-bottom: 3px solid #D4AF37;
    }
    .hero h1 { color: #D4AF37; font-size: 2.5rem; margin-bottom: 0.5rem; }
    .hero p { font-size: 1.2rem; max-width: 800px; margin: 0 auto; }
    .main-container { max-width: 1400px; margin: 2rem auto; padding: 0 2rem; }
    .required-container, .grouping-container, .date-range-container, .grain-container, .elr-container, .premium-container {
        background-color: #F9F9F9;
        border: 2px solid #D4AF37;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .required-container h3, .grouping-container h3, .date-range-container h3, .grain-container h3, .elr-container h3, .premium-container h3 {
        color: #D4AF37;
        margin-top: 0;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
        font-weight: bold;
    }
    .card {
        background-color: #F9F9F9;
        border: 1px solid #D4AF37;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .card h3 { color: #D4AF37; border-bottom: 2px solid #D4AF37; padding-bottom: 0.5rem; }
    .footer {
        background-color: #000000;
        color: #FFFFFF;
        text-align: center;
        padding: 1.5rem;
        border-top: 3px solid #D4AF37;
        margin-top: 3rem;
    }
    .footer a { color: #D4AF37; text-decoration: none; }
    .stButton > button, .stDownloadButton > button {
        background-color: #D4AF37;
        color: #000000;
        border: none;
        border-radius: 4px;
        font-weight: bold;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background-color: #B8960F;
        color: #FFFFFF;
    }
    .stFileUploader { border: 2px dashed #D4AF37; border-radius: 5px; padding: 1rem; }
    .stMultiSelect [data-baseweb="select"], .stSelectbox [data-baseweb="select"] {
        border: 1px solid #D4AF37;
        border-radius: 4px;
    }
    .dataframe { border: 1px solid #D4AF37; border-radius: 8px; overflow: hidden; }
    .data-check-container { background-color: #E3F2FD; border: 2px solid #2196F3; border-radius: 10px; padding: 1rem; margin-bottom: 1rem; }
    .data-check-warning { background-color: #FFF3E0; border: 2px solid #FF9800; border-radius: 10px; padding: 1rem; margin-bottom: 1rem; }
    .data-check-error { background-color: #FFEBEE; border: 2px solid #F44336; border-radius: 10px; padding: 1rem; margin-bottom: 1rem; }
    .elr-table-container { background-color: #F9F9F9; border: 2px solid #D4AF37; border-radius: 10px; padding: 1.5rem; margin: 1rem 0; }
    .stSelectbox div[data-baseweb="select"] { width: 100%; }
</style>
""", unsafe_allow_html=True)

# ---------- CORE FUNCTIONS ----------

def get_accident_period(date, start_date, period_unit):
    """Assign accident period number based on granularity"""
    if period_unit == 'M':
        months_diff = (date.year - start_date.year) * 12 + (date.month - start_date.month)
        return months_diff
    elif period_unit == 'Q':
        date_quarter = date.year * 4 + (date.month - 1) // 3
        start_quarter = start_date.year * 4 + (start_date.month - 1) // 3
        return date_quarter - start_quarter
    else:
        return date.year - start_date.year

def get_development_period(loss_date, report_date, period_unit, n_dev_periods):
    """Calculate development period (lag)"""
    if period_unit == 'M':
        lag = (report_date.year - loss_date.year) * 12 + (report_date.month - loss_date.month)
    elif period_unit == 'Q':
        report_quarter = report_date.year * 4 + (report_date.month - 1) // 3
        loss_quarter = loss_date.year * 4 + (loss_date.month - 1) // 3
        lag = report_quarter - loss_quarter
    else:
        lag = report_date.year - loss_date.year
    return max(0, min(lag, n_dev_periods - 1))

def create_incremental_triangle(data, amount_col, n_dev_periods, max_accident_period):
    """Create incremental triangle from data"""
    inc_triangle = data.pivot_table(
        index='Accident_Period',
        columns='Development_Period',
        values=amount_col,
        aggfunc='sum',
        fill_value=0
    )
    
    for dev in range(n_dev_periods):
        if dev not in inc_triangle.columns:
            inc_triangle[dev] = 0.0
    
    for ap in range(max_accident_period + 1):
        if ap not in inc_triangle.index:
            inc_triangle.loc[ap] = 0.0
    
    inc_triangle = inc_triangle.reindex(sorted(inc_triangle.columns), axis=1)
    inc_triangle = inc_triangle.sort_index()
    inc_triangle = inc_triangle.fillna(0.0)
    
    return inc_triangle

def calculate_development_factors(cum_triangle):
    """Calculate weighted average development factors"""
    n_ay, n_dev = cum_triangle.shape
    dev_factors = []

    for j in range(n_dev - 1):
        sum_next = 0.0
        sum_curr = 0.0

        for i in range(n_ay):
            if i + j + 1 < n_ay:
                curr = cum_triangle.iloc[i, j]
                nxt = cum_triangle.iloc[i, j+1]
                if curr > 0:
                    sum_next += nxt
                    sum_curr += curr

        if sum_curr > 0:
            dev_factors.append(sum_next / sum_curr)
        else:
            dev_factors.append(1.0)

    return dev_factors

def calculate_cdfs(dev_factors):
    """Convert development factors to cumulative development factors (CDF)"""
    cdfs = []
    running_product = 1.0
    for f in reversed(dev_factors):
        running_product *= f
        cdfs.insert(0, running_product)
    return cdfs

def calculate_pct_unpaid(cdfs):
    """Calculate % unpaid for each development period"""
    pct_unpaid = []
    for cdf in cdfs:
        if cdf > 0:
            pct_unpaid.append(1 - (1/cdf))
        else:
            pct_unpaid.append(0)
    return pct_unpaid

def calculate_bf_ibnr(cum_triangle, premiums, elr, start_date, period_unit):
    """Calculate IBNR using Bornhuetter-Ferguson method"""
    n_ay, n_dev = cum_triangle.shape
    
    dev_factors = calculate_development_factors(cum_triangle)
    cdfs = calculate_cdfs(dev_factors)
    pct_unpaid = calculate_pct_unpaid(cdfs)
    
    results = []
    total_bf_ibnr = 0.0
    
    for i in range(n_ay):
        last_obs = -1
        for j in range(n_dev - 1, -1, -1):
            if i + j < n_ay:
                last_obs = j
                break
        
        if last_obs == -1:
            continue
        
        current = cum_triangle.iloc[i, last_obs]
        expected_ultimate = premiums[i] * elr
        
        if last_obs < len(pct_unpaid):
            bf_ibnr = expected_ultimate * pct_unpaid[last_obs]
            pct_dev = (1 - pct_unpaid[last_obs]) * 100
            cdf_val = cdfs[last_obs]
        else:
            bf_ibnr = 0
            pct_dev = 100
            cdf_val = 1.0
        
        bf_ultimate = current + bf_ibnr
        total_bf_ibnr += bf_ibnr
        
        # Create readable label
        if period_unit == 'Y':
            acc_label = str(start_date.year + i)
        elif period_unit == 'Q':
            base_year = start_date.year
            base_quarter = (start_date.month - 1) // 3
            total_quarters = base_year * 4 + base_quarter + i
            year = total_quarters // 4
            quarter = total_quarters % 4 + 1
            acc_label = f"{year}-Q{quarter}"
        else:
            total_months = start_date.year * 12 + start_date.month + i
            year = (total_months - 1) // 12
            month = (total_months - 1) % 12 + 1
            acc_label = f"{year}-{month:02d}"
        
        results.append({
            'Accident_Period': i,
            'Accident_Period_Label': acc_label,
            'Current_Claims': current,
            'Premium': premiums[i],
            'ELR': elr,
            'Expected_Ultimate': expected_ultimate,
            'Last_Observed_Dev': last_obs,
            'CDF_to_Ultimate': cdf_val,
            'Pct_Developed': pct_dev,
            'BF_IBNR': bf_ibnr,
            'BF_Ultimate': bf_ultimate
        })
    
    return {
        'results_df': pd.DataFrame(results),
        'total_bf_ibnr': total_bf_ibnr,
        'dev_factors': dev_factors,
        'cdfs': cdfs,
        'pct_unpaid': pct_unpaid
    }

# ---------- Header ----------
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

# ---------- Hero ----------
st.markdown("""
<div class="hero">
    <h1>Bornhuetter-Ferguson IBNR Calculator</h1>
    <p>Upload claims and premium data. Set a-priori loss ratios per group and calculate IBNR using the Bornhuetter-Ferguson method.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Main Container ----------
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Client name
col1, col2 = st.columns(2)
with col1:
    client_name = st.text_input("Client Name (for file name)", value="Client").strip()
with col2:
    pass

# IBNR Period
st.markdown("""
<div class="date-range-container">
    <h3>IBNR Period</h3>
    <p>Select date range based on Loss Date</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    from_date = st.date_input("From Date", value=date(2021, 1, 1))
    st.caption("Claims with Loss Date on or after this date")
with col2:
    to_date = st.date_input("To Date", value=date(2025, 12, 31))
    st.caption("Claims with Loss Date on or before this date")

from_date = pd.to_datetime(from_date)
to_date = pd.to_datetime(to_date)

st.info(f"**Selected IBNR Period:** {from_date.date()} to {to_date.date()}")

# Grain Selection
st.markdown("""
<div class="grain-container">
    <h3>Triangle Grain</h3>
    <p>Select time unit for grouping periods</p>
</div>
""", unsafe_allow_html=True)

grain_map = {'Yearly': 'Y', 'Quarterly': 'Q', 'Monthly': 'M'}
grain_label = st.selectbox("Select Grain:", options=list(grain_map.keys()), index=0)
grain = grain_map[grain_label]

# Claims File Upload
st.markdown("""
<div class="required-container">
    <h3>Upload Claims Data</h3>
    <p>CSV or Excel with Loss_Date, Report_Date, and claim amounts</p>
</div>
""", unsafe_allow_html=True)

claims_file = st.file_uploader("Choose claims file", type=["csv", "xlsx", "xls"], key="claims_upload")

# Premium File Upload
st.markdown("""
<div class="premium-container">
    <h3>Upload Premium Data</h3>
    <p>CSV or Excel with premium amounts. Rows must match the number of accident periods.</p>
</div>
""", unsafe_allow_html=True)

premium_file = st.file_uploader("Choose premium file", type=["csv", "xlsx", "xls"], key="premium_upload")

if claims_file is not None:
    try:
        original_filename = claims_file.name
        base_filename = re.sub(r'\.[^.]*$', '', original_filename)

        # Read claims file
        ext = claims_file.name.split('.')[-1].lower()
        if ext == 'csv':
            try:
                df = pd.read_csv(claims_file, encoding='utf-8')
            except:
                claims_file.seek(0)
                df = pd.read_csv(claims_file, encoding='cp1252')
        else:
            df = pd.read_excel(claims_file)

        # Clean column names
        df.columns = df.columns.astype(str).str.strip()
        
        # Drop unnamed columns
        unnamed = [c for c in df.columns if c.lower().startswith('unnamed')]
        if unnamed:
            df = df.drop(columns=unnamed)
            st.info(f"Removed {len(unnamed)} unnamed column(s).")

        st.markdown("#### Preview of uploaded claims data")
        st.dataframe(df.head())
        st.markdown("---")

        # --- COLUMN MAPPING ---
        st.markdown("### Map Your Columns")
        all_cols = df.columns.tolist()

        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="required-container">
                <h3>Loss_Date</h3>
                <p>Date when loss occurred</p>
            </div>
            """, unsafe_allow_html=True)
            loss_col = st.selectbox("Loss Date column", options=[""] + all_cols, label_visibility="collapsed")
        
        with col2:
            st.markdown("""
            <div class="required-container">
                <h3>Report_Date</h3>
                <p>Date when claim was reported</p>
            </div>
            """, unsafe_allow_html=True)
            report_col = st.selectbox("Report Date column", options=[""] + all_cols, label_visibility="collapsed")

        if not loss_col or not report_col:
            st.error("Please select Loss Date and Report Date columns.")
            st.stop()

        st.markdown("---")

        # Grouping Column
        st.markdown("""
        <div class="grouping-container">
            <h3>Aggregation Column</h3>
            <p>Select the column to group by (e.g., Line_of_Business). ELR will be set per unique group.</p>
        </div>
        """, unsafe_allow_html=True)
        
        group_options = [c for c in all_cols if c not in [loss_col, report_col]]
        aggregation_col = st.selectbox("Select aggregation column:", options=[""] + group_options)
        
        if not aggregation_col:
            st.error("Please select an aggregation column.")
            st.stop()

        st.markdown("---")

        # Numeric Columns
        st.markdown("### Select Numeric Columns (Claim Amounts)")
        
        num_options = [c for c in all_cols if c not in [loss_col, report_col, aggregation_col]]
        
        if not num_options:
            st.error("No numeric columns found.")
            st.stop()
        
        value_cols = st.multiselect("Select claim amount columns:", options=num_options)
        
        if not value_cols:
            st.error("Please select at least one claim amount column.")
            st.stop()

        st.write(f"Aggregation column: **{aggregation_col}**")
        st.write(f"Claim amount columns: **{', '.join(value_cols)}**")

        # --- PROCESS DATA ---
        df[loss_col] = pd.to_datetime(df[loss_col], errors='coerce')
        df[report_col] = pd.to_datetime(df[report_col], errors='coerce')
        
        df_filtered = df[(df[loss_col] >= from_date) & (df[loss_col] <= to_date)].copy()
        
        if df_filtered.empty:
            st.error("No data for selected period.")
            st.stop()
        
        st.success(f"✅ Filtered: {len(df_filtered)} claims")

        # --- DATA QUALITY CHECKS ---
        st.markdown("### Data Quality Checks")
        
        # Check missing values
        missing = []
        for col in [loss_col, report_col, aggregation_col] + value_cols:
            cnt = df_filtered[col].isna().sum()
            if cnt > 0:
                missing.append(f"{col} ({cnt})")
        
        if missing:
            st.markdown(f'<div class="data-check-error">❌ Missing values: {", ".join(missing)}</div>', unsafe_allow_html=True)
            st.stop()
        
        # Check Report Date before Loss Date
        invalid = df_filtered[df_filtered[report_col] < df_filtered[loss_col]]
        if len(invalid) > 0:
            st.markdown(f'<div class="data-check-error">❌ {len(invalid)} rows with Report Date before Loss Date</div>', unsafe_allow_html=True)
            st.stop()
        
        # Check and remove duplicates
        dup_count = df_filtered.duplicated().sum()
        if dup_count > 0:
            df_filtered = df_filtered.drop_duplicates()
            st.markdown(f'<div class="data-check-warning">⚠️ Removed {dup_count} duplicate rows</div>', unsafe_allow_html=True)
        
        # Clean numeric columns
        for col in value_cols:
            if df_filtered[col].dtype == 'object':
                cleaned = df_filtered[col].astype(str).str.replace(r'[$,€£]', '', regex=True)
                cleaned = cleaned.str.replace(r',', '', regex=False)
                cleaned = cleaned.str.replace(r'^\((.+)\)$', r'-\1', regex=True)
                cleaned = cleaned.str.strip().replace('', '0')
                df_filtered[col] = pd.to_numeric(cleaned, errors='coerce').fillna(0)
        
        st.markdown('<div class="data-check-container">✅ Data quality checks passed</div>', unsafe_allow_html=True)
        st.markdown("---")

        # ====================================================================
        # GET UNIQUE GROUPS AND SET ELR PER GROUP
        # ====================================================================
        
        unique_groups = sorted(df_filtered[aggregation_col].unique())
        
        st.markdown("""
        <div class="elr-container">
            <h3>A-priori Loss Ratio (ELR) per Group</h3>
            <p>Set the expected loss ratio for each unique group in the aggregation column</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="elr-table-container">', unsafe_allow_html=True)
        
        # Create ELR input for each unique group
        elr_dict = {}
        
        # Display as a nice table layout
        cols_per_row = 4
        for i in range(0, len(unique_groups), cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                idx = i + j
                if idx < len(unique_groups):
                    group_name = unique_groups[idx]
                    with cols[j]:
                        st.markdown(f"**{group_name}**")
                        elr_val = st.number_input(
                            f"ELR %",
                            min_value=0.0,
                            max_value=200.0,
                            value=70.0,
                            step=1.0,
                            key=f"elr_{group_name}",
                            label_visibility="collapsed"
                        )
                        elr_dict[group_name] = elr_val / 100
                        st.caption(f"{elr_val:.0f}%")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show ELR summary
        st.write("### ELR Summary")
        elr_summary_df = pd.DataFrame({
            aggregation_col: list(elr_dict.keys()),
            'ELR': [f"{v*100:.1f}%" for v in elr_dict.values()]
        })
        st.dataframe(elr_summary_df, use_container_width=True)

        # ====================================================================
        # CALCULATE TRIANGLE PARAMETERS
        # ====================================================================
        
        if grain == 'M':
            total_months = (to_date.year - from_date.year) * 12 + (to_date.month - from_date.month)
            n_dev_periods = total_months + 1
        elif grain == 'Q':
            start_quarter = from_date.year * 4 + (from_date.month - 1) // 3
            end_quarter = to_date.year * 4 + (to_date.month - 1) // 3
            total_quarters = end_quarter - start_quarter
            n_dev_periods = total_quarters + 1
        else:
            total_years = to_date.year - from_date.year
            n_dev_periods = total_years + 1

        # Create accident and development periods
        df_filtered['Accident_Period'] = df_filtered[loss_col].apply(
            lambda x: get_accident_period(x, from_date, grain)
        )
        df_filtered['Development_Period'] = df_filtered.apply(
            lambda row: get_development_period(row[loss_col], row[report_col], grain, n_dev_periods), 
            axis=1
        )

        accident_periods = sorted(df_filtered['Accident_Period'].unique())
        max_accident_period = max(accident_periods)
        n_accident_periods = max_accident_period + 1

        st.write(f"### Triangle Configuration")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Grain", grain_label)
        with col2:
            st.metric("Development Periods", n_dev_periods)
        with col3:
            st.metric("Accident Periods", n_accident_periods)

        # ====================================================================
        # PROCESS PREMIUM FILE
        # ====================================================================
        
        premiums = None
        premium_data_valid = False
        
        if premium_file is not None:
            st.write("### Premium Data")
            
            try:
                # Read premium file
                prem_ext = premium_file.name.split('.')[-1].lower()
                if prem_ext == 'csv':
                    try:
                        prem_df = pd.read_csv(premium_file, encoding='utf-8')
                    except:
                        premium_file.seek(0)
                        prem_df = pd.read_csv(premium_file, encoding='cp1252')
                else:
                    prem_df = pd.read_excel(premium_file)
                
                # Clean column names
                prem_df.columns = prem_df.columns.astype(str).str.strip()
                
                # Drop unnamed columns
                prem_unnamed = [c for c in prem_df.columns if c.lower().startswith('unnamed')]
                if prem_unnamed:
                    prem_df = prem_df.drop(columns=prem_unnamed)
                
                st.write("Preview of Premium Data:")
                st.dataframe(prem_df.head())
                
                # Select premium column
                prem_cols = prem_df.columns.tolist()
                premium_amount_col = st.selectbox(
                    "Select Premium Amount Column",
                    options=prem_cols
                )
                
                # Validate row count
                if len(prem_df) != n_accident_periods:
                    st.markdown(f'<div class="data-check-error">❌ Premium file has {len(prem_df)} rows but {n_accident_periods} accident periods are required.</div>', unsafe_allow_html=True)
                else:
                    # Clean premium column if needed
                    if prem_df[premium_amount_col].dtype == 'object':
                        cleaned_prem = prem_df[premium_amount_col].astype(str).str.replace(r'[$,€£]', '', regex=True)
                        cleaned_prem = cleaned_prem.str.replace(r',', '', regex=False)
                        cleaned_prem = cleaned_prem.str.strip().replace('', '0')
                        premiums = pd.to_numeric(cleaned_prem, errors='coerce').fillna(0).tolist()
                    else:
                        premiums = prem_df[premium_amount_col].tolist()
                    
                    # Validate premiums
                    missing_prems = sum(1 for p in premiums if pd.isna(p))
                    if missing_prems > 0:
                        st.markdown(f'<div class="data-check-error">❌ {missing_prems} missing premium values found</div>', unsafe_allow_html=True)
                    else:
                        zero_or_neg = sum(1 for p in premiums if p <= 0)
                        if zero_or_neg > 0:
                            st.markdown(f'<div class="data-check-warning">⚠️ {zero_or_neg} periods have zero or negative premiums</div>', unsafe_allow_html=True)
                        
                        premium_data_valid = True
                        st.markdown(f'<div class="data-check-container">✅ Premium data loaded: {len(premiums)} periods, Total Premium: {sum(premiums):,.2f}</div>', unsafe_allow_html=True)
            
            except Exception as e:
                st.markdown(f'<div class="data-check-error">❌ Error reading premium file: {e}</div>', unsafe_allow_html=True)
        else:
            st.info("Please upload a premium file to proceed with BF calculation.")

        # ====================================================================
        # RUN BF CALCULATION FOR EACH GROUP
        # ====================================================================
        
        if premium_data_valid and premiums is not None:
            all_bf_results = []
            all_incremental_triangles = {}
            all_cumulative_triangles = {}
            all_dev_factors = {}
            all_cdfs = {}
            all_pct_unpaid = {}

            with st.spinner("Running Bornhuetter-Ferguson calculation..."):
                for group_name in unique_groups:
                    # Get ELR for this group
                    group_elr = elr_dict[group_name]
                    
                    # Filter data for this group
                    group_data = df_filtered[df_filtered[aggregation_col] == group_name].copy()
                    
                    if len(group_data) == 0:
                        continue
                    
                    for amount_col in value_cols:
                        valid_data = group_data.dropna(subset=[amount_col])
                        
                        if len(valid_data) == 0:
                            continue
                        
                        # Create triangles
                        inc_triangle = create_incremental_triangle(valid_data, amount_col, n_dev_periods, max_accident_period)
                        cum_triangle = inc_triangle.cumsum(axis=1)
                        
                        triangle_key = f"{group_name} | {amount_col}"
                        all_incremental_triangles[triangle_key] = inc_triangle
                        all_cumulative_triangles[triangle_key] = cum_triangle
                        
                        # Run BF with group-specific ELR
                        result = calculate_bf_ibnr(cum_triangle, premiums, group_elr, from_date, grain)
                        
                        all_dev_factors[triangle_key] = result['dev_factors']
                        all_cdfs[triangle_key] = result['cdfs']
                        all_pct_unpaid[triangle_key] = result['pct_unpaid']
                        
                        # Add grouping info
                        result_df = result['results_df'].copy()
                        result_df[aggregation_col] = group_name
                        result_df['Amount_Column'] = amount_col
                        
                        all_bf_results.append(result_df)

            st.success("✅ BF Model fitted successfully!")

            # ====================================================================
            # DISPLAY RESULTS
            # ====================================================================
            
            if all_bf_results:
                combined_results = pd.concat(all_bf_results, ignore_index=True)
                
                # Create summary
                summary_cols = [aggregation_col, 'Amount_Column']
                
                bf_summary = combined_results.groupby(summary_cols).agg({
                    'Current_Claims': 'sum',
                    'Premium': 'sum',
                    'Expected_Ultimate': 'sum',
                    'BF_IBNR': 'sum',
                    'BF_Ultimate': 'sum',
                    'ELR': 'first'
                }).reset_index()
                
                bf_summary['BF_IBNR_Ratio'] = bf_summary['BF_IBNR'] / bf_summary['BF_Ultimate']
                bf_summary['BF_IBNR_Ratio'] = bf_summary['BF_IBNR_Ratio'].fillna(0)

                # Display results header
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader(f"BF IBNR Results for Period: {from_date.date()} to {to_date.date()}")
                st.markdown(f"**Grain:** {grain_label}")
                st.markdown(f"**Aggregation:** {aggregation_col}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display summary table
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("BF IBNR Summary")
                display_summary = bf_summary.copy()
                for col in display_summary.columns:
                    if col not in summary_cols:
                        if 'Ratio' in col:
                            display_summary[col] = display_summary[col].apply(lambda x: f"{x:.2%}" if isinstance(x, (int, float)) else x)
                        elif 'ELR' in col:
                            display_summary[col] = display_summary[col].apply(lambda x: f"{x*100:.1f}%" if isinstance(x, (int, float)) else x)
                        else:
                            display_summary[col] = display_summary[col].apply(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x)
                st.dataframe(display_summary, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Development Factors & CDFs
                with st.expander("📈 Development Factors & CDFs"):
                    for key in all_dev_factors.keys():
                        st.write(f"**{key}**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write("Dev Factors:")
                            st.write([round(x, 4) for x in all_dev_factors[key]])
                        with col2:
                            st.write("CDFs to Ultimate:")
                            st.write([round(x, 4) for x in all_cdfs[key]])
                        with col3:
                            st.write("% Unpaid:")
                            st.write([f"{x*100:.1f}%" for x in all_pct_unpaid[key]])
                        st.write("---")
                
                # Detailed IBNR by Accident Period
                with st.expander("📋 Detailed BF IBNR by Accident Period"):
                    for key in all_incremental_triangles.keys():
                        st.write(f"**{key}**")
                        parts = key.split(' | ')
                        group_val = parts[0]
                        amount_val = parts[-1]
                        
                        detail = combined_results[
                            (combined_results[aggregation_col] == group_val) &
                            (combined_results['Amount_Column'] == amount_val)
                        ]
                        
                        if len(detail) > 0:
                            display_cols = ['Accident_Period_Label', 'Current_Claims', 'Premium', 
                                          'Expected_Ultimate', 'ELR', 'Pct_Developed', 'BF_IBNR', 'BF_Ultimate']
                            display_detail = detail[display_cols].copy()
                            for col in display_detail.columns:
                                if col == 'Accident_Period_Label':
                                    continue
                                elif 'Pct_' in col:
                                    display_detail[col] = display_detail[col].apply(lambda x: f"{x:.1f}%")
                                elif 'ELR' in col:
                                    display_detail[col] = display_detail[col].apply(lambda x: f"{x*100:.1f}%")
                                else:
                                    display_detail[col] = display_detail[col].apply(lambda x: f"{x:,.2f}")
                            st.dataframe(display_detail, use_container_width=True)
                        st.write("---")

                # ====================================================================
                # EXPORT TO EXCEL
                # ====================================================================
                
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # BF IBNR Summary
                    bf_summary.to_excel(writer, index=False, sheet_name='BF_IBNR_Summary')
                    
                    # Detailed BF IBNR
                    combined_results.to_excel(writer, index=False, sheet_name='BF_IBNR_Detail')
                    
                    # ELR by Group
                    elr_summary_df.to_excel(writer, index=False, sheet_name='ELR_By_Group')
                    
                    # Incremental Triangles
                    inc_combined = pd.DataFrame()
                    for key, inc_triangle in all_incremental_triangles.items():
                        temp = inc_triangle.copy()
                        temp['Group_Amount'] = key
                        inc_combined = pd.concat([inc_combined, temp])
                    inc_combined.to_excel(writer, sheet_name='Incremental_Triangles')
                    
                    # Cumulative Triangles
                    cum_combined = pd.DataFrame()
                    for key, cum_triangle in all_cumulative_triangles.items():
                        temp = cum_triangle.copy()
                        temp['Group_Amount'] = key
                        cum_combined = pd.concat([cum_combined, temp])
                    cum_combined.to_excel(writer, sheet_name='Cumulative_Triangles')
                    
                    # Development Factors
                    dev_factors_df = pd.DataFrame()
                    for key, factors in all_dev_factors.items():
                        temp_df = pd.DataFrame({
                            'Group_Amount': key,
                            'Development_Period': range(len(factors)),
                            'Development_Factor': factors
                        })
                        dev_factors_df = pd.concat([dev_factors_df, temp_df], ignore_index=True)
                    dev_factors_df.to_excel(writer, index=False, sheet_name='Development_Factors')
                    
                    # CDFs
                    cdfs_df = pd.DataFrame()
                    for key, cdfs in all_cdfs.items():
                        temp_df = pd.DataFrame({
                            'Group_Amount': key,
                            'Development_Period': range(len(cdfs)),
                            'CDF_to_Ultimate': cdfs
                        })
                        cdfs_df = pd.concat([cdfs_df, temp_df], ignore_index=True)
                    cdfs_df.to_excel(writer, index=False, sheet_name='CDFs_to_Ultimate')
                    
                    # % Unpaid
                    pct_unpaid_df = pd.DataFrame()
                    for key, pcts in all_pct_unpaid.items():
                        temp_df = pd.DataFrame({
                            'Group_Amount': key,
                            'Development_Period': range(len(pcts)),
                            'Pct_Unpaid': pcts
                        })
                        pct_unpaid_df = pd.concat([pct_unpaid_df, temp_df], ignore_index=True)
                    pct_unpaid_df.to_excel(writer, index=False, sheet_name='Pct_Unpaid')
                    
                    # Premiums Used
                    premiums_df = pd.DataFrame({
                        'Accident_Period': range(n_accident_periods),
                        'Premium': premiums
                    })
                    # Add expected ultimate per group
                    for group_name, group_elr in elr_dict.items():
                        premiums_df[f'Expected_Ultimate_{group_name}'] = [p * group_elr for p in premiums]
                    premiums_df.to_excel(writer, index=False, sheet_name='Premiums_Used')
                    
                    # Parameters
                    params_df = pd.DataFrame({
                        'Parameter': ['Start Date', 'End Date', 'Grain', 'Dev Periods', 
                                     'Accident Periods', 'Aggregation Column', 'Amount Columns'],
                        'Value': [
                            str(from_date.date()), str(to_date.date()), grain_label, n_dev_periods,
                            n_accident_periods, aggregation_col, ', '.join(value_cols)
                        ]
                    })
                    params_df.to_excel(writer, index=False, sheet_name='Parameters')
                
                output.seek(0)
                
                safe_client = re.sub(r'[\\/*?:"<>|]', "", client_name).strip() or "Client"
                safe_original = re.sub(r'[\\/*?:"<>|]', "", base_filename).strip() or "Data"
                file_name = f"{safe_client}_{safe_original}_BF_IBNR_{from_date.year}_{to_date.year}.xlsx"
                
                st.download_button("📥 Download Excel Report", data=output, file_name=file_name)

        else:
            if not premium_data_valid:
                st.warning("Please upload a valid premium file to run the BF calculation.")

    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        st.write(traceback.format_exc())

st.markdown('</div>', unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown("""
<div class="footer">
    <p>© 2026 African Actuarial Consultants. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
