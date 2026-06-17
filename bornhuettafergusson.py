# -*- coding: utf-8 -*-
"""
Bornhuetter-Ferguson IBNR Calculator
Flow: Claims Data → Column Mapping → ELR per Group → Premiums per Group → Calculate
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
    .stTextInput label, .stDateInput label, .stSelectbox label, .stMultiselect label,
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
    .section-container {
        background-color: #F9F9F9;
        border: 2px solid #D4AF37;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .section-container h3 {
        color: #D4AF37;
        margin-top: 0;
        margin-bottom: 1rem;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .section-container p {
        color: #666666;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
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
    div[data-testid="stFileUploader"] {
        border: 2px dashed #D4AF37;
        border-radius: 10px;
        padding: 1rem;
        background-color: #FAFAFA;
    }
    div[data-testid="stFileUploader"] section {
        border: none !important;
    }
    .stSelectbox [data-baseweb="select"], .stMultiselect [data-baseweb="select"] {
        border: 1px solid #D4AF37;
        border-radius: 4px;
    }
    .dataframe { border: 1px solid #D4AF37; border-radius: 8px; overflow: hidden; }
    .data-check-info { background-color: #E3F2FD; border: 2px solid #2196F3; border-radius: 10px; padding: 1rem; margin-bottom: 1rem; }
    .data-check-warning { background-color: #FFF3E0; border: 2px solid #FF9800; border-radius: 10px; padding: 1rem; margin-bottom: 1rem; }
    .data-check-error { background-color: #FFEBEE; border: 2px solid #F44336; border-radius: 10px; padding: 1rem; margin-bottom: 1rem; }
    .data-check-success { background-color: #E8F5E9; border: 2px solid #4CAF50; border-radius: 10px; padding: 1rem; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# ---------- CORE FUNCTIONS ----------

def get_accident_period(date, start_date, period_unit):
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
    cdfs = []
    running_product = 1.0
    for f in reversed(dev_factors):
        running_product *= f
        cdfs.insert(0, running_product)
    return cdfs

def calculate_pct_unpaid(cdfs):
    pct_unpaid = []
    for cdf in cdfs:
        if cdf > 0:
            pct_unpaid.append(1 - (1/cdf))
        else:
            pct_unpaid.append(0)
    return pct_unpaid

def calculate_bf_ibnr(cum_triangle, premiums, elr, start_date, period_unit):
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

# ---------- HEADER & HERO ----------
st.markdown("""
<div class="header">
    <div class="nav-links">
        <a href="#">Home</a>
        <a href="#">Services</a>
        <a href="#">Tools</a>
        <a href="#">Contact</a>
    </div>
</div>
<div class="hero">
    <h1>Bornhuetter-Ferguson IBNR Calculator</h1>
    <p>Upload claims data → Map columns → Set ELR per group → Upload premiums → Calculate IBNR</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="main-container">', unsafe_allow_html=True)

# ---------- CLIENT NAME ----------
col1, col2 = st.columns(2)
with col1:
    client_name = st.text_input("Client Name (for file name)", value="Client").strip()

# ---------- STEP 1: DATE RANGE & GRAIN ----------
st.markdown("""
<div class="section-container">
    <h3>Step 1: IBNR Period & Grain</h3>
    <p>Select the date range and triangle granularity</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    from_date = st.date_input("From Date (Accident Date)", value=date(2021, 1, 1))
with col2:
    to_date = st.date_input("To Date (Accident Date)", value=date(2025, 12, 31))

grain_map = {'Yearly': 'Y', 'Quarterly': 'Q', 'Monthly': 'M'}
grain_label = st.selectbox("Triangle Grain:", options=list(grain_map.keys()), index=0)
grain = grain_map[grain_label]

from_date_dt = pd.to_datetime(from_date)
to_date_dt = pd.to_datetime(to_date)
st.info(f"Period: **{from_date}** to **{to_date}** | Grain: **{grain_label}**")

# ---------- STEP 2: UPLOAD CLAIMS DATA ----------
st.markdown("""
<div class="section-container">
    <h3>Step 2: Upload Claims Data</h3>
    <p>Upload CSV or Excel file with Loss_Date, Report_Date, grouping column, and claim amounts</p>
</div>
""", unsafe_allow_html=True)

claims_file = st.file_uploader("Choose claims file", type=["csv", "xlsx", "xls"], key="claims_upload", label_visibility="visible")

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
        unnamed = [c for c in df.columns if c.lower().startswith('unnamed')]
        if unnamed:
            df = df.drop(columns=unnamed)

        st.markdown("**Preview of claims data:**")
        st.dataframe(df.head(5), use_container_width=True)

        # ---------- STEP 3: COLUMN MAPPING ----------
        st.markdown("""
        <div class="section-container">
            <h3>Step 3: Map Your Columns</h3>
            <p>Identify the required columns from your data</p>
        </div>
        """, unsafe_allow_html=True)
        
        all_cols = df.columns.tolist()

        col1, col2 = st.columns(2)
        with col1:
            loss_col = st.selectbox("Loss Date Column:", options=[""] + all_cols, key="loss_col")
        with col2:
            report_col = st.selectbox("Report Date Column:", options=[""] + all_cols, key="report_col")

        if not loss_col or not report_col:
            st.warning("Please select both date columns.")
            st.stop()

        # Grouping column
        group_options = [c for c in all_cols if c not in [loss_col, report_col]]
        aggregation_col = st.selectbox("Aggregation Column (e.g., Line_of_Business):", options=[""] + group_options, key="agg_col")

        if not aggregation_col:
            st.warning("Please select an aggregation column.")
            st.stop()

        # Numeric columns
        num_options = [c for c in all_cols if c not in [loss_col, report_col, aggregation_col]]
        if not num_options:
            st.error("No numeric columns found for claim amounts.")
            st.stop()

        value_cols = st.multiselect("Claim Amount Columns:", options=num_options, key="value_cols")

        if not value_cols:
            st.warning("Please select at least one claim amount column.")
            st.stop()

        # ---------- PROCESS DATA ----------
        df[loss_col] = pd.to_datetime(df[loss_col], errors='coerce')
        df[report_col] = pd.to_datetime(df[report_col], errors='coerce')
        
        df_filtered = df[(df[loss_col] >= from_date_dt) & (df[loss_col] <= to_date_dt)].copy()
        
        if df_filtered.empty:
            st.error("No data for selected period.")
            st.stop()

        # ---------- DATA QUALITY CHECKS ----------
        st.markdown("### Data Quality Checks")
        
        # Missing values
        missing = []
        for col in [loss_col, report_col, aggregation_col] + value_cols:
            cnt = df_filtered[col].isna().sum()
            if cnt > 0:
                missing.append(f"{col} ({cnt})")
        
        if missing:
            st.markdown(f'<div class="data-check-error">❌ Missing values: {", ".join(missing)}</div>', unsafe_allow_html=True)
            st.stop()

        # Report before Loss
        invalid = df_filtered[df_filtered[report_col] < df_filtered[loss_col]]
        if len(invalid) > 0:
            st.markdown(f'<div class="data-check-error">❌ {len(invalid)} rows with Report Date before Loss Date</div>', unsafe_allow_html=True)
            st.stop()

        # Duplicates
        dup_count = df_filtered.duplicated().sum()
        if dup_count > 0:
            df_filtered = df_filtered.drop_duplicates()
            st.markdown(f'<div class="data-check-warning">⚠️ Removed {dup_count} duplicate rows</div>', unsafe_allow_html=True)

        # Clean numeric columns
        for col in value_cols:
            if df_filtered[col].dtype == 'object':
                cleaned = df_filtered[col].astype(str).str.replace(r'[$,€£]', '', regex=True)
                cleaned = cleaned.str.replace(r',', '', regex=False)
                cleaned = cleaned.str.strip().replace('', '0')
                df_filtered[col] = pd.to_numeric(cleaned, errors='coerce').fillna(0)

        st.markdown(f'<div class="data-check-success">✅ {len(df_filtered)} valid records ready for analysis</div>', unsafe_allow_html=True)

        # ---------- GET UNIQUE GROUPS ----------
        unique_groups = sorted(df_filtered[aggregation_col].dropna().unique())
        
        st.markdown(f"**Aggregation Column:** `{aggregation_col}` → **{len(unique_groups)} unique groups:** {', '.join([str(g) for g in unique_groups])}")

        # ---------- CALCULATE TRIANGLE PARAMETERS ----------
        if grain == 'M':
            total_months = (to_date_dt.year - from_date_dt.year) * 12 + (to_date_dt.month - from_date_dt.month)
            n_dev_periods = total_months + 1
        elif grain == 'Q':
            start_quarter = from_date_dt.year * 4 + (from_date_dt.month - 1) // 3
            end_quarter = to_date_dt.year * 4 + (to_date_dt.month - 1) // 3
            n_dev_periods = end_quarter - start_quarter + 1
        else:
            n_dev_periods = to_date_dt.year - from_date_dt.year + 1

        df_filtered['Accident_Period'] = df_filtered[loss_col].apply(lambda x: get_accident_period(x, from_date_dt, grain))
        df_filtered['Development_Period'] = df_filtered.apply(lambda row: get_development_period(row[loss_col], row[report_col], grain, n_dev_periods), axis=1)

        max_accident_period = df_filtered['Accident_Period'].max()
        n_accident_periods = max_accident_period + 1

        st.info(f"Triangle: **{n_accident_periods}** Accident Periods × **{n_dev_periods}** Development Periods")

        # ---------- STEP 4: SET ELR PER GROUP ----------
        st.markdown("""
        <div class="section-container">
            <h3>Step 4: Set A-priori Loss Ratio (ELR) per Group</h3>
            <p>Enter the expected loss ratio for each unique group</p>
        </div>
        """, unsafe_allow_html=True)
        
        elr_dict = {}
        cols = st.columns(min(3, len(unique_groups)))
        
        for i, group_name in enumerate(unique_groups):
            with cols[i % 3]:
                st.markdown(f"**{group_name}**")
                elr_val = st.number_input(
                    "ELR %",
                    min_value=0.0,
                    max_value=200.0,
                    value=70.0,
                    step=1.0,
                    key=f"elr_{group_name}",
                    label_visibility="collapsed"
                )
                elr_dict[group_name] = elr_val / 100

        # Show ELR summary
        elr_summary_df = pd.DataFrame({
            aggregation_col: list(elr_dict.keys()),
            'ELR': [f"{v*100:.1f}%" for v in elr_dict.values()]
        })
        st.dataframe(elr_summary_df, use_container_width=True)

        # ---------- STEP 5: UPLOAD PREMIUM DATA ----------
        st.markdown("""
        <div class="section-container">
            <h3>Step 5: Upload Premium Data</h3>
            <p>Upload CSV or Excel with premium amounts. File must have columns matching the unique groups and rows equal to the number of accident periods.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"**Required:** `{n_accident_periods}` rows (one per accident period) and columns for each group: `{', '.join([str(g) for g in unique_groups])}`")

        premium_file = st.file_uploader("Choose premium file", type=["csv", "xlsx", "xls"], key="premium_upload", label_visibility="visible")

        if premium_file is not None:
            try:
                prem_ext = premium_file.name.split('.')[-1].lower()
                if prem_ext == 'csv':
                    try:
                        prem_df = pd.read_csv(premium_file, encoding='utf-8')
                    except:
                        premium_file.seek(0)
                        prem_df = pd.read_csv(premium_file, encoding='cp1252')
                else:
                    prem_df = pd.read_excel(premium_file)

                prem_df.columns = prem_df.columns.astype(str).str.strip()
                prem_unnamed = [c for c in prem_df.columns if c.lower().startswith('unnamed')]
                if prem_unnamed:
                    prem_df = prem_df.drop(columns=prem_unnamed)

                st.markdown("**Preview of premium data:**")
                st.dataframe(prem_df.head(), use_container_width=True)

                # Map premium columns to groups
                st.markdown("**Map premium columns to groups:**")
                prem_cols = prem_df.columns.tolist()
                
                group_premium_map = {}
                prem_mapping_cols = st.columns(min(3, len(unique_groups)))
                
                for i, group_name in enumerate(unique_groups):
                    with prem_mapping_cols[i % 3]:
                        default_idx = i if i < len(prem_cols) else 0
                        selected_col = st.selectbox(
                            f"Premium column for **{group_name}**",
                            options=prem_cols,
                            index=min(default_idx, len(prem_cols)-1),
                            key=f"prem_map_{group_name}"
                        )
                        group_premium_map[group_name] = selected_col

                # Validate row count
                if len(prem_df) != n_accident_periods:
                    st.markdown(f'<div class="data-check-error">❌ Premium file has {len(prem_df)} rows but {n_accident_periods} accident periods are required.</div>', unsafe_allow_html=True)
                else:
                    # Extract and validate premiums
                    premiums_dict = {}
                    all_valid = True
                    
                    for group_name, prem_col in group_premium_map.items():
                        if prem_df[prem_col].dtype == 'object':
                            cleaned = prem_df[prem_col].astype(str).str.replace(r'[$,€£]', '', regex=True)
                            cleaned = cleaned.str.replace(r',', '', regex=False)
                            cleaned = cleaned.str.strip().replace('', '0')
                            premiums = pd.to_numeric(cleaned, errors='coerce').fillna(0).tolist()
                        else:
                            premiums = prem_df[prem_col].tolist()
                        
                        missing_p = sum(1 for p in premiums if pd.isna(p))
                        if missing_p > 0:
                            st.markdown(f'<div class="data-check-error">❌ {missing_p} missing values in premium column for {group_name}</div>', unsafe_allow_html=True)
                            all_valid = False
                        else:
                            premiums_dict[group_name] = premiums
                    
                    if all_valid:
                        st.markdown(f'<div class="data-check-success">✅ Premium data validated for all {len(unique_groups)} groups</div>', unsafe_allow_html=True)

                # ---------- STEP 6: RUN BF CALCULATION ----------
                if all_valid:
                    st.markdown("""
                    <div class="section-container">
                        <h3>Step 6: Calculate BF IBNR</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("🚀 Run Bornhuetter-Ferguson Calculation", use_container_width=True):
                        all_bf_results = []
                        all_incremental_triangles = {}
                        all_cumulative_triangles = {}
                        all_dev_factors = {}
                        all_cdfs = {}
                        all_pct_unpaid = {}

                        with st.spinner("Calculating BF IBNR for all groups..."):
                            for group_name in unique_groups:
                                group_elr = elr_dict[group_name]
                                group_premiums = premiums_dict[group_name]
                                
                                group_data = df_filtered[df_filtered[aggregation_col] == group_name].copy()
                                
                                if len(group_data) == 0:
                                    continue
                                
                                for amount_col in value_cols:
                                    valid_data = group_data.dropna(subset=[amount_col])
                                    
                                    if len(valid_data) == 0:
                                        continue
                                    
                                    inc_triangle = create_incremental_triangle(valid_data, amount_col, n_dev_periods, max_accident_period)
                                    cum_triangle = inc_triangle.cumsum(axis=1)
                                    
                                    triangle_key = f"{group_name} | {amount_col}"
                                    all_incremental_triangles[triangle_key] = inc_triangle
                                    all_cumulative_triangles[triangle_key] = cum_triangle
                                    
                                    result = calculate_bf_ibnr(cum_triangle, group_premiums, group_elr, from_date_dt, grain)
                                    
                                    all_dev_factors[triangle_key] = result['dev_factors']
                                    all_cdfs[triangle_key] = result['cdfs']
                                    all_pct_unpaid[triangle_key] = result['pct_unpaid']
                                    
                                    result_df = result['results_df'].copy()
                                    result_df[aggregation_col] = group_name
                                    result_df['Amount_Column'] = amount_col
                                    
                                    all_bf_results.append(result_df)

                        if all_bf_results:
                            st.success("✅ BF Calculation Complete!")
                            
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
                            
                            bf_summary['BF_IBNR_Ratio'] = (bf_summary['BF_IBNR'] / bf_summary['BF_Ultimate']).fillna(0)
                            
                            # Add TOTAL row
                            total_row = {aggregation_col: 'TOTAL', 'Amount_Column': 'TOTAL'}
                            total_row['Current_Claims'] = bf_summary['Current_Claims'].sum()
                            total_row['Premium'] = bf_summary['Premium'].sum()
                            total_row['Expected_Ultimate'] = bf_summary['Expected_Ultimate'].sum()
                            total_row['BF_IBNR'] = bf_summary['BF_IBNR'].sum()
                            total_row['BF_Ultimate'] = bf_summary['BF_Ultimate'].sum()
                            total_row['BF_IBNR_Ratio'] = total_row['BF_IBNR'] / total_row['BF_Ultimate'] if total_row['BF_Ultimate'] > 0 else 0
                            total_row['ELR'] = ''
                            
                            bf_summary = pd.concat([bf_summary, pd.DataFrame([total_row])], ignore_index=True)

                            # Display Results
                            st.markdown('<div class="card">', unsafe_allow_html=True)
                            st.subheader(f"BF IBNR Summary: {from_date} to {to_date} | {grain_label}")
                            
                            display_summary = bf_summary.copy()
                            for col in display_summary.columns:
                                if col not in summary_cols:
                                    if 'Ratio' in col:
                                        display_summary[col] = display_summary[col].apply(lambda x: f"{x:.2%}" if isinstance(x, (int, float)) else x)
                                    elif 'ELR' in col:
                                        display_summary[col] = display_summary[col].apply(lambda x: f"{x*100:.1f}%" if isinstance(x, (int, float)) else x)
                                    else:
                                        display_summary[col] = display_summary[col].apply(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x)
                            
                            st.dataframe(display_summary, use_container_width=True, hide_index=True)
                            st.markdown('</div>', unsafe_allow_html=True)

                            # Detailed Results
                            with st.expander("📋 Detailed BF IBNR by Accident Period"):
                                for group_name in unique_groups:
                                    for amount_col in value_cols:
                                        st.write(f"**{group_name} - {amount_col}**")
                                        detail = combined_results[
                                            (combined_results[aggregation_col] == group_name) &
                                            (combined_results['Amount_Column'] == amount_col)
                                        ]
                                        if len(detail) > 0:
                                            display_cols = ['Accident_Period_Label', 'Current_Claims', 'Premium', 
                                                          'Expected_Ultimate', 'Pct_Developed', 'BF_IBNR', 'BF_Ultimate']
                                            display_detail = detail[display_cols].copy()
                                            for c in display_detail.columns:
                                                if c == 'Accident_Period_Label':
                                                    continue
                                                elif 'Pct_' in c:
                                                    display_detail[c] = display_detail[c].apply(lambda x: f"{x:.1f}%")
                                                else:
                                                    display_detail[c] = display_detail[c].apply(lambda x: f"{x:,.2f}")
                                            st.dataframe(display_detail, use_container_width=True, hide_index=True)
                                        st.write("---")

                            # Development Factors
                            with st.expander("📈 Development Factors & CDFs"):
                                for key in all_dev_factors:
                                    st.write(f"**{key}**")
                                    c1, c2, c3 = st.columns(3)
                                    with c1:
                                        st.write("Dev Factors:")
                                        st.write([round(x, 4) for x in all_dev_factors[key]])
                                    with c2:
                                        st.write("CDFs to Ultimate:")
                                        st.write([round(x, 4) for x in all_cdfs[key]])
                                    with c3:
                                        st.write("% Unpaid:")
                                        st.write([f"{x*100:.1f}%" for x in all_pct_unpaid[key]])
                                    st.write("---")

                            # Export to Excel
                            output = BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                bf_summary.to_excel(writer, index=False, sheet_name='BF_IBNR_Summary')
                                combined_results.to_excel(writer, index=False, sheet_name='BF_IBNR_Detail')
                                
                                inc_combined = pd.DataFrame()
                                for key, inc_triangle in all_incremental_triangles.items():
                                    temp = inc_triangle.copy()
                                    temp['Group_Amount'] = key
                                    inc_combined = pd.concat([inc_combined, temp])
                                inc_combined.to_excel(writer, sheet_name='Incremental_Triangles')
                                
                                cum_combined = pd.DataFrame()
                                for key, cum_triangle in all_cumulative_triangles.items():
                                    temp = cum_triangle.copy()
                                    temp['Group_Amount'] = key
                                    cum_combined = pd.concat([cum_combined, temp])
                                cum_combined.to_excel(writer, sheet_name='Cumulative_Triangles')
                                
                                dev_factors_df = pd.DataFrame()
                                for key, factors in all_dev_factors.items():
                                    temp_df = pd.DataFrame({'Group_Amount': key, 'Dev_Period': range(len(factors)), 'Dev_Factor': factors})
                                    dev_factors_df = pd.concat([dev_factors_df, temp_df])
                                dev_factors_df.to_excel(writer, index=False, sheet_name='Development_Factors')
                                
                                cdfs_df = pd.DataFrame()
                                for key, cdfs in all_cdfs.items():
                                    temp_df = pd.DataFrame({'Group_Amount': key, 'Dev_Period': range(len(cdfs)), 'CDF_to_Ultimate': cdfs})
                                    cdfs_df = pd.concat([cdfs_df, temp_df])
                                cdfs_df.to_excel(writer, index=False, sheet_name='CDFs_to_Ultimate')
                                
                                pct_unpaid_df = pd.DataFrame()
                                for key, pcts in all_pct_unpaid.items():
                                    temp_df = pd.DataFrame({'Group_Amount': key, 'Dev_Period': range(len(pcts)), 'Pct_Unpaid': pcts})
                                    pct_unpaid_df = pd.concat([pct_unpaid_df, temp_df])
                                pct_unpaid_df.to_excel(writer, index=False, sheet_name='Pct_Unpaid')
                                
                                params_df = pd.DataFrame({
                                    'Parameter': ['Start Date', 'End Date', 'Grain', 'Dev Periods', 'Accident Periods', 'Aggregation Column', 'Amount Columns'],
                                    'Value': [str(from_date), str(to_date), grain_label, n_dev_periods, n_accident_periods, aggregation_col, ', '.join(value_cols)]
                                })
                                params_df.to_excel(writer, index=False, sheet_name='Parameters')
                            
                            output.seek(0)
                            
                            safe_client = re.sub(r'[\\/*?:"<>|]', "", client_name).strip() or "Client"
                            safe_original = re.sub(r'[\\/*?:"<>|]', "", base_filename).strip() or "Data"
                            file_name = f"{safe_client}_{safe_original}_BF_IBNR_{from_date.year}_{to_date.year}.xlsx"
                            
                            st.download_button("📥 Download Excel Report", data=output, file_name=file_name, use_container_width=True)

            except Exception as e:
                st.error(f"Error processing premium file: {str(e)}")

    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        st.write(traceback.format_exc())

st.markdown('</div>', unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("""
<div class="footer">
    <p>© 2026 African Actuarial Consultants. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
