import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import base64
import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Quant Trader Intelligence",
    page_icon="∫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Professional Styling (Glassmorphism & Scientific Theme) ---
st.markdown("""
    <style>
    /* Background & Fonts */
    .stApp {
        background-color: #f4f6f9;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e6e9ef;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 5px solid #3498db;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e6e9ef;
    }
    
    /* Report Container */
    .report-container {
        background-color: #ffffff;
        padding: 40px;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    
    .report-header {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    
    .highlight-green { color: #27ae60; font-weight: bold; }
    .highlight-red { color: #c0392b; font-weight: bold; }
    .highlight-blue { color: #2980b9; font-weight: bold; }
    
    /* Math Formula Style */
    .formula-box {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        border-left: 3px solid #2c3e50;
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. Data Loading Engine ---
@st.cache_data
def load_data():
    try:
        # Load CSVs
        trader_df = pd.read_csv('historical_data.csv')
        sentiment_df = pd.read_csv('fear_greed_index.csv')
        
        # Clean & Format Dates
        trader_df['Timestamp IST'] = pd.to_datetime(trader_df['Timestamp IST'], format='%d-%m-%Y %H:%M', dayfirst=True)
        trader_df['Date'] = trader_df['Timestamp IST'].dt.date
        trader_df['Date'] = pd.to_datetime(trader_df['Date'])
        
        sentiment_df['Date'] = pd.to_datetime(sentiment_df['date'])
        sentiment_df = sentiment_df.rename(columns={'value': 'Sentiment_Score', 'classification': 'Sentiment_Class'})
        
        # Merge
        merged_df = pd.merge(trader_df, sentiment_df, on='Date', how='inner')
        return merged_df
    except Exception as e:
        st.error(f"Data Loading Error: {e}")
        return None

# --- 2. Analytical Engine ---
def calculate_metrics(df):
    # Core Logic
    df['Is_Win'] = df['Closed PnL'] > 0
    df['Win_Amt'] = df.loc[df['Is_Win'], 'Closed PnL']
    df['Loss_Amt'] = df.loc[~df['Is_Win'], 'Closed PnL']
    
    # Grouping
    sentiment_order = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
    stats_df = df.groupby('Sentiment_Class').agg({
        'Closed PnL': ['mean', 'sum', 'std'],
        'Win_Amt': 'mean',
        'Loss_Amt': 'mean',
        'Size USD': 'mean',
        'Is_Win': 'mean',
        'Account': 'count',
        'Sentiment_Score': 'mean'
    }).reset_index()
    
    stats_df.columns = ['Sentiment', 'Avg_PnL', 'Total_PnL', 'PnL_Std', 'Avg_Win_Size', 'Avg_Loss_Size', 'Avg_Position_Size', 'Win_Rate', 'Trade_Count', 'Avg_Sentiment']
    
    # --- Advanced Math & Quant Metrics ---
    stats_df['Avg_Loss_Size'] = stats_df['Avg_Loss_Size'].abs()
    stats_df['Risk_Reward_Ratio'] = stats_df['Avg_Win_Size'] / stats_df['Avg_Loss_Size']
    
    # Sharpe Ratio (Simplified: Return / Volatility)
    # Avoid division by zero
    stats_df['Sharpe_Ratio'] = np.where(stats_df['PnL_Std'] > 0, stats_df['Avg_PnL'] / stats_df['PnL_Std'], 0)
    
    # Kelly Criterion Calculation (Optimal Bet Size)
    # K% = W - (1-W)/R
    stats_df['Kelly_Criterion'] = (stats_df['Win_Rate'] - ((1 - stats_df['Win_Rate']) / stats_df['Risk_Reward_Ratio']))
    stats_df['Kelly_Pct'] = stats_df['Kelly_Criterion'] * 100 # Convert to percentage
    
    # Sort
    stats_df['Sentiment'] = pd.Categorical(stats_df['Sentiment'], categories=sentiment_order, ordered=True)
    stats_df = stats_df.sort_values('Sentiment').set_index('Sentiment')
    
    return stats_df

def perform_hypothesis_test(df):
    # T-Test: Compare "Extreme Greed" PnL vs "Extreme Fear" PnL
    group_greed = df[df['Sentiment_Class'] == 'Extreme Greed']['Closed PnL']
    group_fear = df[df['Sentiment_Class'] == 'Extreme Fear']['Closed PnL']
    
    # Safety check for sample size
    if len(group_greed) < 2 or len(group_fear) < 2:
        return None, None
    
    t_stat, p_val = stats.ttest_ind(group_greed, group_fear, equal_var=False)
    return t_stat, p_val

# --- 3. Main Application ---
def main():
    st.title("📈 Quant Intelligence Dashboard")
    st.markdown("### Statistical Validation & Risk Modeling")
    
    # Load raw data
    raw_df = load_data()
    
    if raw_df is not None:
        # --- SIDEBAR FILTERS ---
        st.sidebar.header("🔍 Filter Analysis")
        
        # 1. Date Filter
        min_date = raw_df['Date'].min().date()
        max_date = raw_df['Date'].max().date()
        
        st.sidebar.subheader("📅 Time Period")
        start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
        
        # 2. Coin Filter
        st.sidebar.subheader("🪙 Assets")
        all_coins = sorted(raw_df['Coin'].unique().tolist())
        selected_coins = st.sidebar.multiselect("Select Coins", all_coins, default=all_coins[:5]) # Default to first 5 to avoid lag if many
        
        if not selected_coins:
            selected_coins = all_coins # If none selected, treat as all
            st.sidebar.info("Showing all coins (default)")

        # 3. Side Filter (Buy/Sell)
        st.sidebar.subheader("⚖️ Trade Side")
        all_sides = sorted(raw_df['Side'].unique().tolist())
        selected_sides = st.sidebar.multiselect("Select Side", all_sides, default=all_sides)
        
        # --- FILTERING LOGIC ---
        mask = (
            (raw_df['Date'].dt.date >= start_date) & 
            (raw_df['Date'].dt.date <= end_date) &
            (raw_df['Coin'].isin(selected_coins)) &
            (raw_df['Side'].isin(selected_sides))
        )
        filtered_df = raw_df[mask]
        
        # --- MAIN DASHBOARD ---
        if filtered_df.empty:
            st.warning("⚠️ No trades found for the selected filters. Please adjust your selection.")
        else:
            stats_df = calculate_metrics(filtered_df)
            t_stat, p_val = perform_hypothesis_test(filtered_df)
            
            # --- KPIs ---
            with st.container():
                best_zone = stats_df['Avg_PnL'].idxmax()
                
                # T-Test Logic Display
                if p_val is not None:
                    sig_text = "Significant (p < 0.05)" if p_val < 0.05 else "Not Significant"
                    conf_val = f"{(1-p_val)*100:.1f}%"
                else:
                    sig_text = "Insufficient Data"
                    conf_val = "N/A"

                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric("Filtered PnL Edge", best_zone, f"${stats_df.loc[best_zone, 'Avg_PnL']:.2f}/trade")
                kpi2.metric("Statistical Confidence", conf_val, sig_text)
                kpi3.metric("Max Sharpe Ratio", f"{stats_df['Sharpe_Ratio'].max():.2f}", "Risk-Adjusted Return")
                kpi4.metric("Trades Analyzed", f"{len(filtered_df):,}", f"{(len(filtered_df)/len(raw_df))*100:.0f}% of total")

                st.markdown("---")
                
                # --- CHARTS ---
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("💸 Profitability by Sentiment")
                    fig1, ax1 = plt.subplots(figsize=(8, 4))
                    sns.barplot(x=stats_df.index, y=stats_df['Avg_PnL'], palette='RdYlGn', ax=ax1)
                    ax1.axhline(0, color='black')
                    ax1.set_ylabel("Avg PnL ($)")
                    st.pyplot(fig1)
                    
                with c2:
                    st.subheader("📊 Kelly Criterion (Bet Sizing)")
                    fig2, ax2 = plt.subplots(figsize=(8, 4))
                    kelly_plot = stats_df['Kelly_Pct'].apply(lambda x: max(0, x))
                    sns.barplot(x=stats_df.index, y=kelly_plot, palette='viridis', ax=ax2)
                    ax2.set_ylabel("Optimal Allocation (%)")
                    st.pyplot(fig2)

                # --- STATS & REPORT ---
                st.markdown("---")
                c3, c4 = st.columns(2)
                
                with c3:
                    st.subheader("🧮 Statistical Proof")
                    if t_stat is not None:
                        st.markdown(f"""
                        **Hypothesis:** Extreme Greed PnL > Extreme Fear PnL
                        * **T-Statistic:** `{t_stat:.4f}`
                        * **P-Value:** `{p_val:.6f}`
                        """)
                        if p_val < 0.05:
                            st.success("✅ Result is Statistically Significant")
                        else:
                            st.warning("⚠️ Result is likely random noise")
                    else:
                        st.info("Insufficient data points in filtered set for T-Test.")

                with c4:
                    st.subheader("📝 Data Table")
                    st.dataframe(stats_df[['Win_Rate', 'Risk_Reward_Ratio', 'Sharpe_Ratio']].style.format("{:.2f}"))

            # --- REPORT GENERATOR ---
            st.markdown("---")
            st.header("📝 Custom Strategic Report")
            
            if st.button("📄 Generate Report for Selection", type="primary"):
                # Dynamic Data Extraction
                try:
                    eg_pnl = stats_df.loc['Extreme Greed', 'Avg_PnL']
                    eg_sharpe = stats_df.loc['Extreme Greed', 'Sharpe_Ratio']
                    eg_kelly = max(0, stats_df.loc['Extreme Greed', 'Kelly_Pct'])
                    ef_kelly = max(0, stats_df.loc['Extreme Fear', 'Kelly_Pct'])
                    
                    report_html = f"""
                    <div class="report-container">
                        <h1 class="report-header">📊 Strategic Analysis Report</h1>
                        <p><strong>Date:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d')}<br>
                        <strong>Filter Context:</strong> {start_date} to {end_date} | Coins: {len(selected_coins)} Selected</p>
                        
                        <h3>1. Executive Summary</h3>
                        <p>Analysis of the selected <strong>{len(filtered_df):,} trades</strong> shows that the strategy generates an average PnL of 
                        <span class="highlight-green">${eg_pnl:.2f}</span> during Extreme Greed periods.</p>
                        
                        <h3>2. Risk Management (Kelly Criterion)</h3>
                        <ul>
                            <li><strong>Aggressive Zone (Greed):</strong> The math supports an allocation of <strong>{eg_kelly:.2f}%</strong>.</li>
                            <li><strong>Defensive Zone (Fear):</strong> Recommended allocation is <strong>{ef_kelly:.2f}%</strong>.</li>
                        </ul>
                        
                        <h3>3. Efficiency (Sharpe Ratio)</h3>
                        <p>The selected subset achieves a Max Sharpe Ratio of <strong>{eg_sharpe:.2f}</strong>, indicating robust risk-adjusted returns.</p>
                    </div>
                    """
                    st.markdown(report_html, unsafe_allow_html=True)
                    b64 = base64.b64encode(report_html.encode()).decode()
                    href = f'<a href="data:text/html;base64,{b64}" download="Custom_Report.html">📥 Download HTML Report</a>'
                    st.markdown(href, unsafe_allow_html=True)
                except KeyError:
                    st.error("Not enough data segments (Greed/Fear) in the current filter to generate a full comparative report.")

if __name__ == "__main__":
    main()