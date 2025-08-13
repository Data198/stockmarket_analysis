import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sqlalchemy import create_engine, text
from urllib.parse import quote
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Options Analysis Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>📊 Options Analysis Pro</h1>
    <p>Advanced 3-minute OI & Volume Analysis with Professional Insights</p>
</div>
""", unsafe_allow_html=True)

# DB connection with cache_resource
@st.cache_resource(show_spinner=False)
def get_db_engine():
    user = st.secrets["postgres"]["user"]
    password = quote(st.secrets["postgres"]["password"])
    host = st.secrets["postgres"]["host"]
    port = st.secrets["postgres"]["port"]
    database = st.secrets["postgres"]["database"]
    url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    return create_engine(url)

engine = get_db_engine()

# Enhanced fetch function with error handling
def fetch_distinct_values(column: str, 
                         trade_date: str = None, 
                         symbol: str = None,
                         expiry_date: str = None,
                         strike_price: float = None,
                         option_type: str = None):
    try:
        filters = {}
        where_clauses = []
        if trade_date is not None:
            filters["trade_date"] = trade_date
            where_clauses.append("trade_date = :trade_date")
        if symbol is not None:
            filters["symbol"] = symbol
            where_clauses.append("symbol = :symbol")
        if expiry_date is not None:
            filters["expiry_date"] = expiry_date
            where_clauses.append("expiry_date = :expiry_date")
        if strike_price is not None:
            filters["strike_price"] = strike_price
            where_clauses.append("strike_price = :strike_price")
        if option_type is not None:
            filters["option_type"] = option_type
            where_clauses.append("option_type = :option_type")
        
        sql = f"SELECT DISTINCT {column} FROM option_3min_ohlc"
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)
        sql += f" ORDER BY {column}"
        
        query = text(sql)
        df = pd.read_sql(query, engine, params=filters)
        return df[column].tolist()
    except Exception as e:
        st.error(f"Error fetching {column}: {str(e)}")
        return []

# Enhanced data loading with validation
def load_option_data(trade_date, symbol, expiry_date, strike_price, option_type):
    if hasattr(trade_date, 'isoformat'):
        trade_date = trade_date.isoformat()
    if hasattr(expiry_date, 'isoformat'):
        expiry_date = expiry_date.isoformat()
    
    try:
        strike_price = float(strike_price)
    except Exception:
        st.error("Strike price must be a number.")
        return pd.DataFrame()
    
    if not all([trade_date, symbol, expiry_date, strike_price, option_type]):
        st.warning("Please select all filters.")
        return pd.DataFrame()

    try:
        sql = """
            SELECT timestamp, close, open_interest, volume
            FROM option_3min_ohlc
            WHERE trade_date = :trade_date
              AND symbol = :symbol
              AND expiry_date = :expiry_date
              AND strike_price = :strike_price
              AND option_type = :option_type
            ORDER BY timestamp
        """
        query = text(sql)
        params = {
            "trade_date": trade_date,
            "symbol": symbol,
            "expiry_date": expiry_date,
            "strike_price": strike_price,
            "option_type": option_type
        }
        df = pd.read_sql(query, engine, params=params)
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Enhanced analysis function with advanced metrics
def analyze_oi_volume(df, k=2):
    if df.empty:
        return pd.DataFrame()
    
    df = df.copy()
    
    # Basic calculations
    df['Price_Change'] = df['close'].diff()
    df['Price_Change_Pct'] = df['close'].pct_change() * 100
    df['OI_Change'] = df['open_interest'].diff()
    df['OI_Change_Pct'] = df['open_interest'].pct_change() * 100
    
    # Volume analysis
    df['Vol_Mean'] = df['volume'].expanding(min_periods=1).mean()
    df['Vol_Std'] = df['volume'].expanding(min_periods=2).std().fillna(0)
    df['Vol_ZScore'] = (df['volume'] - df['Vol_Mean']) / df['Vol_Std'].replace(0, 1)
    
    # OI analysis
    df['OI_Mean'] = df['OI_Change'].expanding(min_periods=1).mean()
    df['OI_Std'] = df['OI_Change'].expanding(min_periods=2).std().fillna(0)
    df['OI_ZScore'] = (df['OI_Change'] - df['OI_Mean']) / df['OI_Std'].replace(0, 1)
    
    # Abnormal detection
    df['Abnormal_Volume'] = df['volume'] > (df['Vol_Mean'] + k * df['Vol_Std'])
    df['Abnormal_OI_Change'] = df['OI_Change'].abs() > (df['OI_Mean'].abs() + k * df['OI_Std'].abs())
    
    # Advanced metrics
    df['Volume_OI_Ratio'] = df['volume'] / df['open_interest'].replace(0, 1)
    df['Price_Volatility'] = df['Price_Change_Pct'].rolling(window=5, min_periods=1).std()
    df['OI_Trend'] = df['open_interest'].rolling(window=10, min_periods=1).mean()
    df['Volume_Trend'] = df['volume'].rolling(window=10, min_periods=1).mean()
    
    # Enhanced interpretation
    def interpret_row(row):
        price_ch = row['Price_Change']
        oi_ch = row['OI_Change']
        vol = row['volume']
        vol_abn = row['Abnormal_Volume']
        oi_abn = row['Abnormal_OI_Change']
        vol_z = row['Vol_ZScore']
        oi_z = row['OI_ZScore']
        
        # Base interpretation
        if price_ch > 0 and oi_ch > 0 and vol > 0:
            base = "🟢 Bullish activity"
        elif price_ch < 0 and oi_ch < 0 and vol > 0:
            base = "🔴 Bearish activity"
        elif price_ch > 0 and oi_ch < 0:
            base = "🟡 Bullish price, Bearish OI (Profit taking)"
        elif price_ch < 0 and oi_ch > 0:
            base = "🟡 Bearish price, Bullish OI (Accumulation)"
        else:
            base = "⚪ Neutral/No clear trend"
        
        # Abnormal activity indicators
        indicators = []
        if vol_abn:
            if vol_z > 2:
                indicators.append("🚨 Extreme Volume")
            elif vol_z > 1.5:
                indicators.append("⚠️ High Volume")
            else:
                indicators.append("📈 Above Average Volume")
        
        if oi_abn:
            if oi_z > 2:
                indicators.append("🚨 Extreme OI Change")
            elif oi_z > 1.5:
                indicators.append("⚠️ High OI Change")
            else:
                indicators.append("📊 Above Average OI Change")
        
        if indicators:
            return f"{base} + {' & '.join(indicators)}"
        return base
    
    df['Interpretation'] = df.apply(interpret_row, axis=1)
    
    # 🎯 FOUR-SECTION FLOW CATEGORIZATION
    def categorize_flow_section(row):
        price_ch = row['Price_Change']
        oi_ch = row['OI_Change']
        vol_abn = row['Abnormal_Volume']
        oi_abn = row['Abnormal_OI_Change']
        vol_z = row['Vol_ZScore']
        oi_z = row['OI_ZScore']
        
        # 🟢 Long Building (LB) - Price up + OI up + Volume
        if (price_ch > 0 and oi_ch > 0 and vol_abn):
            if vol_z > 1.5 and oi_z > 1.5:
                return "🟢 STRONG LONG BUILDING"
            else:
                return "🟢 LONG BUILDING"
        
        # 🔴 Short Building (SB) - Price down + OI up + Volume
        elif (price_ch < 0 and oi_ch > 0 and vol_abn):
            if vol_z > 1.5 and oi_z > 1.5:
                return "🔴 STRONG SHORT BUILDING"
            else:
                return "🔴 SHORT BUILDING"
        
        # 🔄 Shorts Covering (SC) - Price up + OI down + Volume
        elif (price_ch > 0 and oi_ch < 0 and vol_abn):
            if vol_z > 1.5 and abs(oi_z) > 1.5:
                return "🔄 STRONG SHORTS COVERING"
            else:
                return "🔄 SHORTS COVERING"
        
        # 💰 Longs Unwinding (LU) - Price down + OI down + Volume
        elif (price_ch < 0 and oi_ch < 0 and vol_abn):
            if vol_z > 1.5 and abs(oi_z) > 1.5:
                return "💰 STRONG LONGS UNWINDING"
            else:
                return "💰 LONGS UNWINDING"
        
        # ⚪ Neutral/No Clear Flow
        else:
            return "⚪ NEUTRAL FLOW"
    
    # Trading Signal Generation (Enhanced with Flow Analysis)
    def generate_trading_signal(row):
        flow_section = row['Flow_Section']
        
        if "STRONG LONG BUILDING" in flow_section:
            return "🚀 STRONG BUY - Heavy long accumulation"
        elif "STRONG SHORT BUILDING" in flow_section:
            return "💥 STRONG SELL - Heavy short building"
        elif "STRONG SHORTS COVERING" in flow_section:
            return "🚀 SHORT SQUEEZE - Shorts covering aggressively"
        elif "STRONG LONGS UNWINDING" in flow_section:
            return "📉 PROFIT TAKING - Longs exiting positions"
        elif "LONG BUILDING" in flow_section:
            return "📈 BUY - Long positions building"
        elif "SHORT BUILDING" in flow_section:
            return "📉 SELL - Short positions building"
        elif "SHORTS COVERING" in flow_section:
            return "🔄 SHORT COVERING - Shorts exiting"
        elif "LONGS UNWINDING" in flow_section:
            return "💰 PROFIT BOOKING - Longs taking profits"
        else:
            return "⏳ WAIT - No clear flow, monitor for setup"
    
    # Apply flow categorization
    df['Flow_Section'] = df.apply(categorize_flow_section, axis=1)
    df['Trading_Signal'] = df.apply(generate_trading_signal, axis=1)
    
    # Risk assessment
    df['Risk_Level'] = df.apply(lambda row: 
        'High' if (row['Vol_ZScore'] > 2 or row['OI_ZScore'] > 2) else
        'Medium' if (row['Vol_ZScore'] > 1.5 or row['OI_ZScore'] > 1.5) else 'Low', axis=1)
    
    # Confidence Score (0-100)
    def calculate_confidence(row):
        score = 0
        if row['Abnormal_Volume']: score += 25
        if row['Abnormal_OI_Change']: score += 25
        if abs(row['Vol_ZScore']) > 1.5: score += 20
        if abs(row['OI_ZScore']) > 1.5: score += 20
        if abs(row['Price_Change_Pct']) > 1: score += 10
        return min(score, 100)
    
    df['Confidence_Score'] = df.apply(calculate_confidence, axis=1)
    
    # 🎯 FLOW TRACKING AND WEIGHTED AVERAGE CALCULATIONS
    def calculate_flow_metrics(df):
        """Calculate cumulative flow metrics with weighted averages"""
        flow_metrics = {}
        
        # Initialize flow tracking
        df['Flow_Volume'] = df['volume']
        df['Flow_Time_Weight'] = range(1, len(df) + 1)  # Time weight (1, 2, 3...)
        df['Weighted_Flow'] = df['Flow_Volume'] * df['Flow_Time_Weight']
        
        # Categorize flows
        long_building = df[df['Flow_Section'].str.contains('LONG BUILDING')]
        short_building = df[df['Flow_Section'].str.contains('SHORT BUILDING')]
        shorts_covering = df[df['Flow_Section'].str.contains('SHORTS COVERING')]
        longs_unwinding = df[df['Flow_Section'].str.contains('LONGS UNWINDING')]
        
        # Calculate cumulative flows
        flow_metrics['Long_Building_Total'] = long_building['Flow_Volume'].sum()
        flow_metrics['Short_Building_Total'] = short_building['Flow_Volume'].sum()
        flow_metrics['Shorts_Covering_Total'] = shorts_covering['Flow_Volume'].sum()
        flow_metrics['Longs_Unwinding_Total'] = longs_unwinding['Flow_Volume'].sum()
        
        # Calculate weighted averages
        flow_metrics['Long_Building_Weighted'] = long_building['Weighted_Flow'].sum() / max(long_building['Flow_Time_Weight'].sum(), 1)
        flow_metrics['Short_Building_Weighted'] = short_building['Weighted_Flow'].sum() / max(short_building['Flow_Time_Weight'].sum(), 1)
        flow_metrics['Shorts_Covering_Weighted'] = shorts_covering['Weighted_Flow'].sum() / max(shorts_covering['Flow_Time_Weight'].sum(), 1)
        flow_metrics['Longs_Unwinding_Weighted'] = longs_unwinding['Weighted_Flow'].sum() / max(longs_unwinding['Flow_Time_Weight'].sum(), 1)
        
        # Net flow calculations
        flow_metrics['Net_Long_Flow'] = flow_metrics['Long_Building_Total'] - flow_metrics['Longs_Unwinding_Total']
        flow_metrics['Net_Short_Flow'] = flow_metrics['Short_Building_Total'] - flow_metrics['Shorts_Covering_Total']
        flow_metrics['Net_Flow_Strength'] = flow_metrics['Net_Long_Flow'] - flow_metrics['Net_Short_Flow']
        
        # Flow momentum (recent vs earlier periods)
        recent_periods = min(5, len(df))
        recent_df = df.tail(recent_periods)
        
        recent_lb = recent_df[recent_df['Flow_Section'].str.contains('LONG BUILDING')]['Flow_Volume'].sum()
        recent_sb = recent_df[recent_df['Flow_Section'].str.contains('SHORT BUILDING')]['Flow_Volume'].sum()
        recent_sc = recent_df[recent_df['Flow_Section'].str.contains('SHORTS COVERING')]['Flow_Volume'].sum()
        recent_lu = recent_df[recent_df['Flow_Section'].str.contains('LONGS UNWINDING')]['Flow_Volume'].sum()
        
        flow_metrics['Recent_Long_Momentum'] = recent_lb - recent_lu
        flow_metrics['Recent_Short_Momentum'] = recent_sb - recent_sc
        
        return flow_metrics
    
    # Calculate flow metrics
    flow_metrics = calculate_flow_metrics(df)
    df.attrs['flow_metrics'] = flow_metrics  # Store in dataframe attributes
    
    return df

# Advanced analytics functions
def calculate_advanced_metrics(df):
    if df.empty:
        return {}
    
    metrics = {}
    
    # Price metrics
    metrics['Total_Price_Change'] = df['close'].iloc[-1] - df['close'].iloc[0]
    metrics['Total_Price_Change_Pct'] = ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100
    metrics['Max_Price'] = df['close'].max()
    metrics['Min_Price'] = df['close'].min()
    metrics['Price_Volatility'] = df['close'].pct_change().std() * np.sqrt(252 * 24 * 20) * 100
    
    # Volume metrics
    metrics['Total_Volume'] = df['volume'].sum()
    metrics['Avg_Volume'] = df['volume'].mean()
    metrics['Max_Volume'] = df['volume'].max()
    metrics['Volume_Volatility'] = df['volume'].std() / df['volume'].mean() * 100
    
    # OI metrics
    metrics['OI_Start'] = df['open_interest'].iloc[0]
    metrics['OI_End'] = df['open_interest'].iloc[-1]
    metrics['OI_Change_Total'] = metrics['OI_End'] - metrics['OI_Start']
    metrics['OI_Change_Pct'] = ((metrics['OI_End'] / metrics['OI_Start']) - 1) * 100 if metrics['OI_Start'] > 0 else 0
    
    # Abnormal activity metrics
    abnormal_volume_count = df['Abnormal_Volume'].sum()
    abnormal_oi_count = df['Abnormal_OI_Change'].sum()
    total_periods = len(df)
    
    metrics['Abnormal_Volume_Pct'] = (abnormal_volume_count / total_periods) * 100
    metrics['Abnormal_OI_Pct'] = (abnormal_oi_count / total_periods) * 100
    metrics['High_Risk_Periods'] = (df['Risk_Level'] == 'High').sum()
    
    return metrics

# Sidebar configuration
with st.sidebar:
    st.header("🔧 Analysis Configuration")
    
    # Date selection
    trade_dates = fetch_distinct_values("trade_date")
    if trade_dates:
        selected_date = st.selectbox("📅 Trade Date", trade_dates, index=len(trade_dates)-1 if trade_dates else 0)
    else:
        selected_date = None
        st.error("No trade dates available")
    
    # Symbol selection
    if selected_date:
        symbols = fetch_distinct_values("symbol", trade_date=selected_date)
        if symbols:
            selected_symbol = st.selectbox("🏢 Symbol", symbols)
        else:
            selected_symbol = None
            st.error("No symbols available for selected date")
    else:
        selected_symbol = None
    
    # Expiry selection
    if selected_symbol:
        expiries = fetch_distinct_values("expiry_date", trade_date=selected_date, symbol=selected_symbol)
        if expiries:
            selected_expiry = st.selectbox("📆 Expiry Date", expiries)
        else:
            selected_expiry = None
            st.error("No expiries available")
    else:
        selected_expiry = None
    
    # Strike selection
    if selected_expiry:
        strikes = fetch_distinct_values("strike_price", trade_date=selected_date, symbol=selected_symbol, expiry_date=selected_expiry)
        if strikes:
            selected_strike = st.selectbox("💰 Strike Price", strikes)
        else:
            selected_strike = None
            st.error("No strikes available")
    else:
        selected_strike = None
    
    # Option type selection
    if selected_strike:
        option_types = fetch_distinct_values("option_type", trade_date=selected_date, symbol=selected_symbol, expiry_date=selected_expiry, strike_price=selected_strike)
        if option_types:
            selected_option_type = st.selectbox("📊 Option Type", option_types)
        else:
            selected_option_type = None
            st.error("No option types available")
    else:
        selected_option_type = None
    
    # Analysis parameters
    st.subheader("📊 Analysis Parameters")
    k_value = st.slider("Standard Deviation Multiplier (K)", min_value=1.0, max_value=5.0, value=2.0, step=0.1)
    
    # Update button
    if st.button("🔄 Update Analysis", use_container_width=True):
        st.rerun()

# Main content
if all([selected_date, selected_symbol, selected_expiry, selected_strike, selected_option_type]):
    # Load data
    with st.spinner("Loading option data..."):
        df_data = load_option_data(
            trade_date=selected_date,
            symbol=selected_symbol,
            expiry_date=selected_expiry,
            strike_price=selected_strike,
            option_type=selected_option_type
        )
    
    if not df_data.empty:
        # Perform analysis
        analyzed_df = analyze_oi_volume(df_data, k=k_value)
        metrics = calculate_advanced_metrics(analyzed_df)
        
        # Header with contract info
        st.subheader(f"📊 Contract Analysis: {selected_symbol} {selected_expiry} {selected_strike} {selected_option_type}")
        
        # Key metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Price Change",
                f"₹{metrics['Total_Price_Change']:.2f}",
                f"{metrics['Total_Price_Change_Pct']:.2f}%"
            )
        
        with col2:
            st.metric(
                "Total Volume",
                f"{metrics['Total_Volume']:,.0f}",
                f"{metrics['Abnormal_Volume_Pct']:.1f}% abnormal"
            )
        
        with col3:
            st.metric(
                "OI Change",
                f"{metrics['OI_Change_Total']:,.0f}",
                f"{metrics['OI_Change_Pct']:.1f}%"
            )
        
        with col4:
            st.metric(
                "Risk Level",
                f"{metrics['High_Risk_Periods']} periods",
                f"{(metrics['High_Risk_Periods']/len(analyzed_df)*100):.1f}% of time"
            )
        
        # 🎯 TRADING DECISION DASHBOARD
        st.subheader("🎯 Live Trading Decision Dashboard")
        
        # Get latest signals
        latest_data = analyzed_df.iloc[-1]
        latest_signal = latest_data['Trading_Signal']
        confidence = latest_data['Confidence_Score']
        risk_level = latest_data['Risk_Level']
        
        # 🚀 FOUR-SECTION FLOW ANALYSIS DASHBOARD
        st.subheader("🚀 Four-Section Flow Analysis Dashboard")
        
        # Get flow metrics
        flow_metrics = analyzed_df.attrs.get('flow_metrics', {})
        
        # Flow Overview - 4 columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); 
                        padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
                <h3>🟢 Long Building</h3>
                <p style="font-size: 1.5em; font-weight: bold;">{:,}</p>
                <p style="font-size: 0.9em;">Total Volume</p>
                <p style="font-size: 1.2em; font-weight: bold;">{:.0f}</p>
                <p style="font-size: 0.9em;">Weighted Avg</p>
            </div>
            """.format(
                flow_metrics.get('Long_Building_Total', 0),
                flow_metrics.get('Long_Building_Weighted', 0)
            ), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #dc3545 0%, #fd7e14 100%); 
                        padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
                <h3>🔴 Short Building</h3>
                <p style="font-size: 1.5em; font-weight: bold;">{:,}</p>
                <p style="font-size: 0.9em;">Total Volume</p>
                <p style="font-size: 1.2em; font-weight: bold;">{:.0f}</p>
                <p style="font-size: 0.9em;">Weighted Avg</p>
            </div>
            """.format(
                flow_metrics.get('Short_Building_Total', 0),
                flow_metrics.get('Short_Building_Weighted', 0)
            ), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #17a2b8 0%, #6f42c1 100%); 
                        padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
                <h3>🔄 Shorts Covering</h3>
                <p style="font-size: 1.5em; font-weight: bold;">{:,}</p>
                <p style="font-size: 0.9em;">Total Volume</p>
                <p style="font-size: 1.2em; font-weight: bold;">{:.0f}</p>
                <p style="font-size: 0.9em;">Weighted Avg</p>
            </div>
            """.format(
                flow_metrics.get('Shorts_Covering_Total', 0),
                flow_metrics.get('Shorts_Covering_Weighted', 0)
            ), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%); 
                        padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
                <h3>💰 Longs Unwinding</h3>
                <p style="font-size: 1.5em; font-weight: bold;">{:,}</p>
                <p style="font-size: 0.9em;">Total Volume</p>
                <p style="font-size: 1.2em; font-weight: bold;">{:.0f}</p>
                <p style="font-size: 0.9em;">Weighted Avg</p>
            </div>
            """.format(
                flow_metrics.get('Longs_Unwinding_Total', 0),
                flow_metrics.get('Longs_Unwinding_Weighted', 0)
            ), unsafe_allow_html=True)
        
        # Net Flow Analysis
        st.subheader("📊 Net Flow Analysis & Momentum")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            net_long_flow = flow_metrics.get('Net_Long_Flow', 0)
            net_short_flow = flow_metrics.get('Net_Short_Flow', 0)
            
            if net_long_flow > net_short_flow:
                color = "#28a745"
                status = "🟢 Longs Dominating"
            else:
                color = "#dc3545"
                status = "🔴 Shorts Dominating"
            
            st.markdown(f"""
            <div style="background: {color}; 
                        padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
                <h3>⚖️ Net Flow Strength</h3>
                <p style="font-size: 1.5em; font-weight: bold;">{net_long_flow - net_short_flow:+,}</p>
                <p style="font-size: 1.1em;">{status}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            recent_long_momentum = flow_metrics.get('Recent_Long_Momentum', 0)
            if recent_long_momentum > 0:
                color = "#28a745"
                icon = "📈"
            else:
                color = "#dc3545"
                icon = "📉"
            
            st.markdown(f"""
            <div style="background: {color}; 
                        padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
                <h3>🚀 Recent Long Momentum</h3>
                <p style="font-size: 2em; font-weight: bold;">{icon}</p>
                <p style="font-size: 1.2em; font-weight: bold;">{recent_long_momentum:+,}</p>
                <p style="font-size: 0.9em;">Last 5 periods</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            recent_short_momentum = flow_metrics.get('Recent_Short_Momentum', 0)
            if recent_short_momentum > 0:
                color = "#dc3545"
                icon = "📉"
            else:
                color = "#28a745"
                icon = "📈"
            
            st.markdown(f"""
            <div style="background: {color}; 
                        padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
                <h3>💥 Recent Short Momentum</h3>
                <p style="font-size: 2em; font-weight: bold;">{icon}</p>
                <p style="font-size: 1.2em; font-weight: bold;">{recent_short_momentum:+,}</p>
                <p style="font-size: 0.9em;">Last 5 periods</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Flow Timeline Analysis
        st.subheader("⏰ Flow Timeline Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Flow Section Timeline (Last 10 periods):**")
            flow_timeline = analyzed_df.tail(10)[['timestamp', 'Flow_Section', 'volume']].copy()
            flow_timeline['timestamp'] = flow_timeline['timestamp'].dt.strftime('%H:%M:%S')
            st.dataframe(flow_timeline, use_container_width=True)
        
        with col2:
            st.write("**Flow Section Distribution:**")
            flow_counts = analyzed_df['Flow_Section'].value_counts()
            fig_flow = px.pie(
                values=flow_counts.values,
                names=flow_counts.index,
                title="Flow Section Distribution"
            )
            st.plotly_chart(fig_flow, use_container_width=True)
        
        # 🚨 FLOW-BASED ALERTS & OPPORTUNITIES
        st.subheader("🚨 Flow-Based Alerts & Trading Opportunities")
        
        # Generate flow-based alerts
        alerts = []
        
        # Short squeeze detection
        if flow_metrics.get('Shorts_Covering_Total', 0) > flow_metrics.get('Short_Building_Total', 0) * 0.8:
            alerts.append("🚨 **SHORT SQUEEZE ALERT**: Shorts covering aggressively, potential squeeze setup!")
        
        # Long accumulation detection
        if flow_metrics.get('Long_Building_Total', 0) > flow_metrics.get('Longs_Unwinding_Total', 0) * 2:
            alerts.append("🟢 **LONG ACCUMULATION**: Heavy long building, bullish momentum building!")
        
        # Short momentum detection
        if flow_metrics.get('Short_Building_Total', 0) > flow_metrics.get('Shorts_Covering_Total', 0) * 2:
            alerts.append("🔴 **SHORT MOMENTUM**: Heavy short building, bearish momentum building!")
        
        # Flow reversal detection
        recent_flow = analyzed_df.tail(3)['Flow_Section'].tolist()
        if len(recent_flow) >= 3:
            if recent_flow[-1] != recent_flow[-2] and recent_flow[-2] != recent_flow[-3]:
                alerts.append("🔄 **FLOW REVERSAL**: Flow pattern changing, monitor for new setup!")
        
        # Display alerts
        if alerts:
            for alert in alerts:
                st.warning(alert)
        else:
            st.info("📊 No significant flow alerts at this time. Monitor for pattern changes.")
        
        # Flow Strength Comparison
        st.subheader("💪 Flow Strength Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Long vs Short strength
            long_strength = flow_metrics.get('Net_Long_Flow', 0)
            short_strength = flow_metrics.get('Net_Short_Flow', 0)
            
            fig_strength = go.Figure()
            fig_strength.add_trace(go.Bar(
                x=['Long Flow', 'Short Flow'],
                y=[long_strength, short_strength],
                marker_color=['#28a745', '#dc3545']
            ))
            fig_strength.update_layout(
                title="Net Flow Strength Comparison",
                yaxis_title="Volume",
                showlegend=False
            )
            st.plotly_chart(fig_strength, use_container_width=True)
        
        with col2:
            # Recent momentum comparison
            recent_long = flow_metrics.get('Recent_Long_Momentum', 0)
            recent_short = flow_metrics.get('Recent_Short_Momentum', 0)
            
            fig_momentum = go.Figure()
            fig_momentum.add_trace(go.Bar(
                x=['Long Momentum', 'Short Momentum'],
                y=[recent_long, recent_short],
                marker_color=['#28a745', '#dc3545']
            ))
            fig_momentum.update_layout(
                title="Recent Momentum (Last 5 periods)",
                yaxis_title="Volume",
                showlegend=False
            )
            st.plotly_chart(fig_momentum, use_container_width=True)
        
        # Decision summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
                <h3>📊 Current Signal</h3>
                <p style="font-size: 1.2em; font-weight: bold;">{}</p>
            </div>
            """.format(latest_signal), unsafe_allow_html=True)
        
        with col2:
            # Color code confidence
            if confidence >= 80:
                color = "#28a745"  # Green
                status = "🟢 High Confidence"
            elif confidence >= 60:
                color = "#ffc107"  # Yellow
                status = "🟡 Medium Confidence"
            else:
                color = "#dc3545"  # Red
                status = "🔴 Low Confidence"
            
            st.markdown(f"""
            <div style="background: {color}; 
                        padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
                <h3>🎯 Signal Confidence</h3>
                <p style="font-size: 2em; font-weight: bold;">{confidence}%</p>
                <p style="font-size: 1.1em;">{status}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Color code risk
            if risk_level == 'High':
                color = "#dc3545"  # Red
                icon = "🚨"
            elif risk_level == 'Medium':
                color = "#ffc107"  # Yellow
                icon = "⚠️"
            else:
                color = "#28a745"  # Green
                icon = "✅"
            
            st.markdown(f"""
            <div style="background: {color}; 
                        padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
                <h3>⚠️ Risk Level</h3>
                <p style="font-size: 2em; font-weight: bold;">{icon}</p>
                <p style="font-size: 1.1em;">{risk_level} Risk</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Actionable Trading Recommendations
        st.subheader("💡 Actionable Trading Recommendations")
        
        # Generate recommendations based on latest data
        recommendations = []
        
        if "STRONG BUY" in latest_signal:
            recommendations.append("🚀 **IMMEDIATE ACTION**: Enter long position with tight stop loss")
            recommendations.append("📈 **Target**: Look for 2-3% move in next 15-30 minutes")
            recommendations.append("🛡️ **Stop Loss**: Set at recent support level")
        elif "STRONG SELL" in latest_signal:
            recommendations.append("💥 **IMMEDIATE ACTION**: Enter short position or exit long positions")
            recommendations.append("📉 **Target**: Look for 2-3% decline in next 15-30 minutes")
            recommendations.append("🛡️ **Stop Loss**: Set at recent resistance level")
        elif "BUY" in latest_signal:
            recommendations.append("📈 **ACTION**: Consider long entry on pullbacks")
            recommendations.append("⏰ **Timing**: Wait for confirmation in next 5-10 minutes")
            recommendations.append("📊 **Monitor**: Watch for increasing volume confirmation")
        elif "SELL" in latest_signal:
            recommendations.append("📉 **ACTION**: Consider short entry on rallies")
            recommendations.append("⏰ **Timing**: Wait for confirmation in next 5-10 minutes")
            recommendations.append("📊 **Monitor**: Watch for increasing volume confirmation")
        elif "ACCUMULATE" in latest_signal:
            recommendations.append("🔄 **ACTION**: Start accumulating in small quantities")
            recommendations.append("📈 **Strategy**: Scale in on further dips")
            recommendations.append("⏰ **Timeframe**: This is a longer-term setup")
        elif "TAKE PROFIT" in latest_signal:
            recommendations.append("💰 **ACTION**: Book partial profits if holding long")
            recommendations.append("📊 **Monitor**: Watch for reversal signals")
            recommendations.append("🔄 **Re-entry**: Wait for fresh accumulation signals")
        else:
            recommendations.append("⏳ **ACTION**: No immediate action required")
            recommendations.append("👀 **Monitor**: Watch for new signal development")
            recommendations.append("📊 **Analysis**: Focus on other opportunities")
        
        # Display recommendations
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
        
        # Market Context Analysis
        st.subheader("🔍 Market Context Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Recent Signal History (Last 5 periods):**")
            recent_signals = analyzed_df.tail(5)[['timestamp', 'Trading_Signal', 'Confidence_Score']].copy()
            recent_signals['timestamp'] = recent_signals['timestamp'].dt.strftime('%H:%M:%S')
            st.dataframe(recent_signals, use_container_width=True)
        
        with col2:
            st.write("**Signal Strength Distribution:**")
            signal_counts = analyzed_df['Trading_Signal'].value_counts()
            fig_signals = px.bar(
                x=signal_counts.values,
                y=signal_counts.index,
                orientation='h',
                title="Trading Signal Distribution"
            )
            st.plotly_chart(fig_signals, use_container_width=True)
        
        # Charts section
        st.subheader("📈 Price & Volume Analysis")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Price Movement', 'Volume Analysis', 'Open Interest'),
            row_width=[0.4, 0.3, 0.3]
        )
        
        # Price chart
        fig.add_trace(
            go.Scatter(
                x=analyzed_df['timestamp'],
                y=analyzed_df['close'],
                name='Close Price',
                line=dict(color='#667eea', width=2)
            ),
            row=1, col=1
        )
        
        # Volume chart with abnormal indicators
        colors = ['red' if row['Abnormal_Volume'] else 'blue' for _, row in analyzed_df.iterrows()]
        fig.add_trace(
            go.Bar(
                x=analyzed_df['timestamp'],
                y=analyzed_df['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # OI chart
        fig.add_trace(
            go.Scatter(
                x=analyzed_df['timestamp'],
                y=analyzed_df['open_interest'],
                name='Open Interest',
                line=dict(color='#764ba2', width=2)
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f"{selected_symbol} {selected_expiry} {selected_strike} {selected_option_type}",
            height=600,
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Abnormal activity analysis
        st.subheader("🚨 Abnormal Activity Analysis")
        
        abnormal_df = analyzed_df[(analyzed_df['Abnormal_Volume']) | (analyzed_df['Abnormal_OI_Change'])]
        
        if not abnormal_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Abnormal Volume Periods:**")
                vol_abnormal = analyzed_df[analyzed_df['Abnormal_Volume']]
                if not vol_abnormal.empty:
                    st.dataframe(
                        vol_abnormal[['timestamp', 'close', 'volume', 'Vol_ZScore', 'Interpretation']],
                        use_container_width=True
                    )
            
            with col2:
                st.write("**Abnormal OI Change Periods:**")
                oi_abnormal = analyzed_df[analyzed_df['Abnormal_OI_Change']]
                if not oi_abnormal.empty:
                    st.dataframe(
                        oi_abnormal[['timestamp', 'close', 'OI_Change', 'OI_ZScore', 'Interpretation']],
                        use_container_width=True
                    )
        else:
            st.info("No abnormal activity detected with current parameters.")
        
        # Risk analysis
        st.subheader("⚠️ Risk Assessment")
        
        risk_summary = analyzed_df['Risk_Level'].value_counts()
        col1, col2 = st.columns(2)
        
        with col1:
            fig_risk = px.pie(
                values=risk_summary.values,
                names=risk_summary.index,
                title="Risk Level Distribution"
            )
            st.plotly_chart(fig_risk, use_container_width=True)
        
        with col2:
            st.write("**Risk Level Breakdown:**")
            for level, count in risk_summary.items():
                percentage = (count / len(analyzed_df)) * 100
                st.write(f"**{level} Risk:** {count} periods ({percentage:.1f}%)")
        
        # Full analysis table
        st.subheader("📋 Complete Analysis Data")
        
        # Format display columns
        display_df = analyzed_df.copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        display_df['Price_Change'] = display_df['Price_Change'].round(4)
        display_df['Price_Change_Pct'] = display_df['Price_Change_Pct'].round(2)
        display_df['OI_Change'] = display_df['OI_Change'].round(0)
        display_df['Vol_ZScore'] = display_df['Vol_ZScore'].round(2)
        display_df['OI_ZScore'] = display_df['OI_ZScore'].round(2)
        
        st.dataframe(
            display_df[['timestamp', 'close', 'Price_Change', 'Price_Change_Pct', 
                       'volume', 'Vol_ZScore', 'Abnormal_Volume',
                       'open_interest', 'OI_Change', 'OI_ZScore', 'Abnormal_OI_Change',
                       'Risk_Level', 'Flow_Section', 'Trading_Signal', 'Confidence_Score', 'Interpretation']],
            use_container_width=True,
            height=400
        )
        
        # Export functionality
        st.subheader("📤 Export Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        def df_to_excel_bytes(df, filename):
            import io
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name='Analysis')
                
                # Add summary sheet
                summary_df = pd.DataFrame([metrics])
                summary_df.to_excel(writer, index=False, sheet_name='Summary')
                
                # Add flow analysis sheet
                flow_summary = pd.DataFrame([flow_metrics])
                flow_summary.to_excel(writer, index=False, sheet_name='Flow_Analysis')
            return output.getvalue()
        
        with col1:
            st.download_button(
                label="📥 Full Analysis Excel",
                data=df_to_excel_bytes(analyzed_df, "full_analysis"),
                file_name=f"options_analysis_{selected_symbol}_{selected_expiry}_{selected_strike}_{selected_option_type}_{selected_date}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col2:
            st.download_button(
                label="📥 Abnormal Activity Excel",
                data=df_to_excel_bytes(abnormal_df, "abnormal_activity"),
                file_name=f"abnormal_activity_{selected_symbol}_{selected_expiry}_{selected_strike}_{selected_option_type}_{selected_date}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col3:
            st.download_button(
                label="📥 Summary Metrics Excel",
                data=df_to_excel_bytes(pd.DataFrame([metrics]), "summary"),
                file_name=f"summary_metrics_{selected_symbol}_{selected_expiry}_{selected_strike}_{selected_option_type}_{selected_date}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        # 🎯 COMPREHENSIVE TRADING STRATEGY GUIDE
        st.subheader("🎯 Comprehensive Trading Strategy Guide")
        
        with st.expander("📚 **How to Use This Analysis for Live Trading**", expanded=False):
            st.markdown("""
            ### 🚀 **Live Market Trading Strategy**
            
            #### **1. Signal Strength Hierarchy**
            - **🚀 STRONG BUY/SELL**: Highest confidence, immediate action required
            - **📈📉 BUY/SELL**: Medium confidence, wait for confirmation
            - **🔄 ACCUMULATE**: Smart money accumulation, longer-term setup
            - **💰 TAKE PROFIT**: Profit booking signal, consider exiting
            - **⏳ WAIT**: No clear signal, monitor for setup
            
            #### **2. Real-Time Decision Making**
            - **Monitor every 3 minutes** during market hours
            - **Look for signal changes** from WAIT to actionable signals
            - **Use confidence scores** to gauge signal reliability
            - **Combine with price action** for entry confirmation
            
            #### **3. Entry Strategy**
            - **Strong Signals**: Enter immediately with tight stop loss
            - **Moderate Signals**: Wait for pullback/rally confirmation
            - **Accumulation**: Scale in gradually on dips
            - **Volume Confirmation**: Always wait for volume spike
            
            #### **4. Risk Management**
            - **Stop Loss**: Set at recent support/resistance levels
            - **Position Size**: 1-2% risk per trade
            - **Exit Strategy**: Book profits at 2-3% moves
            - **Trailing Stop**: Move stop loss to breakeven after 1% move
            """)
        
        with st.expander("🎯 **Four-Section Flow Trading Strategy**", expanded=False):
            st.markdown("""
            ### 🚀 **Institutional Flow-Based Trading Strategy**
            
            #### **🟢 Long Building (LB) - BULLISH MOMENTUM**
            - **When**: Price ↑ + OI ↑ + High Volume
            - **Action**: **BUY** - Smart money accumulating
            - **Stop Loss**: Below recent support
            - **Target**: 2-3% move in 15-30 minutes
            
            #### **🔴 Short Building (SB) - BEARISH MOMENTUM**
            - **When**: Price ↓ + OI ↑ + High Volume
            - **Action**: **SELL** - Smart money building shorts
            - **Stop Loss**: Above recent resistance
            - **Target**: 2-3% decline in 15-30 minutes
            
            #### **🔄 Shorts Covering (SC) - SHORT SQUEEZE**
            - **When**: Price ↑ + OI ↓ + High Volume
            - **Action**: **BUY** - Shorts exiting, potential squeeze
            - **Stop Loss**: Below entry point
            - **Target**: 3-5% move as shorts panic
            
            #### **💰 Longs Unwinding (LU) - PROFIT TAKING**
            - **When**: Price ↓ + OI ↓ + High Volume
            - **Action**: **EXIT LONG** - Protect profits
            - **Strategy**: Book partial profits, move stop to breakeven
            
            #### **🎯 Flow Momentum Analysis**
            - **Net Long Flow**: LB - LU (positive = bullish)
            - **Net Short Flow**: SB - SC (positive = bearish)
            - **Recent Momentum**: Last 5 periods flow strength
            - **Weighted Averages**: Recent flows have higher importance
            """)
        
        with st.expander("📊 **Advanced Pattern Recognition**", expanded=False):
            st.markdown("""
            ### 🔍 **Pattern Recognition for Better Entries**
            
            #### **Bullish Patterns**
            1. **Price ↑ + OI ↑ + High Volume**: Strong accumulation
            2. **Price ↓ + OI ↑ + High Volume**: Smart money buying dips
            3. **Low Volume + Price ↑**: Weak move, wait for volume
            
            #### **Bearish Patterns**
            1. **Price ↓ + OI ↓ + High Volume**: Strong distribution
            2. **Price ↑ + OI ↓ + High Volume**: Profit taking
            3. **Low Volume + Price ↓**: Weak move, wait for volume
            
            #### **Reversal Patterns**
            1. **Extreme Volume + Price Reversal**: Potential trend change
            2. **OI Divergence**: Price and OI moving opposite
            3. **Volume Drying Up**: Trend losing momentum
            """)
        
        with st.expander("⏰ **Timing Your Trades**", expanded=False):
            st.markdown("""
            ### ⏰ **Optimal Trading Timing**
            
            #### **Market Hours Strategy**
            - **9:15-10:00 AM**: High volatility, wait for clear signals
            - **10:00-11:30 AM**: Best time for trend following trades
            - **11:30-2:00 PM**: Lower volatility, focus on scalping
            - **2:00-3:30 PM**: High volatility, momentum trades
            
            #### **Signal Confirmation Timeframes**
            - **Strong Signals**: 5-10 minutes confirmation
            - **Moderate Signals**: 15-30 minutes confirmation
            - **Accumulation**: 1-2 hours for full position
            - **Exit Signals**: Immediate action required
            
            #### **Volume Analysis Timing**
            - **Abnormal Volume**: Look for 2-3 consecutive periods
            - **OI Changes**: Monitor for sustained direction
            - **Price Action**: Confirm with candlestick patterns
            """)
        
        with st.expander("🛡️ **Risk Management & Psychology**", expanded=False):
            st.markdown("""
            ### 🛡️ **Risk Management Framework**
            
            #### **Position Sizing**
            - **Conservative**: 0.5-1% risk per trade
            - **Moderate**: 1-2% risk per trade
            - **Aggressive**: 2-3% risk per trade (not recommended)
            
            #### **Stop Loss Strategy**
            - **Tight Stop**: 0.5-1% for strong signals
            - **Normal Stop**: 1-2% for moderate signals
            - **Wide Stop**: 2-3% for accumulation trades
            
            #### **Profit Taking**
            - **Partial Exit**: 50% at 1% profit
            - **Full Exit**: Remaining at 2-3% profit
            - **Trailing Stop**: Move to breakeven after 0.5% profit
            
            #### **Psychological Rules**
            - **Never chase losses** with bigger positions
            - **Stick to your plan** regardless of emotions
            - **Take breaks** after 3 consecutive losses
            - **Review trades** daily for improvement
            """)
        
    else:
        st.warning("No data found for selected filters.")
        st.info("Please check your selection and try again.")

else:
    st.info("👈 Please select all filters from the sidebar to begin analysis.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>📊 Options Analysis Pro | Advanced OI & Volume Analysis</p>
    <p>⚠️ This is for educational purposes only. Not financial advice.</p>
    <p>🎯 Use signals responsibly and always manage your risk!</p>
</div>
""", unsafe_allow_html=True)
