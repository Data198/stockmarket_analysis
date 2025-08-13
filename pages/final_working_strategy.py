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
            SELECT timestamp, close, open_interest, volume, high, low, open
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
    
    # Risk assessment
    df['Risk_Level'] = df.apply(lambda row: 
        'High' if (row['Vol_ZScore'] > 2 or row['OI_ZScore'] > 2) else
        'Medium' if (row['Vol_ZScore'] > 1.5 or row['OI_ZScore'] > 1.5) else 'Low', axis=1)
    
    return df

# Advanced analytics functions
def calculate_advanced_metrics(df):
    if df.empty:
        return {}
    
    metrics = {}
    
    # Price metrics
    metrics['Total_Price_Change'] = df['close'].iloc[-1] - df['close'].iloc[0]
    metrics['Total_Price_Change_Pct'] = ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100
    metrics['Max_Price'] = df['high'].max()
    metrics['Min_Price'] = df['low'].min()
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
                f"${metrics['Total_Price_Change']:.2f}",
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
                       'Risk_Level', 'Interpretation']],
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
</div>
""", unsafe_allow_html=True)
