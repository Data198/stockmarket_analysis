import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote

# --- Secure DB Connection ---
DB_USER = st.secrets["postgres"]["user"]
DB_PASS = quote(st.secrets["postgres"]["password"])  # encode password safely
DB_HOST = st.secrets["postgres"]["host"]
DB_PORT = st.secrets["postgres"]["port"]
DB_NAME = st.secrets["postgres"]["database"]

db_url = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(db_url)

# --- Analysis function ---
def interpret_oi_data_with_abnormal(df, k=2):
    df = df.sort_values(by='timestamp')
    
    df['Price_Change'] = df['close'].diff()
    df['OI_Change'] = df['open_interest'].diff()
    df['Volume_Change'] = df['volume'].diff()
    
    df['Vol_Mean'] = df['volume'].expanding(min_periods=1).mean()
    df['Vol_Std'] = df['volume'].expanding(min_periods=2).std().fillna(0)
    df['OI_Mean'] = df['OI_Change'].expanding(min_periods=1).mean()
    df['OI_Std'] = df['OI_Change'].expanding(min_periods=2).std().fillna(0)
    
    df['Abnormal_Volume'] = df['volume'] > (df['Vol_Mean'] + k * df['Vol_Std'])
    df['Abnormal_OI_Change'] = df['OI_Change'].abs() > (df['OI_Mean'].abs() + k * df['OI_Std'].abs())
    
    interpretations = []
    for _, row in df.iterrows():
        price_ch = row['Price_Change']
        oi_ch = row['OI_Change']
        vol = row['volume']
        vol_abn = row['Abnormal_Volume']
        oi_abn = row['Abnormal_OI_Change']
        
        if price_ch > 0 and oi_ch > 0 and vol > 0:
            decision = "Bullish activity"
        elif price_ch < 0 and oi_ch < 0 and vol > 0:
            decision = "Bearish activity"
        else:
            decision = "Neutral/No clear trend"
        
        if vol_abn and oi_abn:
            decision += " + Abnormal Volume & OI Change"
        elif vol_abn:
            decision += " + Abnormal Volume"
        elif oi_abn:
            decision += " + Abnormal OI Change"
        
        interpretations.append(decision)
    
    df['Interpretation'] = interpretations
    
    return df[[
        'timestamp', 'close', 'Price_Change', 'open_interest', 'OI_Change', 'volume',
        'Abnormal_Volume', 'Abnormal_OI_Change', 'Interpretation'
    ]]

# --- Streamlit UI ---
st.title("Options 3-min OI & Volume Abnormality Analysis")

# Fetch available trade_dates for selection
@st.cache_data
def get_trade_dates():
    query = "SELECT DISTINCT trade_date FROM option_3min_ohlc ORDER BY trade_date DESC"
    df_dates = pd.read_sql(query, engine)
    return df_dates['trade_date'].tolist()

trade_dates = get_trade_dates()
selected_date = st.sidebar.selectbox("Select Trade Date", trade_dates)

# Fetch symbols for selected date
@st.cache_data
def get_symbols(trade_date):
    query = f"SELECT DISTINCT symbol FROM option_3min_ohlc WHERE trade_date = '{trade_date}' ORDER BY symbol"
    df_syms = pd.read_sql(query, engine)
    return df_syms['symbol'].tolist()

symbols = get_symbols(selected_date)
selected_symbol = st.sidebar.selectbox("Select Symbol", symbols)

# Fetch expiry dates for selected filters
@st.cache_data
def get_expiries(trade_date, symbol):
    query = f"""
        SELECT DISTINCT expiry_date FROM option_3min_ohlc 
        WHERE trade_date = '{trade_date}' AND symbol = '{symbol}' ORDER BY expiry_date
    """
    df_exp = pd.read_sql(query, engine)
    return df_exp['expiry_date'].tolist()

expiries = get_expiries(selected_date, selected_symbol)
selected_expiry = st.sidebar.selectbox("Select Expiry Date", expiries)

# Fetch strikes for selected filters
@st.cache_data
def get_strikes(trade_date, symbol, expiry):
    query = f"""
        SELECT DISTINCT strike_price FROM option_3min_ohlc 
        WHERE trade_date = '{trade_date}' AND symbol = '{symbol}' AND expiry_date = '{expiry}'
        ORDER BY strike_price
    """
    df_strikes = pd.read_sql(query, engine)
    return df_strikes['strike_price'].tolist()

strikes = get_strikes(selected_date, selected_symbol, selected_expiry)
selected_strike = st.sidebar.selectbox("Select Strike Price", strikes)

option_types = ['CE', 'PE']
selected_option_type = st.sidebar.selectbox("Select Option Type", option_types)

# Load data for selected filters
@st.cache_data
def load_data(trade_date, symbol, expiry, strike, option_type):
    query = f"""
        SELECT timestamp, close, open_interest, volume, price_change
        FROM option_3min_ohlc
        WHERE trade_date = '{trade_date}'
          AND symbol = '{symbol}'
          AND expiry_date = '{expiry}'
          AND strike_price = {strike}
          AND option_type = '{option_type}'
        ORDER BY timestamp
    """
    df = pd.read_sql(query, engine)
    return df

df_data = load_data(selected_date, selected_symbol, selected_expiry, selected_strike, selected_option_type)

if df_data.empty:
    st.warning("No data found for selected filters.")
else:
    analyzed_df = interpret_oi_data_with_abnormal(df_data)
    
    st.subheader("Full Analysis")
    st.dataframe(analyzed_df)
    
    abnormal_df = analyzed_df[(analyzed_df['Abnormal_Volume']) | (analyzed_df['Abnormal_OI_Change'])]
    st.subheader("Abnormal Volume or OI Change")
    st.dataframe(abnormal_df)
    
    def to_excel(df):
        import io
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
        return output.getvalue()
    
    excel_all = to_excel(analyzed_df)
    excel_abnormal = to_excel(abnormal_df)
    
    st.download_button("Download Full Analysis Excel", data=excel_all, file_name="options_oi_full_analysis.xlsx")
    st.download_button("Download Abnormal Rows Excel", data=excel_abnormal, file_name="options_oi_abnormal_rows.xlsx")
