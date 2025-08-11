import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote

# --- Database connection setup ---
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

# --- Data retrieval functions ---
@st.cache_data(show_spinner=False)
def fetch_distinct_values(column: str, 
                         trade_date: str = None, 
                         symbol: str = None,
                         expiry_date: str = None,
                         strike_price: float = None,
                         option_type: str = None,
                         order_by: str = None):
    order_by = order_by or column
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
    sql += f" ORDER BY {order_by}"
    
    query = text(sql)
    df = pd.read_sql(query, engine, params=filters)
    return df[column].tolist()

@st.cache_data(show_spinner=False)
def load_option_data(trade_date: str,
                     symbol: str,
                     expiry_date: str,
                     strike_price: float,
                     option_type: str):
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
    return df

# --- Analysis function ---
def analyze_oi_volume(df, k=2):
    df = df.sort_values("timestamp").copy()
    df['Price_Change'] = df['close'].diff()
    df['OI_Change'] = df['open_interest'].diff()
    df['Vol_Mean'] = df['volume'].expanding(min_periods=1).mean()
    df['Vol_Std'] = df['volume'].expanding(min_periods=2).std().fillna(0)
    df['OI_Mean'] = df['OI_Change'].expanding(min_periods=1).mean()
    df['OI_Std'] = df['OI_Change'].expanding(min_periods=2).std().fillna(0)

    df['Abnormal_Volume'] = df['volume'] > (df['Vol_Mean'] + k * df['Vol_Std'])
    df['Abnormal_OI_Change'] = df['OI_Change'].abs() > (df['OI_Mean'].abs() + k * df['OI_Std'].abs())

    def interpret_row(row):
        price_ch = row['Price_Change']
        oi_ch = row['OI_Change']
        vol = row['volume']
        vol_abn = row['Abnormal_Volume']
        oi_abn = row['Abnormal_OI_Change']

        if price_ch > 0 and oi_ch > 0 and vol > 0:
            base = "Bullish activity"
        elif price_ch < 0 and oi_ch < 0 and vol > 0:
            base = "Bearish activity"
        else:
            base = "Neutral/No clear trend"

        if vol_abn and oi_abn:
            return f"{base} + Abnormal Volume & OI Change"
        if vol_abn:
            return f"{base} + Abnormal Volume"
        if oi_abn:
            return f"{base} + Abnormal OI Change"
        return base

    df['Interpretation'] = df.apply(interpret_row, axis=1)

    return df[[
        'timestamp', 'close', 'Price_Change', 'open_interest', 'OI_Change', 'volume',
        'Abnormal_Volume', 'Abnormal_OI_Change', 'Interpretation'
    ]]

# --- Streamlit UI ---
st.title("Options 3-min OI & Volume Abnormality Analysis")

trade_dates = fetch_distinct_values("trade_date")
selected_date = st.sidebar.selectbox("Select Trade Date", trade_dates)

symbols = fetch_distinct_values("symbol", trade_date=selected_date)
selected_symbol = st.sidebar.selectbox("Select Symbol", symbols)

expiries = fetch_distinct_values("expiry_date", trade_date=selected_date, symbol=selected_symbol)
selected_expiry = st.sidebar.selectbox("Select Expiry Date", expiries)

strikes = fetch_distinct_values("strike_price", trade_date=selected_date, symbol=selected_symbol, expiry_date=selected_expiry)
selected_strike = st.sidebar.selectbox("Select Strike Price", strikes)

option_types = fetch_distinct_values("option_type", trade_date=selected_date, symbol=selected_symbol, expiry_date=selected_expiry, strike_price=selected_strike)
selected_option_type = st.sidebar.selectbox("Select Option Type", option_types)

df_data = load_option_data(
    trade_date=selected_date,
    symbol=selected_symbol,
    expiry_date=selected_expiry,
    strike_price=selected_strike,
    option_type=selected_option_type
)

if df_data.empty:
    st.warning("No data found for selected filters.")
else:
    analyzed_df = analyze_oi_volume(df_data)

    st.subheader("Full Analysis")
    st.dataframe(analyzed_df, use_container_width=True)

    abnormal_df = analyzed_df[(analyzed_df['Abnormal_Volume']) | (analyzed_df['Abnormal_OI_Change'])]
    st.subheader("Abnormal Volume or OI Change")
    st.dataframe(abnormal_df, use_container_width=True)

    # Excel export function
    def df_to_excel_bytes(df):
        import io
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        return output.getvalue()

    st.download_button(
        label="Download Full Analysis Excel",
        data=df_to_excel_bytes(analyzed_df),
        file_name="options_oi_full_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    st.download_button(
        label="Download Abnormal Rows Excel",
        data=df_to_excel_bytes(abnormal_df),
        file_name="options_oi_abnormal_rows.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
