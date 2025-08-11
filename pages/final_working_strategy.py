import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote

def get_db_engine():
    user = st.secrets["postgres"]["user"]
    password = quote(st.secrets["postgres"]["password"])
    host = st.secrets["postgres"]["host"]
    port = st.secrets["postgres"]["port"]
    database = st.secrets["postgres"]["database"]
    url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    return create_engine(url)

engine = get_db_engine()

def fetch_distinct_values(column: str, trade_date=None, symbol=None, expiry_date=None, strike_price=None, option_type=None):
    print(f"Fetching distinct {column} with filters: trade_date={trade_date}, symbol={symbol}")
    filters = {}
    where_clauses = []
    if trade_date:
        filters["trade_date"] = trade_date
        where_clauses.append("trade_date = :trade_date")
    if symbol:
        filters["symbol"] = symbol
        where_clauses.append("symbol = :symbol")
    if expiry_date:
        filters["expiry_date"] = expiry_date
        where_clauses.append("expiry_date = :expiry_date")
    if strike_price:
        filters["strike_price"] = strike_price
        where_clauses.append("strike_price = :strike_price")
    if option_type:
        filters["option_type"] = option_type
        where_clauses.append("option_type = :option_type")
    sql = f"SELECT DISTINCT {column} FROM option_3min_ohlc"
    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)
    sql += f" ORDER BY {column}"
    query = text(sql)
    df = pd.read_sql(query, engine, params=filters)
    return df[column].tolist()

def load_option_data(trade_date, symbol, expiry_date, strike_price, option_type):
    print(f"Loading option data for {symbol} {strike_price} {option_type} on {trade_date}")
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

st.write(f"Loaded {len(df_data)} rows")

if df_data.empty:
    st.warning("No data found for selected filters.")
else:
    st.dataframe(df_data)
