import streamlit as st
import pandas as pd
import plotly.express as px
from pymongo import MongoClient
from unicorn_binance_websocket_api.manager import BinanceWebSocketApiManager
import threading
import time
from datetime import datetime, timezone

# --- DATABASE SETUP ---
# PASTE YOUR CONNECTION STRING HERE
MONGO_URI = "mongodb+srv://<USERNAME>:<PASSWORD>@student.xxxx.mongodb.net/?retryWrites=true&w=majority"

@st.cache_resource
def init_mongodb():
    client = MongoClient(MONGO_URI)
    # This creates a NEW database called WhaleTracker in your student cluster
    db = client["WhaleTracker"] 
    col = db["LiveTrades"]
    # Auto-delete data older than 24 hours (86,400 seconds)
    col.create_index("createdAt", expireAfterSeconds=86400)
    return col

collection = init_mongodb()

# --- THE TRACKER ENGINE ---
def binance_worker():
    manager = BinanceWebSocketApiManager(exchange="binance.com")
    while True:
        # Get the coins you want to watch from the UI
        current_watchlist = st.session_state.get('watchlist', ['BTCUSDT'])
        for coin in current_watchlist:
            if not manager.get_stream_id_by_label(coin.lower()):
                manager.create_stream('aggTrade', coin.lower(), label=coin.lower())
        
        if manager.is_update_availables():
            old_data = manager.pop_stream_data_from_stream_buffer()
            if old_data and 'data' in old_data:
                d = old_data['data']
                price, qty = float(d['p']), float(d['q'])
                if (price * qty) >= 50000:
                    collection.insert_one({
                        "createdAt": datetime.now(timezone.utc),
                        "symbol": d['s'],
                        "side": "Sell" if d['m'] else "Buy",
                        "price": price,
                        "qty": qty,
                        "value": price * qty
                    })
        time.sleep(0.01)

# Start the tracker in the background
if 'started' not in st.session_state:
    threading.Thread(target=binance_worker, daemon=True).start()
    st.session_state.started = True

# --- DASHBOARD UI ---
st.set_page_config(page_title="24H Whale Tracker", layout="wide")
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ['BTCUSDT', 'ETHUSDT']

# Sidebar for Add/Remove
with st.sidebar:
    st.header("Watchlist")
    add_name = st.text_input("Add Coin (e.g., SOLUSDT)").upper()
    if st.button("Add"):
        if add_name and add_name not in st.session_state.watchlist:
            st.session_state.watchlist.append(add_name)
            st.rerun()

# Display Data
selected = st.selectbox("Select Coin", st.session_state.watchlist)
cursor = collection.find({"symbol": selected}).sort("createdAt", -1)
df = pd.DataFrame(list(cursor))


if not df.empty:
    st.dataframe(df.drop(columns=['_id']), use_container_width=True)
    fig = px.line(df, x="createdAt", y="price", title=f"{selected} Price History (Whale Trades Only)")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No big trades detected in the last 24 hours.")

time.sleep(10)
st.rerun()