import streamlit as st
import pandas as pd
import subprocess
import threading
import time
from monitor import check_device_status
from streamlit_autorefresh import st_autorefresh
from streamlit.runtime.scriptrunner import add_script_run_ctx

# ------------------ 页面配置 ------------------
st.set_page_config(page_title="FedFM Dashboard", layout="wide")
st.title("🌼 FedFM — Federated Learning Monitor")

# ------------------ 初始化 Session State ------------------
st.session_state.setdefault("device_status", None)
st.session_state.setdefault("last_ping_time", 0)
st.session_state.setdefault("flwr_proc", None)
st.session_state.setdefault("flwr_logs", [])

# ------------------ 自动设备刷新 ------------------
PING_INTERVAL = 10  # 秒
now = time.time()

rank_agx = 64
rank_orin = 16
rank_rpi = 4

DEVICE_TABLE = [
    {"node": "server",   "name": "laptop",      "ip": "127.0.0.1", "rank": "/"},
    {"node": "client 1", "name": "agx-orin",    "ip": "127.0.0.21","rank": rank_agx},
    {"node": "client 2", "name": "orin-nano-1", "ip": "127.0.0.31","rank": rank_orin},
    {"node": "client 3", "name": "orin-nano-2", "ip": "192.168.0.32","rank": rank_orin},
    {"node": "client 4", "name": "orin-nano-3", "ip": "192.168.0.33","rank": rank_orin},
    {"node": "client 5", "name": "orin-nano-4", "ip": "192.168.0.34","rank": rank_orin},
    {"node": "client 6", "name": "rpi-5-1",     "ip": "127.0.0.41","rank": rank_rpi},
    {"node": "client 7", "name": "rpi-5-2",     "ip": "192.168.0.42","rank": rank_rpi},
    {"node": "client 8", "name": "rpi-5-3",     "ip": "192.168.0.43","rank": rank_rpi},
    {"node": "client 9", "name": "rpi-5-4",     "ip": "192.168.0.44","rank": rank_rpi},
    {"node": "client 10","name": "rpi-5-5",     "ip": "192.168.0.45","rank": rank_rpi},
]

# 自动刷新
st_autorefresh(interval=10_000, key="auto_ping")

if st.session_state.device_status is None or now - st.session_state.last_ping_time > PING_INTERVAL:
    with st.spinner("🔄 Pinging devices..."):
        st.session_state.device_status = check_device_status(DEVICE_TABLE)
        st.session_state.last_ping_time = now

# ------------------ 启动 FLWR 服务函数 ------------------
def read_output():
    try:
        for line in st.session_state.flwr_proc.stdout:
            append_log(line)
    except Exception as e:
        append_log(f"\n[ERROR] {e}\n")

def start_flwr_server():
    if st.session_state.flwr_proc is not None:
        return

    proc = subprocess.Popen(
        ["flwr", "run", ".", "local-deployment", "--stream"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True
    )
    st.session_state.flwr_proc = proc
    append_log(">>> FLWR server started...\n")

    thread = threading.Thread(target=read_output, daemon=True)
    add_script_run_ctx(thread)
    thread.start()

def stop_flwr_server():
    if st.session_state.flwr_proc is not None:
        st.session_state.flwr_proc.terminate()
        st.session_state.flwr_proc = None
        append_log(">>> FLWR server stopped.\n")

MAX_LOG_LINES = 500

def append_log(line: str):
    logs = st.session_state.get("flwr_logs", [])
    logs.append(line)
    if len(logs) > MAX_LOG_LINES:
        logs = logs[-MAX_LOG_LINES:]
    st.session_state.flwr_logs = logs

# ------------------ 侧边栏 ------------------
st.sidebar.header("🛠️ FL Control")
total_rounds = st.sidebar.selectbox("FL Total Rounds", [20, 30, 50, 100], index=2)
clients_per_round = st.sidebar.selectbox("FL Clients per Round", [0.1, 0.2, 0.5, 1], index=1)
agg_method = st.sidebar.selectbox("Aggregation Method", ["FedFFT", "FedLoRA-ZeroPadding", "FlexLoRA", "FLoRA", "FLASH"], index=4)
scaling_method = st.sidebar.selectbox("Scaling Method", ["fixed", "normal", "sqrt"], index=1)

start = st.sidebar.button("🚀 Start Server", key="start_btn")
stop = st.sidebar.button("⛔ Stop Server", key="stop_btn")

# ------------------ 页面布局 ------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📡 Device Status")
    df = pd.DataFrame(st.session_state.device_status)
    df.columns = ["Node", "Device Name", "IP", "Rank", "Status"]
    styled_df = df.style.set_table_styles(
        [{"selector":"th","props":[("text-align","center")]},
         {"selector":"td","props":[("text-align","center")]}]
    ).hide(axis="index")
    st.markdown(styled_df.to_html(escape=False), unsafe_allow_html=True)

with col2:
    st.subheader("🖥️ Training Terminal")

    if start:
        append_log(">>> Launching FLWR server...\n")
        start_flwr_server()
    elif stop:
        stop_flwr_server()

    # 初始化 log
    if "flwr_logs" not in st.session_state or not st.session_state.flwr_logs:
        st.session_state.flwr_logs = [">>> Waiting for server to start...\n"]

    # 实时日志输出窗口（主线程更新）
    log_box = st.empty()
    log_box.code("".join(st.session_state.flwr_logs[-50:]), language="bash")


