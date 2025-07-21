import streamlit as st
import pandas as pd
from ip_monitor import check_device_status
from tomlkit import parse, dumps
from tomlkit.toml_document import TOMLDocument
import time

def update_pyproject_toml(
    rank_agx, rank_orin, rank_rpi, total_rounds, ratio_per_round, agg_method,
    file_path="pyproject.toml"
):
    with open(file_path, "r", encoding="utf-8") as f:
        doc: TOMLDocument = parse(f.read())

    fl_method_map = {
        "FedLoRA-ZeroPadding": "zero-padding",
        "FlexLoRA": "svd",
        "FLoRA": "nbias",
        "FLASH": "vanilla",
    }

    config = doc["tool"]["flwr"]["app"]["config"]

    config["num-server-rounds"] = total_rounds
    config["strategy"]["fraction-fit"]= ratio_per_round
    config["fl"]["rank-choices"] = f"{rank_agx},{rank_orin},{rank_rpi}"
    config["fl"]["fl-method"] = fl_method_map.get(agg_method, "svd")
    config["fl"]["peft-name"] = "ffa" if agg_method == "FLASH" else "lora"
    config["fl"]["scaling-method"] = "sqrt" if agg_method == "FLASH" else "normal"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(dumps(doc))

# ------------------ Page Config ------------------
st.set_page_config(page_title="FLASH - Demo 2", layout="wide")
st.title("🌼 FLASH - Demo 2")

# ------------------ Init Session State ------------------
st.session_state.setdefault("device_status", None)
st.session_state.setdefault("flwr_proc", None)

# ------------------ Sidebar FL Config ------------------
st.sidebar.header("💾 LoRA Rank Settings")
rank_agx = st.sidebar.number_input("AGX Orin Rank", min_value=1, max_value=192, value=64, step=8)
rank_orin = st.sidebar.number_input("Orin Nano Rank", min_value=1, max_value=192, value=16, step=8)
rank_rpi = st.sidebar.number_input("Raspberry Pi 5 Rank", min_value=1, max_value=192, value=4, step=8)

st.sidebar.header("🛠️ FL Parameter Settings")
total_rounds = st.sidebar.number_input("Total Rounds", min_value=1, max_value=200, value=3, step=1)
ratio_per_round = st.sidebar.number_input("Client Ratio per Round", min_value=0.0, max_value=1.0, value=1.0, step=0.1)
agg_method = st.sidebar.selectbox("Aggregation Method", ["FedLoRA-ZeroPadding", "FlexLoRA", "FLoRA", "FLASH"], index=3)

save = st.sidebar.button("🚀 Save Configurations", key="save_btn")

def get_current_device_status():
    updated_table = [
        {"node": "server",   "name": "laptop",      "ip": "192.168.0.11", "rank": "/"},
        {"node": "client 1", "name": "agx-orin",    "ip": "192.168.0.21","rank": rank_agx},
        {"node": "client 2", "name": "orin-nano-1", "ip": "192.168.0.31","rank": rank_orin},
        {"node": "client 3", "name": "orin-nano-2", "ip": "192.168.0.32","rank": rank_orin},
        {"node": "client 4", "name": "orin-nano-3", "ip": "192.168.0.33","rank": rank_orin},
        {"node": "client 5", "name": "orin-nano-4", "ip": "192.168.0.34","rank": rank_orin},
        {"node": "client 6", "name": "rpi-5-1",     "ip": "192.168.0.41","rank": rank_rpi},
        {"node": "client 7", "name": "rpi-5-2",     "ip": "192.168.0.42","rank": rank_rpi},
        {"node": "client 8", "name": "rpi-5-3",     "ip": "192.168.0.43","rank": rank_rpi},
        {"node": "client 9", "name": "rpi-5-4",     "ip": "192.168.0.44","rank": rank_rpi},
        {"node": "client 10","name": "rpi-5-5",     "ip": "192.168.0.45","rank": rank_rpi},
    ]
    return updated_table

def _emoji(ok: bool) -> str:
    return "🟢" if ok else "🔴"

if st.button("🔍 Ping Devices"):
    with st.spinner("Pinging devices..."):
        st.session_state.device_status = check_device_status(get_current_device_status())

if st.session_state.device_status is None:
    st.session_state.device_status = [
        {**device, "status": _emoji(False)} for device in get_current_device_status()
    ]
else:
    for i, device in enumerate(get_current_device_status()):
        st.session_state.device_status[i]["rank"] = device["rank"]

if save:
    update_pyproject_toml(rank_agx, rank_orin, rank_rpi, total_rounds, ratio_per_round, agg_method)
    success_box = st.empty()
    success_box.success("Configuration saved to pyproject.toml")

    time.sleep(2)
    success_box.empty()

# ------------------ Page Layout ------------------
st.subheader("📡 Device Status")

df = pd.DataFrame(st.session_state.device_status)
df.columns = ["Node", "Device Name", "IP", "Rank", "Status"]

styled_df = df.style.set_table_styles([
    {"selector": "th", "props": [("text-align", "center"), ("padding", "6px")]},
    {"selector": "td", "props": [("text-align", "center"), ("word-wrap", "break-word"), ("max-width", "150px")]},
]).hide(axis="index")

html = styled_df.to_html(escape=False)
html = html.replace('<table ', '<table style="width: 100%; table-layout: fixed;" ')
st.markdown(html, unsafe_allow_html=True)

# ------------------ Compare Videos Section ------------------
st.subheader("🎥 Object Detection Comparison")

gif_pairs = [
    ("/home/fw407/workspace/FedFM/detection/output/gif/1_base.gif", "/home/fw407/workspace/FedFM/detection/output/gif/1_ft.gif"),
    ("/home/fw407/workspace/FedFM/detection/output/gif/2_base.gif", "/home/fw407/workspace/FedFM/detection/output/gif/2_ft.gif"),
    ("/home/fw407/workspace/FedFM/detection/output/gif/3_base.gif", "/home/fw407/workspace/FedFM/detection/output/gif/3_ft.gif"),
    ("/home/fw407/workspace/FedFM/detection/output/gif/4_base.gif", "/home/fw407/workspace/FedFM/detection/output/gif/4_ft.gif"),
]

for i, (base_gif, fine_gif) in enumerate(gif_pairs, start=1):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"#### Group {i}: Base Model (Deformable DETR)", unsafe_allow_html=True)
        st.markdown("<div style='text-align: center; '>", unsafe_allow_html=True)
        st.image(base_gif)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("#### Fine-tuned Model (with new classes on LVIS)", unsafe_allow_html=True)
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.image(fine_gif)
        st.markdown("</div>", unsafe_allow_html=True)




