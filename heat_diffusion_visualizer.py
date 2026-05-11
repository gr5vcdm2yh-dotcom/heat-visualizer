import streamlit as st
import numpy as np
import time

# --- 物理設定 ---
GRID_SIZE = 41  # 41x41の正方形（中心が特定しやすい）
CENTER = 20     # 中心座標
AMBIENT_TEMP = 20.0
SOURCE_TEMP = 100.0

# 材料データ（名前, 熱拡散率, 表示色RGB）
MAT_DEFS = {
    0: {"name": "空気",   "diff": 0.02,  "color": [255, 255, 255]}, # 白
    1: {"name": "金属",   "diff": 0.25,  "color": [100, 100, 110]}, # 濃灰
    2: {"name": "木材",   "diff": 0.08,  "color": [139, 69, 19]},   # 茶
    3: {"name": "断熱材", "diff": 0.005, "color": [173, 216, 230]}  # 水色
}

# --- 高速物理演算 ---
def simulate_step(temp, material):
    D = np.vectorize(lambda x: MAT_DEFS[x]["diff"])(material).astype(float)
    Tp = np.pad(temp, 1, mode="edge")
    Dp = np.pad(D, 1, mode="edge")
    
    c = Tp[1:-1, 1:-1]
    u, d, l, r = Tp[:-2, 1:-1], Tp[2:, 1:-1], Tp[1:-1, :-2], Tp[1:-1, 2:]
    Du, Dd = 0.5*(D + Dp[:-2, 1:-1]), 0.5*(D + Dp[2:, 1:-1])
    Dl, Dr = 0.5*(D + Dp[1:-1, :-2]), 0.5*(D + Dp[1:-1, 2:])
    
    # 熱伝導方程式の差分法
    dT = (Du*(u-c) + Dd*(down-center) + Dl*(left-center) + Dr*(right-center)) # 簡易化
    # 修正版
    dT = (Du*(u-c) + Dd*(d-c) + Dl*(l-c) + Dr*(r-c))
    new_temp = c + 0.2 * dT
    new_temp[CENTER, CENTER] = SOURCE_TEMP # 中心は常に熱源
    return np.clip(new_temp, AMBIENT_TEMP, SOURCE_TEMP)

# --- UI構築 ---
st.set_page_config(page_title="熱拡散シミュレーター", layout="centered")
st.title("🌋 中央熱源シミュレーター (軽量版)")

if 'temp' not in st.session_state:
    st.session_state.temp = np.full((GRID_SIZE, GRID_SIZE), AMBIENT_TEMP)
    st.session_state.temp[CENTER, CENTER] = SOURCE_TEMP
    st.session_state.material = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

# --- 操作パネル ---
with st.sidebar:
    st.header("🛠️ 配置と設定")
    mat_type = st.selectbox("配置する材料", [1, 2, 3], 
                            format_func=lambda x: MAT_DEFS[x]["name"])
    
    col1, col2 = st.columns(2)
    with col1: tx = st.number_input("X座標", 0, GRID_SIZE-1, CENTER+5)
    with col2: ty = st.number_input("Y座標", 0, GRID_SIZE-1, CENTER)
    
    if st.button("指定座標に 3x3 配置", use_container_width=True):
        st.session_state.material[max(0,ty-1):ty+2, max(0,tx-1):tx+2] = mat_type
        st.toast(f"({tx}, {ty}) に{MAT_DEFS[mat_type]['name']}を置きました")

    st.divider()
    sim_speed = st.slider("シミュレーション速度", 1, 10, 5)
    if st.button("全リセット", use_container_width=True):
        st.session_state.temp.fill(AMBIENT_TEMP)
        st.session_state.temp[CENTER, CENTER] = SOURCE_TEMP
        st.session_state.material.fill(0)
        st.rerun()

# --- シミュレーション表示 ---
def get_image():
    # 材料の色をベースにする
    img = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)
    for m_id, info in MAT_DEFS.items():
        img[st.session_state.material == m_id] = info["color"]
    
    # 温度を赤色として重ね合わせる
    t_factor = (st.session_state.temp - AMBIENT_TEMP) / (SOURCE_TEMP - AMBIENT_TEMP)
    t_factor = np.expand_dims(t_factor, axis=2)
    # 熱をオレンジ〜赤のグラデーションで表現
    heat_rgb = np.array([255, 60, 0]) * t_factor
    final_img = np.clip(img * (1 - t_factor * 0.7) + heat_rgb, 0, 255).astype(np.uint8)
    return final_img

st.write("### 現在の配置状況")
placeholder = st.empty()
placeholder.image(get_image(), width=500)

if st.button("🚀 シミュレーション開始！", type="primary", use_container_width=True):
    for _ in range(100): # 100ステップ実行
        # 計算をまとめて行う（表示回数を減らして高速化）
        for _ in range(sim_speed):
            st.session_state.temp = simulate_step(st.session_state.temp, st.session_state.material)
        
        placeholder.image(get_image(), width=500)
        time.sleep(0.01)
    st.success("シミュレーション完了")
