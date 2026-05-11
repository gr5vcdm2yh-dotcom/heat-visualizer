
import streamlit as st
import numpy as np
import time

# --- 物理定数 ---
GRID_SIZE = 41
CENTER = 20
AMBIENT_TEMP = 20.0
SOURCE_TEMP = 100.0

# 材料の定義（拡散率, 表示色）
MAT_DEFS = {
    0: {"name": "空気",   "diff": 0.02,  "color": [255, 255, 255]},
    1: {"name": "金属",   "diff": 0.25,  "color": [120, 120, 130]},
    2: {"name": "木材",   "diff": 0.08,  "color": [140, 70, 20]},
    3: {"name": "断熱材", "diff": 0.005, "color": [170, 220, 240]}
}

def simulate_step(temp, material):
    # 材料ごとの熱拡散率行列を作成
    D = np.vectorize(lambda x: MAT_DEFS[x]["diff"])(material).astype(float)
    
    # 境界条件のためのパディング
    Tp = np.pad(temp, 1, mode="edge")
    Dp = np.pad(D, 1, mode="edge")
    
    c = Tp[1:-1, 1:-1]
    u, d, l, r = Tp[:-2, 1:-1], Tp[2:, 1:-1], Tp[1:-1, :-2], Tp[1:-1, 2:]
    
    # 隣接セルとの拡散率の平均
    Du = 0.5 * (D + Dp[:-2, 1:-1])
    Dd = 0.5 * (D + Dp[2:, 1:-1])
    Dl = 0.5 * (D + Dp[1:-1, :-2])
    Dr = 0.5 * (D + Dp[1:-1, 2:])
    
    # 熱伝導方程式の差分計算（エラー箇所を修正）
    dT = (Du*(u-c) + Dd*(d-c) + Dl*(l-c) + Dr*(r-c))
    
    # 温度更新（中心の熱源は常にSOURCE_TEMP）
    new_temp = c + 0.2 * dT
    new_temp[CENTER, CENTER] = SOURCE_TEMP
    return np.clip(new_temp, AMBIENT_TEMP, SOURCE_TEMP)

# --- UI設定 ---
st.set_page_config(page_title="熱拡散シミュレーター", layout="centered")
st.title("🌋 中央熱源シミュレーター")

if 'temp' not in st.session_state:
    st.session_state.temp = np.full((GRID_SIZE, GRID_SIZE), AMBIENT_TEMP)
    st.session_state.temp[CENTER, CENTER] = SOURCE_TEMP
    st.session_state.material = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

# --- 操作サイドバー ---
with st.sidebar:
    st.header("🛠️ 材料配置")
    mat_type = st.selectbox("材料", [1, 2, 3], format_func=lambda x: MAT_DEFS[x]["name"])
    
    col1, col2 = st.columns(2)
    tx = col1.number_input("X座標", 0, GRID_SIZE-1, CENTER+5)
    ty = col2.number_input("Y座標", 0, GRID_SIZE-1, CENTER)
    
    if st.button("3x3 ブロックを配置", use_container_width=True):
        st.session_state.material[max(0,ty-1):ty+2, max(0,tx-1):tx+2] = mat_type
        st.rerun()

    st.divider()
    sim_steps = st.slider("計算ステップ数", 50, 500, 150)
    
    if st.button("全リセット", use_container_width=True):
        st.session_state.temp.fill(AMBIENT_TEMP)
        st.session_state.temp[CENTER, CENTER] = SOURCE_TEMP
        st.session_state.material.fill(0)
        st.rerun()

# --- 描画ロジック ---
def generate_frame():
    # ベース色
    img = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)
    for m_id, info in MAT_DEFS.items():
        img[st.session_state.material == m_id] = info["color"]
    
    # 温度オーバーレイ
    t_factor = (st.session_state.temp - AMBIENT_TEMP) / (SOURCE_TEMP - AMBIENT_TEMP)
    t_factor = np.expand_dims(t_factor, axis=2)
    heat_color = np.array([255, 50, 0]) * t_factor # オレンジ〜赤
    
    return np.clip(img * (1 - t_factor * 0.6) + heat_color, 0, 255).astype(np.uint8)

# --- メイン表示 ---
view = st.empty()
view.image(generate_frame(), width=500, caption="配置中（中央が熱源です）")

if st.button("🚀 シミュレーション開始！", type="primary", use_container_width=True):
    for i in range(sim_steps):
        # 1回表示するごとに5ステップ計算（高速化）
        for _ in range(5):
            st.session_state.temp = simulate_step(st.session_state.temp, st.session_state.material)
        
        view.image(generate_frame(), width=500, caption=f"計算中... {i}/{sim_steps}")
        time.sleep(0.01)
    st.success("完了！")
