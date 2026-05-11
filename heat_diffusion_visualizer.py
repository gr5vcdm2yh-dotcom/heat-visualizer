
import streamlit as st
import numpy as np
import time

# --- 設定 ---
GRID_W = 60
GRID_H = 36
AMBIENT_TEMP = 20.0
MAX_TEMP = 100.0

DIFFUSIVITY = {0: 0.015, 1: 0.30, 2: 0.06, 3: 0.005}

def heat_color(temp):
    t = np.clip((temp - AMBIENT_TEMP) / (MAX_TEMP - AMBIENT_TEMP), 0.0, 1.0)
    blue, yellow, red = np.array([40, 80, 220]), np.array([255, 230, 40]), np.array([230, 40, 30])
    if t < 0.5:
        rgb = (1 - t/0.5) * blue + (t/0.5) * yellow
    else:
        a = (t - 0.5) / 0.5
        rgb = (1 - a) * yellow + a * red
    return rgb.astype(int)

def simulate_step(temp_grid, material_grid):
    D = np.vectorize(DIFFUSIVITY.get)(material_grid).astype(float)
    Tp, Dp = np.pad(temp_grid, 1, mode="edge"), np.pad(D, 1, mode="edge")
    center, up, down, left, right = Tp[1:-1, 1:-1], Tp[:-2, 1:-1], Tp[2:, 1:-1], Tp[1:-1, :-2], Tp[1:-1, 2:]
    Du, Dd, Dl, Dr = 0.5*(D+Dp[:-2,1:-1]), 0.5*(D+Dp[2:,1:-1]), 0.5*(D+Dp[1:-1,:-2]), 0.5*(D+Dp[1:-1,2:])
    dt, cooling = 0.18, 0.008
    dT = (Du*(up-center) + Dd*(down-center) + Dl*(left-center) + Dr*(right-center)) - cooling*(center-AMBIENT_TEMP)
    return np.clip(center + dt * dT, AMBIENT_TEMP, MAX_TEMP)

st.set_page_config(page_title="熱拡散シミュレーター", layout="wide")
st.title("🔥 熱の広がり方可視化アプリ")
if 'temp' not in st.session_state:
    st.session_state.temp = np.full((GRID_H, GRID_W), AMBIENT_TEMP)
    st.session_state.material = np.zeros((GRID_H, GRID_W), dtype=int)

st.sidebar.header("設定と操作")
mode = st.sidebar.radio("操作モード", ["材料を配置", "加熱する", "消しゴム"])
mat_type = st.sidebar.selectbox("材料選択", ["金属", "木材", "断熱材"])
mat_map = {"金属": 1, "木材": 2, "断熱材": 3}
pos_x = st.sidebar.slider("X座標", 0, GRID_W-1, GRID_W//2)
pos_y = st.sidebar.slider("Y座標", 0, GRID_H-1, GRID_H//2)

if st.sidebar.button("実行"):
    y_r, x_r = slice(max(0,pos_y-1), pos_y+2), slice(max(0,pos_x-1), pos_x+2)
    if mode == "材料を配置": st.session_state.material[y_r, x_r] = mat_map[mat_type]
    elif mode == "加熱する": st.session_state.temp[y_r, x_r] = MAX_TEMP
    else: 
        st.session_state.material[y_r, x_r] = 0
        st.session_state.temp[y_r, x_r] = AMBIENT_TEMP
if st.sidebar.button("リセット"):
    st.session_state.temp.fill(AMBIENT_TEMP)
    st.session_state.material.fill(0)

placeholder = st.empty()
while True:
    st.session_state.temp = simulate_step(st.session_state.temp, st.session_state.material)
    img = np.zeros((GRID_H, GRID_W, 3), dtype=int)
    for r in range(GRID_H):
        for c in range(GRID_W):
            img[r, c] = heat_color(st.session_state.temp[r, c])
            if st.session_state.material[r, c] != 0: img[r, c] = np.clip(img[r, c] + 20, 0, 255)
    placeholder.image(img.astype(np.uint8), use_container_width=True)
    time.sleep(0.05)
