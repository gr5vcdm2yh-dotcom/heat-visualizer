import streamlit as st
import numpy as np
import time
from streamlit_drawable_canvas import st_canvas

# --- 設定 ---
GRID_W, GRID_H = 60, 36
AMBIENT_TEMP = 20.0
MAX_TEMP = 100.0

# 材料ごとの物性と表示色
MATERIALS = {
    0: {"name": "空気", "diff": 0.015, "color": [240, 240, 240]}, # 薄グレー
    1: {"name": "金属", "diff": 0.30,  "color": [100, 100, 110]}, # 濃グレー
    2: {"name": "木材", "diff": 0.06,  "color": [139, 69, 19]},   # 茶色
    3: {"name": "断熱材", "diff": 0.005, "color": [0, 255, 255]}   # 水色
}

# --- 物理計算 ---
def simulate_step(temp_grid, material_grid):
    # 拡散率のマップ作成
    D = np.vectorize(lambda x: MATERIALS[x]["diff"])(material_grid).astype(float)
    Tp, Dp = np.pad(temp_grid, 1, mode="edge"), np.pad(D, 1, mode="edge")
    
    center = Tp[1:-1, 1:-1]
    up, down = Tp[:-2, 1:-1], Tp[2:, 1:-1]
    left, right = Tp[1:-1, :-2], Tp[1:-1, 2:]
    
    # 境界での熱伝導率の平均
    Du, Dd = 0.5*(D + Dp[:-2, 1:-1]), 0.5*(D + Dp[2:, 1:-1])
    Dl, Dr = 0.5*(D + Dp[1:-1, :-2]), 0.5*(D + Dp[1:-1, 2:])

    dt, cooling = 0.15, 0.005
    dT = (Du*(up-center) + Dd*(down-center) + Dl*(left-center) + Dr*(right-center)) - cooling*(center-AMBIENT_TEMP)
    return np.clip(center + dt * dT, AMBIENT_TEMP, MAX_TEMP)

# --- UI設定 ---
st.set_page_config(page_title="熱拡散シミュレーター Pro", layout="wide")
st.title("🌡️ マウスで描ける熱拡散シミュレーター")

if 'temp' not in st.session_state:
    st.session_state.temp = np.full((GRID_H, GRID_W), AMBIENT_TEMP)
    st.session_state.material = np.zeros((GRID_H, GRID_W), dtype=int)

# --- サイドバー ---
with st.sidebar:
    st.header("🖊️ 描画ツール")
    tool = st.radio("モード選択", ["材料を配置", "熱を加える", "消しゴム"])
    
    selected_mat = 1
    if tool == "材料を配置":
        mat_choice = st.selectbox("材料の種類", ["金属", "木材", "断熱材"])
        selected_mat = {"金属": 1, "木材": 2, "断熱材": 3}[mat_choice]
    
    st.divider()
    if st.button("リセット"):
        st.session_state.temp.fill(AMBIENT_TEMP)
        st.session_state.material.fill(0)
        st.rerun()

    st.write("※マウスで右のキャンバスをなぞってください")

# --- メインエリア：キャンバス描画 ---
# 背景画像の作成（現在の材料と温度を合成）
def generate_preview():
    img = np.zeros((GRID_H, GRID_W, 3), dtype=np.uint8)
    for m_id, m_info in MATERIALS.items():
        mask = st.session_state.material == m_id
        img[mask] = m_info["color"]
    
    # 温度による赤みをオーバーレイ
    t_factor = (st.session_state.temp - AMBIENT_TEMP) / (MAX_TEMP - AMBIENT_TEMP)
    img[:, :, 0] = np.clip(img[:, :, 0] + t_factor * 200, 0, 255) # 赤を強く
    return img

bg_img = generate_preview()

col_canvas, col_info = st.columns([3, 1])

with col_canvas:
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=2,
        stroke_color="#000000",
        background_image=None, 
        update_streamlit=True,
        height=GRID_H * 15,
        width=GRID_W * 15,
        drawing_mode="freedraw",
        key="canvas",
    )

# --- 描画データの反映 ---
if canvas_result.image_data is not None:
    # キャンバスの描画データからグリッド座標へ変換
    mask = canvas_result.image_data[:, :, 3] > 0 # 透明じゃない部分
    if np.any(mask):
        import cv2 # サイズ調整用
        mask_resized = cv2.resize(mask.astype(np.uint8), (GRID_W, GRID_H), interpolation=cv2.INTER_NEAREST)
        
        if tool == "材料を配置":
            st.session_state.material[mask_resized > 0] = selected_mat
        elif tool == "熱を加える":
            st.session_state.temp[mask_resized > 0] = MAX_TEMP
        elif tool == "消しゴム":
            st.session_state.material[mask_resized > 0] = 0
            st.session_state.temp[mask_resized > 0] = AMBIENT_TEMP

# --- シミュレーション実行と表示 ---
placeholder = st.empty()

# 実行ループ
while True:
    st.session_state.temp = simulate_step(st.session_state.temp, st.session_state.material)
    
    # 最終的な表示用画像
    display_img = generate_preview()
    placeholder.image(display_img, use_container_width=True)
    
    time.sleep(0.05)
