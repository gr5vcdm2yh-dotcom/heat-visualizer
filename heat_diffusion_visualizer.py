import streamlit as st
import numpy as np
import time
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# --- 物理設定（解像度を少し下げて軽量化） ---
GRID_W, GRID_H = 50, 30
AMBIENT_TEMP = 20.0
MAX_TEMP = 100.0

MAT_DEFS = {
    0: {"name": "空気", "diff": 0.015, "color": [240, 240, 240]},
    1: {"name": "金属", "diff": 0.30,  "color": [100, 100, 110]},
    2: {"name": "木材", "diff": 0.06,  "color": [139, 69, 19]},
    3: {"name": "断熱材", "diff": 0.005, "color": [173, 216, 230]}
}

def simulate_step(temp, material):
    D = np.vectorize(lambda x: MAT_DEFS[x]["diff"])(material).astype(float)
    Tp, Dp = np.pad(temp, 1, mode="edge"), np.pad(D, 1, mode="edge")
    c = Tp[1:-1, 1:-1]
    u, d, l, r = Tp[:-2, 1:-1], Tp[2:, 1:-1], Tp[1:-1, :-2], Tp[1:-1, 2:]
    Du, Dd = 0.5*(D + Dp[:-2, 1:-1]), 0.5*(D + Dp[2:, 1:-1])
    Dl, Dr = 0.5*(D + Dp[1:-1, :-2]), 0.5*(D + Dp[1:-1, 2:])
    dt, cooling = 0.12, 0.004 # 安定のためdtを少し小さく
    dT = (Du*(u-c) + Dd*(d-c) + Dl*(l-c) + Dr*(r-c)) - cooling*(c-AMBIENT_TEMP)
    return np.clip(c + dt * dT, AMBIENT_TEMP, MAX_TEMP)

# --- UI設定 ---
st.set_page_config(page_title="熱拡散シミュレーター", layout="wide")
st.title("🌡️ 熱拡散シミュレーター：軽量版")

if 'temp' not in st.session_state:
    st.session_state.temp = np.full((GRID_H, GRID_W), AMBIENT_TEMP)
    st.session_state.material = np.zeros((GRID_H, GRID_W), dtype=int)
    st.session_state.run_sim = True

# --- サイドバー：操作をここに集約 ---
with st.sidebar:
    st.header("🎨 配置・制御")
    st.session_state.run_sim = st.checkbox("シミュレーション実行", value=True)
    tool = st.radio("モード", ["材料を置く", "熱を加える", "消しゴム"])
    
    selected_mat = 1
    if tool == "材料を置く":
        mat_label = st.selectbox("材料", ["金属", "木材", "断熱材"])
        selected_mat = {"金属": 1, "木材": 2, "断熱材": 3}[mat_label]
    
    if st.button("全リセット"):
        st.session_state.temp.fill(AMBIENT_TEMP)
        st.session_state.material.fill(0)
        st.rerun()

# --- メインエリア ---
# キャンバス（描画時のみサーバーへ通信するように設定）
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=10,
    stroke_color="#000000",
    update_streamlit=False, # ← これが重要！マウスを離した時だけ更新
    height=300,
    width=500,
    drawing_mode="freedraw",
    key="canvas",
)

# 描画された内容の反映
if canvas_result.image_data is not None:
    draw_mask = canvas_result.image_data[:, :, 3] > 0
    if np.any(draw_mask):
        mask_small = Image.fromarray(draw_mask).resize((GRID_W, GRID_H), resample=Image.NEAREST)
        mask_np = np.array(mask_small)
        if tool == "材料を置く": st.session_state.material[mask_np] = selected_mat
        elif tool == "熱を加える": st.session_state.temp[mask_np] = MAX_TEMP
        elif tool == "消しゴム": 
            st.session_state.material[mask_np] = 0
            st.session_state.temp[mask_np] = AMBIENT_TEMP

# --- シミュレーション表示 ---
placeholder = st.empty()

# 描画間隔を調整して負荷軽減
for _ in range(100): # 無限ループではなく回数を区切るか、適切にsleepを入れる
    if st.session_state.run_sim:
        # 計算は複数ステップまとめて行うと高速
        for _ in range(3):
            st.session_state.temp = simulate_step(st.session_state.temp, st.session_state.material)
        
        # 表示用画像の作成
        img = np.zeros((GRID_H, GRID_W, 3), dtype=np.uint8)
        for m_id, info in MAT_DEFS.items():
            img[st.session_state.material == m_id] = info["color"]
        
        t_factor = (st.session_state.temp - AMBIENT_TEMP) / (MAX_TEMP - AMBIENT_TEMP)
        t_factor = np.expand_dims(t_factor, axis=2)
        heat_overlay = np.array([255, 50, 0]) * t_factor
        final_img = np.clip(img * (1 - t_factor*0.6) + heat_overlay, 0, 255).astype(np.uint8)
        
        placeholder.image(final_img, width=800)
    
    time.sleep(0.1) # 描画頻度を下げる（0.1秒 = 10fps）
