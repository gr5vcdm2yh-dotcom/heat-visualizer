import streamlit as st
import numpy as np
import time
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# --- 物理設定 ---
GRID_W, GRID_H = 60, 36
AMBIENT_TEMP = 20.0
MAX_TEMP = 100.0

# 材料データ（名前, 熱拡散率, 表示色）
MAT_DEFS = {
    0: {"name": "空気", "diff": 0.015, "color": [255, 255, 255]}, # 白
    1: {"name": "金属", "diff": 0.30,  "color": [128, 128, 128]}, # グレー
    2: {"name": "木材", "diff": 0.06,  "color": [139, 69, 19]},   # 茶色
    3: {"name": "断熱材", "diff": 0.005, "color": [173, 216, 230]}  # 水色
}

# --- 物理計算ロジック ---
def simulate_step(temp, material):
    D = np.vectorize(lambda x: MAT_DEFS[x]["diff"])(material).astype(float)
    Tp, Dp = np.pad(temp, 1, mode="edge"), np.pad(D, 1, mode="edge")
    c = Tp[1:-1, 1:-1]
    u, d, l, r = Tp[:-2, 1:-1], Tp[2:, 1:-1], Tp[1:-1, :-2], Tp[1:-1, 2:]
    Du, Dd = 0.5*(D + Dp[:-2, 1:-1]), 0.5*(D + Dp[2:, 1:-1])
    Dl, Dr = 0.5*(D + Dp[1:-1, :-2]), 0.5*(D + Dp[1:-1, 2:])
    dt, cooling = 0.15, 0.005
    dT = (Du*(u-c) + Dd*(d-c) + Dl*(l-c) + Dr*(r-c)) - cooling*(c-AMBIENT_TEMP)
    return np.clip(c + dt * dT, AMBIENT_TEMP, MAX_TEMP)

# --- 表示用画像生成 ---
def generate_display_image(temp, material):
    img = np.zeros((GRID_H, GRID_W, 3), dtype=np.uint8)
    for m_id, info in MAT_DEFS.items():
        img[material == m_id] = info["color"]
    
    # 熱を「赤み」として合成
    t_factor = (temp - AMBIENT_TEMP) / (MAX_TEMP - AMBIENT_TEMP)
    t_factor = np.expand_dims(t_factor, axis=2)
    heat_overlay = np.array([255, 50, 0]) * t_factor # オレンジ〜赤
    img = np.clip(img * (1 - t_factor*0.5) + heat_overlay, 0, 255).astype(np.uint8)
    return img

# --- UI設定 ---
st.set_page_config(page_title="熱拡散シミュレーター", layout="wide")
st.title("🌡️ 熱拡散シミュレーター：マウスでお絵描き配置")

if 'temp' not in st.session_state:
    st.session_state.temp = np.full((GRID_H, GRID_W), AMBIENT_TEMP)
    st.session_state.material = np.zeros((GRID_H, GRID_W), dtype=int)

# --- サイドバー操作 ---
with st.sidebar:
    st.header("🎨 配置ツール")
    tool = st.radio("モード", ["材料を置く", "熱を加える", "消しゴム"])
    
    selected_mat = 1
    if tool == "材料を置く":
        mat_label = st.selectbox("材料を選択", ["金属", "木材", "断熱材"])
        selected_mat = {"金属": 1, "木材": 2, "断熱材": 3}[mat_label]
    
    st.divider()
    if st.button("全リセット"):
        st.session_state.temp.fill(AMBIENT_TEMP)
        st.session_state.material.fill(0)
        st.rerun()
    
    st.write("右のキャンバスをマウスでなぞって描いてください。")

# --- キャンバス表示と入力 ---
# 現在の状態をプレビューとして表示
bg_img = generate_display_image(st.session_state.temp, st.session_state.material)
bg_pil = Image.fromarray(bg_img).resize((GRID_W*10, GRID_H*10), resample=Image.NEAREST)

st.write("### 描画エリア (マウスでなぞってください)")
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=15,
    stroke_color="#000000",
    background_image=bg_pil,
    update_streamlit=True,
    height=GRID_H * 10,
    width=GRID_W * 10,
    drawing_mode="freedraw",
    key="canvas",
)

# 描画された内容をグリッドに反映
if canvas_result.image_data is not None:
    # 描画マスクを取得
    draw_mask = canvas_result.image_data[:, :, 3] > 0
    if np.any(draw_mask):
        # 描画サイズ(600x360)をグリッドサイズ(60x36)に縮小
        mask_small = Image.fromarray(draw_mask).resize((GRID_W, GRID_H), resample=Image.NEAREST)
        mask_np = np.array(mask_small)
        
        if tool == "材料を置く":
            st.session_state.material[mask_np] = selected_mat
        elif tool == "熱を加える":
            st.session_state.temp[mask_np] = MAX_TEMP
        elif tool == "消しゴム":
            st.session_state.material[mask_np] = 0
            st.session_state.temp[mask_np] = AMBIENT_TEMP

# --- シミュレーション実行と表示 ---
placeholder = st.empty()

# ループ処理
while True:
    st.session_state.temp = simulate_step(st.session_state.temp, st.session_state.material)
    
    # 最終的な表示用画像
    display_img = generate_display_image(st.session_state.temp, st.session_state.material)
    placeholder.image(display_img, use_container_width=True, caption="シミュレーション実行中...")
    
    time.sleep(0.05)
