import streamlit as st
import numpy as np
import time

# --- 基本設定 ---
GRID_W = 60
GRID_H = 36
AMBIENT_TEMP = 20.0
MAX_TEMP = 100.0

# 物理定数（拡散率）
DIFFUSIVITY = {0: 0.015, 1: 0.30, 2: 0.06, 3: 0.005}

# --- 物理計算ロジック ---
def simulate_step(temp_grid, material_grid):
    D = np.vectorize(DIFFUSIVITY.get)(material_grid).astype(float)
    Tp, Dp = np.pad(temp_grid, 1, mode="edge"), np.pad(D, 1, mode="edge")
    center = Tp[1:-1, 1:-1]
    
    # 上下左右の温度と拡散率
    up, down = Tp[:-2, 1:-1], Tp[2:, 1:-1]
    left, right = Tp[1:-1, :-2], Tp[1:-1, 2:]
    Du, Dd = 0.5*(D + Dp[:-2, 1:-1]), 0.5*(D + Dp[2:, 1:-1])
    Dl, Dr = 0.5*(D + Dp[1:-1, :-2]), 0.5*(D + Dp[1:-1, 2:])

    dt, cooling = 0.18, 0.005
    dT = (Du*(up-center) + Dd*(down-center) + Dl*(left-center) + Dr*(right-center)) - cooling*(center-AMBIENT_TEMP)
    return np.clip(center + dt * dT, AMBIENT_TEMP, MAX_TEMP)

def get_rgb_image(temp_grid, material_grid, px=None, py=None):
    # 温度を0-1に正規化してヒートマップ作成
    t_norm = (temp_grid - AMBIENT_TEMP) / (MAX_TEMP - AMBIENT_TEMP)
    t_norm = np.clip(t_norm, 0, 1)
    
    img = np.zeros((GRID_H, GRID_W, 3), dtype=np.uint8)
    # 簡易色計算 (Blue -> Yellow -> Red)
    img[:, :, 0] = (t_norm * 255).astype(np.uint8) # Red
    img[:, :, 1] = (np.sin(t_norm * np.pi) * 200).astype(np.uint8) # Green
    img[:, :, 2] = ((1 - t_norm) * 200).astype(np.uint8) # Blue
    
    # 材料の輪郭をうっすら表示
    img[material_grid != 0] = np.clip(img[material_grid != 0].astype(int) + 30, 0, 255).astype(np.uint8)
    
    # プレビュー用カーソルの描画
    if px is not None and py is not None:
        img[max(0,py-1):py+2, px, :] = [255, 255, 255] # 白い十字
        img[py, max(0,px-1):px+2, :] = [255, 255, 255]
        
    return img

# --- Streamlit UI構成 ---
st.set_page_config(page_title="熱拡散シミュレーター Pro", layout="wide")
st.title("🔥 物理シミュレーション：熱の拡散と断熱")

if 'temp' not in st.session_state:
    st.session_state.temp = np.full((GRID_H, GRID_W), AMBIENT_TEMP)
    st.session_state.material = np.zeros((GRID_H, GRID_W), dtype=int)
    st.session_state.running = True

# サイドバー：操作パネル
with st.sidebar:
    st.header("🛠 操作ツール")
    mode = st.radio("モード", ["材料を配置", "加熱する", "消しゴム", "温度を測る"])
    
    if mode == "材料を配置":
        mat_type = st.selectbox("材料", ["金属 (速い)", "木材 (普通)", "断熱材 (遅い)"])
        mat_id = {"金属 (速い)": 1, "木材 (普通)": 2, "断熱材 (遅い)": 3}[mat_type]
    
    st.divider()
    st.subheader("📍 位置調整 (Preview)")
    px = st.slider("X座標", 0, GRID_W-1, GRID_W//2)
    py = st.slider("Y座標", 0, GRID_H-1, GRID_H//2)
    
    col1, col2 = st.columns(2)
    if col1.button("実行 (Apply)", use_container_width=True):
        y_r, x_r = slice(max(0,py-1), py+2), slice(max(0,px-1), px+2)
        if mode == "材料を配置": st.session_state.material[y_r, x_r] = mat_id
        elif mode == "加熱する": st.session_state.temp[y_r, x_r] = MAX_TEMP
        elif mode == "消しゴム": 
            st.session_state.material[y_r, x_r] = 0
            st.session_state.temp[y_r, x_r] = AMBIENT_TEMP

    if col2.button("全リセット", use_container_width=True):
        st.session_state.temp.fill(AMBIENT_TEMP)
        st.session_state.material.fill(0)

    st.divider()
    st.session_state.running = st.checkbox("シミュレーションを実行中", value=True)

# メインエリア：情報表示
info_col, chart_col = st.columns([1, 3])

with info_col:
    current_t = st.session_state.temp[py, px]
    st.metric("選択地点の温度", f"{current_t:.2f} ℃")
    mat_name = {0:"空気", 1:"金属", 2:"木材", 3:"断熱材"}[st.session_state.material[py, px]]
    st.write(f"材質: **{mat_name}**")

# メイン画面の描画
placeholder = st.empty()

# 実行ループ（最適化版）
while True:
    if st.session_state.running:
        st.session_state.temp = simulate_step(st.session_state.temp, st.session_state.material)
    
    # 画像生成（プレビューカーソル付き）
    display_img = get_rgb_image(st.session_state.temp, st.session_state.material, px, py)
    
    # 描画更新
    placeholder.image(display_img, use_container_width=True, clamp=True)
    
    # 負荷軽減のためのスリープ
    time.sleep(0.03)
    
    # 操作があった場合に反映させるための再読み込みトリガー（Streamlitの仕様対応）
    if not st.session_state.running:
        break
