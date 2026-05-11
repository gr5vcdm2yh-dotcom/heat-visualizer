
import streamlit as st
import numpy as np
import time

# --- 定数設定 ---
GRID_W, GRID_H = 41, 41 # 中心を特定しやすいよう奇数に設定
CENTER = 20
AMBIENT_TEMP = 20.0
SOURCE_TEMP = 100.0

# 材料の物性（熱拡散率と表示色）
MAT_DEFS = {
    0: {"name": "空気", "diff": 0.02, "color": "⬜"},
    1: {"name": "金属", "diff": 0.25, "color": "⬛"},
    2: {"name": "木材", "diff": 0.08, "color": "🟫"},
    3: {"name": "断熱材", "diff": 0.005, "color": "🟦"}
}

# --- 物理計算エンジン ---
def simulate_step(temp, material):
    D = np.vectorize(lambda x: MAT_DEFS[x]["diff"])(material).astype(float)
    Tp = np.pad(temp, 1, mode="edge")
    Dp = np.pad(D, 1, mode="edge")
    
    c = Tp[1:-1, 1:-1]
    u, d, l, r = Tp[:-2, 1:-1], Tp[2:, 1:-1], Tp[1:-1, :-2], Tp[1:-1, 2:]
    Du, Dd = 0.5*(D + Dp[:-2, 1:-1]), 0.5*(D + Dp[2:, 1:-1])
    Dl, Dr = 0.5*(D + Dp[1:-1, :-2]), 0.5*(D + Dp[1:-1, 2:])
    
    dT = (Du*(u-c) + Dd*(d-c) + Dl*(l-c) + Dr*(r-c))
    new_temp = c + 0.2 * dT
    
    # 中央の熱源を固定
    new_temp[CENTER, CENTER] = SOURCE_TEMP
    return np.clip(new_temp, AMBIENT_TEMP, SOURCE_TEMP)

# --- UI設定 ---
st.set_page_config(page_title="熱拡散シミュレーター", layout="centered")
st.title("🌋 中央熱源シミュレーター")

if 'temp' not in st.session_state:
    st.session_state.temp = np.full((GRID_H, GRID_W), AMBIENT_TEMP)
    st.session_state.temp[CENTER, CENTER] = SOURCE_TEMP
    st.session_state.material = np.zeros((GRID_H, GRID_W), dtype=int)

# --- サイドバー：設定 ---
with st.sidebar:
    st.header("🛠 設定")
    selected_mat = st.selectbox("配置する材料", [1, 2, 3], 
                                format_func=lambda x: f"{MAT_DEFS[x]['color']} {MAT_DEFS[x]['name']}")
    
    steps = st.slider("シミュレーション時間（ステップ数）", 10, 200, 50)
    
    if st.button("全リセット", use_container_width=True):
        st.session_state.temp.fill(AMBIENT_TEMP)
        st.session_state.temp[CENTER, CENTER] = SOURCE_TEMP
        st.session_state.material.fill(0)
        st.rerun()

# --- メインエリア：配置（クリック操作） ---
st.write("### 1. 材料を配置する (クリックで 3x3 ブロック配置)")
# クリック位置を受け取るための簡易的な仕組み
# (Streamlitの標準機能で最も軽くクリックを判定できる「列配置」を使用)
cols = st.columns(5)
with cols[0]:
    target_x = st.number_input("X座標", 0, GRID_W-1, CENTER + 5)
with cols[1]:
    target_y = st.number_input("Y座標", 0, GRID_H-1, CENTER)
if st.button("ここに配置"):
    # 3x3のブロックを配置
    st.session_state.material[max(0,target_y-1):target_y+2, max(0,target_x-1):target_x+2] = selected_mat
    st.success(f"座標({target_x}, {target_y}) に配置しました")

# --- 実行ボタン ---
st.divider()
if st.button("🔥 シミュレーション開始！", type="primary", use_container_width=True):
    progress_bar = st.progress(0)
    status_text = st.empty()
    image_placeholder = st.empty()
    
    for i in range(steps):
        st.session_state.temp = simulate_step(st.session_state.temp, st.session_state.material)
        
        if i % 5 == 0: # 5ステップごとに描画して負荷軽減
            # ヒートマップ表示（材料の枠線も薄く表示）
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 5))
            im = ax.imshow(st.session_state.temp, cmap='hot', vmin=AMBIENT_TEMP, vmax=SOURCE_TEMP)
            ax.contour(st.session_state.material, levels=[0.5], colors='cyan', linewidths=0.5)
            ax.set_axis_off()
            image_placeholder.pyplot(fig)
            plt.close()
            
            progress_bar.progress((i + 1) / steps)
            status_text.text(f"計算中... {i}/{steps} ステップ")
    
    status_text.text("シミュレーション完了")
    st.balloons()
else:
    # 待機中の表示
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(st.session_state.temp, cmap='hot', vmin=AMBIENT_TEMP, vmax=SOURCE_TEMP)
    ax.contour(st.session_state.material, levels=[0.5], colors='cyan', linewidths=1)
    ax.set_axis_off()
    st.pyplot(fig)
    plt.close()
