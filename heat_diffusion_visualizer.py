
import tkinter as tk
from tkinter import ttk
import numpy as np

# ------------------------------------------------------------
# Heat diffusion visualizer
# - Drag a material button onto the grid to place a block.
# - Press and hold on a material block to heat it.
# - "初期化" clears all blocks and resets temperatures.
# ------------------------------------------------------------

CELL_SIZE = 14
GRID_W = 60
GRID_H = 36

AMBIENT_TEMP = 20.0
MAX_TEMP = 100.0

BLOCK_SIZE = 3  # each dropped material becomes a square block of this many cells


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def lerp(a, b, t):
    return a + (b - a) * t


def rgb_to_hex(rgb):
    r, g, b = (int(clamp(x, 0, 255)) for x in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def heat_color(temp):
    """
    Low -> blue, middle -> yellow, high -> red.
    """
    t = (temp - AMBIENT_TEMP) / (MAX_TEMP - AMBIENT_TEMP)
    t = clamp(t, 0.0, 1.0)

    blue = np.array([40, 80, 220], dtype=float)
    yellow = np.array([255, 230, 40], dtype=float)
    red = np.array([230, 40, 30], dtype=float)

    if t < 0.5:
        a = t / 0.5
        rgb = (1 - a) * blue + a * yellow
    else:
        a = (t - 0.5) / 0.5
        rgb = (1 - a) * yellow + a * red
    return tuple(rgb.tolist())


class HeatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("熱の広がり方可視化アプリ")
        self.root.resizable(False, False)

        self.canvas_w = GRID_W * CELL_SIZE
        self.canvas_h = GRID_H * CELL_SIZE

        # Material map:
        # 0 = empty, 1 = metal, 2 = wood, 3 = insulation
        self.material = np.zeros((GRID_H, GRID_W), dtype=np.int8)
        self.temp = np.full((GRID_H, GRID_W), AMBIENT_TEMP, dtype=float)

        # Thermal diffusivity-like coefficients for demo purposes.
        self.diffusivity = {
            0: 0.015,  # air / empty space
            1: 0.30,   # metal
            2: 0.06,   # wood
            3: 0.005,  # insulation
        }
        self.base_tint = {
            0: np.array([245, 245, 245], dtype=float),
            1: np.array([180, 185, 195], dtype=float),
            2: np.array([165, 125, 75], dtype=float),
            3: np.array([205, 230, 245], dtype=float),
        }

        self.drag_material = None
        self.drag_preview = None
        self.dragging_heat = False
        self.last_heat_cell = None

        self._build_ui()
        self._bind_events()

        self._running = True
        self._tick()

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.BOTH)

        self.canvas = tk.Canvas(
            top,
            width=self.canvas_w,
            height=self.canvas_h,
            bg="white",
            highlightthickness=1,
            highlightbackground="#999",
        )
        self.canvas.grid(row=0, column=0, columnspan=6, sticky="nsew")

        info = (
            "材料ボタンをドラッグして配置 → 構造物を長押しすると、その場所から熱が広がります。"
        )
        ttk.Label(top, text=info).grid(row=1, column=0, columnspan=6, sticky="w", pady=(8, 4))

        ttk.Label(top, text="金属 / 木材 / 断熱材").grid(row=2, column=0, sticky="w")

        self.palette = ttk.Frame(top)
        self.palette.grid(row=3, column=0, columnspan=3, sticky="w", pady=(4, 8))

        self.material_names = {1: "金属", 2: "木材", 3: "断熱材"}
        self.material_colors = {
            1: "#c0c5cf",
            2: "#9b6b39",
            3: "#d4ecff",
        }

        for idx, m in enumerate([1, 2, 3]):
            lbl = tk.Label(
                self.palette,
                text=self.material_names[m],
                padx=14,
                pady=8,
                relief="raised",
                bd=2,
                bg=self.material_colors[m],
                fg="black",
                cursor="hand2",
            )
            lbl.grid(row=0, column=idx, padx=6)
            lbl.bind("<ButtonPress-1>", lambda e, mm=m: self._start_drag_material(mm, e))
            lbl.bind("<B1-Motion>", self._drag_material_motion)
            lbl.bind("<ButtonRelease-1>", self._drop_material)

        self.reset_btn = ttk.Button(top, text="初期化", command=self.reset_all)
        self.reset_btn.grid(row=3, column=4, padx=(24, 6), pady=(2, 8), sticky="e")

        self.clear_heat_btn = ttk.Button(top, text="温度だけリセット", command=self.reset_temperature)
        self.clear_heat_btn.grid(row=3, column=5, padx=(6, 0), pady=(2, 8), sticky="e")

        self.legend = ttk.Label(
            top,
            text="色: 青=低温 → 黄=中温 → 赤=高温 / 物性: 金属=速い、木材=中くらい、断熱材=かなり遅い",
        )
        self.legend.grid(row=4, column=0, columnspan=6, sticky="w")

        self.status_var = tk.StringVar(value="配置してください")
        ttk.Label(top, textvariable=self.status_var).grid(row=5, column=0, columnspan=6, sticky="w", pady=(4, 0))

        self._draw_grid_lines()
        self._redraw_all()

    def _bind_events(self):
        self.canvas.bind("<ButtonPress-1>", self._canvas_press)
        self.canvas.bind("<B1-Motion>", self._canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._canvas_release)

    def _draw_grid_lines(self):
        # faint grid
        for x in range(0, self.canvas_w, CELL_SIZE):
            self.canvas.create_line(x, 0, x, self.canvas_h, fill="#ececec")
        for y in range(0, self.canvas_h, CELL_SIZE):
            self.canvas.create_line(0, y, self.canvas_w, y, fill="#ececec")

    def cell_from_xy(self, x, y):
        c = int(x // CELL_SIZE)
        r = int(y // CELL_SIZE)
        if 0 <= c < GRID_W and 0 <= r < GRID_H:
            return r, c
        return None

    def _start_drag_material(self, material_id, event):
        self.drag_material = material_id
        self.status_var.set(f"{self.material_names[material_id]} をドラッグ中")

    def _drag_material_motion(self, event):
        # Preview follows the cursor when dragging a palette item.
        if self.drag_material is None:
            return

        x = self.canvas.canvasx(event.x_root - self.canvas.winfo_rootx())
        y = self.canvas.canvasy(event.y_root - self.canvas.winfo_rooty())

        self._update_preview(x, y)

    def _drop_material(self, event):
        if self.drag_material is None:
            return

        x = self.canvas.canvasx(event.x_root - self.canvas.winfo_rootx())
        y = self.canvas.canvasy(event.y_root - self.canvas.winfo_rooty())

        cell = self.cell_from_xy(x, y)
        if cell is not None:
            self.place_block(cell[0], cell[1], self.drag_material)
            self.status_var.set(f"{self.material_names[self.drag_material]} を配置しました")

        self.drag_material = None
        self._clear_preview()
        self._redraw_all()

    def _canvas_press(self, event):
        cell = self.cell_from_xy(event.x, event.y)
        if cell is None:
            return

        r, c = cell
        if self.material[r, c] != 0:
            self.dragging_heat = True
            self.last_heat_cell = (r, c)
            self.status_var.set("長押し加熱中")
            self._apply_heat(r, c)
        else:
            self.status_var.set("構造物の上で長押しすると加熱できます")

    def _canvas_drag(self, event):
        cell = self.cell_from_xy(event.x, event.y)
        if cell is None:
            return

        if self.dragging_heat:
            r, c = cell
            if self.material[r, c] != 0:
                self.last_heat_cell = (r, c)
                self._apply_heat(r, c)

    def _canvas_release(self, event):
        self.dragging_heat = False
        self.last_heat_cell = None

    def place_block(self, r, c, material_id):
        half = BLOCK_SIZE // 2
        for rr in range(r - half, r - half + BLOCK_SIZE):
            for cc in range(c - half, c - half + BLOCK_SIZE):
                if 0 <= rr < GRID_H and 0 <= cc < GRID_W:
                    self.material[rr, cc] = material_id
                    self.temp[rr, cc] = AMBIENT_TEMP

    def _apply_heat(self, r, c):
        # Heat a small neighborhood around the pressed cell.
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < GRID_H and 0 <= cc < GRID_W and self.material[rr, cc] != 0:
                    self.temp[rr, cc] = min(MAX_TEMP, self.temp[rr, cc] + 10.0)
        self._redraw_all()

    def reset_all(self):
        self.material.fill(0)
        self.temp.fill(AMBIENT_TEMP)
        self.status_var.set("全て初期化しました")
        self._redraw_all()

    def reset_temperature(self):
        self.temp.fill(AMBIENT_TEMP)
        self.status_var.set("温度だけ初期化しました")
        self._redraw_all()

    def _update_preview(self, x, y):
        self._clear_preview()
        cell = self.cell_from_xy(x, y)
        if cell is None:
            return

        r, c = cell
        half = BLOCK_SIZE // 2
        x0 = (c - half) * CELL_SIZE
        y0 = (r - half) * CELL_SIZE
        x1 = x0 + BLOCK_SIZE * CELL_SIZE
        y1 = y0 + BLOCK_SIZE * CELL_SIZE

        self.drag_preview = self.canvas.create_rectangle(
            x0, y0, x1, y1,
            outline="#333",
            width=2,
            dash=(4, 2),
        )

    def _clear_preview(self):
        if self.drag_preview is not None:
            self.canvas.delete(self.drag_preview)
            self.drag_preview = None

    def _simulate_step(self):
        # Vectorized finite-difference style update with material-dependent diffusivity.
        T = self.temp
        D = np.vectorize(self.diffusivity.get)(self.material).astype(float)

        Tp = np.pad(T, 1, mode="edge")
        Dp = np.pad(D, 1, mode="edge")

        center = Tp[1:-1, 1:-1]
        up = Tp[:-2, 1:-1]
        down = Tp[2:, 1:-1]
        left = Tp[1:-1, :-2]
        right = Tp[1:-1, 2:]

        Dc = D
        Du = 0.5 * (Dc + Dp[:-2, 1:-1])
        Dd = 0.5 * (Dc + Dp[2:, 1:-1])
        Dl = 0.5 * (Dc + Dp[1:-1, :-2])
        Dr = 0.5 * (Dc + Dp[1:-1, 2:])

        dt = 0.18
        cooling = 0.008

        dT = (
            Du * (up - center)
            + Dd * (down - center)
            + Dl * (left - center)
            + Dr * (right - center)
        )

        # Mild ambient cooling so hot spots fade after a while.
        dT -= cooling * (center - AMBIENT_TEMP)

        T[:] = center + dt * dT
        np.clip(T, AMBIENT_TEMP, MAX_TEMP, out=T)

        # Keep heating the last pressed cell while the pointer is held down.
        if self.dragging_heat and self.last_heat_cell is not None:
            r, c = self.last_heat_cell
            self._apply_heat(r, c)

    def _blend_color(self, material_id, temp):
        base = self.base_tint.get(material_id, self.base_tint[0]).copy()
        hot = np.array(heat_color(temp), dtype=float)

        # As temperature rises, increase the influence of the heat map.
        t = (temp - AMBIENT_TEMP) / (MAX_TEMP - AMBIENT_TEMP)
        t = clamp(t, 0.0, 1.0)
        mix = 0.15 + 0.85 * t

        rgb = (1 - mix) * base + mix * hot
        return rgb_to_hex(rgb)

    def _redraw_all(self):
        self.canvas.delete("cell")

        for r in range(GRID_H):
            for c in range(GRID_W):
                x0 = c * CELL_SIZE
                y0 = r * CELL_SIZE
                x1 = x0 + CELL_SIZE
                y1 = y0 + CELL_SIZE

                mat = int(self.material[r, c])
                temp = float(self.temp[r, c])

                if mat == 0 and abs(temp - AMBIENT_TEMP) < 0.05:
                    fill = "#ffffff"
                else:
                    fill = self._blend_color(mat, temp)

                self.canvas.create_rectangle(
                    x0, y0, x1, y1,
                    fill=fill,
                    outline="",
                    tags="cell",
                )

                # Light material markers so the structure type remains visible.
                if mat != 0:
                    outline = {1: "#555555", 2: "#7a4f2b", 3: "#7bb8ff"}[mat]
                    self.canvas.create_rectangle(
                        x0, y0, x1, y1,
                        fill="",
                        outline=outline,
                        width=1,
                        tags="cell",
                    )

        # overlay for stronger contrast at hot spots
        for r in range(GRID_H):
            for c in range(GRID_W):
                if self.temp[r, c] >= 72:
                    x = c * CELL_SIZE + CELL_SIZE // 2
                    y = r * CELL_SIZE + CELL_SIZE // 2
                    self.canvas.create_oval(
                        x - 1, y - 1, x + 1, y + 1,
                        fill="#ffffff",
                        outline="",
                        tags="cell",
                    )

        self._clear_preview()

    def _tick(self):
        if not self._running:
            return

        self._simulate_step()
        self._redraw_all()
        self.root.after(40, self._tick)


def main():
    root = tk.Tk()

    # Use the themed widgets if available.
    try:
        style = ttk.Style()
        style.theme_use("clam")
    except Exception:
        pass

    app = HeatApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
