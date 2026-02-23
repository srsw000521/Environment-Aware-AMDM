#!/usr/bin/env python3
"""
make_perlin_noise_map_dual_mask_6panel_dualtau.py

Interactive 6-panel viewer for TWO Perlin-noise maps (A/B) generated with the same params
but different random seeds. You can tune thresholds independently (tauA, tauB).

Panels (left->right, top->bottom):
  [0] A mapped (0..1)
  [1] A binary (walkable=1)
  [2] A masked (A walkable but NOT overlapping with B walkable)
  [3] B mapped (0..1)
  [4] B binary (walkable=1)
  [5] Overlap (A walkable AND B walkable)  -> shown as 1 where overlap exists

Notes:
- "walkable=1" means the agent can step there (your convention).
- Thresholding rule: walkable = (mapped <= tau)
- Mask rule: A_masked = A_walkable AND (NOT B_walkable)
  (i.e., A and B's shared walkable region becomes blocked in A_masked)

Works on Python 3.7+ (no PEP604 unions).

Examples:
  python make_perlin_noise_map_dual_mask_6panel_dualtau.py --interactive --size 500 --seed 0 --seed2 1
  python make_perlin_noise_map_dual_mask_6panel_dualtau.py --interactive --size 500 --tauA 0.71 --tauB 0.80
"""
import argparse
import os
from typing import Optional

import numpy as np
from PIL import Image


def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6 - 15) + 10)


def _lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    return a + t * (b - a)


def _perlin2d(width: int, height: int, res_x: int, res_y: int, rng: np.random.Generator) -> np.ndarray:
    # Gradient vectors on a (res_y+1, res_x+1) lattice
    angles = rng.uniform(0.0, 2.0 * np.pi, size=(res_y + 1, res_x + 1))
    grads = np.stack([np.cos(angles), np.sin(angles)], axis=-1)

    # Grid of points in lattice space
    ys = np.linspace(0, res_y, height, endpoint=False, dtype=np.float32)
    xs = np.linspace(0, res_x, width,  endpoint=False, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)

    x0 = np.floor(X).astype(np.int32)
    y0 = np.floor(Y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    fx = X - x0
    fy = Y - y0

    g00 = grads[y0, x0]
    g10 = grads[y0, x1]
    g01 = grads[y1, x0]
    g11 = grads[y1, x1]

    d00 = np.stack([fx,     fy    ], axis=-1)
    d10 = np.stack([fx - 1, fy    ], axis=-1)
    d01 = np.stack([fx,     fy - 1], axis=-1)
    d11 = np.stack([fx - 1, fy - 1], axis=-1)

    n00 = np.sum(g00 * d00, axis=-1)
    n10 = np.sum(g10 * d10, axis=-1)
    n01 = np.sum(g01 * d01, axis=-1)
    n11 = np.sum(g11 * d11, axis=-1)

    wx = _fade(fx)
    wy = _fade(fy)

    nx0 = _lerp(n00, n10, wx)
    nx1 = _lerp(n01, n11, wx)
    nxy = _lerp(nx0, nx1, wy)

    # Normalize to [0,1]
    nxy = (nxy - nxy.min()) / (nxy.max() - nxy.min() + 1e-8)
    return nxy.astype(np.float32)


def perlin_fractal_2d(width: int, height: int, base_res: int = 4, octaves: int = 3,
                      persistence: float = 0.5, seed: int = 0) -> np.ndarray:
    """
    Fractal Brownian Motion (fBm) Perlin noise, normalized to [0,1].
    """
    rng = np.random.default_rng(seed)
    noise = np.zeros((height, width), dtype=np.float32)
    amp = 1.0
    amp_sum = 0.0

    for o in range(octaves):
        res = base_res * (2 ** o)
        noise += amp * _perlin2d(width, height, res_x=res, res_y=res, rng=rng)
        amp_sum += amp
        amp *= persistence

    noise /= max(amp_sum, 1e-8)
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    return noise


def save_grayscale01(img01: np.ndarray, out_path: str) -> None:
    img = (img01 * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(out_path)
    print(f"Saved: {out_path}")


def save_binary01(bin01: np.ndarray, out_path: str) -> None:
    img = (bin01 * 255).astype(np.uint8)
    Image.fromarray(img).save(out_path)
    print(f"Saved: {out_path}")

def generate_A_masked_binary01(
    size: int = 500,
    seedA: int = 0,
    seedB: int = 1,
    base_res: int = 3,
    octaves: int = 3,
    persistence: float = 0.3,
    tauA: float = 0.71,
    tauB: float = 0.71,
    gamma: float = 1.7,
    use_midband: bool = False,
    invert_B: bool = False,
) -> np.ndarray:
    """
    반환: (H,W) uint8 {0,1} 의 A_masked (walkable=1)
    interactive_ui_dual_tau의 compute() 로직과 동일:
      bA = (mappedA <= tauA)
      bB = (mappedB <= tauB); bB = 1 - bB
      overlap 제거: A_masked = bA; A_masked[overlap]=0
    :contentReference[oaicite:3]{index=3}
    """
    def apply_midband(x: np.ndarray, on: bool, g: float) -> np.ndarray:
        if not on:
            return x.astype(np.float32)
        m = 1.0 - np.abs(2.0 * x - 1.0)
        m = np.clip(m, 0.0, 1.0)
        m = np.power(m, max(float(g), 1e-6))
        return m.astype(np.float32)

    nA = perlin_fractal_2d(size, size, base_res=base_res, octaves=octaves,
                           persistence=persistence, seed=int(seedA))
    nB = perlin_fractal_2d(size, size, base_res=base_res, octaves=octaves,
                           persistence=persistence, seed=int(seedB))

    baseA = nA
    baseB = (1.0 - nB) if invert_B else nB

    mappedA = apply_midband(baseA, use_midband, gamma)
    mappedB = apply_midband(baseB, use_midband, gamma)

    bA = (mappedA <= tauA).astype(np.uint8)
    bB = (mappedB <= tauB).astype(np.uint8)
    bB = (1 - bB).astype(np.uint8)  # 기존 로직 그대로 :contentReference[oaicite:4]{index=4}

    overlap = ((bA == 1) & (bB == 1)).astype(np.uint8)
    A_masked = bA.copy()
    A_masked[overlap == 1] = 0
    return A_masked

def interactive_ui_dual_tau(noiseA01: np.ndarray,
                            noiseB01: np.ndarray,
                            init_tauA: float,
                            init_tauB: float,
                            init_gamma: float,
                            invert: bool,
                            out_prefix: Optional[str]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button


    #baseA = (1.0 - noiseA01) #if invert else noiseA01
    baseA = noiseA01
    baseB = (1.0 - noiseB01) if invert else noiseB01

    state = {
        "use_midband": False,
        "gamma": float(init_gamma),
    }

    def apply_midband(x: np.ndarray, use_midband: bool, gamma: float) -> np.ndarray:
        if not use_midband:
            return x
        # peak at 0.5
        m = 1.0 - np.abs(2.0 * x - 1.0)
        m = np.clip(m, 0.0, 1.0)
        m = np.power(m, max(gamma, 1e-6))
        return m.astype(np.float32)

    def compute(tauA: float, tauB: float, gamma: float):
        mappedA = apply_midband(baseA, state["use_midband"], gamma)
        mappedB = apply_midband(baseB, state["use_midband"], gamma)

        bA = (mappedA <= tauA).astype(np.uint8)  # 1=walkable
        bB = (mappedB <= tauB).astype(np.uint8)  # 1=walkable
        bB = (1 - bB).astype(np.uint8)

        overlap = ((bA == 1) & (bB == 1)).astype(np.uint8)   # 1 where both walkable
        A_masked = bA.copy()
        A_masked[overlap == 1] = 0  # overlap을 "못 가는 곳(0)"으로

        return mappedA, bA, A_masked, mappedB, bB, overlap

    tauA0 = float(np.clip(init_tauA, 0.0, 1.0))
    tauB0 = float(np.clip(init_tauB, 0.0, 1.0))
    g0 = float(np.clip(init_gamma, 0.2, 6.0))

    mappedA0, bA0, A_masked0, mappedB0, bB0, overlap0 = compute(tauA0, tauB0, g0)

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.5))
    fig.suptitle("Dual Perlin → Binary (independent thresholds τA / τB)", fontsize=12)

    axA_noise, axA_bin, axA_mask = axes[0]
    axB_noise, axB_bin, axOverlap = axes[1]

    imA0 = axA_noise.imshow(mappedA0, vmin=0.0, vmax=1.0, interpolation="nearest")
    axA_noise.set_title("A mapped (0..1)")
    axA_noise.set_xticks([]); axA_noise.set_yticks([])

    imA1 = axA_bin.imshow(bA0, vmin=0, vmax=1, interpolation="nearest")
    axA_bin.set_title(f"A binary (walkable=1)  τA={tauA0:.2f}")
    axA_bin.set_xticks([]); axA_bin.set_yticks([])

    imA2 = axA_mask.imshow(A_masked0, vmin=0, vmax=1, interpolation="nearest")
    axA_mask.set_title("A masked (overlap→blocked)")
    axA_mask.set_xticks([]); axA_mask.set_yticks([])

    imB0 = axB_noise.imshow(mappedB0, vmin=0.0, vmax=1.0, interpolation="nearest")
    axB_noise.set_title("B mapped (0..1)")
    axB_noise.set_xticks([]); axB_noise.set_yticks([])

    imB1 = axB_bin.imshow(bB0, vmin=0, vmax=1, interpolation="nearest")
    axB_bin.set_title(f"B binary (walkable=1)  τB={tauB0:.2f}")
    axB_bin.set_xticks([]); axB_bin.set_yticks([])

    imO = axOverlap.imshow(overlap0, vmin=0, vmax=1, interpolation="nearest")
    axOverlap.set_title("Overlap (A & B walkable)")
    axOverlap.set_xticks([]); axOverlap.set_yticks([])

    # ---- sliders area
    plt.subplots_adjust(bottom=0.22, top=0.92, hspace=0.18, wspace=0.10)

    tauA_ax = fig.add_axes([0.12, 0.13, 0.62, 0.03])
    tauA_slider = Slider(tauA_ax, "threshold τA", 0.0, 1.0, valinit=tauA0, valstep=0.01)

    tauB_ax = fig.add_axes([0.12, 0.09, 0.62, 0.03])
    tauB_slider = Slider(tauB_ax, "threshold τB", 0.0, 1.0, valinit=tauB0, valstep=0.01)

    gamma_ax = fig.add_axes([0.12, 0.05, 0.62, 0.03])
    gamma_slider = Slider(gamma_ax, "midband γ", 0.2, 6.0, valinit=g0, valstep=0.1)

    btn_toggle_ax = fig.add_axes([0.77, 0.05, 0.20, 0.06])
    btn_toggle = Button(btn_toggle_ax, "Midband: ON")

    btn_save_ax = fig.add_axes([0.77, 0.12, 0.20, 0.06])
    btn_save = Button(btn_save_ax, "Save 6")

    btn_print_ax = fig.add_axes([0.02, 0.10, 0.08, 0.06])
    btn_print = Button(btn_print_ax, "Print")

    def redraw():
        tauA = float(tauA_slider.val)
        tauB = float(tauB_slider.val)
        state["gamma"] = float(gamma_slider.val)

        mappedA, bA, A_masked, mappedB, bB, overlap = compute(tauA, tauB, state["gamma"])

        imA0.set_data(mappedA)
        imA1.set_data(bA)
        imA2.set_data(A_masked)
        imB0.set_data(mappedB)
        imB1.set_data(bB)
        imO.set_data(overlap)

        mode = "ON" if state["use_midband"] else "OFF"
        axA_noise.set_title(f"A mapped (midband {mode}, γ={state['gamma']:.1f})")
        axB_noise.set_title(f"B mapped (midband {mode}, γ={state['gamma']:.1f})")
        axA_bin.set_title(f"A binary (walkable=1)  τA={tauA:.2f}")
        axB_bin.set_title(f"B binary (walkable=1)  τB={tauB:.2f}")

        fig.canvas.draw_idle()

    tauA_slider.on_changed(lambda _v: redraw())
    tauB_slider.on_changed(lambda _v: redraw())
    gamma_slider.on_changed(lambda _v: redraw())

    def _on_toggle(_event):
        state["use_midband"] = not state["use_midband"]
        btn_toggle.label.set_text("Midband: ON" if state["use_midband"] else "Midband: OFF")
        redraw()

    def _on_print(_event):
        print(f"τA={float(tauA_slider.val):.2f}, τB={float(tauB_slider.val):.2f}, midband={'ON' if state['use_midband'] else 'OFF'}, γ={float(gamma_slider.val):.1f}")

    def _on_save(_event):
        tauA = float(tauA_slider.val)
        tauB = float(tauB_slider.val)
        g = float(gamma_slider.val)
        mappedA, bA, A_masked, mappedB, bB, overlap = compute(tauA, tauB, g)

        mode = "mid" if state["use_midband"] else "raw"
        if out_prefix is None:
            prefix = f"perlin6_{mappedA.shape[0]}_{mappedA.shape[1]}_{mode}_tauA{tauA:.2f}_tauB{tauB:.2f}_g{g:.1f}"
        else:
            prefix = out_prefix

        save_grayscale01(mappedA, prefix + "_A_mapped.png")
        save_binary01(bA, prefix + "_A_bin.png")
        save_binary01(A_masked, prefix + "_A_masked.png")
        save_grayscale01(mappedB, prefix + "_B_mapped.png")
        save_binary01(bB, prefix + "_B_bin.png")
        save_binary01(overlap, prefix + "_overlap.png")

    btn_toggle.on_clicked(_on_toggle)
    btn_save.on_clicked(_on_save)
    btn_print.on_clicked(_on_print)

    try:
        fig.canvas.manager.set_window_title("Dual Perlin dual-threshold UI")
    except Exception:
        pass

    plt.show()




def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=500, help="Image size (size x size).")
    p.add_argument("--seed", type=int, default=5, help="Seed for map A.")
    p.add_argument("--seed2", type=int, default=100, help="Seed for map B.")
    p.add_argument("--base_res", type=int, default=3, help="Base lattice resolution for octave 0.")
    p.add_argument("--octaves", type=int, default=3, help="Number of octaves.")
    p.add_argument("--persistence", type=float, default=0.3, help="Amplitude multiplier per octave.")
    p.add_argument("--invert", action="store_true", help="Invert grayscale (1-noise).")
    p.add_argument("--interactive", action="store_true", help="Open interactive UI.")
    p.add_argument("--tauA", type=float, default=0.71, help="Initial threshold for A (walkable=1 if mapped<=tauA).")
    p.add_argument("--tauB", type=float, default=0.71, help="Initial threshold for B (walkable=1 if mapped<=tauB).")
    p.add_argument("--gamma", type=float, default=1.7, help="Initial midband gamma.")
    p.add_argument("--out_prefix", type=str, default=None, help="Prefix for saved images when clicking 'Save 6'.")
    args = p.parse_args()

    nA = perlin_fractal_2d(args.size, args.size, base_res=args.base_res, octaves=args.octaves,
                           persistence=args.persistence, seed=args.seed)
    nB = perlin_fractal_2d(args.size, args.size, base_res=args.base_res, octaves=args.octaves,
                           persistence=args.persistence, seed=args.seed2)

    if args.interactive:
        interactive_ui_dual_tau(nA, nB, init_tauA=args.tauA, init_tauB=args.tauB,
                                init_gamma=args.gamma, invert=args.invert, out_prefix=args.out_prefix)
    else:
        # non-interactive: just dump the two raw noises as grayscale
        save_grayscale01(nA, "perlin_A.png")
        save_grayscale01(nB, "perlin_B.png")


if __name__ == "__main__":
    main()