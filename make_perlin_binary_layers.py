#!/usr/bin/env python3
"""
make_perlin_binary_layers.py

Generate Perlin fBm noise -> (optional) midband mapping -> binary maps with fixed params,
and compute set-difference between two binary maps:

  diff = walkable(g_big) AND NOT walkable(g_small)

This matches: "흰(g 큰) - 흰(g 작은)" if white means walkable=1.

Notes:
- Convention: walkable=1 (white), blocked=0 (black)
- Binary rule: walkable = (mapped <= tau)

Example:
  python make_perlin_binary_layers.py \
    --size 500 --seed 0 --base_res 4 --octaves 3 --persistence 0.5 \
    --midband --tau 0.80 --g_small 0.9 --g_big 2.6 \
    --out_prefix binary_500_500_mid_tau0.80
"""
import argparse
import numpy as np
from PIL import Image


def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

def _lerp(a, b, t):
    return a + t * (b - a)

def _perlin2d(width, height, res_x, res_y, rng: np.random.Generator):
    angles = rng.uniform(0.0, 2.0 * np.pi, size=(res_y + 1, res_x + 1))
    grads = np.stack([np.cos(angles), np.sin(angles)], axis=-1)

    ys = np.linspace(0, res_y, height, endpoint=False, dtype=np.float32)
    xs = np.linspace(0, res_x, width,  endpoint=False, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)

    x0 = np.floor(X).astype(int)
    y0 = np.floor(Y).astype(int)
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

    nxy = (nxy - nxy.min()) / (nxy.max() - nxy.min() + 1e-8)
    return nxy.astype(np.float32)

def perlin_fractal_2d(width, height, base_res=4, octaves=3, persistence=0.5, seed=0):
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

def midband_map(x01: np.ndarray, gamma: float) -> np.ndarray:
    m = 1.0 - np.abs(2.0 * x01 - 1.0)
    m = np.clip(m, 0.0, 1.0)
    m = np.power(m, max(float(gamma), 1e-6))
    return m.astype(np.float32)

def to_png01(arr01: np.ndarray, out_path: str):
    img = (arr01.astype(np.uint8) * 255)
    Image.fromarray(img).save(out_path)
    print(f"Saved: {out_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=500, help="Map size (size x size).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base_res", type=int, default=4)
    p.add_argument("--octaves", type=int, default=3)
    p.add_argument("--persistence", type=float, default=0.5)
    p.add_argument("--invert", action="store_true")
    p.add_argument("--midband", action="store_true", help="Apply midband mapping before threshold.")
    p.add_argument("--tau", type=float, default=0.80, help="Threshold τ for walkable = (mapped <= τ).")
    p.add_argument("--g_small", type=float, default=0.9, help="Gamma for first (small) map.")
    p.add_argument("--g_big", type=float, default=2.6, help="Gamma for second (big) map.")
    p.add_argument("--out_prefix", type=str, default="binary_map", help="Output prefix (no extension).")
    args = p.parse_args()

    n = perlin_fractal_2d(args.size, args.size,
                          base_res=args.base_res, octaves=args.octaves,
                          persistence=args.persistence, seed=args.seed)
    if args.invert:
        n = 1.0 - n

    def make_binary(gamma: float) -> np.ndarray:
        mapped = midband_map(n, gamma) if args.midband else n
        walkable = (mapped <= float(args.tau)).astype(np.uint8)  # 1=walkable(white), 0=blocked(black)
        return walkable

    b_small = make_binary(args.g_small)
    b_big   = make_binary(args.g_big)

    # set difference: white(big) - white(small)
    diff = ((b_big == 1) & (b_small == 0)).astype(np.uint8)

    to_png01(b_small, f"{args.out_prefix}_g{args.g_small}.png")
    to_png01(b_big,   f"{args.out_prefix}_g{args.g_big}.png")
    to_png01(diff,    f"{args.out_prefix}_diff_g{args.g_big}_minus_g{args.g_small}.png")

if __name__ == "__main__":
    main()