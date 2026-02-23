from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import os
import math

# Load image
img_path = "/home/sura/AMDM/perlin6_500_500_raw_tauA0.80_tauB0.89_g1.7_A_masked.png"
img = Image.open(img_path).convert("L")  # grayscale
arr = np.array(img)

h, w = arr.shape

# Walkable = white (>200)
walkable = arr > 200

boundary_block_m = 2.0
world_size_m = 50.0

m_per_px = world_size_m / w           # 0.1 m/px
margin_px = int(math.ceil(boundary_block_m / m_per_px))  # 20px

walkable[:margin_px, :] = False
walkable[-margin_px:, :] = False
walkable[:, :margin_px] = False
walkable[:, -margin_px:] = False

# Distance between points
def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

# Bresenham line check (direct path blocked?)
def blocked(p1, p2):
    x0, y0 = p1
    x1, y1 = p2
    dx = abs(x1-x0)
    dy = abs(y1-y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx-dy

    x, y = x0, y0
    while True:
        if not walkable[y, x]:
            return True
        if x == x1 and y == y1:
            break
        e2 = 2*err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return False


# Collect all walkable pixels
ys, xs = np.where(walkable)
points = list(zip(xs, ys))

pairs = []

MIN_D = 10
MAX_D = 50

tries = 0
while len(pairs) < 20 and tries < 500000:
    tries += 1

    s = random.choice(points)
    g = random.choice(points)

    d = dist(s, g)

    if d < MIN_D or d > MAX_D:
        continue

    # Must be blocked (needs detour)
    if not blocked(s, g):
        continue

    # Diversity: avoid too close starts/goals
    ok = True
    for ps, pg in pairs:
        if dist(ps, s) < 25 and dist(pg, g) < 25:
            ok = False
            break

    if not ok:
        continue

    pairs.append((s, g))

print("Generated:", len(pairs))

# Annotated image
rgb = img.convert("RGBA")          # <-- RGB 대신 RGBA (alpha 쓰려고)
draw = ImageDraw.Draw(rgb)

try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
except:
    font = ImageFont.load_default()

for i, (s, g) in enumerate(pairs):
    # 1) start-goal line (semi-transparent yellow)
    draw.line([s, g], fill=(255, 255, 0, 120), width=2)

    # 2) start: red, goal: blue
    draw.ellipse([s[0]-4, s[1]-4, s[0]+4, s[1]+4], fill=(255,0,0,255))
    draw.ellipse([g[0]-4, g[1]-4, g[0]+4, g[1]+4], fill=(0,0,255,255))

    draw.text((s[0]+5, s[1]+2), str(i), fill=(255,0,0,255), font=font)
    draw.text((g[0]+5, g[1]+2), str(i), fill=(0,0,255,255), font=font)

out_img = "/home/sura/AMDM/detour_pairs_60_map2.png"
rgb.convert("RGB").save(out_img)   # PNG로 저장 (RGB로 변환해도 되고 RGBA 그대로 저장해도 OK)


# Save npz
starts = np.array([p[0] for p in pairs], dtype=np.float32)  # (N,2) with (x,y)
goals  = np.array([p[1] for p in pairs], dtype=np.float32)

# -----------------------------
# Pixel -> World (match PerlinNavMap.pixel_to_world)
# - use pixel CENTER: +0.5
# - flip z: (0.5 - v/H)
# -----------------------------
u  = starts[:, 0] + 0.5
v  = starts[:, 1] + 0.5
ug = goals[:, 0]  + 0.5
vg = goals[:, 1]  + 0.5

world_x  = (u  / w - 0.5) * 50.0
world_z  = (0.5 - v  / h) * 50.0   # <-- 중요: y-down 이미지를 z-up world로 뒤집기

world_xg = (ug / w - 0.5) * 50.0
world_zg = (0.5 - vg / h) * 50.0

npz_path = "/home/sura/AMDM/start_goal_pairs_map2_60.npz"
np.savez(
    npz_path,
    start_px=starts,
    goal_px=goals,
    start_world=np.stack([world_x, world_z], 1),
    goal_world=np.stack([world_xg, world_zg], 1),
)

bad = 0
for (x, y) in starts.astype(int):
    if not walkable[y, x]:
        bad += 1
print("bad start pixels:", bad, "/", len(starts))
