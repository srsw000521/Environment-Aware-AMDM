import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from pathlib import Path

# SciPy가 설치되어 있어야 합니다.
from scipy.ndimage import binary_closing, label

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Tuple
from make_perlin_noise_map import generate_A_masked_binary01

def make_disk(radius_px: int) -> np.ndarray:
    """원형 구조요소(마스크) 생성"""
    r = int(radius_px)
    y, x = np.ogrid[-r:r+1, -r:r+1]
    return (x*x + y*y) <= r*r

class PerlinNavMap:
    def __init__(
        self,
        img_path,
        world_size_m: float = 50.0,   # 50m x 50m
        boundary_block_m: float = 2.0,
        white_threshold: int = 127,   # 픽셀 밝기 임계치(흰=walkable) (파일/그레이스케일 입력일 때만 사용)
        input_is_binary01: bool = False,  # True이면 img_path는 (H,W) {0,1} 배열/텐서로 간주
    ):
        """
        img_path는 다음 중 하나를 받을 수 있습니다.
        - str / Path: 이미지 파일 경로 (grayscale로 읽어서 threshold)
        - np.ndarray: (H,W) 또는 (H,W,1) 배열
        - torch.Tensor: (H,W) 또는 (1,H,W) 또는 (H,W,1)

        input_is_binary01=True이면 입력을 {0,1}의 walkable 맵으로 그대로 사용합니다.
        """
        self.world_size_m = float(world_size_m)

        # 1) 입력 로드 → (H,W) uint8
        arr = None

        # (a) torch.Tensor 입력
        if torch.is_tensor(img_path):
            t = img_path.detach()
            if t.is_cuda:
                t = t.cpu()
            # shape normalize
            if t.dim() == 3 and t.shape[0] == 1:
                t = t[0]
            if t.dim() == 3 and t.shape[-1] == 1:
                t = t[..., 0]
            arr = t.to(torch.uint8).numpy()

        # (b) numpy 입력
        elif isinstance(img_path, np.ndarray):
            a = img_path
            if a.ndim == 3 and a.shape[-1] == 1:
                a = a[..., 0]
            if a.ndim != 2:
                raise ValueError(f"np.ndarray 입력은 (H,W) 또는 (H,W,1) 이어야 합니다. got {a.shape}")
            arr = a.astype(np.uint8, copy=False)

        # (c) 파일 경로 입력
        else:
            img = Image.open(img_path).convert("L")
            arr = np.array(img, dtype=np.uint8)  # (H,W)

        if arr is None or arr.ndim != 2:
            raise ValueError("입력 이미지를 (H,W) 형태로 변환하지 못했습니다.")

        self.H, self.W = arr.shape

        # 2) walkable(1)/blocked(0) 구성
        if input_is_binary01:
            # {0,1}로 강제 (혹시 0/255 같은 게 들어와도 처리)
            walkable = (arr > 0).astype(np.uint8)
        else:
            walkable = (arr > white_threshold).astype(np.uint8)

        # 3) 외곽 boundary_block_m를 unwalkable로
        self.res_m_per_px = self.world_size_m / float(self.W)  # m/px (정사각 이미지 가정)
        margin_px = int(np.ceil(boundary_block_m / self.res_m_per_px))
        if margin_px > 0:
            walkable[:margin_px, :] = 0
            walkable[-margin_px:, :] = 0
            walkable[:, :margin_px] = 0
            walkable[:, -margin_px:] = 0

        self.walkable = walkable  # (H,W) uint8 in {0,1}

        # reachability 관련은 build_reachability_map() 호출 시 생성
        self.reach_map = None
        self.cc_labels = None
        self.num_cc = None

    def to_torch(self, device="cuda", use_reach_map: bool = False):
        """
        맵을 GPU에 상주시킴. 한 번만 호출.
        """
        self.device = torch.device(device)

        # (H,W) -> (1,1,H,W), float32
        self.walkable_t = torch.from_numpy(self.walkable.astype("float32"))[None, None].to(self.device)

        self.reach_map_t = None
        if use_reach_map:
            if self.reach_map is None:
                raise RuntimeError("use_reach_map=True이면 build_reachability_map()를 먼저 호출해야 합니다.")
            self.reach_map_t = torch.from_numpy(self.reach_map.astype("float32"))[None, None].to(self.device)

        # ★ 추가: cc_labels도 torch로 캐싱 (goal sampling에 필요)
        self.cc_labels_t = None
        if self.cc_labels is not None:
            # (H,W) int32 -> torch.int64 (indexing/compare용)
            self.cc_labels_t = torch.from_numpy(self.cc_labels.astype("int32")).to(self.device)

        # 상수도 torch로 캐싱
        self.W_t = torch.tensor(float(self.W), device=self.device)
        self.H_t = torch.tensor(float(self.H), device=self.device)
        self.world_size_t = torch.tensor(float(self.world_size_m), device=self.device)

    def _pixel_to_world_torch(self, u: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x = (u/W - 0.5) * world_size
        # z = (0.5 - v/H) * world_size
        x = (u / self.W_t - 0.5) * self.world_size_t
        z = (0.5 - v / self.H_t) * self.world_size_t
        return x, z

    def _world_to_pixel_torch(self, x: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # u = (x/world + 0.5) * W
        # v = (0.5 - z/world) * H
        u = (x / self.world_size_t + 0.5) * self.W_t
        v = (0.5 - z / self.world_size_t) * self.H_t
        return u, v

    # ----------------------------
    # 좌표 변환: world (x,z in m) <-> pixel (u,v)
    # ----------------------------

    def _world_to_pixel(self, x_m: float, z_m: float) -> Tuple[float, float]:
        return self.world_to_pixel(x_m, z_m)

    def world_to_pixel(self, x_m: float, z_m: float) -> Tuple[float, float]:
        """
        world: center=(0,0), x는 오른쪽(+), z는 위쪽(+)
        pixel: u는 오른쪽(+), v는 아래쪽(+)
        """
        u = (x_m / self.world_size_m + 0.5) * self.W
        v = (0.5 - z_m / self.world_size_m) * self.H
        return u, v

    def pixel_to_world(self, u: float, v: float) -> Tuple[float, float]:
        x = (u / self.W - 0.5) * self.world_size_m
        z = (0.5 - v / self.H) * self.world_size_m
        return x, z

    def in_bounds_uv(self, u: float, v: float) -> bool:
        return (0 <= u < self.W) and (0 <= v < self.H)

    def _draw_world_box(
            self,
            ax,
            center_x: float,
            center_z: float,
            size_m: float,
            color: str = "yellow",
            lw: float = 2.0,
            alpha: float = 1.0,
            label: str = None
    ):
        """world 좌표에서 center 기준 size_m x size_m 박스를 그림 (pixel space로 변환하여 그리기)"""
        half = size_m * 0.5
        x0 = center_x - half
        z0 = center_z - half
        x1 = center_x + half
        z1 = center_z + half

        W, H = self.W, self.H
        # (x,z) -> (u,v) 4 corners
        u0, v1 = self.world_to_pixel(x0, z0)  # bottom-left
        u1, v0 = self.world_to_pixel(x1, z1)  # top-right

        rect = Rectangle(
            (u0, v0),
            u1 - u0,
            v1 - v0,
            fill=False,
            edgecolor=color,
            linewidth=lw,
            alpha=alpha
        )
        ax.add_patch(rect)

        if label is not None:
            ax.text(u0, v0 - 5, label, color=color, fontsize=10, va="bottom")

    def debug_show_maps(
            self,
            start: Tuple[float, float] = (0.0, 0.0),
            goal: Tuple[float, float] = None,
            roi_size_m: float = 10.0,
            local_grid_size_m: float = 4.0,
            show_reach_map: bool = True,
            show_cc_labels: bool = True,
            show_local_grid_box: bool = True,
            figsize: Tuple[int, int] = (16, 6),
            title_prefix: str = "[Debug] ",
    ):
        """
        - walkable(경계 반영) / reachability map / connected component 라벨을 표시
        - start/goal 점, ROI 박스, local grid(4mx4m) 박스를 overlay
        """
        sx, sz = start
        gx, gz = goal if goal is not None else (None, None)

        # reach_map / cc_labels 없을 수 있으니 안전 처리
        reach_map = self.reach_map if (show_reach_map and self.reach_map is not None) else None
        cc_labels = self.cc_labels if (show_cc_labels and self.cc_labels is not None) else None

        ncols = 1 + (reach_map is not None) + (cc_labels is not None)
        fig, axes = plt.subplots(1, ncols, figsize=figsize)
        if ncols == 1:
            axes = [axes]

        # start/goal -> pixel
        su, sv = self.world_to_pixel(sx, sz)
        if goal is not None:
            gu, gv = self.world_to_pixel(gx, gz)

        col = 0

        # --- walkable ---
        ax = axes[col];
        col += 1
        ax.set_title(f"{title_prefix}Walkable map (boundary applied)")
        ax.imshow(self.walkable, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

        ax.scatter([su], [sv], s=80, c="red", marker="o", label="start")
        if goal is not None:
            ax.scatter([gu], [gv], s=100, c="cyan", marker="*", label="goal")

        # ROI + local grid box
        self._draw_world_box(ax, sx, sz, roi_size_m, color="yellow", lw=2, label=f"ROI {roi_size_m}m")
        if show_local_grid_box:
            self._draw_world_box(ax, sx, sz, local_grid_size_m, color="lime", lw=2, label=f"Local {local_grid_size_m}m")

        ax.legend(loc="lower right")

        # --- reach map ---
        if reach_map is not None:
            ax = axes[col];
            col += 1
            ax.set_title(f"{title_prefix}Reachability map")
            ax.imshow(reach_map, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")

            ax.scatter([su], [sv], s=80, c="red", marker="o")
            if goal is not None:
                ax.scatter([gu], [gv], s=100, c="cyan", marker="*")

            self._draw_world_box(ax, sx, sz, roi_size_m, color="yellow", lw=2)
            if show_local_grid_box:
                self._draw_world_box(ax, sx, sz, local_grid_size_m, color="lime", lw=2)

        # --- CC labels ---
        if cc_labels is not None:
            ax = axes[col];
            col += 1
            ax.set_title(f"{title_prefix}Connected components (n={self.num_cc})")
            ax.imshow(cc_labels, cmap="nipy_spectral")
            ax.axis("off")

            ax.scatter([su], [sv], s=80, c="red", marker="o")
            if goal is not None:
                ax.scatter([gu], [gv], s=100, c="cyan", marker="*")

            self._draw_world_box(ax, sx, sz, roi_size_m, color="yellow", lw=2)
            if show_local_grid_box:
                self._draw_world_box(ax, sx, sz, local_grid_size_m, color="lime", lw=2)

            # start label 표시
            sui, svi = int(np.floor(su)), int(np.floor(sv))
            if 0 <= sui < self.W and 0 <= svi < self.H:
                start_label = int(cc_labels[svi, sui])
                ax.text(
                    10, 20, f"start CC label = {start_label}",
                    color="white", fontsize=12,
                    bbox=dict(facecolor="black", alpha=0.5, pad=4)
                )

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)

    # ----------------------------
    # (2) 로컬 16x16 grid 자체를 heatmap으로 표시
    # ----------------------------
    def debug_show_local_grid(
            self,
            x_m: float,
            z_m: float,
            yaw_rad: float,
            grid_N: int = 16,
            grid_size_m: float = 4.0,
            sub_div: int = 2,
            use_original_walkable: bool = True,
            figsize: Tuple[int, int] = (4, 4),
            title_prefix: str = "[Debug] "
    ) -> np.ndarray:
        """
        로컬 grid 생성 후 heatmap으로 표시하고 grid를 반환합니다.
        """
        grid = self.make_local_grid(
            x_m=x_m, z_m=z_m, yaw_rad=yaw_rad,
            grid_N=grid_N, grid_size_m=grid_size_m,
            sub_div=sub_div,
            use_original_walkable=use_original_walkable
        )

        xt = torch.tensor([x_m], device="cuda", dtype=torch.float32)
        zt = torch.tensor([z_m], device="cuda", dtype=torch.float32)
        yawt = torch.tensor([yaw_rad], device="cuda", dtype=torch.float32)

        grid_torch = self.make_local_grid_torch_batch(
            x_m = xt, z_m = zt, yaw_rad = yawt,
            grid_N=16,
            grid_size_m=4.0,
            sub_div=2,
            use_original_walkable=True
        )

        grid_gpu_np = grid_torch[0].detach().cpu().numpy()  # (16,16)

        plt.figure(figsize=figsize)
        src_name = "walkable" if use_original_walkable else "reach_map"
        plt.title(f"{title_prefix}Local grid ({grid_N}x{grid_N}, {grid_size_m}m) from {src_name}")
        plt.imshow(grid, cmap="gray", vmin=0, vmax=1)
        plt.axis("off")
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)

        plt.figure(figsize=figsize)
        plt.imshow(grid_gpu_np, cmap="gray", vmin=0, vmax=1)
        plt.axis("off")
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)

        return grid

    def sample_walkable_nearest(self, x_m: float, z_m: float) -> float:
        """월드 좌표에서 가장 가까운 픽셀(nearest)로 walkable 샘플 (0/1)"""
        u, v = self.world_to_pixel(x_m, z_m)
        ui = int(np.floor(u))
        vi = int(np.floor(v))
        if 0 <= ui < self.W and 0 <= vi < self.H:
            return float(self.walkable[vi, ui])
        return 0.0  # 지도 밖은 불가능

    # ----------------------------
    # Reachability Map 생성 + Connected Component 라벨링
    # ----------------------------
    def build_reachability_map(self, jump_bridge_m: float = 0.5):
        """
        jump_bridge_m: '이 정도 폭의 좁은 갭/개울은 넘을 수 있다'를 반영하는 파라미터.
        - 너무 크게 잡으면 도달 불가능도 reachable로 과대평가되므로 주의.
        """
        radius_px = int(np.ceil((jump_bridge_m * 0.5) / self.res_m_per_px))
        radius_px = max(radius_px, 0)

        src = self.walkable.astype(bool)

        if radius_px > 0:
            se = make_disk(radius_px)
            reach = binary_closing(src, structure=se)  # 좁은 틈 메우기(연결 강화)
        else:
            reach = src

        # 외곽은 여전히 막아두는 것이 안전 (원본 walkable에서 이미 막았으니 그대로 유지)
        self.reach_map = reach.astype(np.uint8)

        # connected component 라벨 (4-연결/8-연결 선택 가능)
        # 보폭/이동을 부드럽게 보려면 8-연결을 많이 씁니다.
        structure8 = np.array([[1,1,1],
                               [1,1,1],
                               [1,1,1]], dtype=np.uint8)
        labels, n = label(self.reach_map.astype(bool), structure=structure8)
        self.cc_labels = labels
        self.num_cc = n

    @torch.no_grad()
    def sample_start_torch_update_by_indices(
            self,
            start_x: torch.Tensor,  # (B,) float32 (기존)
            start_z: torch.Tensor,  # (B,) float32 (기존)
            indices: torch.Tensor,  # (M,) long
            use_reach_map: bool = True,
            require_walkable_on_original: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        start_x/start_z는 유지하고, indices 위치만 새로 샘플해서 덮어씀.
        """
        if not hasattr(self, "walkable_t"):
            raise RuntimeError("먼저 to_torch(device=..., use_reach_map=...)를 호출하세요.")

        device = self.device
        start_x = start_x.to(device=device, dtype=torch.float32)
        start_z = start_z.to(device=device, dtype=torch.float32)
        indices = indices.to(device=device, dtype=torch.long)

        # 복사본에 덮어쓰기
        out_x = start_x.clone()
        out_z = start_z.clone()

        if use_reach_map:
            if self.cc_labels_t is None:
                raise RuntimeError("use_reach_map=True이면 build_reachability_map() 후 to_torch()에서 cc_labels_t가 필요합니다.")
            cand = (self.cc_labels_t != 0)
        else:
            cand = (self.walkable_t[0, 0] == 1)

        if require_walkable_on_original:
            cand = cand & (self.walkable_t[0, 0] == 1)

        ys, xs = torch.where(cand)  # (P,)
        if xs.numel() == 0:
            raise RuntimeError("start로 쓸 수 있는 위치가 없습니다.")

        M = indices.numel()
        ridx = torch.randint(0, xs.numel(), (M,), device=device)
        u = xs[ridx].to(torch.float32) + 0.5
        v = ys[ridx].to(torch.float32) + 0.5
        x, z = self._pixel_to_world_torch(u, v)  # (M,)

        out_x[indices] = x
        out_z[indices] = z
        return out_x, out_z

    def sample_start(self, use_reach_map: bool = True, require_walkable_on_original: bool = True) -> Tuple[float, float]:
        if use_reach_map:
            if self.cc_labels is None:
                raise RuntimeError("build_reachability_map()를 먼저 호출해야 합니다.")

        if use_reach_map:
            # reachable 영역 (label != 0)
            cand = (self.cc_labels != 0)
        else:
            cand = (self.walkable == 1)

        if require_walkable_on_original:
            cand = cand & (self.walkable == 1)

            # -----------------------------
            # 후보 좌표 뽑기
            # -----------------------------

        ys, xs = np.where(cand)

        if len(xs) == 0:
            raise RuntimeError("start로 쓸 수 있는 walkable 위치가 없습니다.")

        # 랜덤 선택
        idx = np.random.randint(0, len(xs))
        u = xs[idx]
        v = ys[idx]

        # -----------------------------
        # pixel → world 변환
        # -----------------------------

        x, z = self.pixel_to_world(u + 0.5, v + 0.5)

        return x, z

    @torch.no_grad()
    def sample_goal_near_start_torch_update_by_indices(
            self,
            goal_x: torch.Tensor,  # (B,) 기존 goal
            goal_z: torch.Tensor,  # (B,) 기존 goal
            start_x: torch.Tensor,  # (B,) start
            start_z: torch.Tensor,  # (B,) start
            indices: torch.Tensor,  # (M,)
            roi_size_m: float = 10.0,
            use_reach_map: bool = True,
            require_walkable_on_original: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        goal_x/goal_z는 유지하고, indices 위치만 start 주변 ROI에서 샘플해서 덮어씀.
        파이썬 루프 없이 벡터화 (ROI patch gather).
        """
        if not hasattr(self, "walkable_t"):
            raise RuntimeError("먼저 to_torch(device=..., use_reach_map=...)를 호출하세요.")

        device = self.device
        goal_x = goal_x.to(device=device, dtype=torch.float32)
        goal_z = goal_z.to(device=device, dtype=torch.float32)
        start_x = start_x.to(device=device, dtype=torch.float32)
        start_z = start_z.to(device=device, dtype=torch.float32)
        indices = indices.to(device=device, dtype=torch.long)

        out_gx = goal_x.clone()
        out_gz = goal_z.clone()

        if use_reach_map and (self.cc_labels_t is None):
            raise RuntimeError("use_reach_map=True이면 build_reachability_map() 후 to_torch()에서 cc_labels_t가 필요합니다.")

        # ---- (1) indices에 해당하는 start만 뽑기 (M,)
        sx = start_x[indices]
        sz = start_z[indices]

        # start -> pixel center (float)
        su, sv = self._world_to_pixel_torch(sx, sz)  # (M,)
        # pixel index (long)
        sui = torch.floor(su).to(torch.long)
        svi = torch.floor(sv).to(torch.long)

        # 지도 밖 start는 샘플 제외
        inside = (sui >= 0) & (sui < self.W) & (svi >= 0) & (svi < self.H)

        if inside.sum() == 0:
            return out_gx, out_gz  # 아무 것도 못함

        # 유효한 subset만 처리
        idx_in = indices[inside]  # (Mi,)
        sui_in = sui[inside]  # (Mi,)
        svi_in = svi[inside]  # (Mi,)
        Mi = sui_in.numel()

        # start label (Mi,)
        if use_reach_map:
            start_label = self.cc_labels_t[svi_in, sui_in]  # int
            # label==0이면 불가
            ok_label = (start_label != 0)
            if ok_label.sum() == 0:
                return out_gx, out_gz
            idx_in = idx_in[ok_label]
            sui_in = sui_in[ok_label]
            svi_in = svi_in[ok_label]
            start_label = start_label[ok_label]
            Mi = sui_in.numel()

        # ---- (2) ROI 픽셀 오프셋 만들기: K = (2du+1)*(2dv+1)
        half = float(roi_size_m) * 0.5
        du = int(torch.ceil(torch.tensor(half / self.res_m_per_px)).item())
        dv = int(torch.ceil(torch.tensor(half / self.res_m_per_px)).item())

        off_u = torch.arange(-du, du + 1, device=device, dtype=torch.long)  # (2du+1)
        off_v = torch.arange(-dv, dv + 1, device=device, dtype=torch.long)  # (2dv+1)
        grid_v, grid_u = torch.meshgrid(off_v, off_u, indexing="ij")  # (2dv+1,2du+1)
        off_u_flat = grid_u.reshape(-1)  # (K,)
        off_v_flat = grid_v.reshape(-1)  # (K,)
        K = off_u_flat.numel()

        # ---- (3) 각 샘플에 대해 ROI 좌표 (Mi,K)
        u = sui_in[:, None] + off_u_flat[None, :]  # (Mi,K)
        v = svi_in[:, None] + off_v_flat[None, :]  # (Mi,K)

        # clamp (경계 밖은 경계로 붙음)  -> 마스크로 배제도 가능하지만 간단히 clamp 후 insideROI mask로 배제
        u_cl = u.clamp(0, self.W - 1)
        v_cl = v.clamp(0, self.H - 1)

        # ROI의 원래 inside 마스크 (clamp로 인해 중복/왜곡 방지 목적)
        roi_inside = (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)  # (Mi,K)

        # ---- (4) 후보 마스크 만들기 (Mi,K)
        if use_reach_map:
            lab = self.cc_labels_t[v_cl, u_cl]  # (Mi,K)
            cand = (lab == start_label[:, None]) & roi_inside
        else:
            cand = (self.walkable_t[0, 0, v_cl, u_cl] == 1) & roi_inside

        if require_walkable_on_original:
            cand = cand & (self.walkable_t[0, 0, v_cl, u_cl] == 1)

        # ---- (5) 각 행에서 랜덤 1개 선택 (multinomial)
        weights = cand.to(torch.float32)  # (Mi,K)
        row_sum = weights.sum(dim=1)  # (Mi,)
        has_any = row_sum > 0  # (Mi,)

        if has_any.sum() == 0:
            return out_gx, out_gz

        # multinomial은 sum>0인 행만
        weights_ok = weights[has_any]  # (Mj,K)
        pick = torch.multinomial(weights_ok, num_samples=1, replacement=True).squeeze(1)  # (Mj,)

        u_pick = u_cl[has_any, :].gather(1, pick[:, None]).squeeze(1)  # (Mj,)
        v_pick = v_cl[has_any, :].gather(1, pick[:, None]).squeeze(1)  # (Mj,)

        # 픽셀 -> 월드 (0.5 center)
        u_f = u_pick.to(torch.float32) + 0.5
        v_f = v_pick.to(torch.float32) + 0.5
        gx, gz = self._pixel_to_world_torch(u_f, v_f)  # (Mj,)

        # 덮어쓰기
        idx_ok = idx_in[has_any]  # (Mj,)
        out_gx[idx_ok] = gx
        out_gz[idx_ok] = gz
        return out_gx, out_gz

    # ----------------------------
    # Start 주변 ROI에서 reachable goal 샘플링
    # ----------------------------
    def sample_goal_near_start(
        self,
        start_x: float,
        start_z: float,
        roi_size_m: float = 10.0,   # 10m x 10m
        max_tries: int = 2000,
        use_reach_map: bool = True,
        require_walkable_on_original: bool = True,
    ) -> Tuple[float, float]:
        """
        start 주변 roi_size_m 박스 안에서 goal 샘플.
        - use_reach_map=True: reachability CC 기준으로 같은 컴포넌트에서만 샘플
        - require_walkable_on_original=True: 최종 goal은 원본 walkable에서도 1이어야 함(권장)
        """

        if use_reach_map:
            if self.cc_labels is None:
                raise RuntimeError("build_reachability_map()를 먼저 호출해야 합니다.")

        # start 픽셀
        su, sv = self.world_to_pixel(start_x, start_z)
        sui, svi = int(np.floor(su)), int(np.floor(sv))
        if not (0 <= sui < self.W and 0 <= svi < self.H):
            raise ValueError("start가 지도 밖입니다.")

        if use_reach_map:
            start_label = int(self.cc_labels[svi, sui])
            if start_label == 0:
                raise ValueError("start가 reachability map에서 막힌 위치입니다.")

        # ROI를 픽셀 박스로 변환
        half = roi_size_m * 0.5
        du_px = int(np.ceil(half / self.res_m_per_px))
        dv_px = int(np.ceil(half / self.res_m_per_px))

        u0 = max(sui - du_px, 0)
        u1 = min(sui + du_px + 1, self.W)
        v0 = max(svi - dv_px, 0)
        v1 = min(svi + dv_px + 1, self.H)

        # 후보 마스크 만들기
        if use_reach_map:
            cand = (self.cc_labels[v0:v1, u0:u1] == start_label)
        else:
            cand = (self.walkable[v0:v1, u0:u1] == 1)

        if require_walkable_on_original:
            cand = cand & (self.walkable[v0:v1, u0:u1] == 1)

        ys, xs = np.where(cand)
        if len(xs) == 0:
            raise RuntimeError("ROI 내에 reachable goal 후보가 없습니다. ROI를 키우거나 맵/파라미터를 조정해 주세요.")

        # 랜덤 선택
        idx = np.random.randint(0, len(xs))
        gu = u0 + xs[idx]
        gv = v0 + ys[idx]
        gx, gz = self.pixel_to_world(gu + 0.5, gv + 0.5)
        return gx, gz

    @torch.no_grad()
    def make_local_grid_torch_batch(
            self,
            x_m: torch.Tensor,  # (B,)
            z_m: torch.Tensor,  # (B,)
            yaw_rad: torch.Tensor,  # (B,)
            grid_N: int = 16,
            grid_size_m: float = 4.0,
            sub_div: int = 2,
            use_original_walkable: bool = True,
    ) -> torch.Tensor:
        """
        CPU make_local_grid와 픽셀 선택 규칙까지 '완전 동일'하게 맞춘 버전.
        - floor(U), floor(V)로 정수 픽셀 인덱스 고정 선택
        - bounds 밖은 0
        반환: (B,N,N) float32
        """

        if not hasattr(self, "walkable_t"):
            raise RuntimeError("먼저 to_torch(device=...)를 호출해서 맵을 GPU에 올려야 합니다.")

        if x_m.dim() != 1 or z_m.dim() != 1 or yaw_rad.dim() != 1:
            raise ValueError("x_m, z_m, yaw_rad는 모두 (B,) 1D 텐서여야 합니다.")

        B = x_m.shape[0]
        device = self.device

        x_m = x_m.to(device=device, dtype=torch.float32)
        z_m = z_m.to(device=device, dtype=torch.float32)
        yaw_rad = yaw_rad.to(device=device, dtype=torch.float32)

        # src 선택 (1,1,H,W)
        if use_original_walkable:
            src = self.walkable_t
        else:
            if self.reach_map_t is None:
                raise RuntimeError("use_original_walkable=False이면 to_torch(use_reach_map=True)로 reach_map도 올려야 합니다.")
            src = self.reach_map_t

        # (B,1,H,W)로 확장
        srcB = src.expand(B, -1, -1, -1)  # float32

        N = grid_N
        cell = grid_size_m / float(N)
        half = grid_size_m * 0.5

        # grid cell centers (N,N)
        xs = (-half + (torch.arange(N, device=device, dtype=torch.float32) + 0.5) * cell)
        zs = (half - (torch.arange(N, device=device, dtype=torch.float32) + 0.5) * cell)
        Zc, Xc = torch.meshgrid(zs, xs, indexing="ij")  # (N,N)

        # subcell offsets (K,)
        offs = (torch.arange(sub_div, device=device, dtype=torch.float32) + 0.5) / float(sub_div) - 0.5
        offs = offs * cell
        oz, ox = torch.meshgrid(offs, offs, indexing="ij")
        off_x = ox.reshape(-1)  # (K,)
        off_z = oz.reshape(-1)  # (K,)
        K = off_x.numel()

        # (B,N,N,K)
        X = (Xc[..., None] + off_x).unsqueeze(0).expand(B, -1, -1, -1)
        Z = (Zc[..., None] + off_z).unsqueeze(0).expand(B, -1, -1, -1)

        # rotate + translate
        c = torch.cos(yaw_rad).view(B, 1, 1, 1)
        s = torch.sin(yaw_rad).view(B, 1, 1, 1)
        Xw = c * X - s * Z + x_m.view(B, 1, 1, 1)
        Zw = s * X + c * Z + z_m.view(B, 1, 1, 1)

        # world -> pixel (same formula as numpy version)
        U = (Xw / self.world_size_t + 0.5) * self.W_t
        V = (0.5 - Zw / self.world_size_t) * self.H_t

        # ★ 핵심: floor로 픽셀 선택을 CPU와 동일하게
        ui = torch.floor(U).to(torch.int64)
        vi = torch.floor(V).to(torch.int64)

        # bounds mask
        valid = (ui >= 0) & (ui < self.W) & (vi >= 0) & (vi < self.H)

        # gather
        # srcB: (B,1,H,W) -> (B,H,W)
        srcBH = srcB[:, 0]  # (B,H,W)
        vals = torch.zeros((B, N, N, K), device=device, dtype=torch.float32)

        # batch index for advanced indexing
        b_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, N, N, K)

        vals[valid] = srcBH[b_idx[valid], vi[valid], ui[valid]]

        # subcell 평균 -> (B,N,N)
        grid = vals.mean(dim=-1)
        return grid

    @torch.no_grad()
    def make_local_grid_two_views(
            self,
            x_m: torch.Tensor,  # (B,)
            z_m: torch.Tensor,  # (B,)
            yaw_rad: torch.Tensor,  # (B,) agent yaw
            grid_N: int = 16,
            grid_size_m: float = 4.0,
            sub_div: int = 2,
            use_original_walkable: bool = True,
    ):
        """
        return:
          map_world: (B,N,N)  yaw=0 (world-aligned)
          map_yaw:   (B,N,N)  yaw=yaw_rad (agent-aligned)
        """
        device = self.device
        B = x_m.shape[0]
        yaw0 = torch.zeros((B,), device=device, dtype=torch.float32)

        map_world = self.make_local_grid_torch_batch(
            x_m, z_m, yaw0,
            grid_N=grid_N, grid_size_m=grid_size_m, sub_div=sub_div,
            use_original_walkable=use_original_walkable
        )
        map_yaw = self.make_local_grid_torch_batch(
            x_m, z_m, yaw_rad,
            grid_N=grid_N, grid_size_m=grid_size_m, sub_div=sub_div,
            use_original_walkable=use_original_walkable
        )
        return map_world, map_yaw

    @torch.no_grad()
    def make_local_grid_two_views_fast(
            self,
            x_m: torch.Tensor, z_m: torch.Tensor, yaw_rad: torch.Tensor,
            grid_N: int = 16, grid_size_m: float = 4.0, sub_div: int = 2,
            use_original_walkable: bool = True,
    ):
        device = self.device
        B = x_m.shape[0]

        x2 = torch.cat([x_m, x_m], dim=0)
        z2 = torch.cat([z_m, z_m], dim=0)
        yaw0 = torch.zeros((B,), device=device, dtype=torch.float32)
        yaw2 = torch.cat([yaw0, yaw_rad], dim=0)

        grid2 = self.make_local_grid_torch_batch(
            x2, z2, yaw2,
            grid_N=grid_N, grid_size_m=grid_size_m, sub_div=sub_div,
            use_original_walkable=use_original_walkable
        )  # (2B,N,N)

        map_world = grid2[:B]
        map_yaw = grid2[B:]
        return map_world, map_yaw


    def make_local_grid(
            self,
            x_m: float,
            z_m: float,
            yaw_rad: float,
            grid_N: int = 16,
            grid_size_m: float = 4.0,
            sub_div: int = 2,
            use_original_walkable: bool = True,
    ) -> np.ndarray:
        """
        NumPy vectorized version of make_local_grid (no python loops)
        """

        cell = grid_size_m / grid_N
        half = grid_size_m * 0.5

        # ------------------------
        # Local grid centers
        # ------------------------

        xs = (-half + (np.arange(grid_N) + 0.5) * cell)
        zs = (half - (np.arange(grid_N) + 0.5) * cell)

        Xc, Zc = np.meshgrid(xs, zs)  # (N,N)

        # ------------------------
        # Subcell offsets
        # ------------------------

        offs = (np.arange(sub_div) + 0.5) / sub_div - 0.5
        offs = offs * cell
        off_x, off_z = np.meshgrid(offs, offs)
        off_x = off_x.reshape(-1)
        off_z = off_z.reshape(-1)  # (K,)

        K = len(off_x)

        # ------------------------
        # Expand to all samples
        # ------------------------

        X = Xc[..., None] + off_x  # (N,N,K)
        Z = Zc[..., None] + off_z

        # ------------------------
        # Rotate: local → world
        # ------------------------

        c = np.cos(yaw_rad)
        s = np.sin(yaw_rad)

        Xw = c * X - s * Z + x_m
        Zw = s * X + c * Z + z_m

        # ------------------------
        # World → Pixel
        # ------------------------

        U = (Xw / self.world_size_m + 0.5) * self.W
        V = (0.5 - Zw / self.world_size_m) * self.H

        ui = np.floor(U).astype(np.int32)
        vi = np.floor(V).astype(np.int32)

        # ------------------------
        # Sampling
        # ------------------------

        src = self.walkable if use_original_walkable else self.reach_map
        if src is None:
            raise RuntimeError("reach_map 없음")

        valid = (
                (ui >= 0) & (ui < self.W) &
                (vi >= 0) & (vi < self.H)
        )

        vals = np.zeros_like(ui, dtype=np.float32)
        vals[valid] = src[vi[valid], ui[valid]]

        # ------------------------
        # Average subcells
        # ------------------------

        grid = vals.mean(axis=-1)

        return grid.astype(np.float32)

    @torch.no_grad()
    def query_walkable_batch(
            self,
            x_m: torch.Tensor,  # (B,)
            z_m: torch.Tensor,  # (B,)
            use_original_walkable: bool = True,
    ) -> torch.Tensor:
        """
        반환: (B,) float32 in [0,1]
          - 1이면 walkable
          - 0이면 obstacle 또는 bounds 밖
        """
        if not hasattr(self, "walkable_t"):
            raise RuntimeError("to_torch(device=...)를 먼저 호출해서 walkable_t를 만들어야 합니다.")

        device = self.device
        x_m = x_m.to(device=device, dtype=torch.float32).view(-1)
        z_m = z_m.to(device=device, dtype=torch.float32).view(-1)
        B = x_m.shape[0]

        if use_original_walkable:
            src = self.walkable_t  # (1,1,H,W)
        else:
            if self.reach_map_t is None:
                raise RuntimeError("use_original_walkable=False이면 reach_map_t가 필요합니다.")
            src = self.reach_map_t

        srcHW = src[0, 0]  # (H,W)

        # world -> pixel (numpy 버전과 동일한 공식)
        U = (x_m / self.world_size_t + 0.5) * self.W_t
        V = (0.5 - z_m / self.world_size_t) * self.H_t
        ui = torch.floor(U).to(torch.int64)
        vi = torch.floor(V).to(torch.int64)

        valid = (ui >= 0) & (ui < self.W) & (vi >= 0) & (vi < self.H)

        out = torch.zeros((B,), device=device, dtype=torch.float32)
        out[valid] = srcHW[vi[valid], ui[valid]]
        return out

def assert_close(a, b, eps=1e-5, msg=""):
    if not torch.allclose(a, b, atol=eps, rtol=0):
        raise AssertionError(msg + f" max_abs={(a-b).abs().max().item()}")

@torch.no_grad()
def test_indices_only_update(nav, B=512, M=32):
    device = nav.device
    indices = torch.randperm(B, device=device)[:M]

    # 초기 start/goal (의미 없는 값으로 채우되, 유지되는지 확인용)
    start_x0 = torch.linspace(-1, 1, B, device=device)
    start_z0 = torch.linspace( 2, 3, B, device=device)
    goal_x0  = torch.linspace(-4, -3, B, device=device)
    goal_z0  = torch.linspace( 7, 8, B, device=device)

    # start 업데이트
    start_x1, start_z1 = nav.sample_start_torch_update_by_indices(
        start_x0, start_z0, indices, use_reach_map=True
    )

    # indices 밖은 그대로여야 함
    mask = torch.ones(B, device=device, dtype=torch.bool)
    mask[indices] = False
    assert_close(start_x1[mask], start_x0[mask], msg="start_x changed outside indices")
    assert_close(start_z1[mask], start_z0[mask], msg="start_z changed outside indices")

    # goal 업데이트
    goal_x1, goal_z1 = nav.sample_goal_near_start_torch_update_by_indices(
        goal_x0, goal_z0, start_x1, start_z1, indices, roi_size_m=10.0,
        use_reach_map=True
    )

    # indices 밖은 그대로여야 함
    assert_close(goal_x1[mask], goal_x0[mask], msg="goal_x changed outside indices")
    assert_close(goal_z1[mask], goal_z0[mask], msg="goal_z changed outside indices")

    print("[OK] indices-only update works.")



@torch.no_grad()
def test_goal_constraints(nav, B=512, M=64, roi_size_m=10.0):
    device = nav.device
    indices = torch.randperm(B, device=device)[:M]

    # start/goal init
    start_x = torch.zeros(B, device=device)
    start_z = torch.zeros(B, device=device)
    goal_x  = torch.zeros(B, device=device)
    goal_z  = torch.zeros(B, device=device)

    start_x, start_z = nav.sample_start_torch_update_by_indices(start_x, start_z, indices, use_reach_map=True)
    goal_x, goal_z   = nav.sample_goal_near_start_torch_update_by_indices(
        goal_x, goal_z, start_x, start_z, indices, roi_size_m=roi_size_m, use_reach_map=True
    )

    # start/goal -> pixel
    su, sv = nav._world_to_pixel_torch(start_x[indices], start_z[indices])
    gu, gv = nav._world_to_pixel_torch(goal_x[indices],  goal_z[indices])

    sui = torch.floor(su).long().clamp(0, nav.W-1)
    svi = torch.floor(sv).long().clamp(0, nav.H-1)
    gui = torch.floor(gu).long().clamp(0, nav.W-1)
    gvi = torch.floor(gv).long().clamp(0, nav.H-1)

    # label check
    s_label = nav.cc_labels_t[svi, sui]
    g_label = nav.cc_labels_t[gvi, gui]
    same = (s_label == g_label) & (s_label != 0)

    # ROI check (world distance in x,z, axis-aligned ROI)
    half = roi_size_m * 0.5
    dx = (goal_x[indices] - start_x[indices]).abs()
    dz = (goal_z[indices] - start_z[indices]).abs()
    in_roi = (dx <= half + 1e-6) & (dz <= half + 1e-6)

    ok = same & in_roi
    ratio = ok.float().mean().item()
    print(f"[CHECK] reachable&ROI 만족 비율: {ratio*100:.1f}% (M={M})")

    # 너무 낮으면 샘플링 로직/맵/ROI가 문제일 가능성이 큼
    if ratio < 0.95:
        bad = (~ok).nonzero(as_tuple=False).squeeze(1)[:10]
        print("예시 bad index(최대 10개):", bad.tolist())



# ----------------------------
# 사용 예시
# ----------------------------
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    '''nav = PerlinNavMap(
        #img_path="/mnt/data/perlin6_500_500_mid_tauA0.92_tauB0.13_g1.7_overlap.png",
        img_path=BASE_DIR / "perlin6_500_500_raw_tauA0.71_tauB0.71_g1.7_A_masked.png",
        world_size_m=50.0,
        boundary_block_m=2.0,
        white_threshold=127,
    )'''

    A = generate_A_masked_binary01(size=500, seedA=5, seedB=100,tauA = 0.71,
    tauB = 0.71)  # (500,500) {0,1}

    nav = PerlinNavMap(
        img_path=A,
        world_size_m=50.0,
        boundary_block_m=2.0,
        input_is_binary01=True,  # 중요!
    )

    # goal 샘플링을 위해 reachability map 생성(점프/보폭으로 넘길 폭 반영)
    nav.build_reachability_map(jump_bridge_m=0.6)
    nav.to_torch(device="cuda",use_reach_map=True)

    # 예: b=7만 확인
    b = 7
    b2= 9
    B = 512
    device = nav.device
    indices = torch.tensor([b,b2], device=device)


    start_x = torch.zeros(B, device=device)
    start_z = torch.zeros(B, device=device)
    goal_x = torch.zeros(B, device=device)
    goal_z = torch.zeros(B, device=device)

    start_x, start_z = nav.sample_start_torch_update_by_indices(start_x, start_z, indices, use_reach_map=True)
    goal_x, goal_z = nav.sample_goal_near_start_torch_update_by_indices(goal_x, goal_z, start_x, start_z, indices,
                                                                        roi_size_m=10.0)

    start2_x, start2_z = nav.sample_start_torch_update_by_indices(start_x, start_z, indices, use_reach_map=True)
    goal2_x, goal2_z = nav.sample_goal_near_start_torch_update_by_indices(goal_x, goal_z, start2_x, start2_z, indices, roi_size_m=10.0)

    start = (float(start_x[b].item()), float(start_z[b].item()))
    goal = (float(goal_x[b].item()), float(goal_z[b].item()))

    start2 = (float(start2_x[b2].item()), float(start2_z[b2].item()))
    goal2 = (float(goal2_x[b2].item()), float(goal2_z[b2].item()))

    nav.debug_show_maps(start=start, goal=goal, roi_size_m=10.0)
    nav.debug_show_maps(start=start2, goal=goal2, roi_size_m=10.0)

    nav.debug_show_local_grid(x_m=start[0], z_m=start[1], yaw_rad=0)
    nav.debug_show_local_grid(x_m=start[0], z_m=start[1], yaw_rad=3.141592/4)

    nav.debug_show_local_grid(x_m=start2[0], z_m=start2[1], yaw_rad=0)
    nav.debug_show_local_grid(x_m=start2[0], z_m=start2[1], yaw_rad=3.141592 / 4)

    print("Goal:", goal[0], goal[1])

    test_indices_only_update(nav, B=512, M=32)
    test_goal_constraints(nav, B=512, M=64, roi_size_m=10.0)

    import matplotlib.pyplot as plt
    plt.show()
    # 로컬 grid 생성(현재 yaw=0이면 전방이 +z, grid 위쪽)
    #yaw = 0.3
    #local_grid = nav.make_local_grid(start_x, start_z, yaw, grid_N=16, grid_size_m=4.0, sub_div=2)
    #print("Local grid shape:", local_grid.shape, "range:", local_grid.min(), local_grid.max())