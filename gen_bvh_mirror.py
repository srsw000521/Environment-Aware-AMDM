#!/usr/bin/env python
import argparse
import dataset.util.bvh as bvh_util
import numpy as np
from pathlib import Path


LR_PAIRS = [
    ("LeftUpLeg", "RightUpLeg"),
    ("LeftLeg", "RightLeg"),
    ("LeftFoot", "RightFoot"),
    ("LeftToe", "RightToe"),
    ("LeftShoulder", "RightShoulder"),
    ("LeftArm", "RightArm"),
    ("LeftForeArm", "RightForeArm"),
    ("LeftHand", "RightHand"),
]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate mirrored BVH files for LaFAN1."
    )
    parser.add_argument(
        "--src_dir",
        type=str,
        default="/home/sura/AMDM/data/LAFAN1",
        help="원본 BVH들이 들어있는 디렉토리",
    )
    parser.add_argument(
        "--dst_dir",
        type=str,
        default=None,
        help="미러 BVH를 저장할 디렉토리 (기본값: src_dir와 동일)",
    )
    return parser.parse_args()

def mirror_bvh(in_path: Path, out_path: Path):
    motion = bvh_util.import_bvh(str(in_path), end_eff=False)

    positions = motion._positions.copy()      # (F, J, 3)
    rotations = motion._rotations.copy()      # (F, J, 3, 3)

    # 1) 좌우(Z축) 방향 위치 반사: z -> -z
    positions[..., 2] *= -1.0

    # 2) 회전 반사: R' = M R M, M = diag(1,1,-1)
    M = np.diag([1.0, 1.0, -1.0])
    F, J, _, _ = rotations.shape
    rot_flat = rotations.reshape(-1, 3, 3)
    rot_mir_flat = M @ rot_flat @ M
    rot_mir = rot_mir_flat.reshape(F, J, 3, 3)

    # 3) Left / Right 조인트 스왑
    joint_lst = motion._skeleton._joint_lst
    name_to_idx = {jnt._name: jnt._idx for jnt in joint_lst}

    for left_name, right_name in LR_PAIRS:
        if left_name not in name_to_idx or right_name not in name_to_idx:
            continue
        li = name_to_idx[left_name]
        ri = name_to_idx[right_name]

        tmp_pos = positions[:, li, :].copy()
        positions[:, li, :] = positions[:, ri, :]
        positions[:, ri, :] = tmp_pos

        tmp_rot = rot_mir[:, li, :, :].copy()
        rot_mir[:, li, :, :] = rot_mir[:, ri, :, :]
        rot_mir[:, ri, :, :] = tmp_rot

    motion._positions = positions
    motion._rotations = rot_mir

    bvh_util.export_bvh(str(out_path), motion)
    print(f"[OK] {in_path.name} -> {out_path.name} (Z축 미러 + LR 스왑)")

def mirror_foot(in_path: Path, out_path: Path):

    with open(in_path, "r") as f:
        lines = f.readlines()

    out_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            out_lines.append("\n")
            continue

        a, b = line.split()
        out_lines.append(f"{b} {a}\n")  # swap

    with open(out_path, "w") as f:
        f.writelines(out_lines)

    print(f"[OK] {in_path.name} -> {out_path.name} (foot swap)")


def main():
    args = parse_args()

    src_dir = Path(args.src_dir).expanduser().resolve()
    dst_dir = Path(args.dst_dir).expanduser().resolve() if args.dst_dir else src_dir

    dst_dir.mkdir(parents=True, exist_ok=True)

    bvh_paths = sorted(src_dir.glob("*.bvh"))
    foot_paths = sorted(src_dir.glob("*.txt"))

    for in_path in bvh_paths:
        out_name = in_path.stem + "_mirror" + in_path.suffix  # 원래이름 + _mirror.bvh
        out_path = dst_dir / out_name

        print(f"[INFO] {in_path.name}  ->  {out_path.name}")
        mirror_bvh(in_path, out_path)

    for in_path in foot_paths:
        out_name = in_path.stem + "_mirror" + in_path.suffix
        out_path = dst_dir / out_name

        print(f"[INFO] {in_path.name}  ->  {out_path.name}")
        mirror_foot(in_path, out_path)

if __name__ == "__main__":
    main()
