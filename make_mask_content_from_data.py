import os
import os.path as osp

import numpy
import numpy as np
import copy
import pandas as pd

import dataset.util.geo as geo_util
import dataset.util.unit as unit_util
import dataset.util.plot as plot_util
from dataset.util.motion_struct import Skeleton, Joint, Motion
from scipy.spatial.transform import Rotation as R


def import_bvh(bvh_file, root_joint_name=None, end_eff=False):
    with open(bvh_file, "rb") as file:
        items = [w.decode() for line in file for w in line.strip().split()]
        n_items = len(items)

    cnt, depth = 0, 0
    joint_stack = [None, None]
    joint_list = []
    skeleton = Skeleton()

    # build skeleton
    while cnt < n_items and len(joint_stack) > 0:
        joint_cur = joint_stack[-1]
        item = items[cnt].lower()

        if item in ["root", "joint"]:
            name = items[cnt + 1]
            joint = Joint(name=name, idx=len(joint_list))

            if item == "joint":
                joint.add_parent(joint_cur)
            else:
                skeleton.set_root(joint)
            joint_stack.append(joint)
            joint_list.append(joint)
            cnt += 2

        elif item == "end" and items[cnt + 1].lower() == 'site':
            name = 'end_eff_{}'.format(joint_cur._name)
            if end_eff:
                joint = Joint(name=name, idx=len(joint_list))
                joint_stack.append(joint)
                joint_list.append(joint)
                joint.add_parent(joint_cur)
            cnt += 2

        elif item == "offset":
            if (end_eff and name[:7] == 'end_eff') or name[:7] != 'end_eff':
                x = float(items[cnt + 1])
                y = float(items[cnt + 2])
                z = float(items[cnt + 3])

                if joint_cur._parent_idx is not None:
                    coords = np.array([x, y, z])
                else:
                    coords = np.array([0, 0, 0])
                joint_cur.set_offset(coords)
            cnt += 4

        elif item == "channels":
            ndof = int(items[cnt + 1])

            axis_order = []
            assert ndof in [1, 2, 3, 6], "unsupported num of dof {} for joint {}".format(ndof, joint_cur._name)
            joint_cur.set_dof(ndof)

            for i in range(ndof):
                axis = items[cnt + 2 + i]
                if axis[1:] == 'position':
                    continue
                else:
                    axis_order.append(axis[0])

            joint_cur.set_rot_axis_order(''.join(axis_order))
            cnt += 2
            cnt += ndof

        elif item == "{":
            depth += 1
            cnt += 1

        elif item == "}":

            depth -= 1
            if not items[cnt - 4].lower() == 'offset' and not end_eff and name[:7] == 'end_eff':
                joint_stack.pop()
            cnt += 1

            if depth == 0:
                break

        elif item == "hierarchy":
            cnt += 1

        else:
            raise Exception("Unknown Token {}: {} {}".format(cnt, item, items[cnt - 5:cnt + 5]))

    skeleton.add_joints(joint_list)

    num_jnt = len(joint_list)
    motion = Motion(skeleton)

    # load motion info
    while cnt < n_items:
        item = items[cnt].lower()
        if item == 'motion':
            cnt += 1

        elif item == 'frames:':
            num_frames = int(items[cnt + 1])
            motion.set_num_frames(num_frames)
            cnt += 2

        elif item == 'frame' and items[cnt + 1].lower() == 'time:':
            fps = round(1.0 / float(items[cnt + 2]))
            motion.set_fps(fps)
            cnt += 3
            break

    # load motion
    rotations = np.zeros((num_frames, num_jnt, 3, 3))
    root_trans = np.zeros((num_frames, 3))
    for idx_frame in range(num_frames):
        for idx_jnt in range(num_jnt):
            dof = joint_list[idx_jnt]._ndof
            rot_axis_order = joint_list[idx_jnt]._rot_axis_order
            vec = np.array([float(x) for x in items[cnt: cnt + dof]])
            if dof == 6:
                if joint_list[idx_jnt]._parent_idx is None:
                    root_trans[idx_frame] = vec[:3]
                    rotations[idx_frame, idx_jnt] = R.from_euler(rot_axis_order, vec[3:], degrees=True).as_matrix()
                else:
                    if root_joint_name is not None and joint_list[idx_jnt]._name == root_joint_name:
                        root_trans[idx_frame] = vec[:3]
                    rotations[idx_frame, idx_jnt] = R.from_euler(rot_axis_order, vec[3:], degrees=True).as_matrix()
                cnt += dof

            elif 0 < dof <= 3:
                rotations[idx_frame, idx_jnt] = R.from_euler(rot_axis_order, vec[:dof], degrees=True).as_matrix()
                cnt += dof

    motion.set_motion_frames(root_trans, rotations)

    return motion


def compute_onset_indices(contact):             # 0->1로 바뀌는 순간의 frame idx
    T = len(contact)
    onset = np.zeros(T, dtype=bool)
    onset[1:] = (contact[1:] == 1) & (contact[:-1] == 0)
    return np.where(onset)[0]

def map_next_onset_index(onset_idx, T):
    next_onset = np.full(T, -1, dtype=int)
    for t in range(T):
        jpos = np.searchsorted(onset_idx, t+1, side='left')
        if jpos < len(onset_idx):
            next_onset[t] = onset_idx[jpos]
    return next_onset

def make_foot_dxz( world_heading_t, world_foot_pos_j, world_root_pos_t ) :

    v = world_foot_pos_j - world_root_pos_t
    global_heading_rot = geo_util.rot_yaw(world_heading_t)
    v_local = global_heading_rot @ v

    return v_local[[0,2]]

def read_bvh_loco(path, foot_path, root_rot_offset=0, frame_start=None, frame_end=None, is_mirrored=False):

    motion = import_bvh(path, end_eff=False)
    positions = motion._positions * 0.01  # (frames, joints, 3)
    rotations = motion._rotations  #
    root_idx = motion._skeleton._root._idx

    total_len = motion._num_frames

    if frame_start is not None and frame_end is not None:
        end_idx = total_len - frame_end
        positions = positions[frame_start:end_idx]
        rotations = rotations[frame_start:end_idx]

    source_fps = motion._fps

    nfrm, njoint, _ = positions.shape

    positions_world = positions.copy()

    ori = copy.deepcopy(positions[0, root_idx])

    y_min = np.min(positions[0, :, 1])
    ori[1] = y_min

    positions = positions - ori
    velocities_root = positions[1:, root_idx, :] - positions[:-1, root_idx, :]

    positions[:, :, 0] -= positions[:, 0, :1]
    positions[:, :, 2] -= positions[:, 0, 2:]

    global_heading = -np.arctan2(rotations[:, root_idx, 0, 2], rotations[:, root_idx, 2, 2])
    global_heading += root_rot_offset / 180 * np.pi

    global_heading_diff = global_heading[1:] - global_heading[:-1]

    global_heading_rot = np.array([geo_util.rot_yaw(x) for x in global_heading])
    # global_heading_rot_inv = global_heading_rot.transpose(0,2,1)

    positions_no_heading = np.matmul(np.repeat(global_heading_rot[:, None, :, :], njoint, axis=1), positions[..., None])

    velocities_no_heading = positions_no_heading[1:] - positions_no_heading[:-1]
    velocities_root_xy_no_heading = np.matmul(global_heading_rot[:-1], velocities_root[:, :, None]).squeeze()[..., [0, 2]]

    rotations[:, 0, ...] = np.matmul(global_heading_rot, rotations[:, 0, ...])

    size_frame = 3 + njoint * 3 + njoint * 3 + njoint * 6
    final_x = np.zeros((nfrm, size_frame))

    final_x[1:, :2] = velocities_root_xy_no_heading
    final_x[1:, 2] = global_heading_diff
    final_x[:, 3:3 + 3 * njoint] = np.reshape(positions_no_heading, (nfrm, -1))
    final_x[1:, 3 + 3 * njoint:3 + 6 * njoint] = np.reshape(velocities_no_heading, (nfrm - 1, -1))
    final_x[:, 3 + 6 * njoint:3 + 12 * njoint] = np.reshape(rotations[..., :, :2, :], (nfrm, -1))

    # --------------- 발 접지 관련 전처리 부분 --------------------
    data = np.loadtxt(foot_path).astype(np.int32)  # shape [T,2]

    if frame_start is not None and frame_end is not None:
        end_idx = total_len - frame_end
        data = data[frame_start:end_idx]

    if is_mirrored:  # ex) basename.endswith('_mirror')
        # L/R 스왑
        data = data[:, [1, 0]]

    contact_L = data[:, 0]  # 왼발 contact 플래그
    contact_R = data[:, 1]  # 오른발 contact 플래그

    l_toe_idx = 4
    r_toe_idx = 8

    # 착지 시점 계산
    onset_idx_L = compute_onset_indices(contact_L)  # [K_L]
    onset_idx_R = compute_onset_indices(contact_R)  # [K_R]

    numpy.set_printoptions(threshold=numpy.inf)

    # 각 프레임에서 등장하는 첫 번째 idx 없으면 -1
    next_onset_L = map_next_onset_index(onset_idx_L, nfrm)  # [T]
    next_onset_R = map_next_onset_index(onset_idx_R, nfrm)  # [T]

    # 결과 버퍼
    left_dx_dz = np.zeros((nfrm, 2), dtype=np.float32)
    right_dx_dz = np.zeros((nfrm, 2), dtype=np.float32)

    for t in range(nfrm):

        global_heading_t = global_heading[t]
        root_t = positions_world[t, root_idx]

        jL = next_onset_L[t]
        if jL != -1 :
            foot_L = positions_world[jL, l_toe_idx]
            left_dx_dz[t] = make_foot_dxz(global_heading_t, foot_L, root_t)
        else:
            foot_L = positions_world[t, l_toe_idx]
            left_dx_dz[t] = make_foot_dxz(global_heading_t, foot_L, root_t)

        jR = next_onset_R[t]
        if jR != -1 :
            foot_R = positions_world[jR, r_toe_idx]
            right_dx_dz[t] = make_foot_dxz(global_heading_t, foot_R, root_t)
        else:
            foot_R = positions_world[t, r_toe_idx]
            right_dx_dz[t] = make_foot_dxz(global_heading_t, foot_R, root_t)

    extra_cartesian = np.concatenate([
        left_dx_dz,
        right_dx_dz
    ], axis=1).astype(np.float32)  # [nfrm, 4]

    #final_x = np.concatenate([final_x, extra_cartesian], axis=1)

    # ---------------------- polar (θ,d) ----------------------
    left_dx = left_dx_dz[:, 0]
    left_dz = left_dx_dz[:, 1]
    right_dx = right_dx_dz[:, 0]
    right_dz = right_dx_dz[:, 1]

    left_d = np.sqrt(left_dx**2 + left_dz**2)[:, None]
    right_d = np.sqrt(right_dx**2 + right_dz**2)[:, None]

    left_theta = np.arctan2(left_dx, left_dz)
    right_theta = np.arctan2(right_dx, right_dz)

    extra_polar = np.concatenate(
        [left_theta[:, None], left_d,
         right_theta[:, None], right_d],
        axis=1
    ).astype(np.float32)  # [nfrm, 4]

    # 최종 feature에 cartesian + polar 둘 다 붙이기
    final_x = np.concatenate([final_x, extra_cartesian, extra_polar], axis=1)

    return final_x, motion


if __name__ == '__main__':
    read_bvh_loco()
