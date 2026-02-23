
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

def extract_sk_lengths(positions, linked_joints):
    #position: NxJx3
    #single frame rigid body restriction
    lengths = np.zeros((len(linked_joints),positions.shape[0]))
    for i,(st,ed) in enumerate(linked_joints):
        length =  np.linalg.norm(positions[:,st] - positions[:,ed], axis=-1)     
        lengths[i] = length
    return np.mean(lengths,axis=-1)

def get_parent_from_link(links):
    max_index = -1
    parents_dict = dict()
    parents = list()
    for pair in links:
        st, ed = pair
        if st>ed:
            st, ed = ed, st
        max_index = ed if ed>max_index else max_index
        parents_dict[ed] = st
    parents_dict[0] = -1
    for i in range(max_index+1):
        parents.append(parents_dict[i])
    return parents

def load_bvh_info(bvh_file_path):
    joint_name = []
    joint_parent = []
    joint_offset = []
    joint_rot_order = []
    joint_chn_num = []

    cnt = 0
    myStack = []
    root_joint_name = None
    frame_time = 0

    skip_end_site = False  # End Site 블록을 건너뛰기 위한 플래그

    with open(bvh_file_path, 'r') as file_obj:
        for line in file_obj:
            lineList = line.strip().split()
            if not lineList:
                continue

            keyword = lineList[0]

            if keyword == "End":
                skip_end_site = True  # End Site 블록 시작
                continue
            if skip_end_site:
                if keyword == "}":
                    skip_end_site = False  # End Site 블록 종료
                continue

            if keyword == "{":
                myStack.append(cnt)
                cnt += 1

            elif keyword == "}":
                myStack.pop()

            elif keyword == "OFFSET":
                joint_offset.append([float(lineList[1]), float(lineList[2]), float(lineList[3])])

            elif keyword == "JOINT":
                joint_name.append(lineList[1])
                joint_parent.append(myStack[-1])

            elif keyword == "ROOT":
                joint_name.append(lineList[1])
                joint_parent.append(-1)
                root_joint_name = lineList[1]

            elif keyword == "CHANNELS":
                channel_num = int(lineList[1])
                joint_chn_num.append(channel_num)

                # ROOT일 경우 3개 위치 + 3개 회전
                if joint_parent[-1] == -1:
                    rot_lst = lineList[5:]
                else:
                    rot_lst = lineList[2:]

                joint_rot_order.append(''.join([axis[0] for axis in rot_lst]))

            elif keyword == "Frame" and lineList[1] == "Time:":
                frame_time = float(lineList[2])

    joint_offset = np.array(joint_offset).reshape(-1, 3)
    return joint_name, joint_parent, joint_offset, joint_rot_order, joint_chn_num, frame_time

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

        elif item == "end" and items[cnt+1].lower() == 'site':
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
                    coords = np.array([x,y,z])
                else:
                    coords = np.array([0,0,0])
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
            if not items[cnt-4].lower() == 'offset' and not end_eff and name[:7] == 'end_eff':
                joint_stack.pop()
            cnt += 1

            if depth == 0:
                break   

        elif item == "hierarchy":
            cnt += 1

        else:
            raise Exception("Unknown Token {}: {} {}".format(cnt, item, items[cnt-5:cnt+5]))

    skeleton.add_joints(joint_list)

    num_jnt = len(joint_list)
    motion = Motion(skeleton)

    # load motion info
    while cnt < n_items:
        item = items[cnt].lower()
        if item == 'motion':
            cnt += 1

        elif item == 'frames:':
            num_frames = int(items[cnt+1])
            motion.set_num_frames(num_frames)
            cnt += 2

        elif item == 'frame' and items[cnt+1].lower() == 'time:':
            fps = round(1.0/float(items[cnt+2]))
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

def export_bvh(file_path, motion, offset_translate=None, offset_rotate=None):
    root_xyzs = motion._positions[:,0,:]
    joint_rotations = motion._rotations
    
    if offset_translate is not None:
        root_xyzs[:,0] += offset_translate[0]
        root_xyzs[:,2] += offset_translate[1]
    if offset_rotate is not None:
        #TODO
        pass
    
    joint_euler_order = 'ZYX'
    joint_eulers = geo_util.rotation_matrix_to_euler(joint_rotations, joint_euler_order) /np.pi*180

    joint_lst = motion._skeleton._joint_lst
    joint_names = [jnt._name for jnt in joint_lst]
    joint_parent = [jnt._parent_idx if jnt._parent_idx  is not None else -1 for jnt in joint_lst]
    joint_offset = motion._skeleton.get_joint_offset()
    target_fps = motion._fps
    output_as_bvh(file_path, root_xyzs, joint_eulers, joint_euler_order, joint_names, joint_parent, joint_offset, target_fps)

def output_as_bvh(file_path, root_xyz, joint_rot_eulers, joint_rot_order, joint_names, joint_parents, joint_offset, target_fps):
    child_lst = [[] for _ in joint_names]
    root_index = 0
    for i,i_p in enumerate(joint_parents):
        if i_p == -1:
            root_index = i
        else:
            child_lst[i_p].append(i)
    #print(child_lst)

    if isinstance(joint_rot_order, str):
        rot_order = [joint_rot_order for _ in range(len(joint_names))]
    elif isinstance(joint_rot_order, list) and len(joint_rot_order) == len(joint_names):
        rot_order = joint_rot_order
    else:
        raise NotImplementedError
    
    if osp.exists(file_path):
        os.remove(file_path)
    out_file = open(file_path,'w+')
    
    out_str = 'HIERARCHY\n'
    out_str+= 'ROOT {}\n'.format(joint_names[root_index])
    out_str+= '{\n'
    out_str+= ' OFFSET {:6f} {:6f} {:6f}\n'.format(joint_offset[root_index][0],joint_offset[root_index][2],joint_offset[root_index][1])
    out_str+= ' CHANNELS 6 Xposition Yposition Zposition {}rotation {}rotation {}rotation\n'.format(rot_order[root_index][0],rot_order[root_index][1],rot_order[root_index][2])
    
    out_file.write(out_str)
    
    def form_str(file, idx_joint, child_joints, depth):
        #print(child_joints,depth)
        if len(child_joints) == 0:
            end_eff_coord = [0,0,0]
            out_str = ' ' * depth + 'End Site\n'
            out_str += ' ' * depth + '{\n'
            out_str += ' ' * (depth+1) +'OFFSET {:6f} {:6f} {:6f}\n'.format(end_eff_coord[0], end_eff_coord[1], end_eff_coord[2])
            out_str += ' ' * depth + '}\n'
            file.write(out_str)
            return
        
        for i in range(len(child_joints)):
            idx_joint = child_joints[i]
            
            out_str = ' ' * depth + 'JOINT {}\n'.format(joint_names[idx_joint])
            out_str += ' ' * depth + '{\n'

            out_str+= ' ' * (depth +1) + "OFFSET {:6f} {:6f} {:6f}\n".format(joint_offset[idx_joint][0],joint_offset[idx_joint][1],joint_offset[idx_joint][2])
            out_str+= ' ' * (depth +1) + "CHANNELS 3 {}rotation {}rotation {}rotation\n".format(rot_order[idx_joint][0],rot_order[idx_joint][1],rot_order[idx_joint][2]) 
            file.write(out_str)
            form_str(file, idx_joint, child_lst[idx_joint], depth + 1)
            
            file.write(' ' * depth + '}\n')

    form_str(out_file, root_index, child_lst[root_index], 1)
    out_file.write('}\n')
    
    frames = joint_rot_eulers.shape[0]
    out_str = 'MOTION\n'
    out_str += 'Frames: {}\n'.format(frames)
    out_str += 'Frame Time: {:6f}\n'.format(1.0/target_fps)
    out_file.write(out_str)

    for i in range(frames):
        out_str = ''
        out_str += '{:6f} {:6f} {:6f}'.format(root_xyz[i][0],root_xyz[i][1],root_xyz[i][2])
        for r in joint_rot_eulers[i]:
            out_str += ' {:6f} {:6f} {:6f}'.format(r[0],r[1],r[2])
        out_str += '\n'
        out_file.write(out_str)
    out_file.close()

def extract_foot_positions(path, unit, target_fps, root_rot_offset=0, frame_start=None, frame_end=None,
                           velocity_threshold=0.01, height_threshold=0.09):
    motion = import_bvh(path, end_eff=False)
    positions = motion._positions * unit_util.unit_conver_scale(unit)  # (F, J, 3)
    foot_idx = [[3, 4], [7, 8]]  # [left_heel, left_toe], [right_heel, right_toe]

    if frame_start is not None and frame_end is not None:
        positions = positions[frame_start:frame_end]

    source_fps = motion._fps
    if source_fps > target_fps:
        sample_ratio = int(source_fps / target_fps)
        positions = positions[::sample_ratio]

    nfrm, njoint, _ = positions.shape
    final_foot = np.zeros((nfrm, 2), dtype=np.int32)  # [F, 2]

    # --- Velocity 계산 (F - 1)
    velocities = positions[1:] - positions[:-1]  # [F-1, J, 3]
    velocity_magnitudes = np.linalg.norm(velocities, axis=2)  # [F-1, J]

    # 마지막 프레임 보정 (마지막 프레임에 대한 velocity가 없음 → 마지막 값을 복제)
    velocity_magnitudes = np.concatenate(
        [velocity_magnitudes, velocity_magnitudes[-1:, :]], axis=0
    )  # shape: [F, J]

    # --- Height 계산
    heights = positions[:, :, 1]  # y-axis (높이)

    for side in range(2):  # 0: left, 1: right
        contact_mask = np.zeros(nfrm, dtype=bool)
        for j_idx in foot_idx[side]:
            vel_ok = velocity_magnitudes[:, j_idx] < velocity_threshold
            height_ok = heights[:, j_idx] < height_threshold
            joint_contact = np.logical_and(vel_ok, height_ok)
            contact_mask = np.logical_or(contact_mask, joint_contact)

        final_foot[:, side] = contact_mask.astype(np.int32)

    return final_foot  # shape: [F, 2]

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

def make_dxz_from_ref(global_heading_t, target_world, ref_world):
    d = target_world - ref_world                      # (3,)
    d_local = geo_util.rot_yaw(global_heading_t) @ d  # heading-free
    return d_local[[0, 2]].astype(np.float32)         # (2,)

def make_progress_from_contact(contact_01: np.ndarray) -> np.ndarray:
    c = (contact_01 > 0).astype(np.int32)
    T = c.shape[0]
    prog = np.zeros((T, 1), dtype=np.float32)  # ✅ contact==1은 0

    t = 0
    while t < T:
        if c[t] == 0:
            s = t
            while t < T and c[t] == 0:
                t += 1
            e = t  # first 1 or T
            L = e - s

            if L == 1:
                prog[s, 0] = 1.0
            else:
                # ✅ 0구간에서 0->1, 마지막 0이 1
                prog[s:e, 0] = np.linspace(0.0, 1.0, num=L, endpoint=True, dtype=np.float32)

        else:
            t += 1

    return prog

def read_bvh_loco(path, foot_path, unit, target_fps, root_rot_offset=0, frame_start=None, frame_end=None):

    motion = import_bvh(path, end_eff=False)
    positions = motion._positions * unit_util.unit_conver_scale(unit)  # (frames, joints, 3)
    rotations = motion._rotations  #
    root_idx = motion._skeleton._root._idx

    total_len = motion._num_frames

    if frame_start is not None and frame_end is not None:
        end_idx = total_len - frame_end
        positions = positions[frame_start:end_idx]
        rotations = rotations[frame_start:end_idx]

    source_fps = motion._fps

    if source_fps > target_fps:
        sample_ratio = int(source_fps / target_fps)
        positions = positions[::sample_ratio]
        rotations = rotations[::sample_ratio]

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

    left_step_dx_dz = np.zeros((nfrm, 2), dtype=np.float32)  # L_next - L_now
    right_step_dx_dz = np.zeros((nfrm, 2), dtype=np.float32)  # R_next - R_now

    #left_progress_contact = np.zeros((nfrm,1), dtype=np.float32)
    #right_progress_contact = np.zeros((nfrm,1), dtype=np.float32)

    #left_progress_contact = make_progress_from_contact(contact_L)  # (T,1)
    #right_progress_contact = make_progress_from_contact(contact_R)  # (T,1)

    left_next = np.zeros((nfrm, 1), dtype=np.float32)
    right_next = np.zeros((nfrm, 1), dtype=np.float32)
    horizon = 150

    prev_jL = None
    prev_jR = None

    for t in range(nfrm):

        global_heading_t = global_heading[t]

        root_t = positions_world[t, root_idx]

        Lfoot_t = positions_world[t, l_toe_idx]
        Rfoot_t = positions_world[t, r_toe_idx]

        jL = next_onset_L[t]
        if jL != -1: #and (jL - t) <= horizon:
            foot_L = positions_world[jL, l_toe_idx]
        else:
            foot_L = positions_world[t, l_toe_idx]
            #left_progress_contact[t, 0] = 0.0

        jR = next_onset_R[t]
        if jR != -1: #and (jR - t) <= horizon:
            foot_R = positions_world[jR, r_toe_idx]

        else:
            foot_R = positions_world[t, r_toe_idx]
            #right_progress_contact[t, 0] = 0.0

        left_dx_dz[t] = make_foot_dxz(global_heading_t, foot_L, root_t)
        right_dx_dz[t] = make_foot_dxz(global_heading_t, foot_R, root_t)

        left_step_dx_dz[t] = make_dxz_from_ref(global_heading_t, foot_L, Lfoot_t)
        right_step_dx_dz[t] = make_dxz_from_ref(global_heading_t, foot_R, Rfoot_t)

        if (jL == -1) and (jR == -1):
            # 다음 발자국 없음: 정책 선택
            left_next[t, 0] = 0.0
            right_next[t, 0] = 0.0

        elif (jL != -1) and (jR == -1):
            left_next[t, 0] = 1.0
            right_next[t, 0] = 0.0

        elif (jL == -1) and (jR != -1):
            left_next[t, 0] = 0.0
            right_next[t, 0] = 1.0

        else:
            # 둘 다 존재
            if jL < jR:
                left_next[t, 0] = 1.0
                right_next[t, 0] = 0.0
            elif jR < jL:
                left_next[t, 0] = 0.0
                right_next[t, 0] = 1.0
            else:
                # jL == jR : 점프/동시착지
                left_next[t, 0] = 1.0
                right_next[t, 0] = 1.0

    extra_cartesian = np.concatenate([
        left_dx_dz, right_dx_dz,  # root 기준 (기존) 4
        left_step_dx_dz, right_step_dx_dz,  # foot 기준 (추가) 4
        left_next, right_next
        #left_progress_contact, right_progress_contact,
    ], axis=1).astype(np.float32)

    final_x = np.concatenate([final_x, extra_cartesian], axis=1)

    return final_x, motion

def read_bvh_hetero(path, unit, target_fps,  root_rot_offset=0, frame_start=None, frame_end=None):
    motion = import_bvh(path, end_eff=False)
    positions = motion._positions * unit_util.unit_conver_scale(unit) # (frames, joints, 3)
    rotations = motion._rotations #
    root_idx = motion._skeleton._root._idx
    
    if frame_start is not None and frame_end is not None:
        positions = positions[frame_start:frame_end]
        rotations = rotations[frame_start:frame_end]

    source_fps = motion._fps 
    if source_fps > target_fps:
        sample_ratio = int(source_fps/target_fps)
        positions = positions[::sample_ratio]
        rotations = rotations[::sample_ratio]
    
    nfrm, njoint, _ = positions.shape
    
    ori = copy.deepcopy(positions[0,root_idx])
    
    y_min = np.min(positions[0,:,1])
    ori[1] = y_min

    positions = positions - ori
    velocities_root = positions[1:,root_idx,:] - positions[:-1,root_idx,:]
    
    positions[:,:,0] -= positions[:,0,:1]
    positions[:,:,2] -= positions[:,0,2:]

    global_heading = - np.arctan2(rotations[:,root_idx,0,2], rotations[:, root_idx, 2,2]) 
    global_heading += root_rot_offset/180*np.pi
    global_heading_diff = global_heading[1:] - global_heading[:-1] #% (2*np.pi)
    global_heading_diff_rot = np.array([geo_util.rot_yaw(x) for x in global_heading_diff])
    global_heading_rot = np.array([geo_util.rot_yaw(x) for x in global_heading])
    #global_heading_rot_inv = global_heading_rot.transpose(0,2,1)

    positions_no_heading = np.matmul(np.repeat(global_heading_rot[:, None,:, :], njoint, axis=1), positions[...,None])
    
    velocities_no_heading = positions_no_heading[1:] - positions_no_heading[:-1] #np.matmul(np.repeat(global_heading_rot[:-1, None,:, :], njoint, axis=1), (positions[1:] - positions[:-1])[...,None])
    velocities_root_xy_no_heading = np.matmul(global_heading_rot[:-1], velocities_root[:, :, None]).squeeze()[...,[0,2]]
 
    rotations[:,0,...] = np.matmul(global_heading_rot, rotations[:,0,...]) 

    size_frame = 8+njoint*3+njoint*3+njoint*6
    final_x = np.zeros((nfrm, size_frame))

    final_x[1:,2:8] = geo_util.rotation_matrix_to_6d(global_heading_diff_rot)
    final_x[1:,:2] = velocities_root_xy_no_heading 
    final_x[:,8:8+3*njoint] = np.reshape(positions_no_heading, (nfrm,-1))
    final_x[1:,8+3*njoint:8+6*njoint] = np.reshape(velocities_no_heading, (nfrm-1,-1))
    final_x[:,8+6*njoint:8+12*njoint] = np.reshape(rotations[..., :, :2, :], (nfrm,-1))
    return final_x, motion

if __name__ == '__main__':
    pass
