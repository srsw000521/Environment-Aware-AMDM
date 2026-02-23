import policy.envs.base_env as base_env
from render.realtime.mocap_renderer import PBLMocapViewer
import torch
import numpy as np
import tkinter as tk
import gymnasium as gym
from make_perlin_noise_map import generate_A_masked_binary01  # 네가 추가한 함수 이름에 맞춰서
from image_nav_map import PerlinNavMap


from multiprocessing import Process
from filelock import FileLock
# user_input_lockfile = "miscs/interact_temp/user_text"
import os.path as osp


class TargetEnv(base_env.EnvBase):
    NAME = 'Target'

    def __init__(self, config, model, dataset, device):
        super().__init__(config, model, dataset, device)
        self.device = device
        self.config = config
        self.model = model
        self.dataset = dataset

        self.links = self.dataset.links
        self.valid_idx = self.dataset.valid_idx

        self.index_of_target = 0
        self.arena_length = (-7.0, 7.0)
        self.arena_width = (-7.0, 7.0)

        self.num_future_predictions = 1
        self.num_condition_frames = 1

        self.map_N = 16  # 16x16
        self.map_channels = 2  # world map + yaw-rot map
        map_dim = self.map_channels * (self.map_N * self.map_N)

        target_dim = 2
        self.target = torch.zeros((self.num_parallel, target_dim)).to(self.device)

        # 기존: self.observation_dim = (self.frame_dim * self.num_condition_frames) + target_dim
        self.observation_dim = (self.frame_dim * self.num_condition_frames) + target_dim + map_dim

        high = np.inf * np.ones([self.observation_dim])
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        high = np.inf * np.ones([self.action_dim])
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.target_arr = torch.zeros((self.num_parallel, self.max_timestep, 3))

        A = generate_A_masked_binary01(size=500, seedA=0, seedB=1, tauA=0.71,
                                       tauB=0.71)  # (500,500) {0,1}

        nav = PerlinNavMap(
            img_path=A,
            world_size_m=50.0,
            boundary_block_m=2.0,
            input_is_binary01=True,  # 중요!
        )

        # goal 샘플링을 위해 reachability map 생성(점프/보폭으로 넘길 폭 반영)
        nav.build_reachability_map(jump_bridge_m=0.6)
        nav.to_torch(device=str(self.device), use_reach_map=True)

        self.nav=nav

        self.linear_potential = torch.zeros((self.num_parallel, 1), device=self.device)
        self.angular_potential = torch.zeros((self.num_parallel, 1), device=self.device)
        self.valid_idx_t = torch.as_tensor(self.dataset.valid_idx, device=self.device, dtype=torch.long)

        self.use_sg_pool = True
        self.sg_pool_K = 64
        self.sg_pool_roi_size_m = 10.0

        # 각 env가 현재 어떤 pool 항목을 쓰는지 기록 (B,)
        self.sg_pool_ids = torch.zeros((self.num_parallel,), device=self.device, dtype=torch.long)

        if self.use_sg_pool:
            self._build_start_goal_pool(K=self.sg_pool_K, roi_size_m=self.sg_pool_roi_size_m)

    @torch.no_grad()
    def _build_start_goal_pool(self, K: int, roi_size_m: float = 10.0):
        """
        pool: (K,2) start, (K,2) goal  (모두 torch, device=self.device)
        """
        device = self.device

        # pool용 더미 텐서
        start_x = torch.zeros((K,), device=device, dtype=torch.float32)
        start_z = torch.zeros((K,), device=device, dtype=torch.float32)

        # tidx는 update할 indices 역할. 여기선 0..K-1
        tidx = torch.arange(K, device=device, dtype=torch.long)

        # start 샘플
        start_x, start_z = self.nav.sample_start_torch_update_by_indices(
            start_x, start_z, tidx,
            use_reach_map=True,
            require_walkable_on_original=True,
        )

        goal_x = torch.zeros((K,), device=device, dtype=torch.float32)
        goal_z = torch.zeros((K,), device=device, dtype=torch.float32)

        # goal 샘플 (start 기준 ROI)
        goal_x, goal_z = self.nav.sample_goal_near_start_torch_update_by_indices(
            goal_x, goal_z,
            start_x, start_z,
            tidx,
            roi_size_m=roi_size_m,
            use_reach_map=True,
            require_walkable_on_original=True,
        )

        self.sg_start = torch.stack([start_x, start_z], dim=1)  # (K,2)
        self.sg_goal = torch.stack([goal_x, goal_z], dim=1)  # (K,2)

        # 디버깅용 저장
        np.savez(
            osp.join(self.int_output_dir, "start_goal_pool.npz"),
            start=self.sg_start.detach().cpu().numpy(),
            goal=self.sg_goal.detach().cpu().numpy(),
        )

    @torch.no_grad()
    def _assign_pool_to_envs(self, tidx: torch.Tensor):
        """
        tidx: (M,) env indices
        env마다 pool id를 새로 뽑아서 root_xz/target에 반영
        """
        device = self.device
        K = self.sg_start.shape[0]

        # env별 pool id 샘플
        pool_ids = torch.randint(0, K, (tidx.numel(),), device=device, dtype=torch.long)
        self.sg_pool_ids[tidx] = pool_ids

        # start/goal 적용
        self.root_xz[tidx] = self.sg_start[pool_ids]
        self.target[tidx] = self.sg_goal[pool_ids]

        # potential 갱신(부분 업데이트)
        self.calc_potential(indices=tidx)

    @torch.no_grad()
    def calc_potential(self, indices: torch.Tensor = None):
        """
        indices:
          - None이면 전체 업데이트
          - LongTensor이면 해당 batch만 업데이트
        """
        if indices is None:
            target_delta, target_angle = self.get_target_delta_and_angle()  # (B,2), (B,1)
            self.linear_potential.copy_(-target_delta.norm(dim=1, keepdim=True))
            self.angular_potential.copy_(target_angle.cos())
            return

        tidx = indices.to(device=self.root_xz.device, dtype=torch.long).view(-1)

        # 부분만 계산
        target_delta = self.target[tidx] - self.root_xz[tidx]  # (M,2)
        target_angle = torch.atan2(target_delta[:, 1], target_delta[:, 0]).unsqueeze(1) + self.root_facing[
            tidx]  # (M,1)

        self.linear_potential[tidx] = -target_delta.norm(dim=1).unsqueeze(1)
        self.angular_potential[tidx] = target_angle.cos()

    def get_target_delta_and_angle(self):
        target_delta = self.target - self.root_xz
        target_angle = (
                torch.atan2(target_delta[:, 1], target_delta[:, 0]).unsqueeze(1)
                + self.root_facing
        )
        return target_delta, target_angle

    '''def get_observation_components(self):
        target_delta, _ = self.get_target_delta_and_angle()
        mat = self.get_rotation_matrix(-self.root_facing)
        delta = (mat * target_delta.unsqueeze(1)).sum(dim=2)  # (B,2)

        condition = self.get_cond_frame()  # (B, frame_dim)

        # ---- (추가) 맵 2개: world / yaw-rotated
        B = self.num_parallel
        N = self.map_N
        map_world = torch.ones((B, N, N), device=self.device, dtype=torch.float32)
        map_yaw = torch.ones((B, N, N), device=self.device, dtype=torch.float32)

        # flatten: (B, N*N)
        map_world_flat = map_world.view(B, -1)
        map_yaw_flat = map_yaw.view(B, -1)

        return condition, delta, map_world_flat, map_yaw_flat
'''
    def get_observation_components(self):
        target_delta, _ = self.get_target_delta_and_angle()
        mat = self.get_rotation_matrix(-self.root_facing)
        delta = (mat * target_delta.unsqueeze(1)).sum(dim=2)  # (B,2)

        condition = self.get_cond_frame()  # (B, frame_dim*num_condition_frames) 혹은 (B, frame_dim)

        # -------- 여기부터 "진짜 맵" ----------
        # root_xz: (B,2), root_facing: (B,1) 가정
        x = self.root_xz[:, 0]
        z = self.root_xz[:, 1]
        yaw = self.root_facing[:, 0]

        # (B,N,N), (B,N,N)
        map_world, map_yaw = self.nav.make_local_grid_two_views_fast(
            x, z, yaw,
            use_original_walkable=True,  # 지금은 "전부 걸을 수 있는 맵" 가정이면 walkable_t가 전부 1이면 됨
        )

        B = x.shape[0]
        map_world_flat = map_world.reshape(B, -1)  # (B, N*N)
        map_yaw_flat = map_yaw.reshape(B, -1)  # (B, N*N)

        return condition, delta, map_world_flat, map_yaw_flat

    def reset(self, indices: torch.Tensor = None):
        device = self.root_xz.device

        if indices is None:
            # ---- 전체 reset
            self.root_facing.fill_(0)
            self.root_xz.fill_(0)
            self.reward.fill_(0)
            self.timestep.fill_(0)
            self.substep.fill_(0)
            self.done.fill_(False)
            self.early_stop.fill_(False)

            # (선택) foot states도 초기화
            # self.lf.fill_(0); self.rf.fill_(0)

            self.reset_initial_frames(frame_index=None)

            # nav 기반 start/goal
            #self.reset_start_and_goal_from_nav(indices=None, roi_size_m=10.0)]
            if self.use_sg_pool:
                tidx = torch.arange(self.num_parallel, device=self.device, dtype=torch.long)
                self._assign_pool_to_envs(tidx)
            else:
                self.reset_start_and_goal_from_nav(indices=None, roi_size_m=10.0)
            self._last_reset_indices = torch.arange(self.num_parallel, device=self.device, dtype=torch.long)

        else:
            # ---- 부분 reset
            tidx = indices.to(device=device, dtype=torch.long).view(-1)

            self.root_facing.index_fill_(0, tidx, 0)
            self.root_xz.index_fill_(0, tidx, 0)
            self.reward.index_fill_(0, tidx, 0)
            self.done.index_fill_(0, tidx, False)
            self.early_stop.index_fill_(0, tidx, False)

            # ✅ 부분 reset은 index_fill로만
            self.timestep.index_fill_(0, tidx, 0)
            self.substep.index_fill_(0, tidx, 0)

            self.reset_initial_frames(frame_index=tidx)

            # nav 기반 start/goal (indices만)
            #self.reset_start_and_goal_from_nav(indices=tidx, roi_size_m=10.0)
            if self.use_sg_pool:
                self._assign_pool_to_envs(tidx)
            else:
                self.reset_start_and_goal_from_nav(indices=tidx, roi_size_m=10.0)
            self._last_reset_indices = tidx

        obs_components = self.get_observation_components()
        return torch.cat(obs_components, dim=1)

    def reset_index(self, indices: torch.Tensor = None):
        return self.reset(indices=indices)

    def output_motion(self):
        # flag_pos_hist = np.array(self.flag_pos.detach().cpu())
        f = open('./flag.txt', 'w')
        for st, ed in self.flag_sted:
            f.write("{},{}\n".format(st, ed))
        f.close()
        # np.savez(file='../../bvh_demo/out_info.npz',flag_pos=flag_pos_hist,sted=self.flag_sted)
        return super().output_motion()

    def calc_action_penalty_reward(self):
        prob_energy = self.action[..., self.action_dim_per_step:].abs().mean(-1, keepdim=True)
        return -0.02 * prob_energy

    def reset_initial_frames(self, frame_index=None):
        num_frame_used = len(self.valid_idx)

        if frame_index is None:
            num_init = self.num_parallel
            fidx = None
        else:
            # frame_index는 torch indices (M,) or (M,1)
            fidx = frame_index.view(-1).long().to(self.history.device)
            num_init = fidx.numel()

        # (num_init,) 로 뽑기
        start_index = torch.randint(
            0, num_frame_used - 1,
            (num_init,),
            device=self.history.device
        )

        # valid_idx가 torch 텐서라고 가정 (numpy면 __init__에서 torch로 캐싱 권장)
        start_index = self.valid_idx_t[start_index].view(-1)

        # ✅ numpy indexing이면 CPU numpy로 바꿔서 가져온 뒤 torch로
        start_index_cpu = start_index.detach().cpu().numpy()

        data = torch.as_tensor(
            self.dataset.motion_flattened[start_index_cpu],
            device=self.history.device,
            dtype=self.history.dtype,  # ✅ self.history와 dtype 맞춤 (float32)
        )
        data = data.unsqueeze(1)  # ✅ (B,1,frame_dim)  num_condition_frames=1일 때
        if fidx is None:
            self.init_frame.copy_(data.squeeze(1))  # init_frame: (B, frame_dim)
            self.history[:, :self.num_condition_frames].copy_(data)  # history: (B,1,frame_dim)
        else:
            self.init_frame[fidx] = data.squeeze(1)
            self.history[fidx, :self.num_condition_frames] = data

    @torch.no_grad()
    def reset_start_and_goal_from_nav(self, indices: torch.Tensor = None, roi_size_m: float = 10.0):
        """
        indices에 해당하는 env들만:
          - root_xz를 nav에서 샘플한 start로 덮어쓰기
          - target을 start 기준 reachable goal로 덮어쓰기
        indices:
          - None이면 전체 (0..B-1)
          - torch.LongTensor shape (M,) or (M,1)
        """
        device = self.root_xz.device
        B = self.root_xz.shape[0]

        if indices is None:
            tidx = torch.arange(B, device=device, dtype=torch.long)
        else:
            tidx = indices.to(device=device, dtype=torch.long).view(-1)

        # ----- start 업데이트 (indices만 덮어쓰기)
        start_x = self.root_xz[:, 0]
        start_z = self.root_xz[:, 1]

        start_x, start_z = self.nav.sample_start_torch_update_by_indices(
            start_x, start_z, tidx,
            use_reach_map=True,
            require_walkable_on_original=True,
        )

        self.root_xz[:, 0] = start_x
        self.root_xz[:, 1] = start_z

        # ----- goal 업데이트 (indices만 덮어쓰기)
        goal_x = self.target[:, 0]
        goal_z = self.target[:, 1]

        goal_x, goal_z = self.nav.sample_goal_near_start_torch_update_by_indices(
            goal_x, goal_z,
            start_x, start_z,
            tidx,
            roi_size_m=roi_size_m,
            use_reach_map=True,
            require_walkable_on_original=True,
        )

        self.target[:, 0] = goal_x
        self.target[:, 1] = goal_z

        if self.is_rendered:
            self.target_arr[..., self.index_of_target, :2] = self.target[:, :2]  # .detach().cpu().numpy()
            self.target_arr[..., self.index_of_target, 2] = self.timestep
            self.index_of_target += 1

            np.save(osp.join(self.int_output_dir, 'out_target'), self.target_arr)
            self.viewer.update_target_markers(self.target)

        # target 바뀌었으니 potential 갱신
        # (indices 지원 여부에 따라 분기)
        self.calc_potential(indices=tidx)

    @torch.no_grad()
    def calc_progress_reward(self, reset_indices: torch.Tensor = None):
        old_linear = self.linear_potential.clone()

        # ✅ 항상 전체 업데이트
        self.calc_potential()

        linear_progress = self.linear_potential - old_linear

        if reset_indices is not None:
            ridx = reset_indices.view(-1)
            linear_progress[ridx] = 0.0

        return linear_progress

    def integrate_root_translation(self, pose):

        pose_denorm = self.dataset.denorm_data(pose, device=pose.device)
        dr = self.dataset.get_heading_dr(pose_denorm)[..., None]
        root_xz_vel = self.dataset.get_root_linear_planar_vel(pose_denorm)
        root_rotmat_up = self.get_rotation_matrix(self.root_facing)
        #displacement = (root_rotmat_up * root_xz_vel.unsqueeze(1)).sum(dim=2)
        displacement = torch.matmul(root_rotmat_up, root_xz_vel.unsqueeze(-1)).squeeze(-1)

        lfoot_xz, rfoot_xz = self.dataset.get_foot_xz(pose_denorm)

        lf_world = torch.matmul(root_rotmat_up, lfoot_xz.unsqueeze(-1)).squeeze(-1)  # (..., 2)
        rf_world = torch.matmul(root_rotmat_up, rfoot_xz.unsqueeze(-1)).squeeze(-1)

        lf_world = lf_world + self.root_xz
        rf_world = rf_world + self.root_xz

        self.lf = lf_world
        self.rf = rf_world

        self.root_facing.add_(dr).remainder_(2 * np.pi)
        self.root_xz.add_(displacement)

        self.history = self.history.roll(1, dims=1)
        self.history[:, :self.num_condition_frames].copy_(pose.view(pose.shape[0], -1, pose.shape[-1]))

    def calc_env_state(self, next_frame):
        self.next_frame = next_frame
        is_external_step = (self.substep == 0)

        # substep이 마지막일 때만 timestep 증가 (per-env)
        inc = (self.substep == (self.frame_skip - 1)).to(self.timestep.dtype)  # (B,1) 0/1
        self.timestep += inc

        # substep 업데이트
        self.substep = (self.substep + 1) % self.frame_skip

        self.integrate_root_translation(next_frame)
        reset_idx = getattr(self, "_last_reset_indices", None)
        progress = self.calc_progress_reward(reset_indices=reset_idx)
        self._last_reset_indices = None

        target_dist = -self.linear_potential
        target_is_close = target_dist < 0.8
        dist_reward = 2 * torch.exp(0.5 * self.linear_potential)

        self.reward.copy_(dist_reward)
        self.reward[target_is_close].copy_(5)

        # ---------------------------------------------------------
        # Foot penalty (A 단계): lf/rf가 non-walkable이면 penalty만 부여
        # (done은 time-limit만 유지)
        # ---------------------------------------------------------
        # lf/rf: (B,2) world meters
        # nav query 결과: (B,) in [0,1]
        lf_occ = self.nav.query_walkable_batch(
            self.lf[:, 0], self.lf[:, 1],
            use_original_walkable=True
        ).view(-1, 1)

        rf_occ = self.nav.query_walkable_batch(
            self.rf[:, 0], self.rf[:, 1],
            use_original_walkable=True
        ).view(-1, 1)

        # sub_div 평균이 아니라 point query라 보통 0/1이지만,
        # 안전하게 threshold는 0.5로 둠
        lf_bad = (lf_occ <= 0.5)
        rf_bad = (rf_occ <= 0.5)

        # penalty weight (처음엔 작게)
        w_foot = 2.0
        foot_penalty = w_foot * (lf_bad.float() + rf_bad.float())  # (B,1)

        self.reward.sub_(foot_penalty)

        if target_is_close.any() and self.is_rendered:
            reset_indices = self.parallel_ind_buf.masked_select(
                target_is_close.squeeze(1)
            )

            # self.end_steps += torch.sum(self.steps_parallel[reset_indices.cpu().detach()])

            self.reset_target(indices=reset_indices)
            self._last_reset_indices = reset_indices
            # self.steps_parallel[reset_indices.cpu().detach()] *= 0

        obs_components = self.get_observation_components()
        self.done.copy_(self.timestep >= self.max_timestep)

        # Everytime this function is called, should call render
        # otherwise the fps will be wrong
        self.render()

        return (
            torch.cat(obs_components, dim=1),
            self.reward,
            self.done,
            {"reset": self.timestep >= self.max_timestep,

             },
        )

    def reset_target(self, indices=None, location=None):
        if location is not None:
            self.target[:, 0] = location[0]
            self.target[:, 1] = location[1]
            self.calc_potential()
            return

        # nav 기반 goal만 재샘플 (start는 유지)
        if indices is None:
            tidx = torch.arange(self.num_parallel, device=self.device, dtype=torch.long)
        else:
            tidx = indices.to(device=self.device, dtype=torch.long).view(-1)

        goal_x = self.target[:, 0]
        goal_z = self.target[:, 1]
        start_x = self.root_xz[:, 0]
        start_z = self.root_xz[:, 1]

        goal_x, goal_z = self.nav.sample_goal_near_start_torch_update_by_indices(
            goal_x, goal_z, start_x, start_z, tidx,
            roi_size_m=10.0, use_reach_map=True, require_walkable_on_original=True
        )
        self.target[:, 0] = goal_x
        self.target[:, 1] = goal_z

        if self.is_rendered:
            self.target_arr[..., self.index_of_target, :2] = self.target[:, :2]  # .detach().cpu().numpy()
            self.target_arr[..., self.index_of_target, 2] = self.timestep
            self.index_of_target += 1

            np.save(osp.join(self.int_output_dir, 'out_target'), self.target_arr)
            self.viewer.update_target_markers(self.target)


        self.calc_potential(indices=tidx)


    def step(self, action):
        next_frame = self.get_next_frame(action)
        obs, reward, done, info = self.calc_env_state(next_frame)
        return (obs, reward, done, info)

    def dump_additional_render_data(self):
        return {"extra.csv": {"header": "Target.X, Target.Z", "data": self.target[0]}}

        # if self.is_rendered and self.timestep % 10 == 0:
        #     self.viewer.duplicate_character()

    def get_cond_frame(self):
        condition = self.history[:, :self.num_condition_frames].view(-1, self.frame_dim)
        return condition

    def get_next_frame(self, action):
        self.action = action
        condition = self.get_cond_frame()
        extra_info = self.extra_info

        with torch.no_grad():
            output = self.model.rl_step(condition, action, extra_info)

        # if self.is_rendered:
        #    self.record_motion_seq[:,self.record_timestep,:]= output.cpu().detach().numpy()
        #    self.record_timestep += 1
        #    if self.record_timestep % 90 == 0 and self.record_timestep != 0:
        #        self.save_motion()

        return output

    def render(self, mode="human"):
        frame = self.dataset.denorm_data(self.history[:, 0], device=self.device).detach().cpu().numpy()
        if self.is_rendered:
            self.viewer.render(
                torch.tensor(self.dataset.x_to_jnts(frame, mode='angle'), device=self.device,
                             dtype=self.root_facing.dtype),  # 0 is the newest
                self.root_facing,
                self.root_xz,
                0.0,  # No time in this env
                self.action,
            )