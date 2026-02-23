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

        #A = generate_A_masked_binary01(size=500, seedA=0, seedB=1, tauA=0.71,tauB=0.71)  # (500,500) {0,1}

        nav = PerlinNavMap(
            img_path="/home/sura/AMDM/perlin6_500_500_raw_tauA0.80_tauB0.89_g1.7_A_masked.png",
            world_size_m=50.0,
            boundary_block_m=2.0,
            white_threshold=200,  # ★ 너는 >200을 walkable로 썼음
            input_is_binary01=False,  # ★ PNG grayscale threshold 사용
        )

        # goal 샘플링을 위해 reachability map 생성(점프/보폭으로 넘길 폭 반영)
        nav.build_reachability_map(jump_bridge_m=0.6)
        nav.to_torch(device=str(self.device), use_reach_map=True)

        self.nav=nav

        self.linear_potential = torch.zeros((self.num_parallel, 1), device=self.device)
        self.angular_potential = torch.zeros((self.num_parallel, 1), device=self.device)
        self.valid_idx_t = torch.as_tensor(self.dataset.valid_idx, device=self.device, dtype=torch.long)

        
        # ---------------- Start/Goal Pool (from NPZ) ----------------
        # npz should contain start_world/goal_world (K,2) in meters (x,z)
        self.use_sg_pool = True
        self.sg_pool_npz_path = self.config.get("sg_pool_npz_path", "start_goal_pairs_map2_60.npz")
        self.sg_pool_unique_starts = True

        # round-robin cursor (python int) + per-env pair id / fail streak / last success
        self.pair_cursor = 0
        #self.max_fail_streak = int(self.config.get("sg_pool_max_fail_streak", 100))  # stuck 방지
        self.pair_id = torch.zeros((self.num_parallel,), device=self.device, dtype=torch.long)
        self.fail_streak = torch.zeros((self.num_parallel,), device=self.device, dtype=torch.long)
        self._last_episode_success = torch.zeros((self.num_parallel, 1), device=self.device, dtype=torch.bool)
        self.w_future_foot = float(self.config.get("w_future_foot", 0.5))

        if self.use_sg_pool:
            self._load_start_goal_pool_from_npz(self.sg_pool_npz_path)

            #self.target = self.sg_pool_goals[:self.num_parallel].clone()
            self.pair_id[:] = torch.arange(self.num_parallel, device=self.device) % self.sg_pool_K

            '''occ = self.nav.query_walkable_batch(self.target[:, 0], self.target[:, 1], use_original_walkable=True)
            print("[dbg] target walkable ratio:", (occ > 0.5).float().mean().item())
            # 몇 개 샘플 확인
            bad = (occ <= 0.5).nonzero(as_tuple=False).view(-1)
            if bad.numel() > 0:
                i = bad[0].item()
                print("[dbg] example bad target:", self.target[i].detach().cpu().numpy(), "pair_id",
                      int(self.pair_id[i]))'''

    @torch.no_grad()
    def _load_start_goal_pool_from_npz(self, npz_path: str):
        """
        Loads start/goal pool from npz.
        Required keys (preferred):
          - start_world: (K,2)  float
          - goal_world:  (K,2)  float
        Also accepts starts_world/goals_world.
        Values are interpreted as (x,z) meters in the same world frame as self.root_xz / self.target.
        """
        import os
        path = npz_path
        if not os.path.isabs(path):
            # try relative to cwd
            if os.path.exists(path):
                pass
            else:
                # try relative to int_output_dir if available
                try:
                    cand = os.path.join(self.int_output_dir, path)
                    if os.path.exists(cand):
                        path = cand
                except Exception:
                    pass

        if not os.path.exists(path):
            raise FileNotFoundError(f"Start/goal pool npz not found: {path}")

        data = np.load(path, allow_pickle=True)

        if "start_world" in data and "goal_world" in data:
            starts = data["start_world"]
            goals = data["goal_world"]
        elif "starts_world" in data and "goals_world" in data:
            starts = data["starts_world"]
            goals = data["goals_world"]
        else:
            raise KeyError("npz must contain (start_world, goal_world) or (starts_world, goals_world)")

        if starts.shape != goals.shape or starts.ndim != 2 or starts.shape[1] != 2:
            raise ValueError(f"Invalid pool shapes: starts={starts.shape}, goals={goals.shape}")

        self.sg_pool_starts = torch.as_tensor(starts, device=self.device, dtype=torch.float32)  # (K,2)
        self.sg_pool_goals = torch.as_tensor(goals, device=self.device, dtype=torch.float32)    # (K,2)
        self.sg_pool_K = int(self.sg_pool_starts.shape[0])

        # optional: start/goal pixel coords stored for debugging
        self.sg_pool_starts_px = None
        self.sg_pool_goals_px = None
        if "start_px" in data and "goal_px" in data:
            self.sg_pool_starts_px = data["start_px"]
            self.sg_pool_goals_px = data["goal_px"]
        elif "starts_px" in data and "goals_px" in data:
            self.sg_pool_starts_px = data["starts_px"]
            self.sg_pool_goals_px = data["goals_px"]

        print(f"[TargetEnv] Loaded start/goal pool: K={self.sg_pool_K} from {path}")

    def _next_pair_ids(self, n: int) -> torch.Tensor:
        """Round-robin pair id allocation."""
        K = int(self.sg_pool_K)
        start = int(self.pair_cursor) % K
        ids = (torch.arange(n, device=self.device, dtype=torch.long) + start) % K
        self.pair_cursor = (start + n) % K
        return ids

    @torch.no_grad()
    def _advance_pairs_for_envs(self, indices: torch.Tensor):
        """Assign NEW pair ids (round-robin) to env indices, then apply start/goal."""
        tidx = indices.to(device=self.device, dtype=torch.long).view(-1)
        new_ids = self._next_pair_ids(int(tidx.numel()))
        self.pair_id[tidx] = new_ids
        self.fail_streak[tidx] = 0
        self._last_episode_success[tidx] = False
        self._apply_pair_to_envs(tidx)

    @torch.no_grad()
    def _get_current_feet_world_fk(self):
        frame = self.dataset.denorm_data(self.history[:, 0], device=self.device).detach().cpu().numpy()
        jnts = self.dataset.x_to_jnts(frame, mode="angle")  # (B,J,3)

        l_toe = 4
        r_toe = 8

        # local positions
        lf_local = torch.as_tensor(jnts[:, l_toe, :], device=self.device, dtype=torch.float32)  # (B,3)
        rf_local = torch.as_tensor(jnts[:, r_toe, :], device=self.device, dtype=torch.float32)  # (B,3)

        # planar rotate xz only
        R = self.get_rotation_matrix(self.root_facing)  # (B,2,2)

        lf_local_xz = lf_local[:, [0, 2]]  # (B,2)  <-- x,z
        rf_local_xz = rf_local[:, [0, 2]]

        lf_world_xz = torch.matmul(R, lf_local_xz.unsqueeze(-1)).squeeze(-1) + self.root_xz  # (B,2)
        rf_world_xz = torch.matmul(R, rf_local_xz.unsqueeze(-1)).squeeze(-1) + self.root_xz  # (B,2)

        # height: local y + root_y? (대부분 root_y=0이면 local y 그대로 쓰면 됨)
        # 혹시 root vertical이 있다면 더해줘야 하는데, 너 env는 planar만 쓰는 듯.
        lf_world_h = lf_local[:, 1:2]  # (B,1)
        rf_world_h = rf_local[:, 1:2]  # (B,1)

        return lf_world_xz, rf_world_xz, lf_world_h, rf_world_h

    @torch.no_grad()
    def _estimate_contact_from_fk_height(
            self,
            lf_world_xz: torch.Tensor, rf_world_xz: torch.Tensor,
            lf_h: torch.Tensor, rf_h: torch.Tensor
    ):
        """
        Contact if foot height is near ground (with hysteresis).
        Optional: require low horizontal speed only when height is ambiguous.
        Returns (B,1) bool, (B,1) bool
        """
        B = lf_world_xz.shape[0]
        device = lf_world_xz.device

        # init buffers
        if not hasattr(self, "_prev_lf_fk_world_xz"):
            self._prev_lf_fk_world_xz = lf_world_xz.clone()
            self._prev_rf_fk_world_xz = rf_world_xz.clone()
            self._lcontact_state = torch.zeros((B, 1), device=device, dtype=torch.bool)
            self._rcontact_state = torch.zeros((B, 1), device=device, dtype=torch.bool)
            return self._lcontact_state.clone(), self._rcontact_state.clone()

        # horizontal speed (optional)
        lf_vel = torch.norm(lf_world_xz - self._prev_lf_fk_world_xz, dim=1, keepdim=True)
        rf_vel = torch.norm(rf_world_xz - self._prev_rf_fk_world_xz, dim=1, keepdim=True)
        self._prev_lf_fk_world_xz = lf_world_xz.clone()
        self._prev_rf_fk_world_xz = rf_world_xz.clone()

        # --- height hysteresis thresholds ---
        # on: 이 높이보다 낮으면 "붙는다"
        # off: 이 높이보다 높으면 "떼어진다"
        h_on = float(self.config.get("contact_h_on", 0.06))  # 6cm
        h_off = float(self.config.get("contact_h_off", 0.10))  # 10cm
        if h_on >= h_off:
            h_off = h_on * 1.5

        # optional speed gate when height is in-between (ambiguous band)
        use_speed_gate = bool(self.config.get("contact_use_speed_gate", True))
        v_gate = float(self.config.get("contact_v_gate", 0.08))  # 애매할 때만 속도 제한

        def update_state(state, h, v):
            on = h < h_on
            off = h > h_off

            if use_speed_gate:
                # ambiguous zone: h_on~h_off 사이에서는 속도도 같이 보자
                amb = (~on) & (~off)
                on = on | (amb & (v < v_gate))

            state = torch.where(on, torch.ones_like(state), state)
            state = torch.where(off, torch.zeros_like(state), state)
            return state

        self._lcontact_state = update_state(self._lcontact_state, lf_h, lf_vel)
        self._rcontact_state = update_state(self._rcontact_state, rf_h, rf_vel)

        return self._lcontact_state.clone(), self._rcontact_state.clone()

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
    def _finite_stat(self, name, t):
        if not torch.isfinite(t).all():
            print(f"[ENV NaN] {name} shape={tuple(t.shape)} "
                  f"min/max={float(torch.nanmin(t))}/{float(torch.nanmax(t))}")
            bad = (~torch.isfinite(t)).nonzero(as_tuple=False)
            print(" first bad idx:", bad[:5].tolist())
            raise RuntimeError(f"ENV component non-finite: {name}")

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

        self._finite_stat("condition", condition)
        self._finite_stat("delta", delta)
        self._finite_stat("map_world", map_world)
        self._finite_stat("map_yaw", map_yaw)

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

            self.reset_initial_frames(frame_index=None)

            if self.use_sg_pool:
                # 초기에는 batch 크기만큼 round-robin으로 pair를 배정
                tidx = torch.arange(self.num_parallel, device=device, dtype=torch.long)
                self.pair_id[tidx] = self._next_pair_ids(int(self.num_parallel))
                self.fail_streak[tidx] = 0
                self._last_episode_success[tidx] = False
                self._apply_pair_to_envs(tidx)
            else:
                self.reset_start_and_goal_from_nav(indices=None, roi_size_m=10.0)

            self._last_reset_indices = torch.arange(self.num_parallel, device=device, dtype=torch.long)

        else:
            # ---- 부분 reset (끝난 env들만)
            tidx = indices.to(device=device, dtype=torch.long).view(-1)

            self.root_facing.index_fill_(0, tidx, 0)
            self.root_xz.index_fill_(0, tidx, 0)
            self.reward.index_fill_(0, tidx, 0)
            self.done.index_fill_(0, tidx, False)
            self.early_stop.index_fill_(0, tidx, False)

            self.timestep.index_fill_(0, tidx, 0)
            self.substep.index_fill_(0, tidx, 0)

            self.reset_initial_frames(frame_index=tidx)

            if self.use_sg_pool:
                # --- success/fail에 따라 pair 유지/교체
                success = self._last_episode_success[tidx].view(-1)  # (M,)
                fail = ~success

                # 실패 streak 업데이트 (성공 env는 0으로 리셋)
                if fail.any():
                    self.fail_streak[tidx[fail]] += 1
                if success.any():
                    self.fail_streak[tidx[success]] = 0

                # 교체 조건: 성공했거나, fail_streak이 너무 길어졌거나
                to_advance = success #| (self.fail_streak[tidx] >= self.max_fail_streak)

                adv_idx = tidx[to_advance]
                keep_idx = tidx[~to_advance]

                if adv_idx.numel() > 0:
                    self._advance_pairs_for_envs(adv_idx)

                if keep_idx.numel() > 0:
                    # 같은 pair 재시도: pair_id 유지, success flag만 초기화 후 start/goal 적용
                    self._last_episode_success[keep_idx] = False
                    self._apply_pair_to_envs(keep_idx)

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
        _ = self.calc_progress_reward(reset_indices=reset_idx)
        self._last_reset_indices = None

        # ---------------- base target reward (기존 유지) ----------------
        target_dist = -self.linear_potential
        target_is_close = target_dist < 0.8
        dist_reward = 2 * torch.exp(0.5 * self.linear_potential)

        self.reward.copy_(dist_reward)
        self.reward[target_is_close].copy_(5.0)

        # ---------------- foot penalty (A 단계) ----------------
        # ---------- FK toe + contact 기반 collision ----------
        lf_xz, rf_xz, lf_h, rf_h = self._get_current_feet_world_fk()
        lcontact, rcontact = self._estimate_contact_from_fk_height(lf_xz, rf_xz, lf_h, rf_h)

        lf_occ = self.nav.query_walkable_batch(lf_xz[:, 0], lf_xz[:, 1], use_original_walkable=True).view(-1, 1)
        rf_occ = self.nav.query_walkable_batch(rf_xz[:, 0], rf_xz[:, 1], use_original_walkable=True).view(-1, 1)

        # contact 중에 obstacle이면 collision
        lf_bad = (lf_occ <= 0.5) & lcontact
        rf_bad = (rf_occ <= 0.5) & rcontact
        collision = lf_bad | rf_bad

        # (A) collision penalty + episode terminate
        w_collision = float(self.config.get("w_collision", 20.0))
        self.reward[collision] -= w_collision

        # ---------------- 미래 발자국 shaping (약하게) ----------------
        # self.lf/self.rf: diffusion이 예측한 "미래" 발자국(시점 불명) -> done 금지, 작은 shaping만
        if hasattr(self, "lf") and hasattr(self, "rf"):
            lf_pred_occ = self.nav.query_walkable_batch(
                self.lf[:, 0], self.lf[:, 1],
                use_original_walkable=True
            ).view(-1, 1)

            rf_pred_occ = self.nav.query_walkable_batch(
                self.rf[:, 0], self.rf[:, 1],
                use_original_walkable=True
            ).view(-1, 1)

            # 예측 발이 obstacle(=non-walkable)에 있으면 작은 패널티
            pred_bad = (lf_pred_occ <= 0.5) | (rf_pred_occ <= 0.5)

            # (옵션) 예측이 너무 자주 흔들리면 "둘 다 나쁠 때만" 패널티 주는 것도 안정적
            # pred_bad = (lf_pred_occ <= 0.5) & (rf_pred_occ <= 0.5)

            w_future = float(getattr(self, "w_future_foot", 0.5))
            self.reward[pred_bad] -= w_future

        # ---------------- success definition ----------------
        # 성공: 목표에 도달했고, 해당 step에서 발이 obstacle 위가 아님
        success = target_is_close & (~(lf_bad | rf_bad))

        # per-env done: time-limit OR success(early terminate)
        time_limit = (self.timestep >= self.max_timestep)
        self.done.copy_(time_limit | success | collision)

        if success.any():
            suc_idx = success.view(-1).nonzero(as_tuple=False).view(-1)
            self._last_episode_success[suc_idx] = True

        obs_components = self.get_observation_components()

        self.render()

        return (
            torch.cat(obs_components, dim=1),
            self.reward,
            self.done,
            {
                "reset": time_limit,   # time-limit (rollout truncation)
                "success": success,    # 성공 여부
                "collision": collision,
            },
        )

    def reset_target(self, indices=None, location=None):
        # render에서 마우스로 찍는 경우는 그대로 허용
        if location is not None:
            self.target[:, 0] = location[0]
            self.target[:, 1] = location[1]
            self.calc_potential()
            return

        if self.use_sg_pool:
            # pool 모드에서는 "새 goal만" 바꾸지 말고, pair 자체를 다음으로 넘김 (round-robin)
            if indices is None:
                tidx = torch.arange(self.num_parallel, device=self.device, dtype=torch.long)
            else:
                tidx = indices.to(device=self.device, dtype=torch.long).view(-1)
            self._advance_pairs_for_envs(tidx)
            return

        # ---------------- fallback: 기존 nav 기반 goal 리셋 ----------------
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
            self.target_arr[..., self.index_of_target, :2] = self.target[:, :2]
            self.target_arr[..., self.index_of_target, 2] = self.timestep
            self.index_of_target += 1

            np.save(osp.join(self.int_output_dir, 'out_target'), self.target_arr)
            self.viewer.update_target_markers(self.target)

        self.calc_potential(indices=tidx)

    def step(self, action):
        next_frame = self.get_next_frame(action)
        obs, reward, done, info = self.calc_env_state(next_frame)
        return (obs, reward, done, info)

    @torch.no_grad()
    def _apply_pair_to_envs(self, indices: torch.Tensor):
        tidx = indices.to(device=self.device, dtype=torch.long).view(-1)
        pid = self.pair_id[tidx]
        self.root_xz[tidx] = self.sg_pool_starts[pid]
        self.target[tidx] = self.sg_pool_goals[pid]

        self.calc_potential(indices=tidx)

        # ✅ render일 때 타겟 마커 업데이트
        if self.is_rendered:
            self.viewer.update_target_markers(self.target)

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
