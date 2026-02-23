import os
import torch
import math
import policy.envs.base_env as base_env
from render.realtime.sura_mocap_renderer import PBLMocapViewer
import gymnasium as gym
import numpy as np

class sura_BVH_env(gym.Env):
    NAME = "sura_BVH_env"
    def __init__(self, config, model, dataset, device, oriData = None):
        #self.file_name = config['file_name']
        self.start_timestep = config['start_timestep']
        self.max_timestep = config['max_timestep']
        self.num_parallel = config["num_parallel"]
        self.camera_tracking = config.get('camera_tracking',True)
        self.device = device
        self.is_rendered = True
        self.file_lst = dataset.file_lst

        self.frame_skip = 1
        self.num_condition_frames = 1

        self.dataset = dataset
        self.frame_dim = dataset.frame_dim
        self.data_fps = dataset.fps
        self.sk_dict = dataset.skel_info

        self.links = self.dataset.links
        self.valid_idx = self.dataset.valid_idx
        self.valid_range = self.dataset.valid_range

        self.use_cond = dataset.use_cond
        
        self.timestep = torch.zeros((self.num_parallel, 1)).to(self.device)

        self.file_idx = 0
        self.playing = True
        timestep_range = torch.tensor(self.valid_range[self.file_idx]).squeeze().to(self.device)
        
        self.timestep_start = timestep_range[...,0]
        self.timestep_end = timestep_range[...,1]
        self.clip_timestep = self.timestep_start

        self.lf = torch.zeros((self.num_parallel, 2), dtype=torch.float32, device=self.device)
        self.rf = torch.zeros((self.num_parallel, 2), dtype=torch.float32, device=self.device)

        self.root_facing = torch.zeros((self.num_parallel, 1), dtype=torch.float32, device=self.device)
        #self.root_facing.fill_(90)

        self.root_xz = torch.zeros((self.num_parallel, 2)).to(dtype = torch.float32, device = self.device)
        self.root_y = torch.zeros((self.num_parallel, 1), dtype=torch.float32, device=self.device)

        self.reward = torch.zeros((self.num_parallel, 1)).to(self.device)
        self.done = torch.zeros((self.num_parallel, 1)).bool().to(self.device)

        self.history_size = 5
        self.history = torch.zeros(
            (self.num_parallel, self.history_size, self.frame_dim)
        ).to(self.device)

        self.parallel_ind_buf = (
            torch.arange(0, self.num_parallel).long().to(self.device)
        )

        self.action_space = gym.spaces.Box(-1, 1)
        self.observation_space = gym.spaces.Box(-1, 1)
        self.viewer = PBLMocapViewer(
            self,
            num_characters=self.num_parallel,
            target_fps=self.data_fps,
            camera_tracking=self.camera_tracking,
        )

    def get_valid_range(self, valid_index):
        st = 0
        ed = 0
        skip_flag = False 
        st_ed_lst = []
        for i in range(valid_index[-1]):
            if i not in valid_index and not skip_flag:
                ed = i
                st_ed_lst.append([st, ed])
                skip_flag = True

            elif i in valid_index and skip_flag:
                st = i
                skip_flag = False
        
        return st_ed_lst

    def get_rotation_matrix(self, yaw, dim=2):
        zeros = torch.zeros_like(yaw)
        ones = torch.ones_like(yaw)
        if dim == 3:
            #col1 = torch.cat((yaw.cos(), zeros, yaw.sin()), dim=-1)
            #col2 = torch.cat((zeros, ones, zeros), dim=-1)
            #col3 = torch.cat((-yaw.sin(), zeros, yaw.cos()), dim=-1)
            #matrix = torch.stack((col1, col2, col3), dim=-1)
            col1 = torch.cat((yaw.cos(), yaw.sin(), zeros), dim=-1)
            col2 = torch.cat((-yaw.sin(), yaw.cos(), zeros), dim=-1)
            col3 = torch.cat((zeros, zeros, ones), dim=-1)
            matrix = torch.stack((col1, col2, col3), dim=-1)
        else:
            col1 = torch.cat((yaw.cos(), yaw.sin()), dim=-1)
            col2 = torch.cat((-yaw.sin(), yaw.cos()), dim=-1)
            matrix = torch.stack((col1, col2), dim=-1)
        return matrix

    def get_next_frame(self):
        #cur_time_step
        print(f'clip_timestep : {self.clip_timestep}')

        data = self.dataset.motion_flattened[self.clip_timestep]
        data = self.dataset.denorm_data(data)
        data = torch.tensor(data).to(self.device)
        #data_prev = self.dataset.motion_flattened[self.clip_timestep - 1]
        #data_prev = self.dataset.denorm_data(data_prev)
        #data_prev = torch.tensor(data_prev).to(self.device)

        lfoot_xz = data[..., self.dataset.foot_cartesian[0]: self.dataset.foot_cartesian[1] - 2]
        rfoot_xz = data[..., self.dataset.foot_cartesian[0] + 2: self.dataset.foot_cartesian[1]]

        '''lfoot_polar = data[..., self.dataset.foot_polar[0]: self.dataset.foot_polar[1] - 2]
        rfoot_polar = data[..., self.dataset.foot_polar[0]+2 : self.dataset.foot_polar[1]]

        # L foot
        ltheta = lfoot_polar[..., 0]
        l_d = lfoot_polar[..., 1]
        l_recon_dx = l_d * torch.sin(ltheta)
        l_recon_dz = l_d * torch.cos(ltheta)
        l_recon_xz = torch.stack([l_recon_dx, l_recon_dz], dim=-1)  # (..., 2)

        # R foot
        rtheta = rfoot_polar[..., 0]
        r_d = rfoot_polar[..., 1]

        r_recon_dx = r_d * torch.sin(rtheta)
        r_recon_dz = r_d * torch.cos(rtheta)
        r_recon_xz = torch.stack([r_recon_dx, r_recon_dz], dim=-1)  # (..., 2)'''

        #v_root, dyaw, l_disp, r_disp, foot_label = self._slice_future(self.clip_timestep, horizon=30)

        self.timestep += 1
        self.clip_timestep += 1

        return data, lfoot_xz, rfoot_xz #, l_recon_xz, r_recon_xz

    def get_previous_frame(self):
        #cur_time_step
        print(f'clip_timestep : {self.clip_timestep}')

        data = self.dataset.motion_flattened[self.clip_timestep]
        data = self.dataset.denorm_data(data)
        data = torch.tensor(data).to(self.device)

        self.timestep -= 1
        self.clip_timestep -= 1
        return data

    def reset_initial_frames(self, indices=None):
        timestep_range = torch.tensor(self.valid_range[self.file_idx]).squeeze().to(self.device)
        timestep_start = timestep_range[...,0]
        clip_timestep = torch.tensor(self.timestep_start)
        timestep_end = timestep_range[...,1]
        print(f'self.file_idx : {self.file_idx}')
        print(f'file_lst : {self.file_lst}')

        if indices is not None:
            self.clip_timestep[indices] = clip_timestep[indices]
        else:
            self.timestep_start = timestep_start
            self.clip_timestep = clip_timestep
            self.timestep_end = timestep_end

    def seed(self, seed):
        return 

    def reset_index(self, indices=None):
        self.root_facing.fill_(0)
        self.root_xz.fill_(0)
        self.lf.fill_(0)
        self.rf.fill_(0)
        self.done.fill_(False)
        self.reset_initial_frames(indices)

    def integrate_root_translation(self, pose, lfoot_vec, rfoot_vec):#, lrecon_xz, rrecon_xz):

        mat = self.get_rotation_matrix(self.root_facing)
        root_vec = pose[:2]

        #print(self.root_facing)

        mat = mat.to(device=root_vec.device, dtype=root_vec.dtype)
        displacement = torch.matmul(mat, root_vec.unsqueeze(-1)).squeeze(-1)

        lf_world = torch.matmul(mat, lfoot_vec.unsqueeze(-1)).squeeze(-1)  # (..., 2)
        rf_world = torch.matmul(mat, rfoot_vec.unsqueeze(-1)).squeeze(-1)  # (..., 2)

        '''lf_world_polar = torch.matmul(mat, lrecon_xz.unsqueeze(-1)).squeeze(-1)
        rf_world_polar = torch.matmul(mat, rrecon_xz.unsqueeze(-1)).squeeze(-1)'''

        self.lf = lf_world + self.root_xz
        self.rf = rf_world + self.root_xz

        '''self.lf_polar = lf_world_polar + self.root_xz
        self.rf_polar = rf_world_polar + self.root_xz'''

        dr = pose[2][..., None]
        self.root_facing.add_(dr)
        self.root_facing = (self.root_facing + np.pi) % (2 * np.pi) - np.pi
        self.root_xz.add_(displacement)

        self.history = self.history.roll(1, dims=1)
        self.history[:, 0].copy_(pose)

    def calc_env_state(self, next_frame, lfoot_vec, rfoot_vec):#, lrecon_xz, rrecon_xz):
        if next_frame is None:
            return (
                None,
                self.reward,
                self.done,
                {"reset": True},
            )
        #self.integrate_root_translation(next_frame, lfoot_vec, rfoot_vec, lrecon_xz, rrecon_xz)
        self.integrate_root_translation(next_frame, lfoot_vec, rfoot_vec)
        #foot_slide = self.calc_foot_slide()
        #self.reward.add_(foot_slide.sum(dim=-1, keepdim=True) * -10.0)
        #obs_components = self.get_observation_components()

        done = self.clip_timestep >= self.timestep_end
        
        #print(done.shape, self.clip_timestep.shape, self.timestep_end.shape, (self.clip_timestep>=self.timestep_end).shape)
        self.done = done
        self.render()

        return (
            None,
            self.reward,
            self.done,
            {"reset": done},
        )

    def render(self, mode="human"):
        self.viewer.render(
            torch.tensor(self.dataset.x_to_jnts(self.history[:, 0].cpu().numpy(), mode='angle'),device=self.device, dtype=self.history.dtype),
            #torch.tensor(self.dataset.x_to_jnts(self.history[:, 0].cpu().numpy(), mode='position'), device=self.device, dtype=self.history.dtype),
            self.root_facing,
            self.root_xz,
            self.lf,
            self.rf,
            #self.lf_polar,
            #self.rf_polar,
            0.0,  # No time in this env
            0.0   #self.action,
        )

    def dump_additional_render_data(self):
        pass
