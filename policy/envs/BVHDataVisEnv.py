import policy.envs.base_env as base_env
from render.realtime.bvh_mocap_renderer import PBLMocapViewer
import torch
import numpy as np
import gymnasium as gym

class BVHDataVisEnv(base_env.EnvBase):
    NAME = "BVHDataVisEnv"

    def __init__(self, config, model, dataset, device):
        self.device = device
        self.config = config
        self.model = model
        self.dataset = dataset

        self.cur_extra_info = None
        self.positions = None

        self.links = self.dataset.links
        self.valid_idx = self.dataset.valid_idx

        self.frame_dim = 279
        self.action_dim = 279
        self.valid_range = self.dataset.valid_range
        self.sk_dict = dataset.skel_info
        self.data_fps = self.dataset.fps

        self.is_rendered = True
        self.num_parallel = config.get('num_parallel', 1)
        self.frame_skip = config.get('frame_skip', 1)
        self.max_timestep = config.get('max_timestep', 10000)
        self.camera_tracking = config.get('camera_tracking', False)

        self.num_condition_frames = 1

        self.base_action = torch.zeros((self.num_parallel, 1, self.action_dim)).to(
            self.device
        )
        self.timestep = torch.zeros((self.num_parallel, 1)).to(self.device)
        self.substep = torch.zeros((self.num_parallel, 1)).to(self.device)
        self.done = torch.zeros((self.num_parallel, 1)).bool().to(self.device)

        self.parallel_ind_buf = (
            torch.arange(0, self.num_parallel).long().to(self.device)
        )

        high = np.inf * np.ones([self.action_dim])
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        self.viewer = PBLMocapViewer(
            self,
            num_characters=self.num_parallel,
            target_fps=self.data_fps,
            camera_tracking=self.camera_tracking,
        )

    def get_max_frames(self, bvh_idx):
        return len(self.dataset.positions[bvh_idx])

    def get_file_name(self, bvh_idx):
        return self.dataset.file_name[bvh_idx]

    def get_cond_frame(self):
        condition = self.history[:, :self.num_condition_frames].view(-1, self.frame_dim)
        return condition

    def get_next_frame(self):
        output = self.dataset.get_all_position()
        return output

    def calc_env_state(self, positions):
        self.timestep[self.substep == self.frame_skip - 1] += 1
        self.substep = (self.substep + 1) % self.frame_skip
        self.positions = positions

        self.render()

    def render(self, mode="human"):
        if self.is_rendered:
            self.viewer.render(self.positions)

    def reset(self):
        self.timestep.fill_(0)
        self.substep.fill_(0)
        self.done.fill_(False)
        self.reset_initial_frames()
        return

    def reset_initial_frames(self, index=None):
        start_index = 0

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def close(self):
        if self.is_rendered:
            self.viewer.close()
