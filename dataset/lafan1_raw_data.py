import os.path as osp
import numpy as np
import glob

import tqdm
from dataset.util.bvh import load_bvh_info
from dataset.util.skeleton_info import skel_dict
import dataset.util.bvh as bvh_util

class BVHRawData():
    NAME = 'LAFAN1_BVH'
    # For a directory contains multiple identical file type
    def __init__(self, config):
        self.dataset_name = config["data"]["dataset_name"]

        if self.dataset_name in skel_dict:
            self.skel_info = skel_dict[self.dataset_name]
        elif self.dataset_name.split('_')[0] in skel_dict:
            self.skel_info = skel_dict[self.dataset_name.split('_')[0]]

        self.joint_names = self.skel_info.get("name_joint", None)
        self.end_eff = self.skel_info.get("end_eff", None)
        self.joint_offset = self.skel_info.get("offset_joint", None)

        self.root_idx = self.skel_info.get("root_idx", None)
        self.foot_idx = self.skel_info.get('foot_idx', None)
        self.toe_idx = self.skel_info.get('toe_idx', None)
        self.unit = self.skel_info.get('unit', None)
        self.rotate_order = self.skel_info.get('euler_rotate_order', None)

        self.fps = config["data"]["data_fps"]
        self.path = config["data"]["path"]

        self.valid_idx = []
        self.valid_range = list()
        self.test_valid_idx = list()
        self.file_lst = list()
        self.joint_offset = list()

        stats_legacy = osp.join(self.path, 'stats_origin.npz')
        if osp.exists(stats_legacy):
            with np.load(stats_legacy, allow_pickle=True) as stats:
                self.joint_names = stats['joint_names'].tolist()
                self.joint_offset = stats['joint_offset']
                self.num_jnt = len(self.joint_names)
                self.links = stats['links']

        self.file_name = []
        self.joint_offset = []
        self.joint_parent = None
        self.rotate_order = None
        self.joint_names = None
        self.joint_chn_num = None
        self.frame_time = None
        self.motions = []
        self.skeletons = []
        self.rotations = []
        self.positions = []
        self.links=[]
        self.file_paths = self.get_motion_fpaths()

        self.load_skeleton()
        self.load_motion()

    def get_position(self, bvh_idx, frame_idx):
        return self.positions[bvh_idx][frame_idx]

    def get_all_position(self):
        return self.positions

    def get_motion_fpaths(self):
        return glob.glob(osp.join(self.path, '*.{}'.format('bvh')))

    def load_skeleton(self):
        for i, fname in enumerate(tqdm.tqdm(self.file_paths)):
            self.file_name.append(fname)
            if i == 0:
                joint_name, joint_parent, joint_offset, joint_rot_order, joint_chn_num, frame_time = load_bvh_info(fname)
                self.joint_names = joint_name
                self.num_jnt = len(self.joint_names)
                self.joint_parent = joint_parent
                self.joint_offset.append(joint_offset)
                self.rotate_order = joint_rot_order
                self.joint_chn_num = joint_chn_num
                self.frame_time = frame_time

    def load_motion(self):
        for i, fname in enumerate(tqdm.tqdm(self.file_paths)):
            _motion = bvh_util.import_bvh(fname)
            self.motions.append(_motion)
            self.skeletons.append(_motion._skeleton)
            self.rotations.append(_motion._rotations)
            self.positions.append(_motion._positions)
            self.links.append(_motion._skeleton.get_links())
            self.valid_idx.append(_motion.get_num_frames())
