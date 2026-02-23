import os
import sys
import itertools

import torch
import torch.nn.functional as F
import numpy as np

import matplotlib.cm as mpl_color
from imageio import imwrite

import pybullet as pb
from policy.common.bullet_objects import VSphere, VCylinder, VCapsule, FlagPole, Arrow
from policy.common.bullet_utils import BulletClient, Camera, SinglePlayerStadiumScene
import time

FOOT2METER = 0.01
DEG2RAD = np.pi / 180
FADED_ALPHA = 1.0


def extract_joints_xyz(xyzs):
    x = xyzs[..., 0]
    y = xyzs[..., 1]
    z = xyzs[..., 2]
    return x, y, z


class PBLMocapViewer:
    def __init__(
            self,
            env,
            num_characters=1,
            use_params=True,
            target_fps=0,
            camera_tracking=True,
    ):
        self.device = env.device
        target_fps = env.dataset.fps
        sk_dict = env.sk_dict
        sk_dict['links'] = env.dataset.links

        self.env = env
        self.num_characters = num_characters
        self.use_params = use_params

        self.character_index = 0
        self.controller_autonomy = 1.0
        self.debug = False
        self.gui = False

        self.bvh_idx = 0
        self.cur_frame = 0
        self.max_frames = self.env.get_max_frames(self.bvh_idx)
        self.root_xyzs = None

        self._last_mouse_x = 0.0
        self._last_mouse_y = 0.0
        self._is_dragging = False
        self._is_rotating = False
        self._is_panning = False
        self.playing = False
        self.reverse_playing = False

        self._play_counter = 4
        self.play_speed = 0

        # ==================
        self.camera_tracking = camera_tracking
        # use 1.5 for close up, 3 for normal, 6 with GUI
        self.camera_distance = 6 if self.camera_tracking else 12
        self.camera_smooth = np.array([1, 1, 1])

        connection_mode = pb.GUI if env.is_rendered else pb.DIRECT
        self._p = BulletClient(connection_mode=connection_mode)
        self._p.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
        self._p.configureDebugVisualizer(pb.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
        self._p.configureDebugVisualizer(pb.COV_ENABLE_MOUSE_PICKING, 0)
        self._p.configureDebugVisualizer(pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        self._p.configureDebugVisualizer(pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        self._p.configureDebugVisualizer(pb.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)

        # Disable rendering during creation
        self._p.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)

        self.camera = Camera(
            self._p, fps=target_fps, dist=self.camera_distance, pitch=-10, yaw=45
        )
        scene = SinglePlayerStadiumScene(
            self._p, gravity=9.8, timestep=1 / target_fps, frame_skip=1
        )
        scene.initialize()

        cmap = mpl_color.get_cmap("coolwarm")
        self.colours = cmap(np.linspace(0, 1, self.num_characters))

        if num_characters == 1:
            self.colours[0] = (0.98, 0.54, 0.20, 1)

        self.characters = MultiMocapCharacters(self._p, num_characters, sk_dict, self.colours)

        # Re-enable rendering
        self._p.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)

        self.state_id = self._p.saveState()

    def reset(self):
        # self._p.restoreState(self.state_id)
        self.env.reset()

    def render(self, xyzs):
        if getattr(self, "playing", False):
            self._play_counter += 1
            if self._play_counter >= self.play_speed:
                self._play_counter = 0
                self.cur_frame += 1
                print(f'frame: {self.cur_frame}')
                if self.cur_frame >= self.max_frames:
                    self.cur_frame = 0

        elif getattr(self, "reverse_playing", False):
            self._play_counter += 1
            if self._play_counter >= self.play_speed:
                self._play_counter = 0
                self.cur_frame -= 1
                print(f'frame: {self.cur_frame}')
                if self.cur_frame < 0:
                    self.cur_frame = self.max_frames - 1

        frame_xyz = xyzs[self.bvh_idx][self.cur_frame]  # shape: (22, 3)
        frame_xyz = torch.tensor(frame_xyz, dtype=torch.float32).to(self.device)

        # (1, 22, 3) - batch dimension 추가
        frame_xyz = frame_xyz.unsqueeze(0)

        # (1, 22, 3) → (1, 3, 22)
        joint_xyzs = frame_xyz.permute(0, 2, 1)

        # (x, y, z) → (x, z, y) 로 변환
        x = joint_xyzs[:, 0, :]
        y = joint_xyzs[:, 1, :]
        z = joint_xyzs[:, 2, :]
        joint_xyzs = torch.stack((x, z, y), dim=1)  # (1, 3, 22)

        self.joint_xyzs = joint_xyzs
        joint_xyzs = (joint_xyzs * FOOT2METER).cpu().numpy()
        self.joint_xyzs = joint_xyzs

        for index in range(self.num_characters):
            self.characters.set_joint_positions(joint_xyzs[index], index)

        self.root_xyzs = joint_xyzs[:, :, 0]  # (1,3)

        # --- 입력/카메라 ---
        self._handle_mouse_press()
        self._handle_key_press()
        #if self.use_params:
            #self._handle_parameter_update()

        if not getattr(self, "_is_rotating", False) and not getattr(self, "_is_panning", False):
            if self.camera_tracking:
                self.camera.track(self.root_xyzs[self.character_index], self.camera_smooth)
            else:
                self.camera.wait()
        else:
            time.sleep(1.0 / 60.0)

    def close(self):
        self._p.disconnect()
        sys.exit(0)

    def _setup_debug_parameters(self):
        max_frame = self.env.max_timestep - self.env.num_condition_frames
        self.parameters = [
            {
                # -1 for random start frame
                "default": -1,
                "args": ("Start Frame", -1, max_frame, -1),
                "dest": (self.env, "debug_frame_index"),
                "func": lambda x: int(x),
                "post": lambda: self.env.reset(),
            },
            {
                "default": self.env.data_fps,
                "args": ("Target FPS", 1, 240, self.env.data_fps),
                "dest": (self.camera, "_target_period"),
                "func": lambda x: 1 / (x + 1),
            },
            {
                "default": 1,
                "args": ("Controller Autonomy", 0, 1, 1),
                "dest": (self, "controller_autonomy"),
                "func": lambda x: x,
            },
            {
                "default": 1,
                "args": ("Camera Track Character", 0, 1, int(self.camera_tracking)),
                "dest": (self, "camera_tracking"),
                "func": lambda x: x > 0.5,
            },
        ]

        if self.num_characters > 1:
            self.parameters.append(
                {
                    "default": 1,
                    "args": ("Selected Character", 1, self.num_characters + 0.999, 1),
                    "dest": (self, "character_index"),
                    "func": lambda x: int(x - 1.001),
                }
            )

        max_frame_skip = 1  # self.env.num_future_predictions
        if max_frame_skip > 1:
            self.parameters.append(
                {
                    "default": 1,
                    "args": (
                        "Frame Skip",
                        1,
                        max_frame_skip + 0.999,
                        self.env.frame_skip,
                    ),
                    "dest": (self.env, "frame_skip"),
                    "func": lambda x: int(x),
                }
            )

        if hasattr(self.env, "target_direction"):
            self.parameters.append(
                {
                    "default": 0,
                    "args": ("Target Direction", 0, 359, 0),
                    "dest": (self.env, "target_direction"),
                    "func": lambda x: x / 180 * np.pi,
                    "post": lambda: self.env.reset_target(),
                }
            )

        if hasattr(self.env, "target_speed"):
            self.parameters.append(
                {
                    "default": 0,
                    "args": ("Target Speed", 0.0, 0.8, 0.5),
                    "dest": (self.env, "target_speed"),
                    "func": lambda x: x,
                }
            )

        # setup Pybullet parameters
        for param in self.parameters:
            param["id"] = self._p.addUserDebugParameter(*param["args"])

    def _handle_parameter_update(self):
        for param in self.parameters:
            func = param["func"]
            value = func(self._p.readUserDebugParameter(param["id"]))
            cur_value = getattr(*param["dest"], param["default"])
            if cur_value != value:
                setattr(*param["dest"], value)
                if "post" in param:
                    post_func = param["post"]
                    post_func()

    def _handle_mouse_press(self):
        events = self._p.getMouseEvents()

        for ev in events:
            event_type = ev[0]  # 1: MOVE, 2: BUTTON
            x = ev[1]
            y = ev[2]
            button = ev[3]  # 0: left, 1: middle, 2: right
            state = ev[4]  # 0: move, 3: is down, 4: released

            dx = x - getattr(self, "_last_mouse_x", x)
            dy = y - getattr(self, "_last_mouse_y", y)

            if event_type == 1:
                if getattr(self, "_is_rotating", False):
                    self.camera.rotate(dx, dy)
                elif getattr(self, "_is_panning", False):
                    self.camera.pan(dx, dy)

            elif event_type == 2:
                if button == 0 and state == 3:
                    self._is_rotating = True
                elif button == 2 and state == 3:
                    self._is_panning = True
                elif button in [0, 2] and state == 4:
                    if button == 0:
                        self._is_rotating = False
                    elif button == 2:
                        self._is_panning = False

            self._last_mouse_x = x
            self._last_mouse_y = y

    def _handle_key_press(self, keys=None):
        if keys is None:
            keys = self._p.getKeyboardEvents()
        RELEASED = self._p.KEY_WAS_RELEASED

        # keys is a dict, so need to check key exists
        if keys.get(ord("d")) == RELEASED:
            self.debug = not self.debug
        elif keys.get(ord("g")) == RELEASED:
            self.gui = not self.gui
            self._p.configureDebugVisualizer(pb.COV_ENABLE_GUI, int(self.gui))
        elif keys.get(ord("n")) == RELEASED:
            # doesn't work with pybullet's UserParameter
            self.character_index = (self.character_index + 1) % self.num_characters
            self.camera.lookat(self.root_xyzs[self.character_index])
        elif keys.get(ord("m")) == RELEASED:
            self.camera_tracking = not self.camera_tracking
            print(f'camera_tracking: {self.camera_tracking}')
        elif keys.get(ord("q")) == RELEASED:
            self.close()
        elif keys.get(ord("r")) == RELEASED:
            self.reset()
        elif keys.get(ord("t")) == RELEASED:
            self.env.reset_target()
        elif keys.get(ord("i")) == RELEASED:
            image = self.camera.dump_rgb_array()
            imwrite("image_c.png", image)
        elif keys.get(ord("a")) == RELEASED:
            image = self.camera.dump_orthographic_rgb_array()
            imwrite("image_o.png", image)
        elif keys.get(ord("v")) == RELEASED:
            import datetime

            now_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            filename = "{}.mp4".format(now_string)

            self._p.startStateLogging(self._p.STATE_LOGGING_VIDEO_MP4, filename)

        elif keys.get(ord(" ")) == RELEASED:
            was_playing = getattr(self, "playing", False)
            was_reverse = getattr(self, "reverse_playing", False)

            # 둘 다 꺼져 있으면 forward 재생 시작
            if not was_playing and not was_reverse:
                self.playing = True
                self.reverse_playing = False
            else:
                # 정지
                self.playing = False
                self.reverse_playing = False

            print(f"[INFO] Playing: {self.playing}, Reverse: {self.reverse_playing}")

        elif keys.get(ord("b")) == RELEASED:
            was_playing = getattr(self, "reverse_playing", False)
            was_forward = getattr(self, "playing", False)

            if not was_playing and not was_forward:
                self.reverse_playing = True
                self.playing = False
            else:
                self.reverse_playing = False
                self.playing = False

            print(f"[INFO] Reverse Playing: {self.reverse_playing}")

        elif keys.get(65297) == RELEASED:  # 위쪽 화살표
            self.play_speed = max(1, self.play_speed - 1)
            print(f"[INFO] Increased speed: play_speed = {self.play_speed}")

        elif keys.get(65298) == RELEASED:  # 아래쪽 화살표
            self.play_speed += 1
            print(f"[INFO] Decreased speed: play_speed = {self.play_speed}")

        elif keys.get(ord("1")) == RELEASED:
            if (self.bvh_idx <= 0):
                print("[INFO] Start of BVH.")
            else:
                self.cur_frame = 0
                self.bvh_idx -= 1
                self.max_frames = self.env.get_max_frames(self.bvh_idx)
            print(f'BVH_IDX: {self.bvh_idx}')
            print(f'current filename : {self.env.get_file_name(self.bvh_idx)}')
        elif keys.get(ord("3")) == RELEASED:
            if self.bvh_idx >= 59:
                print("[INFO] End of BVH.")
            else :
                self.bvh_idx += 1
                self.cur_frame = 0
                self.max_frames = self.env.get_max_frames(self.bvh_idx)
            print(f'BVH_IDX: {self.bvh_idx}')
            print(f'current filename : {self.env.get_file_name(self.bvh_idx)}')

        elif keys.get(65296) == RELEASED:  # right key
            if self.cur_frame >= self.max_frames-1:
                print("[INFO] End of motion.")
            else :
                self.cur_frame += 1
            print(f"BVH {self.bvh_idx} : Frame {self.cur_frame+1}/{self.max_frames}")

        elif keys.get(65295) == RELEASED:  # left key
            if (self.cur_frame <= 0):
                print("[INFO] Start of motion.")
            else:
                self.cur_frame -= 1
            print(f"BVH {self.bvh_idx} : Frame {self.cur_frame+1}/{self.max_frames}")

class MultiMocapCharacters:
    def __init__(self, bc, num_characters, sk_dict, colours=None, links=True):
        self._p = bc
        self.has_links = links

        if links:
            self.linked_joints = np.array(sk_dict['links'])
            self.linked_joints = self.linked_joints[0]
            self.head_idx = sk_dict['head_idx']
            self.num_joint = sk_dict['num_joint']

            total_parts = num_characters * self.num_joint
            joints = VSphere(bc, radius=0.03, max=True, replica=total_parts)
            self.ids = joints.ids
            self.links = {
                i: [
                    VCapsule(self._p, radius=0.02, height=0.1, rgba=colours[i])
                    for _ in range(self.linked_joints.shape[0])
                ]
                for i in range(num_characters)
            }
            self.z_axes = np.zeros((self.linked_joints.shape[0], 3))
            self.z_axes[:, 2] = 1

            self.heads = [VSphere(bc, radius=0.12) for _ in range(num_characters)]

        if colours is not None:
            self.colours = colours
            for index, colour in zip(range(num_characters), colours):
                colour[3] = 1
                self.set_colour(colour, index)
                if links:
                    self.heads[index].set_color(colour)

    def set_colour(self, colour, index):
        # start = self.start_index + index * self.num_joints
        start = index * self.num_joint
        joint_ids = self.ids[start: start + self.num_joint]
        for id in joint_ids:
            self._p.changeVisualShape(id, -1, rgbaColor=colour)

    def set_joint_positions(self, xyzs, index):

        self._p.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)

        start = index * self.num_joint
        joint_ids = self.ids[start: start + self.num_joint]

        xyzs = xyzs.transpose()
        for i, id in enumerate(joint_ids):
            self._p.resetBasePositionAndOrientation(id, posObj=xyzs[i], ornObj=(0, 0, 0, 1))

        if self.has_links:
            rgba = self.colours[index].copy()
            rgba[-1] = FADED_ALPHA
            deltas = xyzs[self.linked_joints[:, 1]] - xyzs[self.linked_joints[:, 0]]
            heights = np.linalg.norm(deltas, axis=-1)
            positions = xyzs[self.linked_joints].mean(axis=1)

            a = np.cross(deltas, self.z_axes)
            b = np.linalg.norm(deltas, axis=-1) + (deltas * self.z_axes).sum(-1)
            orientations = np.concatenate((a, b[:, None]), axis=-1)
            orientations[:, [0, 1]] *= -1

            for lid, (delta, height, pos, orn, link) in enumerate(
                    zip(deltas, heights, positions, orientations, self.links[index])
            ):
                # 0.05 feet is about 1.5 cm
                if abs(link.height - height) > 0.05:
                    self._p.removeBody(link.id[0])
                    link = VCapsule(self._p, radius=0.02, height=height, rgba=rgba)
                    self.links[index][lid] = link

                link.set_position(pos, orn)

            self.heads[index].set_position(
                0.5 * (xyzs[self.head_idx[1]] - xyzs[self.head_idx[0]]) + xyzs[self.head_idx[1]])
            # self.dir_link.set_position
        self._p.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)
