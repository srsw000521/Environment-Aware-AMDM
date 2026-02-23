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

import math

from PIL import Image

FOOT2METER = 1.3
DEG2RAD = np.pi / 180
FADED_ALPHA = 1.0

def extract_joints_xyz(xyzs):
    x = xyzs[...,0]
    y = xyzs[...,1]
    z = xyzs[...,2]
    return x, y, z

class PBLMocapViewer:
    def __init__(
        self,
        env,
        num_characters=1,
        use_params=True,
        target_fps = 0,
        camera_tracking=False,
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

        #==================
        self.camera_tracking = camera_tracking
        # use 1.5 for close up, 3 for normal, 6 with GUI
        self.camera_distance = 9 if self.camera_tracking else 12
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
        self._hide_stadium_ground_visual()

        tex_path = "/home/sura/AMDM/perlin6_500_500_raw_tauA0.80_tauB0.89_g1.7_A_masked.png"
        self.ground_id = self.create_textured_ground_mesh(
            tex_path=tex_path,
            size_m=50.0,
            z=0.02,
            unit_scale=FOOT2METER,  # ✅ 핵심
        )
        cmap = mpl_color.get_cmap("coolwarm")
        self.colours = cmap(np.linspace(0, 1, self.num_characters))

        if num_characters == 1:
            self.colours[0] = (0.98, 0.54, 0.20, 1)

        # here order is important for some reason ?
        # self.targets = MultiTargets(self._p, num_characters, self.colours)
        self.characters = MultiMocapCharacters(self._p, num_characters,  sk_dict, self.colours)
        
        # Re-enable rendering
        self._p.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)

        self.state_id = self._p.saveState()

        if self.use_params:
            self._setup_debug_parameters()

        self._last_mouse_x = 0.0
        self._last_mouse_y = 0.0
        self._is_rotating = False
        self._is_panning = False

        self._prev_lf = None
        self._prev_rf = None
        self.l_contact = False
        self.r_contact = False
        self._prev_l_contact = False
        self._prev_r_contact = False

        self.trail_stride = 3  # 2프레임마다 한 번씩 선 추가
        self.trail_width = 1
        self.trail_alpha = 0.6
        self._trail_t = 0
        self._trail_ids = []  # 필요하면 나중에 지우기용
        self.max_trail_lines = 20000

    def reset(self):
        # self._p.restoreState(self.state_id)
        self.env.reset()

    def create_textured_ground_mesh(self, tex_path: str, size_m=50.0, z=0.0, unit_scale=1.0):
        if not os.path.isabs(tex_path):
            here = os.path.dirname(os.path.abspath(__file__))
            tex_path = os.path.join(here, tex_path)
        if not os.path.exists(tex_path):
            raise FileNotFoundError(f"Texture not found: {tex_path}")

        tex_id = self._p.loadTexture(tex_path)

        here = os.path.dirname(os.path.abspath(__file__))
        obj_path = os.path.join(here, "_ground_uv_plane.obj")
        if not os.path.exists(obj_path):
            obj = """# unit plane with UVs (0..1)
    v -0.5 -0.5 0.0
    v  0.5 -0.5 0.0
    v  0.5  0.5 0.0
    v -0.5  0.5 0.0
    vt 0.0 0.0
    vt 1.0 0.0
    vt 1.0 1.0
    vt 0.0 1.0
    vn 0.0 0.0 1.0
    f 1/1/1 2/2/1 3/3/1
    f 1/1/1 3/3/1 4/4/1
    """
            with open(obj_path, "w") as f:
                f.write(obj)

        # ✅ 스케일 먼저 계산
        size_m_scaled = float(size_m) * float(unit_scale)
        mesh_scale = [size_m_scaled, size_m_scaled, 1.0]

        # ✅ visual도 스케일된 mesh_scale로 생성
        vis = self._p.createVisualShape(
            shapeType=self._p.GEOM_MESH,
            fileName=obj_path,
            meshScale=mesh_scale,
            rgbaColor=[1, 1, 1, 1],
            specularColor=[0, 0, 0]
        )

        # collision도 동일 스케일
        thickness = 0.02
        half = size_m_scaled * 0.5
        col = self._p.createCollisionShape(
            self._p.GEOM_BOX,
            halfExtents=[half, half, thickness * 0.5]
        )

        ground_id = self._p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis
        )

        self._p.resetBasePositionAndOrientation(
            ground_id, [0, 0, z - thickness * 0.5], [0, 0, 0, 1]
        )

        # ✅ 텍스처 적용도 다시
        self._p.changeVisualShape(ground_id, -1, textureUniqueId=tex_id, rgbaColor=[1, 1, 1, 1])

        return ground_id

    def _hide_stadium_ground_visual(self):
        """
        stadium/plane/ground로 보이는 body의 visual만 투명하게 만든다.
        (collision은 남을 수 있음. 하지만 우리가 만든 box를 z=0에 두면 실제로는 box가 보이고,
         충돌도 box가 더 '위'라서 대부분 문제 없음.)
        """
        for bid in range(self._p.getNumBodies()):
            try:
                name = self._p.getBodyInfo(bid)[1].decode("utf-8").lower()
            except Exception:
                name = ""

            # ground 후보만 골라서 투명화
            if ("plane" in name) or ("ground" in name) or ("stadium" in name):
                # base link (-1) visual을 투명하게
                try:
                    self._p.changeVisualShape(bid, -1, rgbaColor=[1, 1, 1, 0])
                    # 혹시 link가 여러개면 다 돌려서 투명화
                    for lid in range(self._p.getNumJoints(bid)):
                        self._p.changeVisualShape(bid, lid, rgbaColor=[1, 1, 1, 0])
                except Exception:
                    pass

    def _apply_ground_texture(self, texture_path: str, texture_scaling=(1, 1)):
        # 1) texture load
        if not os.path.isabs(texture_path):
            here = os.path.dirname(os.path.abspath(__file__))
            texture_path = os.path.join(here, texture_path)

        if not os.path.exists(texture_path):
            raise FileNotFoundError(f"Texture not found: {texture_path}")

        tex_id = self._p.loadTexture(texture_path)

        # 2) ground body 찾기: 이름에 stadium/plane/ground 들어간 것 우선
        ground_id = None
        for bid in range(self._p.getNumBodies()):
            name = self._p.getBodyInfo(bid)[1].decode("utf-8").lower()
            if ("stadium" in name) or ("plane" in name) or ("ground" in name):
                ground_id = bid
                break

        # fallback: 못 찾으면 그냥 0번 body 시도
        if ground_id is None and self._p.getNumBodies() > 0:
            ground_id = 0

        if ground_id is None:
            print("[WARN] No ground body found.")
            return

        # 3) texture 적용
        self._p.changeVisualShape(
            ground_id,
            -1,
            textureUniqueId=tex_id,
            rgbaColor=[1, 1, 1, 1],  # 텍스처 색이 그대로 보이게
            textureScaling=[0.02, 0.02]
        )
        print(f"[OK] Applied texture to ground body={ground_id}")

    def add_path_markers(self, path, color=None):

        if not hasattr(self, "path") and color is None:
            num_points = min(150, len(path))
            colours = np.tile([1, 1, 1, 0.5], [num_points, 1])
            self.path = MultiTargets(self._p, num_points, colours)
        else:
            num_points =  min(150, len(path))

        num_points = min(100, len(path))

        #느린 : step 18-20 num_steps : ~30
        #보통 : step 14-16 num_steps : ~40
        #빠른 : step 10~12 num_steps : ~50

        start_left = 10
        #num_steps = 25
        num_steps = 40
        end = len(path) - 1

        if color == 'left':
            base_f = torch.linspace(start_left, end, num_steps)  # float
            indices = base_f.round().long()

        elif color == 'right':
            '''base_f = torch.linspace(start_left, end, num_steps)  # left와 동일한 base
            step = (end - start_left) / (num_steps - 1)  # 평균 간격
            indices = (base_f + step / 2).round().long()  # half-step shift
            indices = indices.clamp(min=0, max=end)'''
            base_f = torch.linspace(start_left, end, num_steps)
            indices = base_f.round().long()  # shift 없음 (jump: 동시착지)
            indices = indices.clamp(min=0, max=end)
        else:
            indices = torch.linspace(0, end, num_points).long()

        positions = F.pad(path[indices] * FOOT2METER, pad=[0, 1], value=0)

        '''for index, position in enumerate(positions.cpu().numpy()):
            self.path.set_position(position, index)'''

        return path[indices]

    def update_target_markers(self, targets):
        #from environments.mocap_envs import JoystickEnv

        render_arrow = True if self.env.NAME == 'JOYSTICK' else False
        if not hasattr(self, "targets"):
            marker = Arrow if render_arrow else FlagPole
            self.targets = MultiTargets(
                self._p, self.num_characters, self.colours, marker
            )

            #data = self.poses[np.random.randint(self.poses.shape[0])]
            #self.targets = MocapCharacterTarget(
            #    self._p, self.num_characters, data, self.colours, None
            #)

        if render_arrow:
            target_xyzs = F.pad(self.env.root_xz, pad=[0, 1]) * FOOT2METER
            target_orns = self.env.target_direction_buf

            for index, (pos, angle) in enumerate(zip(target_xyzs, target_orns)):
                orn = self._p.getQuaternionFromEuler([0, 0, float(angle)-np.pi/2])
                self.targets.set_position(pos, index, orn)
        else:
            if targets.shape[-1] == 2:
                targets = F.pad(targets, pad=[0, 1], value=0)
            target_xyzs = (
                ( targets * FOOT2METER).cpu().numpy()
            )

            for index in range(self.num_characters):
                self.targets.set_position(target_xyzs[index], index)
                #height = target_xyzs[:,1]

    def duplicate_character(self):
        characters = self.characters
        colours = self.colours
        num_characters = self.num_characters
        bc = self._p

        bc.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)

        if self.characters.has_links:
            for index, colour in zip(range(num_characters), colours):
                faded_colour = colour.copy()
                faded_colour[-1] = FADED_ALPHA
                characters.heads[index].set_color(faded_colour)
                characters.links[index] = []

        self.characters = MultiMocapCharacters(bc, num_characters, self.env.sk_dict, colours)

        if hasattr(self, "targets") and self.targets.marker == Arrow:
            self.targets = MultiTargets(
                self._p, self.num_characters, self.colours, Arrow
            )

        bc.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)

    def _draw_filled_foot_ellipse(self, center, a=0.13, b=0.06, color=(0, 1, 0), yaw=None,  n_rings=6, n_seg=24):
        """
        Filled ellipse using debug lines (z-up).
        Ellipse major axis is ALONG +Y (foot forward).
        """

        if yaw is None:
            c, s = 1.0, 0.0
        else:
            c, s = math.cos(yaw), math.sin(yaw)

        # 여러 개의 내부 타원을 그려서 채워진 느낌
        for k in range(1, n_rings + 1):
            t = k / n_rings
            ak = a * t
            bk = b * t

            for i in range(n_seg):
                th0 = 2 * math.pi * i / n_seg
                th1 = 2 * math.pi * (i + 1) / n_seg

                x0, y0 = bk * math.sin(th0), ak * math.cos(th0)
                x1, y1 = bk * math.sin(th1), ak * math.cos(th1)

                # yaw 회전
                X0 = c * x0 - s * y0
                Y0 = s * x0 + c * y0
                X1 = c * x1 - s * y1
                Y1 = s * x1 + c * y1

                p0 = center.copy()
                p1 = center.copy()

                p0[0] += X0
                p0[1] += Y0
                p1[0] += X1
                p1[1] += Y1

                self._p.addUserDebugLine(
                    p0, p1,
                    lineColorRGB=color,
                    lineWidth=2,
                    lifeTime=0
                )

    def _rot2d(self, pts_xy: np.ndarray, yaw: float) -> np.ndarray:
        """pts_xy: (N,2), yaw in rad"""
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s],
                      [s, c]], dtype=np.float32)
        return pts_xy @ R.T

    def _draw_square_x_y(self, center_xy, half_extent_m, yaw_rad=0.0,
                         color=(0, 1, 0), line_width=2, life_time=0.0, z=0.03):
        """
        PyBullet 좌표계: (x,y)가 바닥 평면, z가 위.
        center_xy: (2,) in meters
        half_extent_m: e.g., 3.5 for 7m window
        yaw_rad: 0이면 월드축 정렬(map0), facings면 회전(map1)
        """
        cx, cy = float(center_xy[0]), float(center_xy[1])
        h = float(half_extent_m)

        corners = np.array([
            [-h, -h],
            [h, -h],
            [h, h],
            [-h, h],
        ], dtype=np.float32)

        rc = self._rot2d(corners, yaw_rad)
        rc[:, 0] += cx
        rc[:, 1] += cy

        for i in range(4):
            p0 = [float(rc[i, 0]), float(rc[i, 1]), z]
            p1 = [float(rc[(i + 1) % 4, 0]), float(rc[(i + 1) % 4, 1]), z]
            self._p.addUserDebugLine(
                p0, p1,
                lineColorRGB=color,
                lineWidth=line_width,
                lifeTime=life_time
            )

    '''def render(self, xyzs, facings, root_xzs, lfoot, rfoot, time_remain, action=None):
        x, z, y = extract_joints_xyz(xyzs)
        num_jnt = x.shape[-1]

        mat = self.env.get_rotation_matrix(facings).to(self.device)

        rotated_xy = torch.matmul(mat[...,None,:,:].expand(-1,num_jnt,-1,-1), torch.stack((x, y), dim=-1)[...,None])[...,0]
        #rotated_xy *= 0
        
        poses = torch.cat((rotated_xy, z[...,None]), dim=-1).permute(0, 2, 1)
        root_xyzs = F.pad(root_xzs, pad=[0, 1])
        lfoot_world = F.pad(lfoot, pad=[0, 1]) * FOOT2METER
        rfoot_world = F.pad(rfoot, pad=[0, 1]) * FOOT2METER

        #root_xyzs *= 0
        joint_xyzs_foot = (poses + root_xyzs.unsqueeze(dim=-1))
        joint_xyzs = joint_xyzs_foot * FOOT2METER

        l_toe_jidx, r_toe_jidx = 4, 8
        ground_y = 0.00
        h_thresh = 0.04  # 4cm
        v_thresh = 0.02  # 2cm/frame
        d_thresh = 0.8

        lf = joint_xyzs[:, :, l_toe_jidx]  # (B,3)
        rf = joint_xyzs[:, :, r_toe_jidx]  # (B,3)

        lf_foot = joint_xyzs_foot[:,:, l_toe_jidx]
        rf_foot = joint_xyzs_foot[:,:, r_toe_jidx]

        self.env.world_lfoot = lf_foot[:,:2].detach()
        self.env.world_rfoot = rf_foot[:,:2].detach()

        if self._prev_lf is not None:
            lf_vel = torch.norm(lf[:, :2] - self._prev_lf[:, :2], dim=-1)
            rf_vel = torch.norm(rf[:, :2] - self._prev_rf[:, :2], dim=-1)
        else:
            lf_vel = torch.zeros_like(lf[:,0])
            rf_vel = torch.zeros_like(rf[:,0])

        lf_dist = torch.norm(lf[:, :2] - lfoot_world[:, :2], dim=-1)
        rf_dist = torch.norm(rf[:, :2] - rfoot_world[:, :2], dim=-1)

        # contact 조건
        lf_contact = ((lf[:, 2] - ground_y).abs() < h_thresh) & (lf_vel < v_thresh) & (lf_dist < d_thresh)
        rf_contact = ((rf[:, 2] - ground_y).abs() < h_thresh) & (rf_vel < v_thresh) & (rf_dist < d_thresh)

        # batch 없이 bool 하나로 저장
        self.l_contact = bool(lf_contact[0].item())
        self.r_contact = bool(rf_contact[0].item())

        self.env.lcontact = bool(self.l_contact)
        self.env.rcontact = bool(self.r_contact)

        self._prev_lf = lf.detach()
        self._prev_rf = rf.detach()

        self.root_xyzs = (
            (F.pad(root_xzs, pad=[0, 1], value=3) * FOOT2METER).cpu().numpy()
        )

        self.joint_xyzs = joint_xyzs.cpu().numpy()

        for index in range(self.num_characters):
            self.characters.set_joint_positions_withoutpolar(joint_xyzs[index].cpu().numpy(), index, lfoot_world, rfoot_world)

            if not hasattr(self, "_trail_t"):
                self._trail_t = 0

            # 너무 촘촘하면 느리니 stride로 샘플링
            if (self._trail_t % self.trail_stride) == 0:
                # 한 캐릭터만 (원하면 for index in range(B)로 확장)
                idx = self.character_index if hasattr(self, "character_index") else 0
                pts = joint_xyzs[idx].detach().cpu().numpy()  # (3,J)

                T = self.env.max_timestep
                u = min(self._trail_t / max(T - 1.0, 1.0), 1.0)
                rgb = [0.0, float(u), float(1.0 - u)]

                # 링크 리스트: sk_dict['links']가 (parent, child) 인덱스 쌍이라고 가정
                for (a, b) in self.env.sk_dict["links"]:
                    p1 = pts[:, a]  # (3,)
                    p2 = pts[:, b]
                    # lifeTime=0 => 영구 누적
                    line_id = self._p.addUserDebugLine(
                        p1, p2,
                        lineColorRGB=rgb,
                        lineWidth=self.trail_width,
                        lifeTime=0
                    )
                    # 나중에 지울 필요 있으면 저장
                    if len(self._trail_ids) < self.max_trail_lines:
                        self._trail_ids.append(line_id)

            self._trail_t += 1

            if not hasattr(self, "_footstep_ids"):
                self._footstep_ids = []

            idx = self.character_index if hasattr(self, "character_index") else 0

            eps = 0.00  # z-fighting 방지용

            # --- 왼발 onset ---
            if (not self._prev_l_contact) and self.l_contact:
                p = lf[idx].detach().cpu().numpy().copy()
                p[2] = eps
                self._draw_filled_foot_ellipse(
                    center=p,
                    a=0.13,
                    b=0.06,
                    color=(0.8, 0.0, 0.0),
                    yaw=float(facings),
                    n_rings=8
                )

            # --- 오른발 onset ---
            if (not self._prev_r_contact) and self.r_contact:
                p = rf[idx].detach().cpu().numpy().copy()
                p[2] = eps
                self._draw_filled_foot_ellipse(
                    center=p,
                    a=0.13,
                    b=0.06,
                    color=(0.0, 0.0, 0.8),
                    yaw=float(facings),
                    n_rings = 8
                )

            # 이전 상태 업데이트
            self._prev_l_contact = self.l_contact
            self._prev_r_contact = self.r_contact

            if self.debug and index == self.character_index:
                target_dist = (
                    -float(self.env.linear_potential[index])
                    if hasattr(self.env, "linear_potential")
                    else 0
                )
                print(
                    "FPS: {:4.1f} | Time Left: {:4.1f} | Distance: {:4.1f} ".format(
                        self.camera._fps, float(time_remain), target_dist
                    )
                )
                if action is not None:
                    a = action[index]
                    print(
                        "max: {:4.2f} | mean: {:4.2f} | median: {:4.2f} | min: {:4.2f}".format(
                            float(a.max()),
                            float(a.mean()),
                            float(a.median()),
                            float(a.min()),
                        )
                    )

        self._handle_mouse_press()
        self._handle_key_press()
        if self.use_params:
            self._handle_parameter_update()
        if self.camera_tracking:
            self.camera.track(self.root_xyzs[self.character_index], self.camera_smooth)
        else:
            self.camera.wait()'''


    def render(self, xyzs, facings, root_xzs, time_remain, action):
        x, z, y = extract_joints_xyz(xyzs)
        num_jnt = x.shape[-1]

        mat = self.env.get_rotation_matrix(facings).to(self.device)

        rotated_xy = torch.matmul(mat[..., None, :, :].expand(-1, num_jnt, -1, -1), torch.stack((x, y), dim=-1)[..., None])[
        ..., 0]
        # rotated_xy *= 0

        poses = torch.cat((rotated_xy, z[..., None]), dim=-1).permute(0, 2, 1)
        root_xyzs = F.pad(root_xzs, pad=[0, 1])
        # root_xyzs *= 0
        joint_xyzs = ((poses + root_xyzs.unsqueeze(dim=-1)) * FOOT2METER).cpu().numpy()
        self.root_xyzs = (
            (F.pad(root_xzs, pad=[0, 1], value=3) * FOOT2METER).cpu().numpy()
        )
        self.joint_xyzs = joint_xyzs

        for index in range(self.num_characters):
            self.characters.set_joint_positions(joint_xyzs[index], index)

            if self.debug and index == self.character_index:
                target_dist = (
                    -float(self.env.linear_potential[index])
                    if hasattr(self.env, "linear_potential")
                    else 0
                )
                print(
                    "FPS: {:4.1f} | Time Left: {:4.1f} | Distance: {:4.1f} ".format(
                        self.camera._fps, float(time_remain), target_dist
                    )
                )
                if action is not None:
                    a = action[index]
                    print(
                        "max: {:4.2f} | mean: {:4.2f} | median: {:4.2f} | min: {:4.2f}".format(
                            float(a.max()),
                            float(a.mean()),
                            float(a.median()),
                            float(a.min()),
                        )
                    )

        self._handle_mouse_press()
        self._handle_key_press()
        if self.use_params:
            self._handle_parameter_update()
        if self.camera_tracking:
            self.camera.track(self.root_xyzs[self.character_index], self.camera_smooth)
        else:
            self.camera.wait()

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

        max_frame_skip = 1#self.env.num_future_predictions
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

            #now_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            now_string = "footstep_only"
            filename = "{}.mp4".format(now_string)

            self._p.startStateLogging(self._p.STATE_LOGGING_VIDEO_MP4, filename)
        elif keys.get(ord(" ")) == RELEASED:
            while True:
                keys = self._p.getKeyboardEvents()
                if keys.get(ord(" ")) == RELEASED:
                    break
                elif keys.get(ord("a")) == RELEASED or keys.get(ord("i")) == RELEASED:
                    self._handle_key_press(keys)

class MultiMocapCharacters:
    def __init__(self, bc, num_characters, sk_dict, colours=None, links=True):
        self._p = bc
        self.has_links = links

        #self.dir_link  = 
        #            VCapsule(self._p, radius=0.06, height=0.1, rgba=colours[i])
        #            for i in range(num_characters)
        
        if links:
            self.linked_joints = np.array(sk_dict['links'])
            
            self.head_idx =  sk_dict['head_idx']
            self.num_joint =   sk_dict['num_joint']

            total_parts = num_characters * self.num_joint
            joints = VSphere(bc, radius=0.07, max=True, replica=total_parts)
            self.ids = joints.ids
            self.links = {
                i: [
                    VCapsule(self._p, radius=0.03, height=0.1, rgba=colours[i])
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
        joint_ids = self.ids[start : start + self.num_joint]
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
                    link = VCapsule(self._p, radius=0.06, height=height, rgba=rgba)
                    self.links[index][lid] = link

                link.set_position(pos, orn)

            self.heads[index].set_position(
                0.5 * (xyzs[self.head_idx[1]] - xyzs[self.head_idx[0]]) + xyzs[self.head_idx[1]])
            # self.dir_link.set_position
        self._p.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)

    def set_joint_positions_withoutpolar(self, xyzs, index, lfoot_world, rfoot_world):

        self._p.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)

        start = index * self.num_joint
        joint_ids = self.ids[start: start + self.num_joint]

        xyzs = xyzs.transpose()
        for i, id in enumerate(joint_ids):
            self._p.resetBasePositionAndOrientation(id, posObj=xyzs[i], ornObj=(0, 0, 0, 1))
            # if i==4 :
            # self._p.changeVisualShape(id, -1, rgbaColor=(0,0,1,1))

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
                    link = VCapsule(self._p, radius=0.06, height=height, rgba=rgba)
                    self.links[index][lid] = link

                link.set_position(pos, orn)

            self.heads[index].set_position(
                0.5 * (xyzs[self.head_idx[1]] - xyzs[self.head_idx[0]]) + xyzs[self.head_idx[1]])
            # self.dir_link.set_position

            if not hasattr(self, "lfoot_marker"):
                sphere_shape = self._p.createVisualShape(pb.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1])
                self.lfoot_marker = self._p.createMultiBody(baseVisualShapeIndex=sphere_shape, basePosition=[0, 0, 0])

            if not hasattr(self, "rfoot_marker"):
                sphere_shape = self._p.createVisualShape(pb.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 0, 1, 1])
                self.rfoot_marker = self._p.createMultiBody(baseVisualShapeIndex=sphere_shape, basePosition=[0, 0, 0])

            lfoot_pos3d = [lfoot_world[:, 0], lfoot_world[:, 1], 0.0]
            rfoot_pos3d = [rfoot_world[:, 0], rfoot_world[:, 1], 0.0]

            # 위치 업데이트
            self._p.resetBasePositionAndOrientation(self.lfoot_marker, posObj=lfoot_pos3d, ornObj=(0, 0, 0, 1))
            self._p.resetBasePositionAndOrientation(self.rfoot_marker, posObj=rfoot_pos3d, ornObj=(0, 0, 0, 1))

        self._p.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)

    def set_joint_positions_without_footstep(self, xyzs, index):

        self._p.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)

        start = index * self.num_joint
        joint_ids = self.ids[start: start + self.num_joint]

        xyzs = xyzs.transpose()
        for i, id in enumerate(joint_ids):
            self._p.resetBasePositionAndOrientation(id, posObj=xyzs[i], ornObj=(0, 0, 0, 1))
            # if i==4 :
            # self._p.changeVisualShape(id, -1, rgbaColor=(0,0,1,1))

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
                    link = VCapsule(self._p, radius=0.06, height=height, rgba=rgba)
                    self.links[index][lid] = link

                link.set_position(pos, orn)

            self.heads[index].set_position(
                0.5 * (xyzs[self.head_idx[1]] - xyzs[self.head_idx[0]]) + xyzs[self.head_idx[1]])

        self._p.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)


class MultiTargets:
    def __init__(self, bc, num_characters, colours=None, obj_class=VSphere):
        self._p = bc
        self.marker = obj_class

        # self.start_index = self._p.getNumBodies()
        if self.marker is FlagPole:
            flags = obj_class(self._p, replica= num_characters)
        else :
            flags = obj_class(self._p, radius= 0.04, replica=num_characters)
        self.ids = flags.ids

        if colours is not None:
            for index, colour in zip(range(num_characters), colours):
                if index == 0:
                    colour = [1,1,1,1]
                self.set_colour(colour, index)


        self.target_gnd = VSphere(self._p, radius=0.04, rgba=(colour[0],colour[1],colour[2], 1), max=True, replica=1)
        self.target_ed = VSphere(self._p, radius=0.04, rgba=(colour[0],colour[1],colour[2],1), max=True, replica=10)

    def set_colour(self, colour, index):
        self._p.changeVisualShape(self.ids[index], -1, rgbaColor=(colour[0],colour[1],colour[2],1))

    def set_position(self, xyz, index, orn=(1, 0, 0, 1)):
        xyz = xyz[:3]
        self._p.resetBasePositionAndOrientation(self.ids[index], posObj=xyz, ornObj=orn)
        
        num_sph = int(xyz[2]/0.2)
        xyz[2] = 0.0
        self._p.resetBasePositionAndOrientation(self.target_gnd.ids[0], posObj=xyz, ornObj=orn)
        
        for i in range(10):
            pos = xyz
            if i > num_sph:
                pos[2] = num_sph * 0.2
            else:
                pos[2] = i * 0.2
            self._p.resetBasePositionAndOrientation(self.target_ed.ids[i], posObj=pos, ornObj=orn)


class MocapCharacter:
    def __init__(self, bc, rgba=None):

        self._p = bc
        num_joints = self._p.num_joint

        # useMaximalCoordinates=True is faster for things that don't `move`
        body = VSphere(bc, radius=0.07, rgba=rgba, max=True, replica=num_joints)
        self.joint_ids = body.ids

    def set_joint_positions(self, xyzs):
        for xyz, id in zip(xyzs, self.joint_ids):
            self._p.resetBasePositionAndOrientation(id, posObj=xyz, ornObj=(0, 0, 0, 1))


class MocapCharacterTarget:
    def __init__(self, bc, num_characters, body_data = None, colours=None, obj_class=None):
        assert body_data is not None
        self._p = bc
        num_joints = self._p.num_joint

        # useMaximalCoordinates=True is faster for things that don't `move`
        body = VSphere(bc, radius=0.07, rgba=colours[0], max=True, replica=num_joints*num_characters)
        self.joint_ids = body.ids
        self.joints = body_data
        self.num_joints = num_joints


        self.links = {
            i: [
                VCapsule(self._p, radius=0.06, height=0.1, rgba=colours[i])
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
                self.set_colour(colour, index)
                self.heads[index].set_color(colour)

    def set_colour(self, colour, index):
        # start = self.start_index + index * self.num_joints
        start = index * self.num_joints
        joint_ids = self.joint_ids[start : start + self.num_joints]
        for id in joint_ids:
            self._p.changeVisualShape(id, -1, rgbaColor=colour)


    def set_position(self, xyz, index, orn=(1, 0, 0, 1)):
        #xyz[2] = 0
        self.joints = self.joints + xyz[:3]
        self.set_joint_positions(self.joints,index)
    
    def set_joint_positions(self, xyzs, index):
        self._p.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)

        start = index * self.num_joints
        joint_ids = self.joint_ids[start : start + self.num_joints]
        for xyz, id in zip(xyzs, joint_ids):
            self._p.resetBasePositionAndOrientation(id, posObj=xyz, ornObj=(0, 0, 0, 1))

    