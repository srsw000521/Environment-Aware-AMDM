# mocap_renderer_panda3d.py
# Panda3D mocap viewer:
# (A) current frame: spheres (joints) + cylinders (bones)
# (B) past frames: line trail
# (C) footstep onset: filled ellipse footprints
#
# Install: pip install panda3d
# Usage:
#   viewer = MocapPandaViewer(env=env)   # env optional but recommended for update_from_env()
#   viewer.update_from_env()             # call after env.calc_env_state(...) updated env.history/root_facing/root_xz
#   viewer.run()

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import (
    WindowProperties,
    PerspectiveLens,
    CardMaker,
    NodePath,
    Texture,
    PNMImage,
    Vec3,
    Point3,
    LineSegs,
    AmbientLight, DirectionalLight, BitMask32,
    Vec3,
)

from direct.gui.OnscreenText import OnscreenText
from render.realtime.geom_cylinder import ProceduralCylinder
from render.realtime.geom_camera_controller import OrbitCameraController

import math
import numpy as np

# optional: torch is only needed when you call update_from_env() with env
try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None


FOOT2METER = 1.3  # same constant used in pybullet viewer


def make_checker_texture(
    width=512,
    height=512,
    tile_px=32,
    c0=(130, 150, 180),
    c1=(200, 200, 200),
):
    """Create a procedural checkerboard texture (no external files)."""
    img = PNMImage(width, height)
    for y in range(height):
        for x in range(width):
            tx = x // tile_px
            ty = y // tile_px
            use0 = ((tx + ty) % 2 == 0)
            r, g, b = c0 if use0 else c1
            img.setXelVal(x, y, r, g, b)

    tex = Texture("checker")
    tex.load(img)
    tex.setMinfilter(Texture.FTLinearMipmapLinear)
    tex.setMagfilter(Texture.FTLinear)
    return tex





class MocapPandaViewer(ShowBase):
    """
    - call viewer.update_from_env() after env.calc_env_state(...) updates env.history/root_facing/root_xz
    - or call viewer.render_pose(joint_xyzs_3xJ, facings_rad) directly if you already have world joints
    """

    def __init__(self, env=None):
        super().__init__()

        # IMPORTANT: disable Panda3D default mouse trackball
        self.disableMouse()

        self.env = env  # optional

        props = WindowProperties()
        props.setTitle("Mocap Viewer (Panda3D)")
        self.win.requestProperties(props)

        self.setBackgroundColor(0.03, 0.03, 0.05, 1)
        self.setFrameRateMeter(True)

        # Camera lens: perspective FOV 60
        lens = PerspectiveLens()
        lens.setFov(60)
        lens.setNearFar(0.01, 2000)
        self.cam.node().setLens(lens)

        # Ground plane
        self.ground = self._create_checker_ground(size=50.0, tex_repeat=20)
        self.ground.hide(BitMask32.bit(1))
        self.ground.reparentTo(self.render)


        # Custom camera controller
        self.cam_ctl = OrbitCameraController(self)
        self.cam_ctl.target = Point3(0, 0, 0)
        self.cam_ctl.dist = 18.0

        # Optional axis helper (XYZ)
        self.axis = self._create_axis(length=2.0, thickness=3.0)
        self.axis.hide(BitMask32.bit(1))
        self.axis.reparentTo(self.render)

        # ----------------------------
        # Lighting (simple shading)
        # ----------------------------
        self._setup_lighting()

        self.cam_text = OnscreenText(
            text="",
            pos=(-1.32, 0.92),
            scale=0.045,
            fg=(1, 1, 0.3, 1),
            align=0,
            mayChange=True
        )

        self.paused = False
        self._step_once = False

        self.accept("space", self._toggle_pause)
        self.accept("n", self._request_step_once)

        # (선택) 상태 표시
        self.pause_text = OnscreenText(
            text="",
            pos=(1.25, 0.92),
            scale=0.05,
            fg=(1, 0.6, 0.2, 1),
            align=2,
            mayChange=True
        )

        # ----------------------------
        # (A) current frame geometry
        # ----------------------------


        self._skel_inited = False
        self._skel_root = self.render.attachNewNode("character_root")


        self._joint_nodes = []
        self._bone_wrappers = []  # wrapper NodePaths
        self._bone_geoms = []

        self._links = None
        self._num_joints = None

        self.joint_radius = 0.035  # 기존 0.07의 절반
        self.bone_thickness = 2.0 * self.joint_radius  # sphere 지름과 동일(연결된 느낌)

        self._sphere_model = None
        self._cyl_model = None

        # ----------------------------
        # (B) trail lines
        # ----------------------------
        self._trail_root = self.render.attachNewNode("trail_root")
        self._trail_root.hide(BitMask32.bit(1))                 # 그림자 on/off

        self._trail_nodes = []
        self.trail_stride = 3
        self.max_trail_frames = 200
        self.trail_thickness = 2.0
        self._trail_t = 0

        # ----------------------------
        # (C) footprints
        # ----------------------------
        self._foot_root = self.render.attachNewNode("footprints_root")
        self._prev_lf = None
        self._prev_rf = None
        self._prev_l_contact = False
        self._prev_r_contact = False
        self.l_contact = False
        self.r_contact = False

        # same toe indices used in pybullet viewer
        self.l_toe_jidx = 4
        self.r_toe_jidx = 8

        # Update loop (camera + HUD)
        self.taskMgr.add(self._update, "update")
        self.accept("window-event", self._on_window_event)
        self._on_window_event(self.win)  # 초기 1회 적용

    def _on_window_event(self, win):
        if win is None or win.isClosed():
            # ShowBase의 정상 종료 루틴(태스크/윈도우 정리)
            self.userExit()
            return

        w = win.getXSize()
        h = win.getYSize()
        if h <= 0:
            return

        aspect = w / float(h)

        lens = self.cam.node().getLens()
        lens.setAspectRatio(aspect)

    def _toggle_pause(self):
        self.paused = not self.paused
        # pause를 풀면 step_once 요청은 무의미하니 초기화(선택)
        if not self.paused:
            self._step_once = False

    def _request_step_once(self):
        # pause 상태에서만 1스텝 요청
        if self.paused:
            self._step_once = True

    def consume_step_permission(self) -> bool:
        """
        step_env()에서 호출:
        - paused=False면 항상 True
        - paused=True면 _step_once가 True일 때만 True를 주고, 즉시 False로 소모
        """
        if not self.paused:
            return True
        if self._step_once:
            self._step_once = False
            return True
        return False

    def is_paused(self) -> bool:
        return bool(self.paused)

    def _setup_lighting(self):
        # 기존 light가 있다면 제거(재시작/리로드 대비)
        self.render.clearLight()

        # 약한 ambient (완전 새까맣게 되는 것 방지)
        alight = AmbientLight("ambient")
        alight.setColor((0.55, 0.55, 0.55, 1.0))
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)

        # Directional light
        dlight = DirectionalLight("dir")
        dlight.setColor((1.0, 1.0, 1.0, 1.0))
        dlight.setShadowCaster(True, 2048, 2048)
        dlight.setCameraMask(BitMask32.bit(1))

        dlnp = self.render.attachNewNode(dlight)

        # (중요) directional light는 "어디서 비추는 위치"가 아니라 "방향"이 핵심입니다.
        # Panda3D에서는 light 노드의 forward 방향(-Y)이 빛의 방향으로 사용됩니다.
        #
        # 우리가 원하는 빛 방향: (1,1,-1)
        # 즉, 그 방향으로 "날아오는" 빛.
        #
        # 따라서 노드를 그 반대쪽(-1,-1,1)에 두고 origin을 바라보게 하면 쉽게 맞출 수 있습니다.
        dlnp.setPos(-10, -10, 10)  # (-1,-1,1) 방향에 위치
        dlnp.lookAt(0, 0, 0)

        lens = dlight.getLens()
        lens.setNearFar(1.0, 120.0)  # 장면 크기에 맞춰 조절
        lens.setFilmSize(50, 50)

        self.render.setLight(dlnp)

        # 모델들 shading 활성화
        self.render.setShaderAuto()



    # -----------------------
    # Public API
    # -----------------------
    def update_from_env(self):
        """
        Reproduce the same pipeline as env.render() + pybullet MocapViewer.render():
          frame = dataset.denorm_data(history[:,0])
          xyzs  = dataset.x_to_jnts(frame, mode='angle')
          then compute world joint xyzs using facings/root_xz and FOOT2METER.
        """
        if self.env is None:
            raise RuntimeError("viewer.env is None. Create viewer = MocapPandaViewer(env=env)")

        if torch is None or F is None:
            raise RuntimeError("torch is required for update_from_env(). Please install torch, or call render_pose() directly.")

        env = self.env
        device = getattr(env, "device", None)

        # Build xyzs exactly like env.render() did
        denorm = env.dataset.denorm_data(env.history[:, 0], device=device).cpu().numpy()
        xyzs = torch.tensor(
            env.dataset.x_to_jnts(denorm, mode="angle"),
            device=device,
            dtype=env.history.dtype
        )

        facings = env.root_facing
        root_xzs = env.root_xz

        joint_xyzs, facings0 = self._compute_world_joint_xyzs(xyzs, facings, root_xzs)  # (B,3,J)
        pts_3xJ = joint_xyzs[0].detach().cpu().numpy()
        pts_Jx3 = pts_3xJ.T

        # Initialize geometry lazily
        if not self._skel_inited:
            # links
            links = None
            if hasattr(env, "sk_dict") and "links" in env.sk_dict:
                links = env.sk_dict["links"]
            elif hasattr(env.dataset, "sk_dict") and "links" in env.dataset.sk_dict:
                links = env.dataset.sk_dict["links"]
            elif hasattr(env.dataset, "links"):
                links = env.dataset.links
            if links is None:
                # fallback chain
                links = [(i, i + 1) for i in range(pts_Jx3.shape[0] - 1)]
            self._init_skeleton_geometry(num_joints=pts_Jx3.shape[0], links=links)

        # (A) update current
        self._update_current_pose(pts_Jx3)

        # (B) trail
        self._trail_t += 1
        if (self._trail_t % self.trail_stride) == 0:
            self._push_trail(pts_Jx3)

        # (C) footprints based on contact onset (same conditions as pybullet viewer)
        self._update_foot_contact_and_footprints(joint_xyzs, facings0)

    def render_pose(self, joint_xyzs_3xJ: np.ndarray, facings_rad: float = 0.0):
        """
        If you already computed world joints (3,J), you can call this directly.
        """
        pts_3xJ = np.asarray(joint_xyzs_3xJ, dtype=np.float32)
        pts_Jx3 = pts_3xJ.T

        if not self._skel_inited:
            # need links from env if possible, else chain
            if self.env is not None and hasattr(self.env, "sk_dict") and "links" in self.env.sk_dict:
                links = self.env.sk_dict["links"]
            else:
                links = [(i, i + 1) for i in range(pts_Jx3.shape[0] - 1)]
            self._init_skeleton_geometry(num_joints=pts_Jx3.shape[0], links=links)

        self._update_current_pose(pts_Jx3)

        self._trail_t += 1
        if (self._trail_t % self.trail_stride) == 0:
            self._push_trail(pts_Jx3)

        # simple footprint mode without env targets:
        self._update_footprints_without_targets(pts_3xJ, facings_rad)

    # -----------------------
    # Internal helpers
    # -----------------------
    def _compute_world_joint_xyzs(self, xyzs, facings, root_xzs):
        """
        Matches pybullet viewer math:
          x,z,y = extract_joints_xyz(xyzs)
          rotated_xy = R(facings) * [x,y]
          poses = [rotated_xy, z]
          joint_xyzs_foot = poses + pad(root_xzs)
          joint_xyzs = joint_xyzs_foot * FOOT2METER
        """
        # xyzs: (B,J,3)
        x = xyzs[..., 0]
        z = xyzs[..., 1]
        y = xyzs[..., 2]
        B, J = x.shape[0], x.shape[1]

        mat = self.env.get_rotation_matrix(facings)  # (B,2,2)

        xy = torch.stack((x, y), dim=-1)  # (B,J,2)
        rotated_xy = torch.matmul(
            mat[..., None, :, :].expand(-1, J, -1, -1),
            xy[..., None]
        )[..., 0]  # (B,J,2)

        poses = torch.cat((rotated_xy, z[..., None]), dim=-1).permute(0, 2, 1)  # (B,3,J)
        root_xyzs = F.pad(root_xzs, pad=[0, 1])  # (B,3)
        joint_xyzs_foot = poses + root_xyzs.unsqueeze(dim=-1)  # (B,3,J)
        joint_xyzs = joint_xyzs_foot * FOOT2METER  # (B,3,J)

        # facings0: a python float (radians)
        fac0 = facings[0] if hasattr(facings, "__len__") else facings
        try:
            fac0 = float(fac0.item())
        except Exception:
            fac0 = float(fac0)

        return joint_xyzs, fac0

    def _init_skeleton_geometry(self, num_joints, links):

        self._skel_root.setShaderAuto()
        self._skel_root.setLightOff(False)

        self._skel_inited = True
        self._num_joints = int(num_joints)
        self._links = [(int(a), int(b)) for (a, b) in list(links)]

        self._sphere_model = self.loader.loadModel("models/misc/sphere")
        #self._cyl_model = self.loader.loadModel("models/box")

        self._bone_template = ProceduralCylinder(circular_div=24, caps=True).create_nodepath("bone_cyl")
        self._bone_template.setTextureOff(1)
        self._bone_template.setColor(0.98, 0.54, 0.20, 1.0)


        # joints
        self._joint_nodes = []
        for j in range(self._num_joints):
            np_j = self._sphere_model.copyTo(self._skel_root)
            np_j.setScale(self.joint_radius)
            #np_j.setColor(1.0, 0.0, 0.0, 1.0)  # 빨간색
            np_j.setLightOff(False)
            np_j.setShaderAuto()
            np_j.setColor(0.98, 0.54, 0.20, 1.0)  # 오렌지

            self._joint_nodes.append(np_j)

        # bones (wrapper + geom)
        self._bone_wrappers = []
        self._bone_models  = []
        self._bone_offsets = []

        for (a, b) in self._links:
            wrapper = self._skel_root.attachNewNode(f"bone_{a}_{b}")
            offset = wrapper.attachNewNode("offset")

            #model = self._cyl_model.copyTo(offset)
            model = self._bone_template.copyTo(offset)
            model.setTextureOff(1)
            model.setColor(0.98, 0.54, 0.20, 1.0)
            model.setShaderAuto()
            model.setLightOff(False)

            #min_pt, max_pt = model.getTightBounds()
            #cx = 0.5 * (min_pt.x + max_pt.x)
            #cz = 0.5 * (min_pt.z + max_pt.z)
            #offset.setPos(-cx, -min_pt.y, -cz)

            self._bone_wrappers.append(wrapper)
            self._bone_offsets.append(offset)
            self._bone_models.append(model)


    def _update_current_pose(self, joints_Jx3: np.ndarray):
        # joints
        for j in range(self._num_joints):
            x, y, z = float(joints_Jx3[j, 0]), float(joints_Jx3[j, 1]), float(joints_Jx3[j, 2])
            self._joint_nodes[j].setPos(x, y, z)

        # bones
        for i, (a, b) in enumerate(self._links):
            p0 = self._joint_nodes[a].getPos(self._skel_root)
            p1 = self._joint_nodes[b].getPos(self._skel_root)

            v = p1 - p0
            L = v.length()
            if L < 1e-6:
                self._bone_wrappers[i].hide()
                continue
            self._bone_wrappers[i].show()

            wrapper = self._bone_wrappers[i]
            wrapper.setPos(p0)



            wrapper.lookAt(self._skel_root, p1, Vec3(0,0,1))

            t = self.bone_thickness
            wrapper.setScale(t/2, L, t/2)


    def _push_trail(self, joints_Jx3: np.ndarray):
        ls = LineSegs("trail")
        ls.setThickness(self.trail_thickness)

        # optional: color gradient by time
        # emulate pybullet commented code: u in [0,1], rgb=[0,u,1-u]
        T = getattr(self.env, "max_timestep", 2000) if self.env is not None else 2000
        u = min(self._trail_t / max(T - 1.0, 1.0), 1.0)
        ls.setColor(0.0, float(u), float(1.0 - u), 1.0)

        for (a, b) in self._links:
            ax, ay, az = joints_Jx3[a]
            bx, by, bz = joints_Jx3[b]
            ls.moveTo(float(ax), float(ay), float(az))
            ls.drawTo(float(bx), float(by), float(bz))

        np_line = self._trail_root.attachNewNode(ls.create())
        self._trail_nodes.append(np_line)

        if len(self._trail_nodes) > self.max_trail_frames:
            old = self._trail_nodes.pop(0)
            old.removeNode()

    # -------- Foot contact + footprints (C) --------
    def _update_foot_contact_and_footprints(self, joint_xyzs_B3J, facings0: float):
        """
        Use the same heuristic as pybullet viewer:
          toe idx: 4,8
          height thresh, vel thresh, dist thresh (to target foot pos)
        If env doesn't provide lfoot/rfoot targets, we fallback so dist=0.
        """
        env = self.env
        B = joint_xyzs_B3J.shape[0]

        l_toe = self.l_toe_jidx
        r_toe = self.r_toe_jidx

        ground_y = 0.00
        h_thresh = 0.04
        v_thresh = 0.02
        d_thresh = 0.8

        lf = joint_xyzs_B3J[:, :, l_toe]  # (B,3)
        rf = joint_xyzs_B3J[:, :, r_toe]  # (B,3)

        # velocity
        if self._prev_lf is not None:
            lf_vel = torch.norm(lf[:, :2] - self._prev_lf[:, :2], dim=-1)
            rf_vel = torch.norm(rf[:, :2] - self._prev_rf[:, :2], dim=-1)
        else:
            lf_vel = torch.zeros_like(lf[:, 0])
            rf_vel = torch.zeros_like(rf[:, 0])

        # target foot pos (world) for distance check
        # if env doesn't have them, set targets equal to current toe pos (=> dist=0)
        lfoot_world = None
        rfoot_world = None

        # common candidates
        if hasattr(env, "lfoot_world") and hasattr(env, "rfoot_world"):
            # already in world units? uncertain. keep safe if shapes match
            pass

        # fallback to zeros but match type
        lfoot_world = lf.detach().clone()
        rfoot_world = rf.detach().clone()

        lf_dist = torch.norm(lf[:, :2] - lfoot_world[:, :2], dim=-1)
        rf_dist = torch.norm(rf[:, :2] - rfoot_world[:, :2], dim=-1)

        lf_contact = ((lf[:, 2] - ground_y).abs() < h_thresh) & (lf_vel < v_thresh) & (lf_dist < d_thresh)
        rf_contact = ((rf[:, 2] - ground_y).abs() < h_thresh) & (rf_vel < v_thresh) & (rf_dist < d_thresh)

        # store as plain bool (single character use)
        self.l_contact = bool(lf_contact[0].item())
        self.r_contact = bool(rf_contact[0].item())

        # onset detection (prev False -> now True)
        eps = 0.002  # avoid z-fighting with ground
        if (not self._prev_l_contact) and self.l_contact:
            p = lf[0].detach().cpu().numpy().copy()
            p[2] = eps
            self._draw_filled_foot_ellipse(
                center=p,
                a=0.13,
                b=0.06,
                color=(0.8, 0.0, 0.0, 1.0),
                yaw=facings0,
                n_rings=8,
                n_seg=24,
            )
        if (not self._prev_r_contact) and self.r_contact:
            p = rf[0].detach().cpu().numpy().copy()
            p[2] = eps
            self._draw_filled_foot_ellipse(
                center=p,
                a=0.13,
                b=0.06,
                color=(0.0, 0.6, 1.0, 1.0),
                yaw=facings0,
                n_rings=8,
                n_seg=24,
            )

        # update prev
        self._prev_l_contact = self.l_contact
        self._prev_r_contact = self.r_contact
        self._prev_lf = lf.detach()
        self._prev_rf = rf.detach()

    def _update_footprints_without_targets(self, pts_3xJ: np.ndarray, facings_rad: float):
        """
        Non-env simple contact heuristic: height+vel only.
        """
        l_toe = self.l_toe_jidx
        r_toe = self.r_toe_jidx

        lf = pts_3xJ[:, l_toe]  # (3,)
        rf = pts_3xJ[:, r_toe]

        # make fake B=1 arrays for vel
        lf2 = lf[:2]
        rf2 = rf[:2]

        if self._prev_lf is not None:
            plf = self._prev_lf[0].detach().cpu().numpy()
            prf = self._prev_rf[0].detach().cpu().numpy()
            lf_vel = np.linalg.norm(lf2 - plf[:2])
            rf_vel = np.linalg.norm(rf2 - prf[:2])
        else:
            lf_vel = 0.0
            rf_vel = 0.0

        h_thresh = 0.04
        v_thresh = 0.02
        ground_y = 0.00

        l_contact = (abs(lf[2] - ground_y) < h_thresh) and (lf_vel < v_thresh)
        r_contact = (abs(rf[2] - ground_y) < h_thresh) and (rf_vel < v_thresh)

        eps = 0.002
        if (not self._prev_l_contact) and l_contact:
            p = lf.copy()
            p[2] = eps
            self._draw_filled_foot_ellipse(p, a=0.13, b=0.06, color=(0.8, 0.0, 0.0, 1.0), yaw=facings_rad, n_rings=8)
        if (not self._prev_r_contact) and r_contact:
            p = rf.copy()
            p[2] = eps
            self._draw_filled_foot_ellipse(p, a=0.13, b=0.06, color=(0.0, 0.6, 1.0, 1.0), yaw=facings_rad, n_rings=8)

        # update prev via torch-like placeholders
        if torch is not None:
            self._prev_lf = torch.tensor([[lf[0], lf[1], lf[2]]])
            self._prev_rf = torch.tensor([[rf[0], rf[1], rf[2]]])
        self._prev_l_contact = bool(l_contact)
        self._prev_r_contact = bool(r_contact)

    def _draw_filled_foot_ellipse(self, center, a=0.13, b=0.06, color=(0, 1, 0, 1), yaw=None, n_rings=6, n_seg=24):
        """
        Panda3D version of pybullet _draw_filled_foot_ellipse (z-up).
        Ellipse major axis is along +Y (foot forward). Yaw rotates around Z.
        """
        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])

        if yaw is None:
            c, s = 1.0, 0.0
        else:
            c, s = math.cos(float(yaw)), math.sin(float(yaw))

        ls = LineSegs("footprint")
        ls.setThickness(2.0)
        ls.setColor(float(color[0]), float(color[1]), float(color[2]), float(color[3]))

        # draw multiple inner ellipses to look filled
        for k in range(1, n_rings + 1):
            t = k / n_rings
            ak = a * t
            bk = b * t

            for i in range(n_seg):
                th0 = 2 * math.pi * i / n_seg
                th1 = 2 * math.pi * (i + 1) / n_seg

                # ellipse: x=b*sin, y=a*cos  (major along +Y)
                x0, y0 = bk * math.sin(th0), ak * math.cos(th0)
                x1, y1 = bk * math.sin(th1), ak * math.cos(th1)

                # yaw rotation
                X0 = c * x0 - s * y0
                Y0 = s * x0 + c * y0
                X1 = c * x1 - s * y1
                Y1 = s * x1 + c * y1

                ls.moveTo(cx + X0, cy + Y0, cz)
                ls.drawTo(cx + X1, cy + Y1, cz)

        np_fp = self._foot_root.attachNewNode(ls.create())
        # footprints are persistent; if you want to limit count, manage a list + pop/removeNode()

    # -----------------------
    # Scene primitives
    # -----------------------
    def _create_checker_ground(self, size=10.0, tex_repeat=4) -> NodePath:
        cm = CardMaker("ground")
        cm.setFrame(-size, size, -size, size)  # card is in XZ by default
        ground = NodePath(cm.generate())

        # rotate to XY plane: normal +Z
        ground.setP(-90)
        ground.setPos(0, 0, 0)
        ground.setTwoSided(True)

        tex = make_checker_texture(width=512, height=512, tile_px=64)
        ground.setTexture(tex)

        from panda3d.core import TextureStage
        ground.setTexScale(TextureStage.getDefault(), tex_repeat, tex_repeat)
        return ground

    def _create_axis(self, length=1.0, thickness=2.0) -> NodePath:
        ls = LineSegs("axis")
        ls.setThickness(thickness)

        ls.setColor(1, 0, 0, 1)
        ls.moveTo(0, 0, 0)
        ls.drawTo(length, 0, 0)

        ls.setColor(0, 1, 0, 1)
        ls.moveTo(0, 0, 0)
        ls.drawTo(0, length, 0)

        ls.setColor(0, 0, 1, 1)
        ls.moveTo(0, 0, 0)
        ls.drawTo(0, 0, length)

        return NodePath(ls.create())

    # -----------------------
    # Per-frame update (camera + HUD only)
    # -----------------------
    def _update(self, task: Task):
        self.cam_ctl.update_camera()

        yaw = self.cam_ctl.yaw
        pitch = self.cam_ctl.pitch
        dist = self.cam_ctl.dist
        tx, ty, tz = self.cam_ctl.target

        self.cam_text.setText(
            f"yaw  : {yaw:7.2f}\n"
            f"pitch: {pitch:7.2f}\n"
            f"dist : {dist:7.2f}\n"
            f"tgt  : ({tx:5.2f}, {ty:5.2f}, {tz:5.2f})"
        )

        self.pause_text.setText("PAUSED" if self.paused else "")

        return Task.cont


if __name__ == "__main__":
    viewer = MocapPandaViewer(env=None)
    viewer.run()
