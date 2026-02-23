
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    Point2, Point3,
)

import math


class OrbitCameraController:
    """
    Custom trackball-like camera controller (stable).
    Controls:
      - Left drag: orbit
      - Right drag: pan
      - Wheel: zoom
      - R: reset
    """

    def __init__(self, base: ShowBase):
        self.base = base

        # target point to look at
        self.reset()

        self._orbiting = False
        self._panning = False
        self._last_mouse = None

        self._pan_scale = 2.5

        base.accept("mouse1", self._start_orbit)
        base.accept("mouse1-up", self._stop_orbit)

        base.accept("mouse2", self._start_pan)
        base.accept("mouse2-up", self._stop_pan)

        #base.accept("mouse3", self._start_pan)
        #base.accept("mouse3-up", self._stop_pan)

        base.accept("wheel_up", self._zoom, [-1.0])
        base.accept("wheel_down", self._zoom, [1.0])

        base.accept("r", self.reset)


    def reset(self):
        self.target = Point3(0, 0, 0)
        self.dist = 12.0
        self.yaw = 45.0
        self.pitch = 25.0


    def _start_orbit(self):
        if self.base.mouseWatcherNode.hasMouse():
            self._orbiting = True
            m = self.base.mouseWatcherNode.getMouse()
            self._last_mouse = (m.getX(), m.getY())

    def _stop_orbit(self):
        self._orbiting = False
        self._last_mouse = None

    def _start_pan(self):
        if self.base.mouseWatcherNode.hasMouse():
            self._panning = True
            m = self.base.mouseWatcherNode.getMouse()
            self._last_mouse = (m.getX(), m.getY())

    def _stop_pan(self):
        self._panning = False
        self._last_mouse = None

    def _zoom(self, delta):
        self.dist = float(max(1.0, min(150.0, self.dist + delta)))

    def _mouse_to_world_on_z_plane(self, mx: float, my: float, z_plane: float):
        """
        Screen-space mouse (mx,my in [-1,1]) -> world position on plane z=z_plane.
        Returns Point3 or None (if ray parallel to plane).
        """
        lens = self.base.camLens

        near = Point3()
        far = Point3()
        lens.extrude(Point2(mx, my), near, far)

        # near/far are in camera space -> convert to world(render) space
        near_w = self.base.render.getRelativePoint(self.base.camera, near)
        far_w = self.base.render.getRelativePoint(self.base.camera, far)

        d = far_w - near_w
        if abs(d.z) < 1e-8:
            return None

        t = (z_plane - near_w.z) / d.z
        return near_w + d * t

    def _update_orbit_pan(self):
        if not self.base.mouseWatcherNode.hasMouse():
            return
        mx, my = self.base.mouseWatcherNode.getMouse().getX(), self.base.mouseWatcherNode.getMouse().getY()

        if self._last_mouse is None:
            self._last_mouse = (mx, my)
            return

        dx = mx - self._last_mouse[0]
        dy = my - self._last_mouse[1]


        if self._orbiting:
            self.yaw += dx * 180.0
            self.pitch += dy * 90.0
            self.pitch = float(max(0.0, min(60.0, self.pitch)))

        #if self._panning:
            #right = self.base.camera.getQuat().getRight()
            #up = self.base.camera.getQuat().getUp()

            #s = self._pan_scale * (self.dist / 12.0)
            #self.target += (-right * dx * s) + (-up * dy * s)

        if self._panning:
            z_plane = self.target.z

            p_last = self._mouse_to_world_on_z_plane(self._last_mouse[0], self._last_mouse[1], z_plane)
            p_curr = self._mouse_to_world_on_z_plane(mx, my, z_plane)

            if p_last is not None and p_curr is not None:
                # "바닥을 잡고 끄는" 느낌: target을 (last - curr)만큼 이동
                self.target += (p_last - p_curr)

        self._last_mouse = (mx, my)



    def update_camera(self):
        self._update_orbit_pan()

        yaw = math.radians(self.yaw)
        pitch = math.radians(self.pitch)

        x = self.dist * math.cos(pitch) * math.sin(yaw)
        y = -self.dist * math.cos(pitch) * math.cos(yaw)
        z = self.dist * math.sin(pitch)

        cam_pos = Point3(self.target.x + x, self.target.y + y, self.target.z + z)
        self.base.camera.setPos(cam_pos)
        self.base.camera.lookAt(self.target)