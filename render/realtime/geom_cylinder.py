# render/realtime/geom_cylinder.py
from panda3d.core import (
    Geom, GeomNode, GeomVertexData, GeomVertexFormat,
    GeomVertexWriter, GeomTriangles, NodePath
)
import math


class ProceduralCylinder:
    """
    Y-axis cylinder:
      - radius = 1 on XZ plane
      - height = 1 along +Y
      - base center at (0,0,0), top center at (0,1,0)

    Args:
      circular_div: number of segments around the circle (>=3 recommended)
      caps: whether to create top/bottom caps
    """
    def __init__(self, circular_div: int = 24, caps: bool = True):
        self.circular_div = max(3, int(circular_div))
        self.caps = bool(caps)

    def create_nodepath(self, name: str = "procedural_cylinder") -> NodePath:
        fmt = GeomVertexFormat.getV3n3()
        vdata = GeomVertexData(name, fmt, Geom.UHStatic)

        vw = GeomVertexWriter(vdata, "vertex")
        nw = GeomVertexWriter(vdata, "normal")

        tris = GeomTriangles(Geom.UHStatic)

        div = self.circular_div
        two_pi = 2.0 * math.pi

        # --------------------------
        # Side vertices (2 per div)
        # --------------------------
        # We build a ring of bottom/top vertices with radial normals.
        # Indexing:
        #   side_bottom[i], side_top[i]
        side_bottom = []
        side_top = []

        for i in range(div):
            th = two_pi * i / div
            x = math.cos(th)
            z = math.sin(th)

            # radial normal (x,0,z)
            nx, ny, nz = x, 0.0, z

            # bottom (y=0)
            side_bottom.append(vw.getWriteRow())
            vw.addData3f(x, 0.0, z)
            nw.addData3f(nx, ny, nz)

            # top (y=1)
            side_top.append(vw.getWriteRow())
            vw.addData3f(x, 1.0, z)
            nw.addData3f(nx, ny, nz)

        # side faces: two triangles per segment
        for i in range(div):
            i_next = (i + 1) % div
            b0 = side_bottom[i]
            t0 = side_top[i]
            b1 = side_bottom[i_next]
            t1 = side_top[i_next]

            # (b0, t0, t1) and (b0, t1, b1)
            tris.addVertices(b0, t0, t1)
            tris.addVertices(b0, t1, b1)

        # --------------------------
        # Caps (optional)
        # --------------------------
        if self.caps:
            # bottom cap (y=0), normal (0,-1,0)
            bottom_center = vw.getWriteRow()
            vw.addData3f(0.0, 0.0, 0.0)
            nw.addData3f(0.0, -1.0, 0.0)

            bottom_ring = []
            for i in range(div):
                th = two_pi * i / div
                x = math.cos(th)
                z = math.sin(th)

                bottom_ring.append(vw.getWriteRow())
                vw.addData3f(x, 0.0, z)
                nw.addData3f(0.0, -1.0, 0.0)

            for i in range(div):
                i_next = (i + 1) % div
                v0 = bottom_ring[i]
                v1 = bottom_ring[i_next]
                # winding chosen so normal points -Y
                tris.addVertices(bottom_center, v1, v0)

            # top cap (y=1), normal (0, +1, 0)
            top_center = vw.getWriteRow()
            vw.addData3f(0.0, 1.0, 0.0)
            nw.addData3f(0.0, 1.0, 0.0)

            top_ring = []
            for i in range(div):
                th = two_pi * i / div
                x = math.cos(th)
                z = math.sin(th)

                top_ring.append(vw.getWriteRow())
                vw.addData3f(x, 1.0, z)
                nw.addData3f(0.0, 1.0, 0.0)

            for i in range(div):
                i_next = (i + 1) % div
                v0 = top_ring[i]
                v1 = top_ring[i_next]
                # winding so normal points +Y
                tris.addVertices(top_center, v0, v1)

        geom = Geom(vdata)
        geom.addPrimitive(tris)

        gnode = GeomNode(name)
        gnode.addGeom(geom)

        return NodePath(gnode)
