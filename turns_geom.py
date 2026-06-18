"""Turn-aware intersection geometry: all 12 movements (entry→exit) and their true
pairwise conflict points, read from the SUMO net's internal via-lane shapes.

A "movement" here is a connection from an *_in edge to an *_out edge (right/through/
left), indexed 0..11.  Conflict point CP[i,j] is the first geometric crossing of the
two movements' through-paths (from-lane end + via-lane shape + to-lane start); None
where they don't cross.  This is the turn-general version of cosim_sumo.conflict_points
(which only covered the 4 straight movements).

    import turns_geom as G
    mv = G.movements(net_path)          # list of Movement(idx, frm, to, dir, axis, sgn, turn)
    cpx, cpy = G.cp_tensors(net_path)    # [M,M] conflict-point x/y (NaN where none)
"""
import math
from dataclasses import dataclass

import sumolib

import cosim_sumo as C   # reuse ORIGIN (entry-edge axis/sign), _poly_x

_IN  = ("east_in", "west_in", "north_in", "south_in")
_CACHE = {}


@dataclass
class Movement:
    idx: int
    frm: str
    to: str
    dir: str          # 'r' | 's' | 'l'
    axis: int         # entry-approach axis (0=x/EW, 1=y/NS)
    sgn: float        # entry-approach travel sign (±1)
    shape: list       # SHORT polyline: from-lane end + via + to-lane start (for CP crossing)
    path: list        # FULL polyline: from-lane + via + to-lane (arc-length matches VAR_DISTANCE)


def movements(net_path):
    """All entry→exit connections as indexed Movements, with via-lane geometry."""
    if (net_path, "mv") in _CACHE:
        return _CACHE[(net_path, "mv")]
    net = sumolib.net.readNet(net_path, withInternal=True)
    mvs = []
    for frm in _IN:
        edge = net.getEdge(frm)
        for to_edge in edge.getOutgoing():
            for conn in edge.getConnections(to_edge):
                via = conn.getViaLaneID()
                vshape = list(net.getLane(via).getShape()) if via else []
                shape = ([conn.getFromLane().getShape()[-1]] + vshape
                         + [conn.getToLane().getShape()[0]])
                path = (list(conn.getFromLane().getShape()) + vshape
                        + list(conn.getToLane().getShape()))
                axis, sgn = C.ORIGIN[frm]
                mvs.append(Movement(len(mvs), frm, to_edge.getID(),
                                    conn.getDirection(), axis, sgn, shape, path))
    _CACHE[(net_path, "mv")] = mvs
    return mvs


def conflict_pairs(net_path):
    """CP[(i,j)] = true crossing of movements i and j (None if they don't cross)."""
    if (net_path, "cp") in _CACHE:
        return _CACHE[(net_path, "cp")]
    mv = movements(net_path)
    CP = {}
    for i, a in enumerate(mv):
        for j, b in enumerate(mv):
            if i == j or a.frm == b.frm:        # same approach never "conflicts" via CP
                CP[(i, j)] = None
            else:
                CP[(i, j)] = C._poly_x(a.shape, b.shape)
    _CACHE[(net_path, "cp")] = CP
    return CP


def cp_tensors(net_path):
    import torch
    mv = movements(net_path); M = len(mv); CP = conflict_pairs(net_path)
    cx = torch.full((M, M), float("nan")); cy = torch.full((M, M), float("nan"))
    for i in range(M):
        for j in range(M):
            pt = CP[(i, j)]
            if pt is not None:
                cx[i, j], cy[i, j] = float(pt[0]), float(pt[1])
    return cx, cy


def gate_geometry(net_path):
    """12-movement geometry for the 2-D box gate, analogous to sim_torch.build_geometry
    but turn-aware:
      geo    : {movement_idx: (pts [P,2], cum [P])}  — path polyline + arc-lengths
      s_cp   : [M,M] arc-length of CP(i,j) ALONG movement i's path (NaN if no crossing)
      s_junc : [M] arc-length of i's FIRST conflict point (≈ where it enters the box)
      CONF   : [M,M] bool — movements i,j cross (the turn-general replacement for the
               straight gate's axis_i != axis_j test)
    """
    import torch
    import sim_torch as S
    mv = movements(net_path); M = len(mv)
    CP = conflict_pairs(net_path)
    geo = {}
    for m in mv:
        poly = [m.path[0]]
        for p in m.path[1:]:                               # FULL path (matches VAR_DISTANCE)
            if abs(p[0] - poly[-1][0]) > 1e-6 or abs(p[1] - poly[-1][1]) > 1e-6:
                poly.append(p)
        pts = torch.tensor(poly, dtype=torch.float32)
        seg = (pts[1:] - pts[:-1]).norm(dim=1)
        cum = torch.cat([torch.zeros(1), torch.cumsum(seg, 0)])
        geo[m.idx] = (pts, cum)
    s_cp = torch.full((M, M), float("nan"))
    for i in range(M):
        for j in range(M):
            pt = CP[(i, j)]
            if pt is not None:
                s_cp[i, j] = S._project(geo[i][0], geo[i][1],
                                        torch.tensor(pt, dtype=torch.float32))
    s_junc = torch.nan_to_num(s_cp, nan=1e9).min(dim=1).values
    CONF   = ~torch.isnan(s_cp)
    return geo, s_cp, s_junc, CONF
