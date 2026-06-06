import json
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

import sumo as _sumo_pkg

SUMO_BIN = Path(_sumo_pkg.__file__).parent / "bin"
INPUTS_DIR = Path("inputs")
SUMO_DIR = Path("sumo_files")


def _bin(name: str) -> str:
    exe = SUMO_BIN / (name + (".exe" if os.name == "nt" else ""))
    return str(exe)


def load_config(path: Path = INPUTS_DIR / "intersection.json") -> dict:
    with open(path) as f:
        return json.load(f)


def _pretty_xml(root: ET.Element) -> str:
    raw = ET.tostring(root, encoding="unicode")
    return minidom.parseString(raw).toprettyxml(indent="  ")


def _write_nodes(cfg: dict, out_dir: Path) -> Path:
    L = cfg["arm_length"]
    nodes = ET.Element("nodes")
    for nid, x, y, ntype in [
        ("center", 0,  0,  "priority"),
        ("north",  0,  L,  "dead_end"),
        ("south",  0, -L,  "dead_end"),
        ("east",   L,  0,  "dead_end"),
        ("west",  -L,  0,  "dead_end"),
    ]:
        ET.SubElement(nodes, "node", id=nid, x=str(x), y=str(y), type=ntype)
    path = out_dir / "nodes.xml"
    path.write_text(_pretty_xml(nodes))
    return path


def _write_edges(cfg: dict, out_dir: Path) -> Path:
    s = str(cfg["speed_limit"])
    w = str(cfg["lane_width"])
    edges = ET.Element("edges")
    # incoming: 2 lanes (lane 0 = through+right, lane 1 = left-turn only)
    # outgoing: 1 lane
    for eid, frm, to, n_lanes in [
        ("north_in",  "north",  "center", "2"),
        ("north_out", "center", "north",  "1"),
        ("south_in",  "south",  "center", "2"),
        ("south_out", "center", "south",  "1"),
        ("east_in",   "east",   "center", "2"),
        ("east_out",  "center", "east",   "1"),
        ("west_in",   "west",   "center", "2"),
        ("west_out",  "center", "west",   "1"),
    ]:
        ET.SubElement(edges, "edge", id=eid, **{"from": frm, "to": to},
                      numLanes=n_lanes, speed=s, width=w)
    path = out_dir / "edges.xml"
    path.write_text(_pretty_xml(edges))
    return path


def _write_connections(out_dir: Path) -> Path:
    conns = ET.Element("connections")

    # east_in (westbound): lane 0 = through+right-turn, lane 1 = left-turn only
    ET.SubElement(conns, "connection", **{"from": "east_in", "to": "west_out",  "fromLane": "0", "toLane": "0"})
    ET.SubElement(conns, "connection", **{"from": "east_in", "to": "north_out", "fromLane": "0", "toLane": "0"})
    ET.SubElement(conns, "connection", **{"from": "east_in", "to": "south_out", "fromLane": "1", "toLane": "0"})

    # west_in (eastbound): lane 0 = through+right-turn, lane 1 = left-turn only
    ET.SubElement(conns, "connection", **{"from": "west_in", "to": "east_out",  "fromLane": "0", "toLane": "0"})
    ET.SubElement(conns, "connection", **{"from": "west_in", "to": "south_out", "fromLane": "0", "toLane": "0"})
    ET.SubElement(conns, "connection", **{"from": "west_in", "to": "north_out", "fromLane": "1", "toLane": "0"})

    # north_in (southbound): lane 0 = through+right-turn, lane 1 = left-turn only
    ET.SubElement(conns, "connection", **{"from": "north_in", "to": "south_out", "fromLane": "0", "toLane": "0"})
    ET.SubElement(conns, "connection", **{"from": "north_in", "to": "west_out",  "fromLane": "0", "toLane": "0"})
    ET.SubElement(conns, "connection", **{"from": "north_in", "to": "east_out",  "fromLane": "1", "toLane": "0"})

    # south_in (northbound): lane 0 = through+right-turn, lane 1 = left-turn only
    ET.SubElement(conns, "connection", **{"from": "south_in", "to": "north_out", "fromLane": "0", "toLane": "0"})
    ET.SubElement(conns, "connection", **{"from": "south_in", "to": "east_out",  "fromLane": "0", "toLane": "0"})
    ET.SubElement(conns, "connection", **{"from": "south_in", "to": "west_out",  "fromLane": "1", "toLane": "0"})

    path = out_dir / "connections.xml"
    path.write_text(_pretty_xml(conns))
    return path


def _generate_network(nodes_path: Path, edges_path: Path, conn_path: Path, out_dir: Path) -> Path:
    net_path = out_dir / "intersection.net.xml"
    env = os.environ.copy()
    env["SUMO_HOME"] = str(SUMO_BIN.parent)
    subprocess.run([
        _bin("netconvert"),
        "--node-files",       str(nodes_path),
        "--edge-files",       str(edges_path),
        "--connection-files", str(conn_path),
        "--output-file",      str(net_path),
        "--no-turnarounds",
    ], check=True, env=env)
    return net_path


def _write_routes(cfg: dict, out_dir: Path) -> Path:
    vph_t = int(cfg["vehicles_per_hour"])          # through volume
    vph_r = max(1, int(vph_t * 0.25))             # right-turn  (~25% of through)
    vph_l = max(1, int(vph_t * 0.15))             # left-turn   (~15% of through)
    dur   = str(cfg["simulation_duration"])
    spd   = str(cfg["speed_limit"])

    routes = ET.Element("routes")
    ET.SubElement(routes, "vType", id="car", accel="2.6", decel="4.5",
                  sigma="0.5", length="5", maxSpeed=spd)

    # named route used by the ego vehicle (E → W through)
    ET.SubElement(routes, "route", id="route_ego", edges="east_in west_out")

    def _flow(fid, frm, to, vph, begin="5"):
        ET.SubElement(routes, "flow", id=fid, type="car",
                      **{"from": frm, "to": to},
                      begin=begin, end=dur, vehsPerHour=str(vph),
                      departLane="best",      # SUMO assigns the correct lane per movement
                      departSpeed="desired")  # depart at speed limit only if gap is safe

    # SUMO requires flows sorted by begin time — all begin=5 first, then begin=5.5.
    # EW/WE axis starts at t=5; NS/SN axis at t=5.5 so opposing streams don't
    # arrive at the junction simultaneously.

    # --- begin=5 : EW and WE (all movements) ---
    _flow("flow_ew",   "east_in",  "west_out",   vph_t, begin="5")
    _flow("flow_we",   "west_in",  "east_out",   vph_t, begin="5")
    _flow("flow_ew_r", "east_in",  "north_out",  vph_r, begin="5")
    _flow("flow_we_r", "west_in",  "south_out",  vph_r, begin="5")
    _flow("flow_ew_l", "east_in",  "south_out",  vph_l, begin="5")
    _flow("flow_we_l", "west_in",  "north_out",  vph_l, begin="5")

    # --- begin=5.5 : NS and SN (all movements) ---
    _flow("flow_ns",   "north_in", "south_out",  vph_t, begin="5.5")
    _flow("flow_sn",   "south_in", "north_out",  vph_t, begin="5.5")
    _flow("flow_ns_r", "north_in", "west_out",   vph_r, begin="5.5")
    _flow("flow_sn_r", "south_in", "east_out",   vph_r, begin="5.5")
    _flow("flow_ns_l", "north_in", "east_out",   vph_l, begin="5.5")
    _flow("flow_sn_l", "south_in", "west_out",   vph_l, begin="5.5")

    path = out_dir / "routes.xml"
    path.write_text(_pretty_xml(routes))
    return path


def _write_sumocfg(cfg: dict, net_path: Path, rou_path: Path, out_dir: Path) -> Path:
    config = ET.Element("configuration")

    inp = ET.SubElement(config, "input")
    ET.SubElement(inp, "net-file",    value=net_path.name)
    ET.SubElement(inp, "route-files", value=rou_path.name)

    time = ET.SubElement(config, "time")
    ET.SubElement(time, "begin",       value="0")
    ET.SubElement(time, "end",         value=str(cfg["simulation_duration"]))
    ET.SubElement(time, "step-length", value=str(cfg["step_length"]))

    path = out_dir / "intersection.sumocfg"
    path.write_text(_pretty_xml(config))
    return path


def build(config_path: Path = INPUTS_DIR / "intersection.json") -> Path:
    cfg = load_config(config_path)
    SUMO_DIR.mkdir(exist_ok=True)

    nodes   = _write_nodes(cfg, SUMO_DIR)
    edges   = _write_edges(cfg, SUMO_DIR)
    conns   = _write_connections(SUMO_DIR)
    net     = _generate_network(nodes, edges, conns, SUMO_DIR)
    routes  = _write_routes(cfg, SUMO_DIR)
    sumocfg = _write_sumocfg(cfg, net, routes, SUMO_DIR)

    print(f"[simulator] SUMO files written to {SUMO_DIR}/")
    return sumocfg


if __name__ == "__main__":
    build()
