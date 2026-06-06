import os
import torch
import traci
from pathlib import Path

import sumo as _sumo_pkg
from model import IDMModel
from simulator import build, load_config

SUMO_BIN = Path(_sumo_pkg.__file__).parent / "bin"
TRACI_PORT = 8813


def _bin(name: str) -> str:
    return str(SUMO_BIN / (name + (".exe" if os.name == "nt" else "")))


def _setup_env():
    os.environ["SUMO_HOME"] = str(SUMO_BIN.parent)
    os.environ["PATH"]      = str(SUMO_BIN) + os.pathsep + os.environ.get("PATH", "")


# ---------------------------------------------------------------------------
# Read all vehicle states from SUMO into PyTorch tensors
# ---------------------------------------------------------------------------

def _read_states(vehicle_ids: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    v_list, gap_list, vl_list = [], [], []

    for vid in vehicle_ids:
        v      = traci.vehicle.getSpeed(vid)
        leader = traci.vehicle.getLeader(vid)
        if leader is not None:
            lead_id, gap = leader
            v_lead = traci.vehicle.getSpeed(lead_id)
        else:
            gap    = 100.0
            v_lead = v        # free-flow: no leader

        v_list.append(v)
        gap_list.append(gap)
        vl_list.append(v_lead)

    return (
        torch.tensor(v_list,   dtype=torch.float32),
        torch.tensor(gap_list, dtype=torch.float32),
        torch.tensor(vl_list,  dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# CoSimulator
# ---------------------------------------------------------------------------

class CoSimulator:
    """
    SUMO owns: road network, vehicle spawning/despawning, position updates.
    PyTorch owns: IDM acceleration + Euler speed integration for every vehicle.

    Each step:
      1. simulationStep()  — SUMO moves vehicles, spawns new ones
      2. read states       — get (v, gap, v_lead) for every active vehicle
      3. IDM forward pass  — compute acceleration in PyTorch (graph tracked)
      4. Euler step        — v_next = v + a * dt   (graph tracked)
      5. setSpeed()        — hand v_next back to SUMO
    """

    def __init__(self, sumocfg: Path, dt: float, idm: IDMModel, gui: bool = False):
        self.sumocfg  = sumocfg
        self.dt       = dt
        self.idm      = idm
        self.gui      = gui
        self._known   = set()   # vehicle IDs we have already configured

    def _launch(self):
        _setup_env()
        binary = _bin("sumo-gui") if self.gui else _bin("sumo")

        # sumo-gui TraCI is broken in the pip package on Windows — always use sumo.exe
        traci.start([
            _bin("sumo"), "-c", str(self.sumocfg),
            "--step-length", str(self.dt),
            "--collision.action", "warn",
        ])

    def _sync_vehicle_set(self, current: set[str]) -> tuple[set[str], set[str]]:
        arrived  = current - self._known   # spawned this step
        departed = self._known - current   # left the network this step
        for vid in arrived:
            traci.vehicle.setSpeedMode(vid, 0)
        self._known = current
        return arrived, departed

    def step(self) -> dict:
        traci.simulationStep()

        current = set(traci.vehicle.getIDList())
        arrived, departed = self._sync_vehicle_set(current)

        vids = list(current)
        result = {"arrived": arrived, "departed": departed,
                  "active": vids, "ids": vids}

        if not vids:
            return result

        # --- PyTorch ---
        v, gap, v_lead = _read_states(vids)
        accel  = self.idm(v, gap, v_lead)
        v_next = torch.clamp(v + accel * self.dt, min=0.0)

        # --- SUMO ---
        for i, vid in enumerate(vids):
            traci.vehicle.setSpeed(vid, float(v_next[i]))

        result.update({"v": v, "gap": gap, "v_lead": v_lead,
                       "accel": accel, "v_next": v_next})
        return result

    def reset(self):
        if traci.isLoaded():
            traci.close()
        self._known = set()
        self._launch()

    def close(self):
        if traci.isLoaded():
            traci.close()

    def run(self, max_steps: int = 1000):
        self.reset()
        try:
            for step in range(max_steps):
                info = self.step()
                if info and step % 100 == 0:
                    n = len(info["ids"])
                    v_mean = info["v"].mean().item()
                    print(f"step {step:4d} | vehicles: {n:3d} | mean speed: {v_mean:.2f} m/s")
        finally:
            self.close()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sumocfg = build()
    cfg     = load_config()
    idm     = IDMModel()

    sim = CoSimulator(sumocfg=sumocfg, dt=cfg["step_length"], idm=idm, gui=True)
    sim.run(max_steps=3000)
