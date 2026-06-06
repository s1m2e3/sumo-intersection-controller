import subprocess
from pathlib import Path
import sumo as _sumo_pkg

SUMO_BIN = Path(_sumo_pkg.__file__).parent / "bin"
sumocfg = Path("sumo_files/intersection.sumocfg")

subprocess.run([str(SUMO_BIN / "sumo-gui.exe"), "-c", str(sumocfg)])
