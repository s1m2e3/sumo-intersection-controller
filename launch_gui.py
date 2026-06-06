from simulator import build, load_config
from cosimulator import CoSimulator
from model import IDMModel

sumocfg = build()
cfg     = load_config()
idm     = IDMModel()

sim = CoSimulator(sumocfg=sumocfg, dt=cfg["step_length"], idm=idm, gui=True)
sim.run(max_steps=3000)
