import sys, numpy as np, xml.etree.ElementTree as ET
r = ET.parse(sys.argv[1]).getroot()
steps = r.findall('step')
def col(name): return np.array([float(s.get(name)) for s in steps])
t = col('time'); inserted = col('inserted'); ended = col('ended')
running = col('running'); waiting = col('waiting'); halting = col('halting')
# steady-state arrival rate in t in [40,120]
m = t >= 40.0
end40 = ended[t >= 40.0][0]; endL = ended[-1]
rate = (endL - end40) / (t[-1] - 40.0) * 3600
print(f'  total demand (2000 vph x 120s)     : ~67 veh')
print(f'  inserted into network (total)      : {inserted[-1]:.0f}')
print(f'  still WAITING at source (t=120)    : {waiting[-1]:.0f}  (demand held back)')
print(f'  arrived/ended (total, 120s)        : {ended[-1]:.0f}')
print(f'  steady-state throughput (t40-120)  : {rate:.0f} veh/h')
print(f'  mean halting (queued) over run     : {np.mean(halting):.1f} veh')
