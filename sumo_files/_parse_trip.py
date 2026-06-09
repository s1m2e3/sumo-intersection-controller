import sys, numpy as np, xml.etree.ElementTree as ET
f = sys.argv[1]
r = ET.parse(f).getroot()
trips = r.findall('tripinfo')
n = len(trips)
dur = [float(t.get('duration')) for t in trips]
loss = [float(t.get('timeLoss')) for t in trips]
wait = [float(t.get('waitingTime')) for t in trips]
print(f'  vehicles ARRIVED (cleared in 120s) : {n}')
print(f'  served throughput                  : {n/120*3600:.0f} veh/h')
print(f'  mean travel time                   : {np.mean(dur):.1f} s')
print(f'  mean time loss (vs free flow)      : {np.mean(loss):.1f} s')
print(f'  mean waiting time                  : {np.mean(wait):.1f} s')
print(f'  free-flow travel (400 m / 13.9)    : {400/13.89:.1f} s')
