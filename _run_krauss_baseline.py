import _krauss_turns as K
r = K.run(0, vph={'l':100,'s':200,'r':100})
sv = r['served']
srv = '  '.join(f"{d}:{sv[d][0]}/{sv[d][1]}" for d in ('s','l','r'))
print(f"Krauss seed=0  coll={r['collided']}  vph={r['vph']:.0f}  arrived={r['arrived']}  served[{srv}]")
