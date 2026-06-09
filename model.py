"""
model.py — neural-network interface constants ONLY.

The previous neural network (HybridModel, RKHSPhysicsKernel, IDMPhysics,
PhysicsTuning and the legacy nets) has been removed — the model is being
redone.  What remains here is the *contract* the differentiable simulator
(intersection_env.py) and the SUMO eval loop (demo_hybrid.py) build their
observation tensors against.  A new model must consume tokens of these
dimensions and expose the same forward signature
(see IntersectionEnv.step / demo_hybrid.run).

When you add the new network back, define its class(es) below these
constants (or in a new module) and re-wire the dangling `HybridModel`
imports in demo_hybrid.py and train_hybrid.py.
"""
from __future__ import annotations

# ── ego-token feature layout (D_EGO = 4, all values normalized ∈ [0,1] or signed) ──
# slot 0: v / V_MAX                    normalized speed
# slot 1: min(gap, 100) / 100          normalized gap to leader
# slot 2: (v − v_lead) / V_MAX         signed closing speed
# slot 3: max(0, d_jct) / ARM_LENGTH   normalized distance to junction
D_EGO      = 4
ARM_LENGTH = 200.0   # m — road arm length used for d_jct normalisation

# ── own-stream summary token layout (D_SUM = 4) ──────────────────────────────
# slot 0: P_own / P_SCALE              own stream packing pressure ∈ [0, 1]
# slot 1: n_own / N_SCALE              approaching vehicles in own stream ∈ [0, 1]
# slot 2: mean_v_own / V_REF           mean speed of own queue
# slot 3: v_follower / V_REF           immediate follower speed
D_SUM      = 4       # see social_force._stream_summary

# ── rival-stream token layout (D_RIVAL = 4, one token per conflicting stream) ─
# slot 0: P_k / P_SCALE                rival stream packing pressure ∈ [0, 1]
# slot 1: n_k / N_SCALE                approaching vehicles in rival stream ∈ [0, 1]
# slot 2: mean_v_k / V_REF             mean speed of rival stream
# slot 3: mu_k                         worst conflict gate from this stream ∈ [0, 1]
# Streams with no vehicles and no urgency are padded with zero rows.
D_RIVAL    = 4
K_MAX      = 8       # upper bound on conflicting streams in a 4-way intersection

# ── turning / approach gating constants (consumed by demo_hybrid.py) ──────────
V_TURN_HIGH = 11.0   # m/s — turning gate fully active above this speed
D_APPROACH  = 30.0   # m   — approach gate width for mu_wp
