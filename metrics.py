import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.run_result import RunResult

def l2_error(result, reference, field):
    errs = []
    for s, s_ref in zip(result.states, reference.states):
        q, q_ref = getattr(s, field), getattr(s_ref, field)
        errs.append(np.linalg.norm(q - q_ref) / np.linalg.norm(q_ref))
    return np.array(errs)

def linf_error(result, reference, field):
    errs = []
    for s, s_ref in zip(result.states, reference.states):
        q, q_ref = getattr(s, field), getattr(s_ref, field)
        errs.append(np.max(np.abs(q - q_ref)) / np.max(np.abs(q_ref)))
    return np.array(errs)

def _total(result, field):
    return np.array([np.sum(getattr(s, field)) for s in result.states])

def conservation_drift(result, field):
    totals = _total(result, field)
    return (totals - totals[0]) / totals[0]

def mass_drift(result):
    return conservation_drift(result, "rho")

def div_B_error(state, dx=1.0, dy=1.0):
    if state.rho.ndim < 2:
        raise ValueError("div_B_error requires a 2D+ FieldState")
    Bx, By = state.B[0], state.B[1]
    dBxdx = np.gradient(Bx, dx, axis=1)
    dBydy = np.gradient(By, dy, axis=0)
    divB = dBxdx + dBydy
    B_mag = np.sqrt(Bx**2 + By**2)
    return float(np.linalg.norm(divB) / np.linalg.norm(B_mag))

def div_B_over_time(result, dx=1.0, dy=1.0):
    return np.array([div_B_error(s, dx, dy) for s in result.states])
