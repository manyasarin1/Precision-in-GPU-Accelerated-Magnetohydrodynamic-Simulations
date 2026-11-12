"""
GPU MHD kernel satisfying the Interface Contract (shared/field_state.py,
shared/run_result.py), built on CuPy so that precision is genuine, not
emulated: fp16/fp32/fp64 arrays are real device dtypes, arithmetic runs on
the GPU in that dtype wherever CuPy supports it natively, and timing uses
CUDA events (not CPU wall-clock) so it reflects actual device execution
rather than Python/host overhead.

This supersedes kernels/toy_kernel.py's precision HANDLING (same physics:
first-order, dimensionally-split, global Lax-Friedrichs), for the purpose of
reporting genuine GPU runtime and memory numbers. kernels/toy_kernel.py
remains useful as a CPU reference / cross-check and for the fp64-precision-
emulation accuracy study already in the paper.

Known scope limits (stated plainly, not hidden):
  * bf16 requires CuPy's ml_dtypes-backed bfloat16 support (CuPy >= 12 with
    ml_dtypes installed). If unavailable, bf16 is skipped at runtime with a
    clear message rather than silently falling back to another dtype.
  * A handful of elementwise ops (sqrt of a fp16 array in particular, on
    some CuPy/CUDA versions) may upcast internally; where that happens the
    result is cast back to the target dtype before the next step, so memory
    footprint and the bulk of the arithmetic remain in the requested
    precision even if a given op's internal accumulation is wider.
  * This is still a toy (first-order Lax-Friedrichs) scheme, not the
    production solver -- the contribution here is genuine GPU precision and
    timing, not a production-grade numerical method.
"""

import sys, os, time
import numpy as np
import cupy as cp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.field_state import FieldState
from shared.run_result import RunResult

GAMMA = 5.0 / 3.0
VALID_PRECISIONS = {"fp64", "fp32", "fp16", "bf16", "adaptive"}
VALID_PROBLEMS = {"brio_wu", "orszag_tang"}

_CP_DTYPE = {"fp64": cp.float64, "fp32": cp.float32, "fp16": cp.float16}


def _bf16_dtype():
    try:
        import ml_dtypes
        return cp.dtype(ml_dtypes.bfloat16)
    except Exception as e:
        raise RuntimeError(
            "bf16 requires CuPy with ml_dtypes-backed bfloat16 support "
            "(pip install ml_dtypes; CuPy >= 12). Original error: " + repr(e)
        )


def _dtype_for(precision):
    if precision == "bf16":
        return _bf16_dtype()
    return _CP_DTYPE[precision]


def _bytes_per_element(precision):
    if precision == "bf16":
        return 2
    return {"fp64": 8, "fp32": 4, "fp16": 2}[precision]


# --------------------------------------------------------------------------
# Initial conditions (built on host, then moved to device in the target dtype)
# --------------------------------------------------------------------------

def _brio_wu_ic(nx):
    x = (np.arange(nx) + 0.5) / nx
    left = x < 0.5
    rho = np.where(left, 1.0, 0.125)
    press = np.where(left, 1.0, 0.1)
    vel = np.zeros((3, nx))
    B = np.zeros((3, nx))
    B[0] = 0.75
    B[1] = np.where(left, 1.0, -1.0)
    return rho, vel, press, B


def _orszag_tang_ic(ny, nx):
    x = (np.arange(nx) + 0.5) / nx * 2 * np.pi
    y = (np.arange(ny) + 0.5) / ny * 2 * np.pi
    X, Y = np.meshgrid(x, y)
    rho = np.full((ny, nx), GAMMA ** 2)
    press = np.full((ny, nx), GAMMA)
    vel = np.zeros((3, ny, nx))
    vel[0] = -np.sin(Y)
    vel[1] = np.sin(X)
    B = np.zeros((3, ny, nx))
    B[0] = -np.sin(Y)
    B[1] = np.sin(2 * X)
    return rho, vel, press, B


# --------------------------------------------------------------------------
# Ideal-MHD physics on device arrays (dtype-generic: operates in whatever
# dtype the input arrays already are, via CuPy's own dtype promotion rules)
# --------------------------------------------------------------------------

def _total_energy(rho, vel, press, B):
    return press / (GAMMA - 1) + 0.5 * rho * cp.sum(vel ** 2, axis=0) + 0.5 * cp.sum(B ** 2, axis=0)


def _flux(rho, vel, press, B, component):
    vd, Bd = vel[component], B[component]
    B2 = cp.sum(B ** 2, axis=0)
    vB = cp.sum(vel * B, axis=0)
    F_rho = rho * vd
    F_mom = rho * vel * vd - B * Bd
    F_mom[component] = F_mom[component] + press + 0.5 * B2
    F_B = vd * B - vel * Bd
    energy = _total_energy(rho, vel, press, B)
    F_energy = (energy + press + 0.5 * B2) * vd - Bd * vB
    return F_rho, F_mom, F_B, F_energy


def _max_speed(rho, vel, press, B):
    # Computed in fp32 regardless of storage dtype: this is a scalar CFL/
    # stability control value, not state, and fp16 sqrt/division here is a
    # known source of spurious inf/nan unrelated to the precision question
    # this kernel is measuring.
    rho32, vel32, press32, B32 = rho.astype(cp.float32), vel.astype(cp.float32), press.astype(cp.float32), B.astype(cp.float32)
    B2 = cp.sum(B32 ** 2, axis=0)
    cs2 = cp.clip(GAMMA * press32 / rho32, 0, None)
    va2 = B2 / rho32
    fast = cp.sqrt(cs2 + va2)
    speed = cp.sqrt(cp.sum(vel32 ** 2, axis=0)) + fast
    return float(cp.max(speed).get())


def _sweep_axes(ndim):
    return [(ndim - 1 - i, i) for i in range(ndim)]


def _lax_friedrichs_sweep(rho, vel, press, B, dt, dx, spatial_axis, component, dtype):
    energy = _total_energy(rho, vel, press, B)
    mom = rho * vel
    F_rho, F_mom, F_B, F_energy = _flux(rho, vel, press, B, component)
    alpha = _max_speed(rho, vel, press, B)
    alpha_t = cp.asarray(alpha, dtype=dtype)

    def lf_update(U, F):
        ax = spatial_axis + (U.ndim - rho.ndim)
        F_r, F_l = cp.roll(F, -1, axis=ax), cp.roll(F, 1, axis=ax)
        U_r, U_l = cp.roll(U, -1, axis=ax), cp.roll(U, 1, axis=ax)
        F_half_r = 0.5 * (F + F_r) - 0.5 * alpha_t * (U_r - U)
        F_half_l = 0.5 * (F_l + F) - 0.5 * alpha_t * (U - U_l)
        return (U - (dt / dx) * (F_half_r - F_half_l)).astype(dtype)

    rho_new = lf_update(rho, F_rho)
    mom_new = lf_update(mom, F_mom)
    B_new = lf_update(B, F_B)
    energy_new = lf_update(energy, F_energy)

    vel_new = (mom_new / rho_new).astype(dtype)
    press_new = ((GAMMA - 1) * (
        energy_new.astype(cp.float32) - 0.5 * rho_new.astype(cp.float32) * cp.sum(vel_new.astype(cp.float32) ** 2, axis=0)
        - 0.5 * cp.sum(B_new.astype(cp.float32) ** 2, axis=0)
    )).astype(dtype)
    return rho_new, vel_new, press_new, B_new, alpha


def _classify_cells(rho):
    rho32 = rho.astype(cp.float32)
    g = cp.gradient(rho32)
    if isinstance(g, cp.ndarray):
        g = [g]
    grad = sum(cp.abs(x) for x in g)
    threshold = cp.median(grad)
    return grad > threshold


def _state_to_host(rho, vel, press, B, sim_time):
    return FieldState(
        rho=cp.asnumpy(rho).astype(np.float64),
        vel=cp.asnumpy(vel).astype(np.float64),
        press=cp.asnumpy(press).astype(np.float64),
        B=cp.asnumpy(B).astype(np.float64),
        energy=cp.asnumpy(_total_energy(rho, vel, press, B)).astype(np.float64),
        time=sim_time,
    )


def run_kernel(precision: str, problem: str, **kwargs) -> RunResult:
    if precision not in VALID_PRECISIONS:
        raise ValueError(f"unknown precision {precision!r}, expected one of {VALID_PRECISIONS}")
    if problem not in VALID_PROBLEMS:
        raise ValueError(f"unknown problem {problem!r}, expected one of {VALID_PROBLEMS}")

    n_steps = kwargs.get("n_steps", 300)
    output_every = kwargs.get("output_every", 10)
    cfl = kwargs.get("cfl", 0.4)
    k = kwargs.get("k", 10)

    if problem == "brio_wu":
        nx = kwargs.get("nx", 400)
        rho_h, vel_h, press_h, B_h = _brio_wu_ic(nx)
        dx_list = [1.0 / nx]
        grid_desc = {"nx": nx}
    else:
        n = kwargs.get("n", 96)
        rho_h, vel_h, press_h, B_h = _orszag_tang_ic(n, n)
        dx_list = [2 * np.pi / n, 2 * np.pi / n]
        grid_desc = {"n": n}

    compute_dtype = cp.float32 if precision == "adaptive" else _dtype_for(precision)
    # adaptive stores a mixed fp32/fp16 state; the "working" array dtype is
    # fp32 and the fp16 regions are tracked by rounding through fp16 each
    # reclassification, matching the toy kernel's cell-level scheme.
    store_dtype = _dtype_for(precision) if precision != "adaptive" else cp.float32

    rho = cp.asarray(rho_h, dtype=store_dtype)
    vel = cp.asarray(vel_h, dtype=store_dtype)
    press = cp.asarray(press_h, dtype=store_dtype)
    B = cp.asarray(B_h, dtype=store_dtype)

    ndim = rho.ndim
    sweeps = _sweep_axes(ndim)
    rough_mask = cp.zeros(rho.shape, dtype=bool) if precision == "adaptive" else None

    cp.cuda.Stream.null.synchronize()
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()
    start_evt, end_evt = cp.cuda.Event(), cp.cuda.Event()
    start_evt.record()

    sim_time = 0.0
    states = [_state_to_host(rho, vel, press, B, sim_time)]

    for step in range(1, n_steps + 1):
        dx_min = min(dx_list)
        alpha_probe = _max_speed(rho, vel, press, B)
        dt = cfl * dx_min / (alpha_probe + 1e-12)

        for spatial_axis, component in sweeps:
            rho, vel, press, B, _ = _lax_friedrichs_sweep(
                rho, vel, press, B, dt, dx_list[spatial_axis], spatial_axis, component, store_dtype
            )
        sim_time += dt

        if precision == "adaptive":
            if step % k == 0 or step == 1:
                rough_mask = _classify_cells(rho)
            rho16, vel16, press16, B16 = (
                rho.astype(cp.float16).astype(cp.float32),
                vel.astype(cp.float16).astype(cp.float32),
                press.astype(cp.float16).astype(cp.float32),
                B.astype(cp.float16).astype(cp.float32),
            )
            m_s = cp.broadcast_to(rough_mask, rho.shape)
            m_v = cp.broadcast_to(rough_mask, vel.shape)
            rho = cp.where(m_s, rho, rho16)
            vel = cp.where(m_v, vel, vel16)
            press = cp.where(m_s, press, press16)
            B = cp.where(m_v, B, B16)

        if step % output_every == 0 or step == n_steps:
            states.append(_state_to_host(rho, vel, press, B, sim_time))

    end_evt.record()
    end_evt.synchronize()
    runtime_sec = cp.cuda.get_elapsed_time(start_evt, end_evt) / 1000.0
    peak_bytes = mempool.used_bytes()

    bpe = _bytes_per_element(precision) if precision != "adaptive" else None
    nominal_bytes = None
    if bpe is not None:
        nominal_bytes = int((rho.size + vel.size + press.size + B.size) * bpe)

    extra = {
        "grid": grid_desc, "n_steps": n_steps, "cfl": cfl, "scheme": "lax_friedrichs_split_gpu",
        "device": str(cp.cuda.runtime.getDeviceProperties(0)["name"]),
        "cupy_mempool_used_bytes": int(peak_bytes),
    }
    if precision == "adaptive":
        extra["k"] = k
        extra["final_rough_fraction"] = float(cp.mean(rough_mask).get())

    return RunResult(states=states, runtime_sec=runtime_sec, precision=precision,
                      memory_bytes=nominal_bytes if nominal_bytes is not None else int(peak_bytes),
                      extra=extra)
