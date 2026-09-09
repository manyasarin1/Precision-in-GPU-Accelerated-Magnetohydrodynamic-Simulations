import glob, re
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.field_state import FieldState
from shared.run_result import RunResult

def parse_tab_header(path):
    """Real Athena++ .tab header looks like:
    # Athena++ data at time=... cycle=... variables=prim
    # i    x1v    rho    press    vel1    vel2    vel3    Bcc1    Bcc2    Bcc3
    Column names are just the second comment line, space-separated."""
    with open(path) as f:
        lines = [next(f) for _ in range(2)]
    header_line = lines[1]
    assert header_line.startswith("#"), f"Unexpected header format in {path}"
    return header_line.lstrip("#").split()

def load_athena_tab(path):
    cols = parse_tab_header(path)
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[None, :]
    return {name: data[:, i] for i, name in enumerate(cols)}

def _tab_to_field_state(tab, sim_time, ny=None, nx=None):
    def get(col, default=0.0):
        if col in tab:
            arr = tab[col]
            return arr.reshape(ny, nx) if (ny and nx) else arr
        shape = tab["rho"].shape if not (ny and nx) else (ny, nx)
        return np.full(shape, default)
    rho = get("rho")
    press = get("press")
    energy = get("Etot", default=np.nan)  # not present in this output — will be NaN, that's expected
    vel = np.stack([get("vel1"), get("vel2"), get("vel3")])
    B = np.stack([get("Bcc1"), get("Bcc2"), get("Bcc3")])
    return FieldState(rho=rho, vel=vel, press=press, B=B, energy=energy, time=sim_time)

def load_run(pattern, ny=None, nx=None, precision="fp64"):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")
    states = []
    for f in files:
        tab = load_athena_tab(f)
        # extract time from the first header line instead of a data column
        with open(f) as fh:
            first_line = fh.readline()
        m = re.search(r"time=([\d.eE+-]+)", first_line)
        sim_time = float(m.group(1)) if m else float(len(states))
        states.append(_tab_to_field_state(tab, sim_time, ny=ny, nx=nx))
    return RunResult(states=states, runtime_sec=float("nan"), precision=precision,
                      extra={"source": "athena++", "n_files": len(files)})
