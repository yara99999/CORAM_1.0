"""
run_multistart.py
=================
Multi-start global-search driver for the non-convex (desalination
economies-of-scale) national intertemporal model.

We multi-start over IPOPT SOLVER SETTINGS (barrier/scaling path) from the single
factorisable warm-start — perturbing the primal start crashes MUMPS on the
ill-conditioned KKT.  Each start runs as its OWN SUBPROCESS so that a MUMPS
segfault only loses that seed; the model appends one row per start to a shared
CSV (seed, objective, feasibility, physical totals).

Env vars:
  MS_N       number of starts             (default 100)
  MS_SCEN    No_GW | 10_BCM | 5_BCM       (default 10_BCM)
  A0_MAXITER iteration cap per start       (default 300)
  MS_CSV     output CSV path              (default scratch/multistart_<scen>.csv)
"""
import os, sys, io, time, subprocess

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE     = os.path.dirname(os.path.abspath(__file__))
MODEL    = os.path.join(HERE, "run_intertemporal_model.py")
REPO_ROOT = os.path.dirname(HERE)
N        = int(os.environ.get("MS_N", "100"))
SCEN     = os.environ.get("MS_SCEN", "10_BCM")
if SCEN not in ("No_GW", "10_BCM", "5_BCM"):
    sys.exit(f"MS_SCEN must be No_GW/10_BCM/5_BCM, got {SCEN}")
_DIAG_IDX = {"No_GW": "0", "10_BCM": "1", "5_BCM": "2"}[SCEN]

_SCRATCH = os.path.join(REPO_ROOT, "results", "multistart")
os.makedirs(_SCRATCH, exist_ok=True)
CSV      = os.environ.get("MS_CSV", os.path.join(_SCRATCH, f"multistart_{SCEN}.csv"))
if os.path.exists(CSV):
    os.remove(CSV)

base_env = dict(os.environ)
base_env.update({
    "FORCE_RESOLVE":   "1",
    "DIAG_ONLY":       _DIAG_IDX,
    "DIAG_PRINT_LEVEL":"0",
    "A0_MAXITER":      os.environ.get("A0_MAXITER", "300"),
    "MS_CSV":          CSV,
})

print(f"Multi-start (solver-param): scenario={SCEN} N={N} maxiter={base_env['A0_MAXITER']}\n"
      f"CSV -> {CSV}\n", flush=True)

t_start = time.time()
ok = crash = 0
for seed in range(1, N + 1):
    env = dict(base_env); env["MS_SEED"] = str(seed)
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, "-u", MODEL], cwd=HERE, env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=900)
        rc = r.returncode
    except subprocess.TimeoutExpired:
        rc = -999
    if rc == 0:
        ok += 1; tag = "ok"
    else:
        crash += 1; tag = f"CRASH/rc={rc}"
    print(f"[{seed}/{N}] {tag}  {time.time()-t0:.0f}s  "
          f"(ok={ok} crash={crash}, elapsed {(time.time()-t_start)/60:.1f} min)", flush=True)

print(f"\nDone: {ok} ok, {crash} crashed/failed of {N} in "
      f"{(time.time()-t_start)/60:.1f} min. CSV: {CSV}", flush=True)
