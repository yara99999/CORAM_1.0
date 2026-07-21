"""
plot_multistart.py
==================
Visualize the multi-start local-optima distribution for the non-convex national
model, so we can judge how far apart the local optima are (i.e. whether the
solution is effectively reproducible).

Usage: python plot_multistart.py [csv_path]
Default csv: scratch/multistart_10_BCM.csv   (override with 1st arg or MS_CSV)
"""
import sys, os, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
_SCRATCH  = os.path.join(REPO_ROOT, "results", "multistart")
csv = (sys.argv[1] if len(sys.argv) > 1
       else os.environ.get("MS_CSV", os.path.join(_SCRATCH, "multistart_10_BCM.csv")))
OUT = os.path.join(REPO_ROOT, "results", "multistart", "figures")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(csv)
df = df[np.isfinite(df["objective"])]
scen = df["scenario"].iloc[0] if len(df) else "?"

# Feasibility: use RELATIVE violation (each constraint normalized by its own
# scale) — the absolute metric is dominated by a large constant tonnage
# constraint.  A converged local optimum has tiny relative violation.
viol_cut = float(os.environ.get("MS_VIOL_CUT", "1e-3"))
_vcol = "relviol" if "relviol" in df.columns else "maxviol"
feas = df[df[_vcol] < viol_cut]

print(f"CSV: {csv}")
print(f"Total starts recorded: {len(df)}   feasible(<{viol_cut}): {len(feas)}")
if len(feas):
    o = feas["objective"].values
    print(f"Feasible objective: min={o.min():.4e}  max={o.max():.4e}  "
          f"median={np.median(o):.4e}  spread={ (o.max()-o.min())/o.min()*100:.2f}% of min")
    best = feas.loc[feas["objective"].idxmin()]
    print(f"BEST feasible: seed={int(best['seed'])} obj={best['objective']:.6e} "
          f"GW_sum={best['gw_bcm_sum']:.1f} desal_sum={best['desal_bcm_sum']:.1f} "
          f"domcrop_Mt={best['domcrop_tons_sum']/1e6:.1f} import_Mt={best['import_tons_sum']/1e6:.1f}")

fig, ax = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(f"Multi-start local optima — {scen}  (N={len(df)} starts)",
             fontsize=13, fontweight="bold")

# (a) objective histogram — all vs feasible
ax[0, 0].hist(df["objective"], bins=30, color="lightgray", label="all starts")
if len(feas):
    ax[0, 0].hist(feas["objective"], bins=30, color="tab:blue", label=f"feasible (<{viol_cut})")
ax[0, 0].set_title("Objective distribution"); ax[0, 0].set_xlabel("Objective (penalty-incl.)")
ax[0, 0].set_ylabel("count"); ax[0, 0].legend(fontsize=8)

# (b) objective vs violation (convergence view)
ax[0, 1].scatter(df[_vcol] + 1e-12, df["objective"], s=18, c="tab:red", alpha=0.6)
ax[0, 1].set_xscale("log"); ax[0, 1].axvline(viol_cut, ls="--", c="k", lw=0.8)
ax[0, 1].set_title(f"Objective vs {_vcol} (convergence)")
ax[0, 1].set_xlabel(f"{_vcol} (relative)"); ax[0, 1].set_ylabel("objective")

# (c) feasible objective — zoom
if len(feas):
    ax[1, 0].hist(feas["objective"], bins=20, color="tab:blue")
    ax[1, 0].axvline(feas["objective"].min(), color="green", ls="--", label="best")
    ax[1, 0].set_title(f"Feasible optima (n={len(feas)}) — spread "
                       f"{(feas['objective'].max()-feas['objective'].min())/feas['objective'].min()*100:.1f}%")
    ax[1, 0].set_xlabel("objective"); ax[1, 0].legend(fontsize=8)
else:
    ax[1, 0].text(0.5, 0.5, "no feasible starts yet", ha="center")

# (d) physical spread among feasible: domestic crop vs imports (Mt)
if len(feas):
    ax[1, 1].scatter(feas["domcrop_tons_sum"] / 1e6, feas["import_tons_sum"] / 1e6,
                     s=25, c="tab:green", alpha=0.7)
    ax[1, 1].set_title("Physical spread (feasible starts)")
    ax[1, 1].set_xlabel("Σ domestic crop (Mt)"); ax[1, 1].set_ylabel("Σ imports (Mt)")
else:
    ax[1, 1].text(0.5, 0.5, "no feasible starts yet", ha="center")

fig.tight_layout(rect=[0, 0, 1, 0.96])
path = os.path.join(OUT, f"multistart_{scen}.png")
fig.savefig(path, dpi=140)
print("Saved:", path)
