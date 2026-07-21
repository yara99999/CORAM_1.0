# Saudi Arabia Water-Energy-Food Nexus Model (National)

A national-scale optimization model of Saudi Arabia's water, energy, and food
system, built in [Pyomo](http://www.pyomo.org/) and solved with
[IPOPT](https://coin-or.github.io/Ipopt/). It jointly optimizes water
sourcing (groundwater, desalination, surface dams, treated wastewater),
electricity generation, and domestic crop production/food imports across
Saudi Arabia's 2023-2060 planning horizon, under configurable groundwater
extraction limits and energy-mix scenarios.

This repository contains two models that work together:

| Script | What it does | Horizon |
|---|---|---|
| `model/run_calibration_sequential.py` | **Calibration.** Solves each year 2023-2060 independently (myopic/sequential), tracking inherited capacity from prior investment decisions. Its 2023 output is the observed-data-calibrated baseline the intertemporal model needs. | 1 year at a time |
| `model/run_intertemporal_model.py` | **Future projection.** Solves 2024-2060 as a single simultaneous NLP with an NPV objective (5% discount rate) and explicit groundwater stock depletion. Warm-started from the calibration run's results. | 2024-2060, one shot |

Run the calibration script first — the intertemporal model reads its output
as its 2023 starting point and per-year warm-start.

---

## Quick start

```bash
pip install -r requirements.txt
# IPOPT itself is not a pip package — install via conda (recommended):
conda install -c conda-forge ipopt
```

```bash
cd model

# Step 1: calibration (fast quick-start mode — one scenario, all years)
TEST_SINGLE=1 python run_calibration_sequential.py

# Step 2: intertemporal projection (quick-start scope: STANDARD energy x all 3 GW scenarios)
python run_intertemporal_model.py
# For just one scenario (STANDARD / no GW cap only), set TEST_SINGLE=1 here too:
# TEST_SINGLE=1 python run_intertemporal_model.py
```

On Windows (PowerShell), set env vars separately: `$env:TEST_SINGLE=1; python run_calibration_sequential.py`.

The calibration quick-start (STANDARD, no GW cap, all 38 years) takes about
15-30 minutes. The intertemporal script's default scope (all 3 groundwater
caps, one 37-year solve each) takes longer — expect tens of minutes;
`TEST_SINGLE=1` narrows it to just the no-cap scenario if you want a fast
correctness check instead. Results land in `results/calibration/` and
`results/intertemporal/`.

**Read the "Non-convexity and multi-start" section below before treating any
single intertemporal run as a final result** — a single solve can land in a
poor local optimum, and the recommended way to get a reliable result is the
multi-start driver, not a single `python run_intertemporal_model.py` call.

### Full batch

Both scripts run all 3 groundwater scenarios (no cap / 10 BCM / 5 BCM) by
default — set `TEST_SINGLE=1` to narrow either one down to a single
scenario instead. **The calibration script's full batch is substantially
longer than its quick-start**: 38 years × 3 GW scenarios = 114 individual
solves, versus 38 for the `TEST_SINGLE=1` quick-start. Both scripts support
resuming: set `SKIP_EXISTING = True` (calibration script's default) or just
re-run — already-solved (year, scenario) combinations are skipped.

---

## Repository structure

```
model/
  run_calibration_sequential.py   # calibration / sequential model
  run_intertemporal_model.py      # 2024-2060 single-shot projection
  run_multistart.py               # multi-start driver (see below)
  plot_multistart.py              # analyzes/plots a multi-start CSV
data/
  2023/                           # crop, climate, water-user, water-source,
  2030/ 2040/ 2050/ 2060/         #   investment, and energy-mix data by year
  Population_clean.xlsx
  Calibration_Parameters.xlsx     # 2023 shadow-price duals (PMP calibration)
  Calibrated_Logit for 2023.xlsx  # logit coefficients (crop-mix/import-mix)
  Water_Demand_Anchors.xlsx       # domestic/industrial demand anchor points (BCM/yr)
  Desalination_Fleet_Capacity_Aggregated.xlsx   # see note below
results/
  calibration/                    # calibration script writes here
  intertemporal/                  # intertemporal model writes here
  multistart/                     # multi-start driver writes CSVs + figures here
requirements.txt
LICENSE
```

## A note on the desalination data

The original desalination fleet dataset is plant-by-plant (named facilities,
exact GPS coordinates, individual commissioning/retirement years, and
commercially-sourced capacity/cost figures) and is **not included** in this
release. In its place, `data/Desalination_Fleet_Capacity_Aggregated.xlsx`
provides the derived aggregate this model actually needs: total RO/MSF/MED
capacity (BCM/yr) for every year 2023-2060, computed from the underlying
plant-level data. Both scripts' `get_desal_capacity_bcm(year)` function
returns identical values whether it reads the original plant list or this
aggregated table — nothing about the model's behavior changes, only the
publishability of the input data.

## Scenarios

- **Energy scenario:** `STANDARD` only (renewable-share trajectory and power-plant cost/mix assumptions).
- **Groundwater scenarios** (`GW_SCENARIOS`): no national cap, a 10 BCM/yr cap, or a 5 BCM/yr cap on groundwater extraction from 2024 onward.
- Output filenames encode both: `{year}_{energy_scenario}_{gw_label}.xlsx`, e.g. `2035_STANDARD_5_BCM_GW_Limit.xlsx`.

## Non-convexity and multi-start

The intertemporal model's desalination cost function has economies of scale
(concave), which makes the full 37-year NLP **non-convex with multiple local
optima**. A single `python run_intertemporal_model.py` call can converge to
a poor local optimum — you may see a `Solver Status` of `maxIterations`
alongside an implausibly large `NPV Objective (all yrs)` figure, even though
the underlying per-year physical quantities (population, water demand,
groundwater use, land area) and individual annual cost components look
completely normal. That is not a bug or a convergence failure to "fix" —
it's the model landing in a bad basin of a genuinely non-convex problem, and
it's expected behavior for a single cold-start solve.

**`run_multistart.py`** is the intended way to get a reliable result: it
solves the same scenario from many different IPOPT barrier/scaling
configurations (each a separate subprocess, crash-isolated) and records
every attempt's objective and feasibility to a CSV. **`plot_multistart.py`**
then reads that CSV, filters to feasible starts, reports the best (lowest
feasible objective) seed, and plots the spread across local optima so you
can judge how reproducible the result actually is.

```bash
cd model

# Run N starts for one GW scenario (No_GW / 10_BCM / 5_BCM)
MS_N=20 MS_SCEN=10_BCM A0_MAXITER=300 python run_multistart.py

# Analyze the results
python plot_multistart.py ../results/multistart/multistart_10_BCM.csv
```

`plot_multistart.py` prints the best feasible seed, e.g.
`BEST feasible: seed=42 obj=7.1e11 ...` — to materialize that seed's full
result files (not just the summary CSV row), re-run the model with that seed
fixed: `MS_SEED=42 python run_intertemporal_model.py`. Each start in the
20-100 range typically takes a few minutes; larger `MS_N` gives a more
reliable "best" but takes proportionally longer (`run_multistart.py` runs
starts sequentially, one subprocess at a time).

## Citation

If you use this model in your own work, please cite [add citation details
here — thesis/paper reference].

## License

MIT — see [LICENSE](LICENSE).
