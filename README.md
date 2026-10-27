# CORAM v1.0: Cost-Optimized Resource Allocation Model (Saudi Arabia, national)

CORAM is a nonlinear, intertemporal optimization model of Saudi Arabia's
water–energy–food system. It minimizes the discounted cost of securing water,
food and the associated energy over 2024–2060 in a single solve, jointly
choosing water allocation by source and sector, investment in new reverse
osmosis, wastewater treatment, dams and domestic water-loss reduction, domestic
crop production and food imports. Electricity use and CO₂ emissions of the
water sector are computed from an exogenous grid pathway. The model is written
in [Pyomo](http://www.pyomo.org/) and solved with
[IPOPT](https://coin-or.github.io/Ipopt/), and is calibrated to 2023.

This repository accompanies the paper *Development of Cost Optimization
Resource Allocation Model (CORAM) version 1.0: A Case Study of Saudi Arabia*
(Elborolosy, Hejazi and Wada, submitted to *Geoscientific Model Development*).

## Installation

```bash
conda env create -f environment.yml
conda activate coram
```

`environment.yml` pins the versions used for the paper (Python 3.11,
Pyomo 6.7.3, IPOPT 3.14 from conda-forge, pandas 2.2.2, NumPy 1.26.4).
`requirements.txt` lists the same Python packages for pip, but IPOPT itself
must come from conda (`conda install -c conda-forge ipopt=3.14`).

On Windows, run the model only inside the activated environment: IPOPT and
NumPy need the environment's library folders on `PATH`.

## Running the model

```bash
cd Model
bash run_final_batch.sh          # Linux/macOS/Git Bash
# .\run_final_batch.ps1          # Windows PowerShell
```

The batch script runs, in order:

1. the 2023 calibration (a single-year solve of the same model with existing
   infrastructure only) → `Results/calibration_2023/`;
2. the no-groundwater-limit scenario, 2024–2060 → `Results/intertemporal/`;
3. the 10 BCM and 5 BCM groundwater-cap scenarios, started from the
   no-limit solution and from `Results/warmstart_5BCM/` respectively
   → `Results/intertemporal/`.

It then prints the headline numbers (`results_numbers.py`) and the 2023
calibration fit (`calib_fit.py`). A full run takes roughly 20–40 minutes on a
laptop. Each scenario writes one Excel file per year
(`{year}_STANDARD_{scenario}.xlsx`) with water production and allocation,
crop production and imports, energy use, power plants, investments, costs and
a summary sheet. Solves are deterministic: the same inputs give the same
results.

To run a single scenario directly:

```bash
cd Model
# 0 = no groundwater limit, 1 = 10 BCM cap, 2 = 5 BCM cap
DIAG_ONLY=0 FORCE_RESOLVE=1 RESULTS_DIR_IT=../Results/my_run python run_intertemporal_model.py
```

Without `RESULTS_DIR_IT` the model writes to `Results_scratch/`.

## Main settings

All settings are environment variables; the defaults are those used in the paper.

| Variable | Default | Meaning |
|---|---|---|
| `DIAG_ONLY` | all | Run one groundwater scenario (0, 1 or 2) |
| `RESULTS_DIR_IT` | `Results_scratch` | Output folder |
| `CALIB_MODE` | 0 | 1 = single-year 2023 calibration solve |
| `WS_SOURCE_DIR`, `WS_SOURCE_GW` | sequential results | Warm-start folder and scenario (the batch script starts the capped scenarios from the no-limit solution) |
| `WATER_TRANSPORT_FRAC` | 1.38 | Water conveyance and distribution cost as a multiple of production cost (58 % of total cost) |
| `IRR_EFF` | 0.65 | Irrigation delivery efficiency |
| `ENABLE_NRW` | 1 | Domestic water-loss reduction investment |
| `EXISTING_DESAL_MIN_UTIL` | 0.95 | Minimum use of existing desalination before new reverse osmosis is built |
| `GW_RAMP_YEARS` | 0 | Years over which the groundwater cap is phased in (0 = immediate) |
| `DISCOUNT_RATE` | 0.05 | Social discount rate |

## Repository structure

```
Model/
  run_intertemporal_model.py      # the model (2024-2060 projection and 2023 calibration)
  run_final_batch.sh / .ps1       # calibration + three scenarios + summary
  results_numbers.py              # headline numbers of a results folder
  calib_fit.py                    # 2023 model-versus-observed fit
  Calculations/dates_floor_duals.csv   # weights of the date-production penalty
  Data/                           # inputs: 2023 crops, climate, water users and sources,
                                  #   investment options, population, energy pathways
  Data/2023 - Regional/Desal_Plants_Regional_SWPC.xlsx   # desalination plant list (see below)
  Results/2023 Calibration Results.xlsx                  # observed 2023 data used by calib_fit.py
Calibration_Parameters.xlsx       # PMP shadow prices from the 2023 calibration
Calibrated_Logit for 2023.xlsx    # crop-mix logit coefficients
Results/Results_v2_vintage_sequential/   # 2023 baseline and warm start (see below)
Results/warmstart_5BCM/                  # starting point for the 5 BCM solve (see below)
```

## Data notes

**Desalination plants.** Existing, under-construction and planned plants
(62 plants, commissioning and decommissioning years, capacity and production
factor) come from the Saudi Water Partnership Company's 7-Year Statement
2025–2031, with SWCC plant data where the Statement gives no value. A plant
operates from its commissioning year until the year before its
decommissioning year. **Plant-level unit costs are confidential and are not
distributed.** Every plant is assigned its technology's representative cost
(the capacity-weighted average of the plant-level costs): reverse osmosis
US$0.533 m⁻³, multi-stage flash US$1.093 m⁻³ and multi-effect distillation
US$1.012 m⁻³. Results obtained with this release therefore differ slightly
from those reported in the paper, which use plant-level costs (see
*Reproducing the paper* below). Plant coordinates are not included.

**Baseline and warm start.** `Results/Results_v2_vintage_sequential/` holds the
no-limit results of an earlier year-by-year (sequential) calibration of CORAM.
The model uses its 2023 file as the observed starting point (irrigated area
and crop mix) and its 2024–2060 files as the starting point for the solver.

**Other inputs** are compiled from MEWA, the General Authority of Statistics,
FAO, ERA5, the United Nations population projections, Superwell (Niazi et al.,
2024) and the electricity-system scenarios of Durand-Lasserve (2025); see the
paper for the full list of sources.

**5 BCM warm start.** `Results/warmstart_5BCM/` holds the paper's 5 BCM
solution in the form the model reads as a starting point (irrigated area,
groundwater use, water by user and source with desalination as one unit per
technology, crops and investments; no costs). With technology-average
desalination costs, IPOPT does not converge for the 5 BCM scenario when started
from the no-limit or 10 BCM solution, so the batch script starts it from this
folder. The starting point does not change the optimum.

## Reproducing the paper

`bash run_final_batch.sh` with this release gives the following, compared with
the paper (plant-level desalination costs). Costs are discounted at 5 % to
2024, in billion US$.

| | No limit | 10 BCM | 5 BCM |
|---|---|---|---|
| Total cost, paper | 976 | 1,008 | 1,019 |
| Total cost, this release | 978 | 1,009 | 1,020 |
| Water supply cost, paper | 112.5 | 138.8 | 172.4 |
| Water supply cost, this release | 113.9 | 140.3 | 174.0 |

Only the water supply cost changes (by about 1 %), because existing
desalination is priced at its technology average. Crop production and imports,
water production by source and sector, the desalination fleet and new reverse
osmosis capacity, electricity use, CO₂ emissions, investment and the 2023
calibration fit are the same as in the paper to the reported precision. The one
visible difference is timing in the 10 BCM scenario: new dam capacity is fully
used from 2038 instead of 2046.

The model outputs used in the paper (one Excel file per year and scenario,
with desalination grouped into existing RO, planned RO, MSF, MED and new RO)
and the paper figures are included in the Zenodo archive of this release.

## Citation

If you use CORAM, please cite the paper above and this code archive:

Elborolosy, Y., Hejazi, M., and Wada, Y.: CORAM v1.0: Cost-Optimized Resource
Allocation Model for Saudi Arabia (code, inputs and outputs), Zenodo [code],
https://doi.org/10.5281/zenodo.23693000, 2026.

## License

MIT; see [LICENSE](LICENSE).
