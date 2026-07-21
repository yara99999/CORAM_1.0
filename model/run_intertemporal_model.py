"""
run_intertemporal_model.py
==========================
Single-shot 2024-2060 intertemporal water-energy-food nexus NLP.

Key differences vs sequential myopic model:
  - ALL years 2024-2060 optimised simultaneously in one solve
  - 5 % annual discount rate (NPV objective)
  - Explicit groundwater stock depletion  (non-renewable aquifer, 2500 BCM)
  - Investment decisions are continuous (≥ 0 real-valued) and year-indexed;
    capacity accumulates linearly across cohorts within PLANT_LIFETIME window
  - PMP temporal penalties kept at 50 % weight (single-shot solver already
    sees the full time horizon and needs less smoothing)
  - Warm-started from sequential-model xlsx results for faster convergence

Output: one xlsx per (year, energy_scenario, gw_label) in Results_intertemporal/
        — identical sheet structure to Results_v2_vintage_sequential/ so the
        same plotting scripts work on both result sets.

Run order  : STANDARD  ×  No_GW_Limit → 10_BCM → 5_BCM
Test first : set TEST_SINGLE = True to solve only STANDARD / No_GW_Limit.
"""

import math
import os
import time
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from pyomo.environ import (
    ConcreteModel, sqrt, Var, Objective, Constraint, NonNegativeReals,
    SolverFactory, minimize, value, log, exp, Suffix,
)
from pyomo.opt import TerminationCondition


# =============================================================================
# CONFIGURATION
# =============================================================================

YEARS            = list(range(2024, 2061))    # 37 annual time steps
ENERGY_SCENARIOS = ["STANDARD"]
GW_SCENARIOS     = [
    (False, False),   # No groundwater limit
    (True,  False),   # 10 BCM limit
    (False, True),    # 5 BCM limit
]

# Run-scope flag: TEST_SINGLE=1 solves only STANDARD / No_GW_Limit; otherwise
# all 3 GW scenarios run (STANDARD is the only energy scenario).
TEST_SINGLE = os.environ.get("TEST_SINGLE", "0") == "1"

DISCOUNT_RATE        = float(os.environ.get("DISCOUNT_RATE", "0.05"))    # social discount rate (5 % per year)
PMP_TEMPORAL_WEIGHT  = float(os.environ.get("PMP_TEMPORAL_WEIGHT", "0.5"))  # multiplier on temporal PMP penalty (year-over-year smoothing); raise to iron out sub-% crop-band wiggles
PLANT_LIFETIME       = 40     # flat vintage curve — dams and power plants
GW_INITIAL_STOCK_BCM = 2500   # BCM — Saudi non-renewable aquifer stock
# Grid CO2 calibration: anchor the 2023 capacity-weighted grid emission factor to
# the official DNA/KFUPM Grid Emission Factor (0.568 tCO2/MWh; KFUPM GHG report
# 2024). Bottom-up 2023 mix = 0.628 tCO2/MWh → scale per-fuel factors by 0.568/0.628.
DNA_GRID_EF_2023     = 0.568   # tCO2/MWh (official)
_MODEL_GRID_EF_2023  = 0.628   # tCO2/MWh (bottom-up capacity-weighted, 2023 mix)
CO2_GRID_CALIB       = DNA_GRID_EF_2023 / _MODEL_GRID_EF_2023   # ≈ 0.9045
# Carbon damage cost (social cost of carbon) internalised in the objective, per
# manuscript Eq. 12: C_d = emissions × C_CO2.  C_CO2 = $47/tCO2 for Saudi Arabia
# (Ricke et al., 2018).  Set 0 to disable (report-only).
CARBON_DAMAGE_USD_PER_T = float(os.environ.get("CARBON_DAMAGE_USD_PER_T", "47"))

# Treated-wastewater REUSE efficiency ramp for EXISTING plants.  The base-year
# reuse rate is low (0.185, matching 2023 reuse) because reuse distribution
# infrastructure was limited; as that network matures, existing plants recover a
# growing share of treated wastewater.  The ramp follows the observed agricultural
# reuse growth (87->422 Mm3, 2000-2022, ~7%/yr) extended forward and CAPPED at 0.80.
# The 2023 anchor (0.185) preserves the base-year calibration.  New plants keep
# their design efficiency (0.90).  Env-tunable via TWW_EFF_<year>.
TWW_EFF_ANCHORS = {2023: 0.185, 2030: 0.30, 2040: 0.60, 2045: 0.80, 2060: 0.80}
TWW_EFF_ANCHORS = {y: float(os.environ.get(f"TWW_EFF_{y}", v)) for y, v in TWW_EFF_ANCHORS.items()}
_tww_eff_x = sorted(TWW_EFF_ANCHORS)
_tww_eff_y = [TWW_EFF_ANCHORS[k] for k in _tww_eff_x]
def tww_existing_eff(year):
    """Year-dependent reuse efficiency for the existing TWW fleet (ramps 0.185->0.80)."""
    return float(np.interp(year, _tww_eff_x, _tww_eff_y))

# TWW retirement: not modelled — existing fleet runs at full capacity for entire
# horizon, matching the sequential model convention.


# =============================================================================
# PATH CONSTANTS
# =============================================================================

# All paths are relative to this repository -- clone it anywhere and it runs.
MODEL_DIR       = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT       = os.path.dirname(MODEL_DIR)
DATA_DIR        = os.path.join(REPO_ROOT, "data")
SEQ_RESULTS_DIR = os.environ.get("SEQ_RESULTS_DIR", os.path.join(REPO_ROOT, "results", "calibration"))
RESULTS_DIR_IT  = os.environ.get("RESULTS_DIR_IT",  os.path.join(REPO_ROOT, "results", "intertemporal"))
LOG_FILE        = os.path.join(MODEL_DIR, "run_log_intertemporal.txt")

# NOTE: the original SWCC plant-by-plant fleet file is not published (each row
# identifies a named facility with exact GPS coordinates). SWCC_FILE instead
# points to a derived, aggregated table (Year, RO/MSF/MED capacity in BCM/yr)
# computed from it -- see the "SWCC desalination fleet" block below for how
# it's consumed; get_desal_capacity_bcm() returns identical values either way.
SWCC_FILE  = os.path.join(DATA_DIR, "Desalination_Fleet_Capacity_Aggregated.xlsx")
POP_FILE   = os.path.join(DATA_DIR, "Population_clean.xlsx")
CALIB_FILE = os.path.join(DATA_DIR, "Calibration_Parameters.xlsx")
LOGIT_FILE = os.path.join(DATA_DIR, "Calibrated_Logit for 2023.xlsx")


# =============================================================================
# PRE-COMPUTED LOOKUPS  (identical to run_all_scenarios_vintage_sequential.py)
# =============================================================================

print("Loading lookup tables...")

# 1. Population
_pop_df     = pd.read_excel(POP_FILE)
_pop_lookup = dict(zip(_pop_df["Year"].astype(int), _pop_df["Population"] * 1000))

def get_population(year):
    year = int(year)
    if year in _pop_lookup:
        return _pop_lookup[year]
    nearest = min(_pop_lookup, key=lambda y: abs(y - year))
    return _pop_lookup[nearest]


# 2. Water demand — piecewise linear interpolation.
#    Anchor values (BCM/yr) loaded from Water_Demand_Anchors.xlsx.
_demand_df = pd.read_excel(os.path.join(DATA_DIR, "Water_Demand_Anchors.xlsx")).sort_values("Year")
_dom_interp = interp1d(_demand_df["Year"], _demand_df["Domestic_BCM"],
                       kind="linear", fill_value="extrapolate")
_ind_interp = interp1d(_demand_df["Year"], _demand_df["Industry_BCM"],
                       kind="linear", fill_value="extrapolate")

def get_demand(year):
    return float(_dom_interp(year)), float(_ind_interp(year))


# 3. Renewable share
_RENEWABLE_ANCHORS = {
    "STANDARD": {2023: 0.012, 2030: 0.225, 2040: 0.236, 2050: 0.250, 2060: 0.280},
}
_ren_interps = {
    sc: interp1d(sorted(anc), [anc[y] for y in sorted(anc)],
                 kind="linear", fill_value="extrapolate")
    for sc, anc in _RENEWABLE_ANCHORS.items()
}

def get_renewable_share(year, scenario):
    return float(np.clip(_ren_interps[scenario](year), 0.0, 1.0))


# 4. SWCC desalination fleet -- pre-aggregated by (year, technology) in BCM/yr
# from the underlying plant-by-plant fleet data (not published; see SWCC_FILE
# note near ROOT). Same lookup semantics as filtering active plants directly.
_swcc = pd.read_excel(SWCC_FILE, sheet_name="Sheet1").set_index("Year")

def get_desal_capacity_bcm(year):
    year = int(year)
    if year not in _swcc.index:
        year = min(_swcc.index, key=lambda y: abs(y - year))
    row = _swcc.loc[year]
    return {
        "RO":  float(row["RO_Capacity_BCM"]),
        "MSF": float(row["MSF_Capacity_BCM"]),
        "MED": float(row["MED_Capacity_BCM"]),
    }


# 5. Energy base data (fixed to 2023 per-scenario file)
_SCENARIO_SUFFIX = {"STANDARD": "Baseline"}

def _load_power_plants_base(scenario):
    suffix = _SCENARIO_SUFFIX[scenario]
    path   = f"{DATA_DIR}/2023/Energy_2023_{suffix}.xlsx"
    if not os.path.exists(path):
        path = f"{DATA_DIR}/2030/Energy_2030_{suffix}.xlsx"
    xl    = pd.ExcelFile(path)
    sheet = "PowerPlants_2" if "PowerPlants_2" in xl.sheet_names else "PowerPlants"
    df    = pd.read_excel(path, sheet_name=sheet)
    return {
        row["Fuel"]: {
            "Capacity (MW)": float(row["Capacity (MW)"]),
            "Cost ($/kWh)":  float(row["Cost ($/kwh)"]),
            "Efficiency":    float(row["Eff"]),
            "CO2 (t/kwh)":  float(row["CO2 (t/kwh)"]),
        }
        for _, row in df.iterrows()
    }

_POWER_PLANTS_BASE = {sc: _load_power_plants_base(sc) for sc in _SCENARIO_SUFFIX}


# 5b. Existing-fleet capacity TRAJECTORY (grows 2023->2060) ------------------
# The base fleet is not static: national installed capacity expands over the
# horizon (esp. renewables 1 GW -> 54 GW).  The year-specific planning files
# (Energy_YYYY_*.xlsx) give aggregate Fossil / Renewable totals at the 2030/
# 2040/2050/2060 milestones; we interpolate linearly between milestones and
# split each aggregate across the individual 2023 fuels in proportion to the
# 2023 per-fuel mix (fossil: NG/Crude/HFO/Diesel; renewable: Solar/Wind).
_FOSSIL_FUELS_2023 = {"NG", "Crude", "Diesel", "HFO", "Oil", "Gas"}

def _milestone_totals(scenario):
    """{year: (fossil_MW, renewable_MW)} at 2023 + planning milestones."""
    suffix = _SCENARIO_SUFFIX[scenario]
    base   = _POWER_PLANTS_BASE[scenario]
    foss23 = sum(v["Capacity (MW)"] for k, v in base.items() if k in _FOSSIL_FUELS_2023)
    ren23  = sum(v["Capacity (MW)"] for k, v in base.items() if k not in _FOSSIL_FUELS_2023)
    tot = {2023: (foss23, ren23)}
    for yr in (2030, 2040, 2050, 2060):
        p  = f"{DATA_DIR}/{yr}/Energy_{yr}_{suffix}.xlsx"
        xl = pd.ExcelFile(p)
        sh = "PowerPlants_2" if "PowerPlants_2" in xl.sheet_names else "PowerPlants"
        g  = pd.read_excel(p, sheet_name=sh).groupby("Fuel")["Capacity (MW)"].sum()
        foss = float(sum(v for k, v in g.items()
                         if str(k).lower() == "fossil" or k in _FOSSIL_FUELS_2023))
        ren  = float(sum(v for k, v in g.items()
                         if str(k).lower() == "renewable" or k in ("Solar", "Wind", "Nuclear")))
        tot[yr] = (foss, ren)
    return tot

def _build_capacity_by_year(scenario):
    """{year: {fuel: MW}} for YEARS, per-fuel from interpolated aggregates."""
    base   = _POWER_PLANTS_BASE[scenario]
    tot    = _milestone_totals(scenario)
    myrs   = sorted(tot)
    f_interp = interp1d(myrs, [tot[y][0] for y in myrs], kind="linear",
                        fill_value=(tot[myrs[0]][0], tot[myrs[-1]][0]), bounds_error=False)
    r_interp = interp1d(myrs, [tot[y][1] for y in myrs], kind="linear",
                        fill_value=(tot[myrs[0]][1], tot[myrs[-1]][1]), bounds_error=False)
    foss23 = tot[2023][0] or 1.0
    ren23  = tot[2023][1] or 1.0
    f_share = {k: base[k]["Capacity (MW)"] / foss23 for k in base if k in _FOSSIL_FUELS_2023}
    r_share = {k: base[k]["Capacity (MW)"] / ren23  for k in base if k not in _FOSSIL_FUELS_2023}
    out = {}
    for y in YEARS:
        fy, ry = float(f_interp(y)), float(r_interp(y))
        cap = {k: fy * sh for k, sh in f_share.items()}
        cap.update({k: ry * sh for k, sh in r_share.items()})
        out[y] = cap
    return out

_CAP_BY_YEAR = {sc: _build_capacity_by_year(sc) for sc in _SCENARIO_SUFFIX}


# 6. Crop base prices (2023)
_base_dom_df = pd.read_excel(f"{DATA_DIR}/2023/Crops_2023.xlsx", sheet_name="Domestic")
_base_dom_df.columns = _base_dom_df.columns.str.lower()
_base_imp_df = pd.read_excel(f"{DATA_DIR}/2023/Crops_2023.xlsx", sheet_name="Import")
_BASE_CULCOST = dict(zip(_base_dom_df["crop"], _base_dom_df["culcost ($/ton)"].astype(float)))
_BASE_IMPCOST = dict(zip(_base_imp_df["Crop"], _base_imp_df["Import Cost ($/ton)"].astype(float)))
del _base_dom_df, _base_imp_df


# 7. Calibration dual prices and logit coefficients (fixed to 2023)
_dp_water    = pd.read_excel(CALIB_FILE, sheet_name="Water_2023")
_dp_domestic = pd.read_excel(CALIB_FILE, sheet_name="Dom_2023")
_dp_import   = pd.read_excel(CALIB_FILE, sheet_name="Imp_2023")
_DUAL_PRICES_WATER    = dict(zip(_dp_water["Source"],  _dp_water["Dual"]))
_DUAL_PRICES_DOMESTIC = dict(zip(_dp_domestic["Crop"], _dp_domestic["Dual"]))
_DUAL_PRICES_IMPORT   = dict(zip(_dp_import["Crop"],   _dp_import["Dual"]))
del _dp_water, _dp_domestic, _dp_import

_logit_df     = pd.read_excel(LOGIT_FILE, sheet_name="Dom_Production")
_logit_df.columns = _logit_df.columns.str.lower()
_logit_imp_df = pd.read_excel(LOGIT_FILE, sheet_name="Import_Production")
_logit_imp_df.columns = _logit_imp_df.columns.str.lower()
_LOGIT_COEF_1        = dict(zip(_logit_df["crop"],     _logit_df["coef_1"]))
_LOGIT_COEF_2        = dict(zip(_logit_df["crop"],     _logit_df["coef_2"]))
_LOGIT_COEF_1_IMPORT = dict(zip(_logit_imp_df["crop"], _logit_imp_df["coef_1"]))
_LOGIT_COEF_2_IMPORT = dict(zip(_logit_imp_df["crop"], _logit_imp_df["coef_2"]))
del _logit_df, _logit_imp_df


# 8. Agronomic parameters (all fixed to 2023)
_crop_file_2023    = f"{DATA_DIR}/2023/Crops_2023.xlsx"
_climate_file_2023 = f"{DATA_DIR}/2023/Climate_2023.xlsx"
_wu_file_2023      = f"{DATA_DIR}/2023/WaterUsers_2023.xlsx"
_inv_file_2023     = f"{DATA_DIR}/2023/Investment.xlsx"

_dom_yield_2023     = pd.read_excel(_crop_file_2023, sheet_name="Domestic")
_dom_yield_2023.columns = _dom_yield_2023.columns.str.lower()
_IMP_CROPS_DF_2023  = pd.read_excel(_crop_file_2023, sheet_name="Import")
_DOMESTIC_CROPS_2023 = _dom_yield_2023["crop"].unique()
_YIELD_PER_HA_2023  = {
    c: float(_dom_yield_2023.loc[_dom_yield_2023["crop"] == c, "yield (tons/ha)"].values[0])
    for c in _DOMESTIC_CROPS_2023
    if not pd.isnull(_dom_yield_2023.loc[_dom_yield_2023["crop"] == c, "yield (tons/ha)"].values[0])
}
_combined_2023   = pd.concat([_dom_yield_2023, _IMP_CROPS_DF_2023.rename(columns=str.lower)], ignore_index=True)
_CROP_CATEGORIES = _combined_2023.set_index("crop")["category"].to_dict()
_CROP_CALORIES   = _combined_2023.set_index("crop")["calories per ton"].to_dict()
_CROP_PROTEIN    = _combined_2023.set_index("crop")["protein per ton"].to_dict()
del _combined_2023

_crop_coef_2023  = pd.read_excel(_crop_file_2023, sheet_name="kc")
_phase_month_raw = pd.read_excel(_crop_file_2023, sheet_name="PhaseMonth")
_et0_2023        = pd.read_excel(_crop_file_2023, sheet_name="ET0")
_rain_2023       = pd.read_excel(_climate_file_2023, sheet_name="Rain")
_rain_2023.columns = _rain_2023.columns.str.lower()
_rain_2023 = _rain_2023[["month", "peff (mm)", "ro (mm)"]]
_pm_mcols = _phase_month_raw.columns[3:]
_ph = _phase_month_raw.copy()
for _mc in _pm_mcols:
    _ph[_mc] = _ph["phasefactor"] * _ph[_mc]
_PHASE_PRODUCT_2023 = _ph.groupby("Crop")[_pm_mcols].sum().reset_index()
_crop_coef_2023.columns = _crop_coef_2023.columns.str.lower()
_PHASE_PRODUCT_2023.columns = _PHASE_PRODUCT_2023.columns.str.lower()
_month_cols_2023 = [c.lower() for c in _pm_mcols]
_cwr_parts = []
for _mc in _month_cols_2023:
    _mg = _crop_coef_2023.merge(_PHASE_PRODUCT_2023[["crop", _mc]], on="crop", how="left")
    if f"{_mc}_x" in _mg.columns:
        _mg = _mg.rename(columns={f"{_mc}_x": "kc"})
    elif _mc in _mg.columns:
        _mg = _mg.rename(columns={_mc: "kc"})
    if f"{_mc}_y" in _mg.columns:
        _mg = _mg.rename(columns={f"{_mc}_y": _mc})
    _mg = _mg.merge(_dom_yield_2023[["crop", "irrarea (ha)"]], on="crop", how="left")
    _et0_val  = _et0_2023[_mc].values[0]
    _peff_val = _rain_2023.loc[_rain_2023["month"].str.lower() == _mc, "peff (mm)"].values[0]
    _mg["Crop Water Requirement (m/month)"] = (
        (_mg["kc"] * _mg[_mc] * _et0_val - _peff_val) / 1000
    ).clip(lower=0)
    _mg["Month"] = _mc
    _cwr_parts.append(_mg)
_CROP_WATER_REQS_2023 = pd.concat(_cwr_parts, ignore_index=True)
_CROP_WATER_REQ_DICT  = _CROP_WATER_REQS_2023.groupby("crop")["Crop Water Requirement (m/month)"].sum().to_dict()
del _crop_coef_2023, _phase_month_raw, _et0_2023, _rain_2023, _ph, _cwr_parts

_agr_2023 = pd.read_excel(_wu_file_2023, sheet_name="Agriculture"); _agr_2023.columns = _agr_2023.columns.str.lower()
_dom_2023 = pd.read_excel(_wu_file_2023, sheet_name="Domestic");    _dom_2023.columns = _dom_2023.columns.str.lower()
_ind_2023 = pd.read_excel(_wu_file_2023, sheet_name="Industrial");  _ind_2023.columns = _ind_2023.columns.str.lower()
_IRR_EFF_2023 = float(_agr_2023["irreff"].iloc[0])
_DOM_EFF_2023 = float(_dom_2023["domeff"].iloc[0])
_IND_EFF_2023 = float(_ind_2023["indeff"].iloc[0])
del _agr_2023, _dom_2023, _ind_2023

_water_inv_2023 = pd.read_excel(_inv_file_2023, sheet_name="WaterSources")
_power_inv_2023 = pd.read_excel(_inv_file_2023, sheet_name="PowerPlant")
_power_inv_2023 = _power_inv_2023.rename(columns={"OppCost ($/kwh)": "Cost ($/kwh)", "Capacity": "Capacity (MW)"})
_NEW_WATER_COST      = dict(zip(_water_inv_2023["Name"], _water_inv_2023["Opex Cost ($/m3)"]))
_NEW_WATER_EFF       = dict(zip(_water_inv_2023["Name"], _water_inv_2023["Eff"]))
_NEW_WATER_ENERGY_D  = dict(zip(_water_inv_2023["Name"], _water_inv_2023["Energy (kwh/m3)"]))
_NEW_WATER_CAPEX     = {row["Name"]: float(row["Capex (Million USD)"]) * 1e6
                        for _, row in _water_inv_2023.iterrows()}
_NEW_WATER_UNIT_SIZE = dict(zip(_water_inv_2023["Name"], _water_inv_2023["Capacity (BCM)"]))
_NEW_POWER_COST      = dict(zip(_power_inv_2023["Name"], _power_inv_2023["Cost ($/kwh)"]))
_NEW_POWER_EFF       = dict(zip(_power_inv_2023["Name"], _power_inv_2023["Eff"]))
_NEW_POWER_CO2       = dict(zip(_power_inv_2023["Name"], _power_inv_2023["CO2 (t/kwh)"]))
_NEW_POWER_CAP       = dict(zip(_power_inv_2023["Name"], _power_inv_2023["Capacity (MW)"]))
del _water_inv_2023, _power_inv_2023

# 2023 water source raw data (capacities, costs, efficiencies)
_WATERSOURCES_2023_RAW = pd.read_excel(f"{DATA_DIR}/2023/WaterSources_2023.xlsx", sheet_name=None)

print("Lookup tables ready.")


# =============================================================================
# STATIC WATER-SOURCE STRUCTURES  (computed once from 2023 data)
# =============================================================================

# Build static water supply list from 2023 raw data (same structure as sequential)
_STATIC_WATER_SUPPLY = {}
_STATIC_SOURCE_EFF   = {}
_STATIC_ENERGY_FACTORS = {}
_STATIC_SOURCE_CAPS  = {}   # {source_name: BCM}  — excluding desal (year-specific)

_eff_col_map = {
    "Desalination": "desaleff (%)",
    "SurfaceWater": "dameff (%)",
    "TreatedWW":    "twweff (%)",
}

for _st, _df in _WATERSOURCES_2023_RAW.items():
    _df = _df.copy()
    _df.columns = _df.columns.str.lower()
    _STATIC_WATER_SUPPLY[_st] = []
    for _, _row in _df.iterrows():
        _entry = {"name": _row["name"], "capacity (BCM)": float(_row["capacity (m3)"]) / 1e9}
        if _st in ["TreatedWW", "SurfaceWater"]:
            _entry["cost"] = float(_row["cost ($/m3)"])
        _STATIC_WATER_SUPPLY[_st].append(_entry)
        _skey = f"{_st}_{_row['name']}"
        _STATIC_SOURCE_CAPS[_skey] = float(_row["capacity (m3)"]) / 1e9

    _eff_col = _eff_col_map.get(_st)
    if _eff_col and _eff_col in _df.columns:
        for _, _row in _df.iterrows():
            _STATIC_SOURCE_EFF[f"{_st}_{_row['name']}"] = float(_row[_eff_col])

    _energy_col = next((c for c in _df.columns if "energy (kwh/m3)" in c.lower()), None)
    if _energy_col:
        for _, _row in _df.iterrows():
            _STATIC_ENERGY_FACTORS.setdefault(_st, {})[_row["name"]] = float(_row[_energy_col])

# New-plant efficiencies added below (after _NEW_SHORT_TO_FULL is defined, ~line 455)
# — see that block for why the short->full key mapping matters here too.

# Precompute SWCC desal capacities for every model year
_DESAL_CAPS_BY_YEAR = {y: get_desal_capacity_bcm(y) for y in YEARS}

# User-source type mapping (same as sequential)
_USER_SOURCE_TYPES = {
    "Agriculture": ["Groundwater", "SurfaceWater", "Desalination", "TreatedWW"],
    "Industry":    ["Groundwater", "Desalination", "TreatedWW"],
    "Domestic":    ["Groundwater", "Desalination"],
}
_WATER_NEW_PLANTS = [
    "SurfaceWater_NewSaudiDam", "TreatedWW_NewSaudiTWW", "Desalination_NewRO"
]

# Build expanded_user_sources (identical algorithm to sequential run_model)
_EXPANDED_USER_SOURCES = {}
for _user, _types in _USER_SOURCE_TYPES.items():
    _EXPANDED_USER_SOURCES[_user] = []
    for _st in _types:
        if _st in _STATIC_WATER_SUPPLY:
            _EXPANDED_USER_SOURCES[_user].extend(
                [f"{_st}_{s['name']}" for s in _STATIC_WATER_SUPPLY[_st]]
            )
    for _np in _WATER_NEW_PLANTS:
        if _np not in _EXPANDED_USER_SOURCES[_user]:
            _EXPANDED_USER_SOURCES[_user].append(_np)

_EXPANDED_USER_SOURCES["Domestic"] = [
    s for s in _EXPANDED_USER_SOURCES["Domestic"]
    if not (s.startswith("TreatedWW") or s.startswith("SurfaceWater"))
]
_EXPANDED_USER_SOURCES["Industry"] = [
    s for s in _EXPANDED_USER_SOURCES["Industry"] if not s.startswith("SurfaceWater")
]

# User-source energy factors (for energy requirement computation)
_USER_ENERGY_FACTORS = {}
for _user, _types in _USER_SOURCE_TYPES.items():
    for _st in _types:
        if _st in _STATIC_WATER_SUPPLY:
            for _src in _STATIC_WATER_SUPPLY[_st]:
                _skey = f"{_st}_{_src['name']}"
                _ef = _STATIC_ENERGY_FACTORS.get(_st, {}).get(_src["name"], 0)
                _USER_ENERGY_FACTORS[(_user, _skey)] = _ef
# New plants: store energy factors under the FULL source name (e.g.
# "Desalination_NewRO"), because the model looks up energy_req_fac[(u, s)] with the
# full source name.  Previously keyed by the short name ("NewRO"), so the lookup
# failed and defaulted to 0 -> new RO/desal was treated as ENERGY-FREE, wrongly
# lowering energy cost as production shifted to new plants.
_NEW_SHORT_TO_FULL = {_full.split("_", 1)[1]: _full for _full in _WATER_NEW_PLANTS}
for _name in _NEW_WATER_COST:
    _full = _NEW_SHORT_TO_FULL.get(_name, _name)
    for _user in _USER_SOURCE_TYPES:
        _USER_ENERGY_FACTORS[(_user, _full)] = _NEW_WATER_ENERGY_D.get(_name, 0)

# New-plant efficiencies: same short->full key bug as the energy factors above
# (found 2026-07-11). _STATIC_SOURCE_EFF was keyed by the short name ("NewSaudiDam"
# 0.9, "NewRO" 0.95, "NewSaudiTWW" 0.9), but capacity_rule looks up
# source_eff.get(s, 1.0) with the FULL name ("SurfaceWater_NewSaudiDam" etc.) ->
# miss -> silently defaulted to 1.0 (100% efficient) for every new plant type.
# Does NOT affect existing TreatedWW_SaudiTWW's separate time-varying ramp
# (tww_existing_eff), which is looked up by its own exact full name elsewhere in
# capacity_rule and is untouched by this fix.
for _name in _NEW_WATER_COST:
    _full = _NEW_SHORT_TO_FULL.get(_name, _name)
    _STATIC_SOURCE_EFF[_full] = _NEW_WATER_EFF.get(_name, 1.0)

# Desal cost model (identical to sequential)
# Groundwater: recalibrated 2026-07-09 pm (3rd fix this session) against the
# Superwell groundwater supply-cost model for Saudi Arabia (5 basins, .25
# depletion limit; Superwell_Saudi_.3PD_.25DL.xlsx, Sheet1). The old exp curve
# below (0.8359, 3e-11) — origin undocumented, not traceable to any cited
# source in this codebase — priced groundwater 28-35x above the Superwell
# curve across CORAM's own 5-19 BCM/yr extraction range (e.g. $1.25/m3 vs
# $0.04/m3 at the 13.3 BCM/yr 2024 baseline), and is REPLACED, not kept as a
# fallback. New curve: fit (scipy curve_fit, R2=0.984) to all 469 Year-1
# grid-cell records across the 5 Saudi basins, sorted by unit_cost ascending
# (merit order) with volume_produced_allwells cumulated in that order — i.e.
# reconstructing the exact supply-cost curve, not the paper's power-law
# fit alone (that fit is only valid 0-50 BCM/yr; this rational form matches
# it there — a,b terms — AND captures the reserve-exhaustion elbow above
# ~200 BCM/yr via the added term, which a pure power law structurally cannot).
# cost(V_bcm) = a*(V+eps)^b + c*(V/Vmax)^p / (1-V/Vmax); type "superwell".
# TODO: apply this same recalibration to the regional model
# (model_intertemporal_v1.py), which currently uses its own flat/linear GW
# cost — see CORAM_SESSION_HANDOFF.md.
_DESAL_COST_COEFFS = {
    "Desalination_MSF": {"type": "log", "coeffs": [-0.087, 2.3055]},
    "Desalination_MED": {"type": "log", "coeffs": [-0.137, 2.792]},
    "Desalination_RO":  {"type": "log", "coeffs": [-0.125, 2.3882]},
    "Groundwater":      {"type": "superwell", "coeffs": [0.02068, 0.2475, 0.17610, 4.033, 335.0]},
}
# These log curves were calibrated to hit their reference cost (RO 0.92/MED
# 1.03/MSF 0.78 $/m3, matching the regional model's flat _TECH_DEFAULT_COST) at
# a tiny quantity (~1e5-1e8 m3/yr) and decrease unboundedly from there. At the
# national model's actual scale (1e8-1e10 m3/yr per user/source), RO and MED
# cross zero and go negative (a "profit" for producing more water) — a scale-
# mismatch artifact, not a real economies-of-scale effect. Floors below keep
# the intended economies-of-scale decline (still well below each source's own
# small-Q reference cost) while guaranteeing unit cost never reaches zero.
# Ordered RO < MED < MSF, matching the technologies' relative energy/thermal
# cost burden already encoded elsewhere in the model (MSF/MED energy_req_fac
# >> RO). All sit above New RO's flat $0.25/m3 opex since existing plants are
# older/less efficient than new-build capacity.
_DESAL_COST_FLOOR = {
    "Desalination_RO":  0.35,
    "Desalination_MED": 0.45,
    "Desalination_MSF": 0.55,
}
_BRINE_FACTORS = {
    "Desalination_RO":  0.59, "Desalination_MSF": 0.72,
    "Desalination_MED": 0.74, "Desalination_NewRO": 0.59,
}

# New plant investment parameters
_NEW_PLANT_INVESTMENTS = {
    "SurfaceWater_NewSaudiDam": {
        "unit_size": _NEW_WATER_UNIT_SIZE.get("NewSaudiDam", 0.10),
        "cost":      _NEW_WATER_CAPEX.get("NewSaudiDam", 0.5e9),
    },
    "TreatedWW_NewSaudiTWW": {
        "unit_size": _NEW_WATER_UNIT_SIZE.get("NewSaudiTWW", 0.20),
        "cost":      _NEW_WATER_CAPEX.get("NewSaudiTWW", 0.3e9),
    },
    "Desalination_NewRO": {
        "unit_size": _NEW_WATER_UNIT_SIZE.get("NewRO", 0.75),
        "cost":      _NEW_WATER_CAPEX.get("NewRO", 2.0e9),
    },
    "NewNG":     {"unit_size": 500,  "cost": 0.4e9},
    "NewDiesel": {"unit_size": 500,  "cost": 0.5e9},
    "NewCrude":  {"unit_size": 500,  "cost": 0.5e9},
    "NewHFO":    {"unit_size": 1000, "cost": 1.0e9},
    "NewSolar":  {"unit_size": 1000, "cost": 0.5e9},
    "NewWind":   {"unit_size": 1000, "cost": 1.5e9},
}
_POWER_PLANTS_NEW = ["NewNG", "NewDiesel", "NewCrude", "NewHFO", "NewSolar", "NewWind"]
_ALL_NEW_PLANTS   = list(_NEW_PLANT_INVESTMENTS.keys())

# Existing TWW capacity (2023 installed base)
_TWW_EXISTING_CAP = sum(
    item["capacity (BCM)"] for item in _STATIC_WATER_SUPPLY.get("TreatedWW", [])
)

# No TWW retirement: existing fleet capacity is constant for the full horizon.

_DAM_EXISTING_CAP = sum(
    item["capacity (BCM)"] for item in _STATIC_WATER_SUPPLY.get("SurfaceWater", [])
)


# =============================================================================
# HELPER: LOAD 2023 BASELINE FROM SEQUENTIAL RESULTS
# =============================================================================

def load_2023_baseline(energy_scenario, gw_label):
    """
    Read the sequential model's 2023 result to get the calibrated baseline:
    irrigated area, crop areas, import quantities.  Falls back to sensible
    defaults if the file is missing.

    Always reads the No_GW_Limit sequential file, regardless of gw_label:
    2023 is observed history and must be a single common starting point for
    all three GW scenarios.  The 10/5 BCM sequential 2023 files apply their
    *future* cap retroactively to the calibration year (area pinned to
    ~505k/~221k ha vs the true ~787k ha observed), which silently anchored
    each scenario's area_annual_up/dn ramp (+-3%/yr) to a different, wrong
    starting point and was the root cause of the 10-vs-5-BCM desal-investment
    timing artifact (see CORAM_SESSION_HANDOFF.md, 2026-07-09).
    """
    fpath = os.path.join(SEQ_RESULTS_DIR, f"2023_{energy_scenario}_No_GW_Limit.xlsx")
    baseline = {
        "total_irrigated_area": 787_600,
        "cultivated_area":      {},
        "crop_prod":            {},
        "crop_prod_import":     {},
    }
    if not os.path.exists(fpath):
        print(f"  [WARN] 2023 baseline not found: {fpath} — using defaults")
        return baseline
    try:
        xl = pd.ExcelFile(fpath)
        if "Summary" in xl.sheet_names:
            df = pd.read_excel(fpath, sheet_name="Summary")
            rm = {r["Metric"]: r["Value"] for _, r in df.iterrows()}
            baseline["total_irrigated_area"] = float(rm.get("Total Irrigated Area (ha)", 787_600))
        if "Domestic Crop Production" in xl.sheet_names:
            df = pd.read_excel(fpath, sheet_name="Domestic Crop Production")
            for _, row in df.iterrows():
                c = row["Crop"]
                baseline["cultivated_area"][c] = float(row.get("Cultivated Area (ha)", 0) or 0)
                baseline["crop_prod"][c]        = float(row.get("Production (tons)", 0) or 0)
        if "Imported Crops" in xl.sheet_names:
            df = pd.read_excel(fpath, sheet_name="Imported Crops")
            for _, row in df.iterrows():
                c = row["Crop"]
                baseline["crop_prod_import"][c] = float(row.get("Imported Quantity (tons)", 0) or 0)
    except Exception as e:
        print(f"  [WARN] Could not read 2023 baseline ({e}) — using defaults")
    return baseline


# =============================================================================
# HELPER: LOAD SEQUENTIAL WARM-START VALUES (all years)
# =============================================================================

def load_sequential_warmstart(energy_scenario, gw_label):
    """
    Read every sequential-model xlsx file for this scenario and return a
    dict   {year: {variable_key: value}}
    used to warm-start the intertemporal model before the first IPOPT call.

    Keys stored per year:
      ("production", u, s), ("allocation", u, s),
      ("cultivated_area", c), ("total_irrigated_area",),
      ("crop_prod", c), ("crop_prod_import", c),
      ("num_new_units", p), ("gw_stock",)
    """
    # Warm-start source can be overridden: the binding-GW scenarios (10/5 BCM)
    # converge far better when started from the already-solved *intertemporal*
    # No_GW solution (self-consistent full-horizon, small PMP penalties) than
    # from the myopic sequential results (which start at inf_pr~3e3, huge
    # penalties, and stall IPOPT).  WS_SOURCE_DIR / WS_SOURCE_GW enable this.
    _ws_dir   = os.environ.get("WS_SOURCE_DIR", SEQ_RESULTS_DIR)
    _ws_gw    = os.environ.get("WS_SOURCE_GW", gw_label)
    ws = {}
    cumulative_gw = 0.0
    for y in YEARS:
        fpath = os.path.join(_ws_dir, f"{y}_{energy_scenario}_{_ws_gw}.xlsx")
        yd = {}
        if not os.path.exists(fpath):
            ws[y] = yd
            continue
        try:
            xl = pd.ExcelFile(fpath)
            if "Summary" in xl.sheet_names:
                df = pd.read_excel(fpath, sheet_name="Summary")
                rm = {r["Metric"]: r["Value"] for _, r in df.iterrows()}
                yd[("total_irrigated_area",)] = float(rm.get("Total Irrigated Area (ha)", 787_600))
                gw_prod_y = float(rm.get("GW Production (BCM)", 0) or 0)
                cumulative_gw += gw_prod_y
                yd[("gw_stock",)] = max(0.0, GW_INITIAL_STOCK_BCM - cumulative_gw)
            if "Water Production" in xl.sheet_names:
                df = pd.read_excel(fpath, sheet_name="Water Production")
                for _, row in df.iterrows():
                    yd[("production", row["User"], row["Source"])] = float(row.get("Water Produced (billion m3)", 0) or 0)
            if "Water Allocation" in xl.sheet_names:
                df = pd.read_excel(fpath, sheet_name="Water Allocation")
                for _, row in df.iterrows():
                    yd[("allocation", row["User"], row["Source"])] = float(row.get("Water Allocated (billion m3)", 0) or 0)
            if "Domestic Crop Production" in xl.sheet_names:
                df = pd.read_excel(fpath, sheet_name="Domestic Crop Production")
                for _, row in df.iterrows():
                    c = row["Crop"]
                    yd[("cultivated_area", c)]  = float(row.get("Cultivated Area (ha)", 0) or 0)
                    yd[("crop_prod", c)]         = float(row.get("Production (tons)", 0) or 0)
            if "Imported Crops" in xl.sheet_names:
                df = pd.read_excel(fpath, sheet_name="Imported Crops")
                for _, row in df.iterrows():
                    yd[("crop_prod_import", row["Crop"])] = float(row.get("Imported Quantity (tons)", 0) or 0)
            if "Investments" in xl.sheet_names:
                df = pd.read_excel(fpath, sheet_name="Investments")
                for _, row in df.iterrows():
                    yd[("num_new_units", row["Plant"])] = float(row.get("Number of Units", 0) or 0)
        except Exception as e:
            print(f"  [WARN] warmstart read failed for {y}: {e}")
        ws[y] = yd
    return ws


# =============================================================================
# HELPER: PRECOMPUTE YEAR-SPECIFIC PARAMETERS
# =============================================================================

def precompute_year_params(energy_scenario):
    """
    Return a dict  {year: {...}}  with all scalar parameters that vary by year.
    All heavy numpy/interp work is done here in Python — not inside Pyomo rules.
    """
    params = {}
    for y in YEARS:
        pe  = 1.02 ** (y - 2023)          # 2% price escalation from 2023
        pop = get_population(y)
        dom, ind = get_demand(y)
        ren = get_renewable_share(y, energy_scenario)

        # Crop costs (escalated)
        culcost = {c: _BASE_CULCOST[c] * pe for c in _BASE_CULCOST}
        impcost = {c: _BASE_IMPCOST[c] * pe for c in _BASE_IMPCOST}

        # Logit equilibrium shares for domestic crops (for PMP static penalty)
        costs_sc = {c: culcost[c] / 1000 for c in _DOMESTIC_CROPS_2023 if c in culcost}
        exp_t    = {c: _LOGIT_COEF_1[c] * np.exp(np.clip(_LOGIT_COEF_2[c] * costs_sc[c], -100, 100))
                    for c in _DOMESTIC_CROPS_2023 if c in _LOGIT_COEF_1 and c in costs_sc}
        exp_s    = sum(exp_t.values()) or 1.0
        dom_shares = {c: exp_t[c] / exp_s for c in exp_t}

        # Logit equilibrium shares for import crops
        imp_crops = list(_IMP_CROPS_DF_2023["Crop"])
        costs_sc_imp = {c: impcost[c] / 1000 for c in imp_crops if c in impcost}
        exp_t_imp    = {c: _LOGIT_COEF_1_IMPORT[c] * np.exp(np.clip(
                            _LOGIT_COEF_2_IMPORT[c] * costs_sc_imp[c], -100, 100))
                        for c in imp_crops if c in _LOGIT_COEF_1_IMPORT and c in costs_sc_imp}
        exp_s_imp    = sum(exp_t_imp.values()) or 1.0
        imp_shares   = {c: exp_t_imp[c] / exp_s_imp for c in exp_t_imp}

        # Power plant cost dict (escalated)
        pp_base  = _POWER_PLANTS_BASE[energy_scenario]
        pp_costs = {p: pp_base[p]["Cost ($/kWh)"] * pe for p in pp_base}
        for p in _NEW_POWER_COST:
            pp_costs[p] = _NEW_POWER_COST[p] * pe

        params[y] = {
            "price_esc":        pe,
            "population":       pop,
            "domestic_demand":  dom,
            "industry_demand":  ind,
            "ren_share":        ren,
            "calorie_demand":   pop * 1970 * 365,
            "protein_demand":   pop * 107 * 365 / 1000,
            "culcost":          culcost,
            "impcost":          impcost,
            "dom_shares":       dom_shares,
            "imp_shares":       imp_shares,
            "desal_caps":       _DESAL_CAPS_BY_YEAR[y],
            "pp_costs":         pp_costs,
        }
    return params


# =============================================================================
# MAIN INTERTEMPORAL SOLVER
# =============================================================================

def run_intertemporal(energy_scenario, scenario1, scenario2):
    """
    Build and solve the 2024-2060 intertemporal NLP for one scenario triple
    (energy_scenario, scenario1=10BCM_GW, scenario2=5BCM_GW).

    Saves one xlsx per year per scenario to RESULTS_DIR_IT/.
    """
    gw_label     = ("10_BCM_GW_Limit" if scenario1 else
                    "5_BCM_GW_Limit"  if scenario2 else "No_GW_Limit")
    scenario_tag = f"{energy_scenario}_{gw_label}"

    # ------------------------------------------------------------------
    # SKIP if all year files already exist (allows safe restart without
    # re-running completed scenarios).
    # ------------------------------------------------------------------
    _existing = [
        y for y in YEARS
        if os.path.exists(os.path.join(RESULTS_DIR_IT, f"{y}_{scenario_tag}.xlsx"))
    ]
    if len(_existing) == len(YEARS) and not os.environ.get("FORCE_RESOLVE"):
        print(f"\n  [SKIP] {scenario_tag} — all {len(YEARS)} year files already exist.")
        return
    elif _existing:
        print(f"\n  [PARTIAL] {scenario_tag} — {len(_existing)}/{len(YEARS)} files exist, re-solving.")

    print(f"\n{'='*72}")
    print(f"  INTERTEMPORAL: {energy_scenario} | {gw_label}")
    print(f"  Horizon: {YEARS[0]}-{YEARS[-1]}  |  Discount rate: {DISCOUNT_RATE*100:.0f}%")
    print(f"  GW initial stock: {GW_INITIAL_STOCK_BCM} BCM")
    print(f"{'='*72}")

    # ---- Year-specific parameters ----------------------------------------
    params = precompute_year_params(energy_scenario)

    # ---- 2023 calibration baseline ----------------------------------------
    baseline = load_2023_baseline(energy_scenario, gw_label)
    baseline_area   = baseline["total_irrigated_area"]
    baseline_cprod  = baseline["crop_prod"]          # {crop: tons}
    baseline_cprimp = baseline["crop_prod_import"]   # {crop: tons}

    # ---- Warm-start from sequential results --------------------------------
    # The sequential myopic solution is a feasible (but suboptimal) starting
    # point for the intertemporal NLP.  Starting cold from arbitrary bounds
    # causes MUMPS to fail factorising the KKT matrix → internalSolverError.
    #
    # TWW S-curve clamping: existing TWW capacity shrinks over time per the
    # survival curve; sequential TWW values for later years may exceed the
    # intertemporal capacity limit.  We clamp them in the init loop below.
    #
    # num_new_units: NOT loaded from sequential results — the sequential model
    # tracks cumulative capacity while the intertemporal model tracks annual
    # investment decisions.  Loading cumulative values as annual would be wrong.
    use_warmstart = True
    if use_warmstart:
        print("  Loading sequential warm-start values...")
        ws_all = load_sequential_warmstart(energy_scenario, gw_label)
        ws_available = sum(1 for y in YEARS if ws_all.get(y))
        print(f"  Warm-start data found for {ws_available}/{len(YEARS)} years")
    else:
        print("  Cold-start — no sequential warm-start available")
        ws_all = {}

    # ---- Derive common data structures (identical to sequential) -----------
    domestic_crops   = _DOMESTIC_CROPS_2023
    import_crops     = list(_IMP_CROPS_DF_2023["Crop"])
    yield_per_ha     = _YIELD_PER_HA_2023
    crop_water_req   = _CROP_WATER_REQ_DICT
    crop_categories  = _CROP_CATEGORIES
    crop_calories    = _CROP_CALORIES
    crop_protein     = _CROP_PROTEIN
    expanded_us      = _EXPANDED_USER_SOURCES
    water_users      = list(expanded_us.keys())         # ["Agriculture","Industry","Domestic"]
    source_eff       = _STATIC_SOURCE_EFF
    energy_req_fac   = _USER_ENERGY_FACTORS             # {(u,s): kWh/m3}
    all_sources      = list({s for u in water_users for s in expanded_us[u]})

    # power_plants dict for this energy scenario (base + new, cost escalation applied per year)
    pp_base = {p: dict(v) for p, v in _POWER_PLANTS_BASE[energy_scenario].items()}
    for p in _NEW_POWER_COST:
        if p not in pp_base:
            pp_base[p] = {
                "Cost ($/kWh)":  _NEW_POWER_COST[p],
                "Efficiency":    _NEW_POWER_EFF.get(p, 1.0),
                "CO2 (t/kwh)":  _NEW_POWER_CO2.get(p, 0.0),
                "Capacity (MW)": _NEW_POWER_CAP.get(p, 0.0),
            }
    power_plants    = list(pp_base.keys())
    pp_eff          = {p: pp_base[p]["Efficiency"]   for p in power_plants}
    # Scale all per-fuel CO2 factors so the 2023 capacity-weighted grid emission
    # factor matches the official DNA/KFUPM Grid Emission Factor of 0.568 tCO2/MWh
    # (KFUPM GHG report 2024). The bottom-up 2023 mix gives 0.628 tCO2/MWh, so we
    # apply CO2_GRID_CALIB; the decarbonisation trajectory (falling emissions as the
    # renewable share rises) is preserved because the scaling is uniform.
    pp_co2          = {p: pp_base[p]["CO2 (t/kwh)"] * CO2_GRID_CALIB for p in power_plants}
    pp_cap          = {p: pp_base[p]["Capacity (MW)"] for p in power_plants}
    # Year-dependent existing-fleet capacity (grows 2023->2060, see _CAP_BY_YEAR)
    pp_cap_year     = _CAP_BY_YEAR[energy_scenario]
    pp_new          = _POWER_PLANTS_NEW

    # Food / feed crop lists for calorie/protein constraints
    food_crops = [c for c, cat in crop_categories.items()
                  if isinstance(cat, str) and cat.lower() == "food"]
    feed_crops = [c for c, cat in crop_categories.items()
                  if isinstance(cat, str) and cat.lower() == "feed"]

    # Agriculture desal sources (forced to zero)
    agr_desal_srcs = [s for s in expanded_us["Agriculture"] if s.startswith("Desalination")]

    # PMP calibration — fixed gamma denominators use 2023 baseline to avoid
    # bilinear terms in the intertemporal objective (gamma × Δq²).
    _Q_FLOOR      = 50_000
    _AREA_FLOOR   = 100_000
    _REF_AREA_HA  = 787_600
    _LAM_STATIC   = 100_000
    _LAM_TEMPORAL =  20_000

    gamma_dom = {}
    for c in domestic_crops:
        q_ref = max(baseline_cprod.get(c, _Q_FLOOR), _Q_FLOOR)
        gamma_dom[c] = abs(_DUAL_PRICES_DOMESTIC.get(c, 0)) / q_ref

    gamma_imp = {}
    for c in import_crops:
        q_ref = max(baseline_cprimp.get(c, _Q_FLOOR), _Q_FLOOR)
        gamma_imp[c] = abs(_DUAL_PRICES_IMPORT.get(c, 0)) / q_ref

    gamma_static  = _LAM_STATIC / _REF_AREA_HA
    gamma_temporal = _LAM_TEMPORAL / max(baseline_area, _AREA_FLOOR)

    # Desal cost helper (Pyomo-compatible — uses pyomo log/sqrt/exp)
    def desal_unit_cost(source_type_key, Q):
        info = _DESAL_COST_COEFFS[source_type_key]
        if info["type"] == "log":
            a, b = info["coeffs"]
            raw = a * log(sqrt(Q**2 + 1e-12)) + b
            floor = _DESAL_COST_FLOOR.get(source_type_key)
            if floor is None:
                return raw
            # Smooth max(raw, floor) — keeps the expression IPOPT-differentiable
            # (same soft-max form as _floor_pen below) instead of a hard clip.
            return 0.5 * (raw + floor + sqrt((raw - floor) ** 2 + 1e-9))
        elif info["type"] == "superwell":
            # Superwell-fitted supply curve (see _DESAL_COST_COEFFS comment).
            # Q arrives in m3; the fit is in BCM (V). +eps avoids the infinite
            # slope of V^b (b<1) at V=0; the (1-V/Vmax) denominator is safe for
            # any V the model will ever reach (Vmax=335 BCM dwarfs the <20
            # BCM/yr CORAM range) but is guarded anyway for robustness.
            a, b, c, p, vmax = info["coeffs"]
            V = Q / 1e9
            return a * (V + 1e-6) ** b + c * (V / vmax) ** p / (1 - V / vmax)
        else:
            a, b = info["coeffs"]
            return a * exp(b * Q)

    # Source cost lookup (base $/m3, escalated per year inside objective)
    def get_source_base_cost(s):
        parts = s.split("_", 1)
        if len(parts) < 2:
            return 0.0
        st, sname = parts[0], parts[1]
        if sname in _NEW_WATER_COST:
            return _NEW_WATER_COST[sname]
        items = _STATIC_WATER_SUPPLY.get(st, [])
        for item in items:
            if item["name"] == sname:
                return item.get("cost", 0.0)
        return 0.0

    # ---- Build Pyomo model -------------------------------------------------
    print("  Building Pyomo model...")
    t_build = time.time()
    model = ConcreteModel()
    model.dual = Suffix(direction=Suffix.IMPORT)

    # -- Variable index sets --
    _us_pairs  = [(u, s) for u in water_users for s in expanded_us[u]]
    _ea_index  = [(u, s, p, y) for (u, s) in _us_pairs for p in power_plants for y in YEARS]
    _ca_index  = [(c, y) for c in domestic_crops for y in YEARS]
    _cp_index  = [(c, "Domestic", y) for c in domestic_crops for y in YEARS]
    _ci_index  = [(c, "Import",   y) for c in import_crops   for y in YEARS]
    _al_index  = [(u, s, y) for (u, s) in _us_pairs for y in YEARS]
    _nnu_index = [(p, y) for p in _ALL_NEW_PLANTS  for y in YEARS]

    # -- Variables --
    model.allocation          = Var(_al_index,  domain=NonNegativeReals)
    model.production          = Var(_al_index,  domain=NonNegativeReals)
    model.cultivated_area     = Var(_ca_index,  domain=NonNegativeReals)
    model.crop_prod           = Var(_cp_index,  domain=NonNegativeReals)
    model.crop_prod_import    = Var(_ci_index,  domain=NonNegativeReals)
    model.total_irrigated_area= Var(YEARS,      domain=NonNegativeReals)
    model.num_new_units       = Var(_nnu_index, domain=NonNegativeReals)   # continuous relaxation
    model.energy_allocation   = Var(_ea_index,  domain=NonNegativeReals)
    model.gw_stock            = Var(YEARS,      domain=NonNegativeReals)

    print(f"  Variables: {sum(1 for _ in model.component_data_objects(Var, active=True)):,}")

    # =========================================================================
    # OBJECTIVE: discounted sum of annual costs
    # =========================================================================

    def annual_cost_pyomo(m, y, p):
        """Return Pyomo expression for undiscounted annual cost in year y."""
        pe = p["price_esc"]

        # --- Water cost (nonlinear log/exp) ---
        # Groundwater priced from TOTAL system-wide annual extraction (all
        # sectors combined), not per-sector: the Superwell supply curve is a
        # merit-order curve over the WHOLE shared aquifer for that year, so the
        # unit price every sector pays must reflect the combined draw, not each
        # sector's own (much smaller) volume evaluated in isolation against the
        # cheap end of the curve. Computed once per year, applied per sector.
        _total_gw_alloc = sum(
            m.allocation[u2, s2, y] for u2 in water_users for s2 in expanded_us[u2]
            if s2.startswith("Groundwater")
        )
        _gw_unit_cost = desal_unit_cost("Groundwater", _total_gw_alloc * 1e9)

        water_cost = 0
        for u in water_users:
            for s in expanded_us[u]:
                parts = s.split("_")
                if len(parts) < 2:
                    continue
                if parts[0].lower() == "desalination":
                    stype_key = f"{parts[0]}_{parts[1]}"
                else:
                    stype_key = None

                Q_m3 = m.allocation[u, s, y] * 1e9
                if parts[0] == "Groundwater":
                    water_cost += Q_m3 * _gw_unit_cost
                elif stype_key and stype_key in _DESAL_COST_COEFFS:
                    water_cost += Q_m3 * desal_unit_cost(stype_key, Q_m3)
                else:
                    base_cost = get_source_base_cost(s)
                    cost = base_cost * pe
                    if parts[0] == "SurfaceWater" and u == "Domestic":
                        cost *= 1.10
                    water_cost += Q_m3 * cost

        # --- Crop cost ---
        crop_cost = (
            sum(m.crop_prod[c, "Domestic", y] * p["culcost"].get(c, 0)
                for c in domestic_crops if (c, "Domestic", y) in m.crop_prod)
            + sum(m.crop_prod_import[c, "Import", y] * p["impcost"].get(c, 0)
                  for c in import_crops if (c, "Import", y) in m.crop_prod_import)
        )

        # --- Energy cost ---
        energy_cost = sum(
            m.energy_allocation[u, s, pp, y] * p["pp_costs"].get(pp, 0) * 8_760_000
            for u in water_users for s in expanded_us[u] for pp in power_plants
            if (u, s, pp, y) in m.energy_allocation
        )

        # --- Carbon damage cost (manuscript Eq. 12): emissions × social cost of carbon ---
        # emissions (tCO2) = energy_allocation × pp_co2 (t/kWh) × 8,760,000; priced at $/tCO2.
        carbon_cost = CARBON_DAMAGE_USD_PER_T * sum(
            m.energy_allocation[u, s, pp, y] * pp_co2.get(pp, 0) * 8_760_000
            for u in water_users for s in expanded_us[u] for pp in power_plants
            if (u, s, pp, y) in m.energy_allocation
        )

        # --- Investment cost (CapEx for THIS year's new decisions) ---
        investment_cost = sum(
            m.num_new_units[pl, y] * _NEW_PLANT_INVESTMENTS[pl]["cost"]
            for pl in _ALL_NEW_PLANTS
            if (pl, y) in m.num_new_units
        )

        # --- PMP static logit penalties (full weight) ---
        # PMP_NORMALIZE=1 scales each crop's squared deviation by its 2023
        # production level, so the penalty measures RELATIVE (not absolute-tonnage)
        # deviation.  This stops the penalty from exploding as production grows and
        # keeps it comparable to the real cost (so the objective is economically
        # meaningful).  Default (=0) preserves the original absolute formulation.
        _PMP_NORM = os.environ.get("PMP_NORMALIZE", "0") == "1"
        def _pmp_sc(ref):
            return max(float(ref or 0.0), 1e4) if _PMP_NORM else 1.0

        total_dom_prod = sum(m.crop_prod[c, "Domestic", y] for c in domestic_crops
                             if (c, "Domestic", y) in m.crop_prod)
        soft_pen_dom = sum(
            abs(_DUAL_PRICES_DOMESTIC.get(c, 0)) *
            ((m.crop_prod[c, "Domestic", y] - p["dom_shares"].get(c, 0) * total_dom_prod)
             / _pmp_sc(baseline_cprod.get(c, 0))) ** 2
            for c in domestic_crops if (c, "Domestic", y) in m.crop_prod
        )

        total_imp_prod = sum(m.crop_prod_import[c, "Import", y] for c in import_crops
                             if (c, "Import", y) in m.crop_prod_import)
        soft_pen_imp = sum(
            abs(_DUAL_PRICES_IMPORT.get(c, 0)) *
            ((m.crop_prod_import[c, "Import", y] - p["imp_shares"].get(c, 0) * total_imp_prod)
             / _pmp_sc(baseline_cprimp.get(c, 0))) ** 2
            for c in import_crops
            if (c, "Import", y) in m.crop_prod_import and c in p["imp_shares"]
        )

        # --- PMP temporal adjustment (reduced weight, gamma fixed at 2023 baseline) ---
        if y == YEARS[0]:   # 2024: compare against 2023 scalar baseline
            prev_cprod   = baseline_cprod
            prev_cprimp  = baseline_cprimp
            prev_area    = baseline_area
        else:               # 2025+: compare against prior-year Pyomo variables
            prev_cprod   = {c: m.cultivated_area[c, y-1] * yield_per_ha.get(c, 1.0)
                            for c in domestic_crops if (c, y-1) in m.cultivated_area}
            prev_cprimp  = {c: m.crop_prod_import[c, "Import", y-1]
                            for c in import_crops if (c, "Import", y-1) in m.crop_prod_import}
            prev_area    = m.total_irrigated_area[y-1]

        adj_dom = PMP_TEMPORAL_WEIGHT * sum(
            gamma_dom[c] * ((m.crop_prod[c, "Domestic", y] - prev_cprod.get(c, 0))
                            / _pmp_sc(baseline_cprod.get(c, 0))) ** 2
            for c in domestic_crops if (c, "Domestic", y) in m.crop_prod
        )
        adj_imp = PMP_TEMPORAL_WEIGHT * sum(
            gamma_imp[c] * ((m.crop_prod_import[c, "Import", y] - prev_cprimp.get(c, 0))
                            / _pmp_sc(baseline_cprimp.get(c, 0))) ** 2
            for c in import_crops if (c, "Import", y) in m.crop_prod_import
        )

        # --- Land use PMP penalty (static + temporal) ---
        land_static  = gamma_static  * (_REF_AREA_HA - m.total_irrigated_area[y]) ** 2
        land_temporal = PMP_TEMPORAL_WEIGHT * gamma_temporal * (m.total_irrigated_area[y] - prev_area) ** 2
        land_pen      = land_static + land_temporal

        # --- Water mix soft penalties (desalination, TWW, surface water) ---
        total_wp = sum(m.production[u, s, y] for u in water_users for s in expanded_us[u])
        total_desal = sum(m.production[u, s, y] for u in water_users for s in expanded_us[u]
                          if s.startswith("Desalination"))
        total_tww   = sum(m.production[u, s, y] for u in water_users for s in expanded_us[u]
                          if s.startswith("TreatedWW"))
        total_sw    = sum(m.production[u, s, y] for u in water_users for s in expanded_us[u]
                          if s.startswith("SurfaceWater"))
        # Water-mix calibration anchors as one-sided FLOORS ("clean" formulation).
        # The dual-price keys are "*_Min" — these were conceived as MINIMUM shares.
        # Each penalty applies only when a source's share falls BELOW its base-year
        # target; it no longer penalises the share from above.  The two flexible
        # non-conventional sources are instead bounded by PHYSICAL caps: treated
        # wastewater by the return-flow limit (max_tww: 0.8*(dom+ind) demand) and
        # surface water by the rainfall/runoff ceiling (max_sw, 2.25 BCM); desal by
        # its real cost.  So the desal/TWW split is cost-determined within physical
        # limits and no longer forced to 18% from above.  Smooth positive part
        # p(g)=0.5*(g+sqrt(g^2+eps)) keeps each floor differentiable for IPOPT.
        def _floor_pen(weight, target_share, actual):
            g = target_share * total_wp - actual
            return abs(weight) * (0.5 * (g + sqrt(g ** 2 + 1e-6))) ** 2
        desal_pen = _floor_pen(_DUAL_PRICES_WATER.get("Total_Desalination_Min", 0), 0.18,  total_desal)
        tww_pen   = _floor_pen(_DUAL_PRICES_WATER.get("TreatedWW_Min",          0), 0.025, total_tww)
        sw_pen    = _floor_pen(_DUAL_PRICES_WATER.get("SurfaceWater_Min",        0), 0.05,  total_sw)
        mix_pen   = desal_pen + tww_pen + sw_pen

        _real = water_cost + crop_cost + energy_cost + carbon_cost + investment_cost
        _pen  = soft_pen_dom + soft_pen_imp + adj_dom + adj_imp + land_pen + mix_pen
        annual_cost_pyomo._last = (_real, _pen)   # diagnostic side-channel
        return _real + _pen

    # Proximal (Tikhonov) regularization: the PMP-penalty objective is non-convex
    # and degenerate in the crop-mix direction, so IPOPT reaches a feasible KKT
    # point but cannot terminate cleanly (it wanders between near-equal-cost
    # mixes).  A small strictly-convex proximal term anchored to the warm-start
    # crop levels, normalized by each crop's own scale, makes the objective
    # strictly convex near the operating point -> unique optimum + clean stop.
    # Weight is tiny (relative deviations) so the 2023 calibration is preserved;
    # verify crop R^2 after tuning PMP_REG.
    # Off by default: the non-convexity is the desalination economies-of-scale
    # cost, not the (convex) PMP penalties, so regularization here does not help
    # and distorts.  Kept env-tunable for experiments only.
    _PMP_REG = float(os.environ.get("PMP_REG", "0"))

    def _reg_ref(y, key, c):
        return float(ws_all.get(y, {}).get((key, c), 0.0) or 0.0)

    def total_objective(m):
        total = 0
        m._real_npv = 0   # diagnostic: NPV of real costs
        m._pen_npv  = 0   # diagnostic: NPV of PMP penalties
        m._pen_by_year = {}   # diagnostic: undiscounted penalty per year
        for y in YEARS:
            disc = (1.0 / (1.0 + DISCOUNT_RATE)) ** (y - YEARS[0])
            total += disc * annual_cost_pyomo(m, y, params[y])
            _r, _p = annual_cost_pyomo._last
            m._real_npv += disc * _r
            m._pen_npv  += disc * _p
            m._pen_by_year[y] = _p
        if _PMP_REG > 0:
            reg = 0
            for y in YEARS:
                for c in domestic_crops:
                    if (c, "Domestic", y) in m.crop_prod:
                        ref = _reg_ref(y, "crop_prod", c)
                        sc  = max(ref, 1e4)               # per-crop scale (tons)
                        reg += ((m.crop_prod[c, "Domestic", y] - ref) / sc) ** 2
                for c in import_crops:
                    if (c, "Import", y) in m.crop_prod_import:
                        ref = _reg_ref(y, "crop_prod_import", c)
                        sc  = max(ref, 1e4)
                        reg += ((m.crop_prod_import[c, "Import", y] - ref) / sc) ** 2
            total += _PMP_REG * reg
        return total

    print("  Assembling objective...")
    model.obj = Objective(rule=total_objective, sense=minimize)

    # -------------------------------------------------------------------------
    # IPOPT USER-SCALING (nlp_scaling_method=user-scaling)
    # The objective reaches ~1e15-1e16 (PMP penalties are quadratic in raw
    # tonnage) and tonnage/area variables are ~1e5-1e6, so the KKT system is
    # badly conditioned and the binding-GW (10/5 BCM) solves stall / restoration-
    # fail.  Providing per-variable and objective scale factors lets IPOPT work
    # in non-dimensional space.  This is mathematically exact: IPOPT unscales the
    # solution, so the optimum and the 2023 calibration are UNCHANGED.
    # -------------------------------------------------------------------------
    model.scaling_factor = Suffix(direction=Suffix.EXPORT)
    model.scaling_factor[model.obj] = float(os.environ.get("OBJ_SCALE", "1e-15"))  # obj ~1e16 -> ~O(10)
    _VAR_SCALE = {
        "allocation":            1.0,    # BCM   ~O(1)
        "production":            1.0,    # BCM   ~O(1)
        "num_new_units":         1.0,    # units ~O(1-100)
        "cultivated_area":       1e-5,   # ha    ~1e4-1e5 per crop
        "total_irrigated_area":  1e-6,   # ha    ~8e5
        "crop_prod":             1e-6,   # tons  ~1e5-1e7
        "crop_prod_import":      1e-6,   # tons  ~1e5-1e7
        "gw_stock":              1e-3,   # BCM   ~2.5e3
        "energy_allocation":     1e-4,   # ~avg MW, ~1e4-1e5
    }
    for _vname, _sf in _VAR_SCALE.items():
        _v = getattr(model, _vname, None)
        if _v is None:
            continue
        for _idx in _v:
            model.scaling_factor[_v[_idx]] = _sf

    # =========================================================================
    # CONSTRAINTS  (one block per year, linked across years where needed)
    # =========================================================================
    print("  Adding constraints...")

    # -- (1) Water demand: lower and upper bounds per user-year --
    def demand_lower(m, u, y):
        total = sum(m.allocation[u, s, y] for s in expanded_us[u])
        if u == "Agriculture":
            return total >= 0
        elif u == "Domestic":
            return total >= params[y]["domestic_demand"]
        else:
            return total >= params[y]["industry_demand"]

    def demand_upper(m, u, y):
        if u == "Agriculture":
            return Constraint.Skip
        elif u == "Domestic":
            return sum(m.allocation[u, s, y] for s in expanded_us[u]) <= 1.1 * params[y]["domestic_demand"]
        else:
            return sum(m.allocation[u, s, y] for s in expanded_us[u]) <= 1.1 * params[y]["industry_demand"]

    model.demand_lower = Constraint(
        [(u, y) for u in water_users for y in YEARS], rule=demand_lower)
    model.demand_upper = Constraint(
        [(u, y) for u in water_users for y in YEARS], rule=demand_upper)

    # -- (2) Domestic desalination share ≥ 58 % --
    def dom_desal_min(m, y):
        dom_desal = sum(m.production["Domestic", s, y] for s in expanded_us["Domestic"]
                        if s.startswith("Desalination"))
        dom_total = sum(m.production["Domestic", s, y] for s in expanded_us["Domestic"])
        return dom_desal >= 0.58 * dom_total

    model.domestic_desal_min = Constraint(YEARS, rule=dom_desal_min)

    # -- (3) Agriculture cannot use desalination --
    model.no_agr_desal = Constraint(
        [(s, y) for s in agr_desal_srcs for y in YEARS],
        rule=lambda m, s, y: m.allocation["Agriculture", s, y] == 0,
    )

    # -- (4) Source capacity constraints (existing + new plants) --
    def capacity_rule(m, s, y):
        parts = s.split("_", 1)
        st    = parts[0]
        sname = parts[1] if len(parts) > 1 else ""

        if s in _WATER_NEW_PLANTS:
            # New water plants (dam, RO, new TWW): flat vintage window
            cap_expr = sum(
                m.num_new_units[s, y_] * _NEW_PLANT_INVESTMENTS[s]["unit_size"]
                for y_ in YEARS if y_ <= y and (y - y_) < PLANT_LIFETIME
            )
        elif st == "TreatedWW":
            # Existing TWW: constant capacity for full horizon (no retirement, matches sequential)
            cap_expr = _STATIC_SOURCE_CAPS.get(s, 0)
        elif st == "Desalination" and sname in ("RO", "MSF", "MED"):
            # Existing desal: year-specific SWCC fleet capacity
            cap_expr = params[y]["desal_caps"].get(sname, 0)
        else:
            cap_expr = _STATIC_SOURCE_CAPS.get(s, 0)

        eff  = source_eff.get(s, 1.0)
        if s == "TreatedWW_SaudiTWW":          # existing TWW reuse efficiency ramps over time
            eff = tww_existing_eff(y)
        prod = sum(m.production[u, s, y] for u in water_users if s in expanded_us[u])
        return prod <= cap_expr * eff

    model.capacity_constraint = Constraint(
        [(s, y) for s in all_sources for y in YEARS], rule=capacity_rule)

    # Redundant explicit new-water-plant capacity constraint (matches sequential model)
    def new_water_cap(m, pl, y):
        cap = sum(
            m.num_new_units[pl, y_] * _NEW_PLANT_INVESTMENTS[pl]["unit_size"]
            for y_ in YEARS if y_ <= y and (y - y_) < PLANT_LIFETIME
        )
        prod = sum(m.production[u, pl, y] for u in water_users if pl in expanded_us[u])
        return prod <= cap

    model.new_water_plant_cap = Constraint(
        [(pl, y) for pl in _WATER_NEW_PLANTS for y in YEARS], rule=new_water_cap)

    # -- (5) Delivery: allocation = production × SECTOR delivery efficiency --
    # Two distinct efficiencies: the SOURCE efficiency (how much of the source is
    # produced) is applied once in the capacity constraint above; the SECTOR
    # delivery efficiency (distribution losses to the end user: irrigation 0.60,
    # domestic 0.67, industry 0.85) is applied here.  Previously the source
    # efficiency was (incorrectly) reused here for desal/dam/TWW.
    def _sector_eff(u):
        return (_IRR_EFF_2023 if u == "Agriculture" else
                _DOM_EFF_2023 if u == "Domestic" else _IND_EFF_2023)
    model.production_constraint = Constraint(
        [(u, s, y) for (u, s) in _us_pairs for y in YEARS],
        rule=lambda m, u, s, y:
            m.allocation[u, s, y] == m.production[u, s, y] * _sector_eff(u),
    )

    # -- (6) Crop area ↔ production --
    model.prod_area = Constraint(
        [(c, y) for c in domestic_crops for y in YEARS],
        rule=lambda m, c, y:
            m.cultivated_area[c, y] == m.crop_prod[c, "Domestic", y] / yield_per_ha[c],
    )

    # -- (7) Agricultural water balance --
    def agr_water_bal(m, y):
        water_used = sum(m.allocation["Agriculture", s, y] for s in expanded_us["Agriculture"])
        water_req  = sum(m.cultivated_area[c, y] * 10000 * crop_water_req.get(c, 0)
                         for c in domestic_crops) / 1e9
        return water_used == water_req

    model.agr_water_demand = Constraint(YEARS, rule=agr_water_bal)

    # -- (8) Total irrigated area definition --
    model.total_irrigated_area_def = Constraint(
        YEARS,
        rule=lambda m, y: m.total_irrigated_area[y] == sum(
            m.cultivated_area[c, y] for c in domestic_crops),
    )

    # -- (9) Max irrigated area --
    model.irrigated_area_limit = Constraint(
        YEARS, rule=lambda m, y: m.total_irrigated_area[y] <= 787_600)

    # -- (10) ASYMMETRIC annual area change (cross-year linking) --
    # Expansion is slow (developing new irrigated land takes years) but
    # contraction/fallowing is fast (land can be abandoned quickly, e.g. Saudi's
    # 2008-2016 wheat phase-out). A symmetric ±3%/yr rate (2026-07-11 and earlier)
    # forced the tightest-water scenario (5 BCM) to keep the 2023 baseline area
    # (787,600 ha) under cultivation for ~13 years even though its water budget
    # couldn't irrigate it — and since every irrigated hectare must grow a crop
    # (no fallow state), the model filled the un-irrigable land with the
    # least-thirsty crop (Other Grain, economically loss-making: domestic
    # $1006/t vs $492/t import), an artifact that inflated 5 BCM's early domestic
    # production and dominated Figure 4's early years (52% of 5 BCM land in 2024).
    # Fix: separate, faster DOWNWARD rate (AREA_CHANGE_RATE_DN, default 10%/yr) so
    # area can adjust to what water actually supports, while keeping the slow
    # UPWARD rate (AREA_CHANGE_RATE, 3%/yr). No_GW is unaffected (its area grows,
    # so only the upward bound binds).
    _AREA_RATE    = float(os.environ.get("AREA_CHANGE_RATE", "0.03"))       # upward (expansion)
    _AREA_RATE_DN = float(os.environ.get("AREA_CHANGE_RATE_DN", "0.10"))    # downward (fallowing)
    def area_dn(m, y):
        prev = baseline_area if y == YEARS[0] else m.total_irrigated_area[y - 1]
        return m.total_irrigated_area[y] >= prev * (1 - _AREA_RATE_DN)

    def area_up(m, y):
        # Cannot use min() with a Pyomo variable — split into two bounds.
        # The hard 787,600 ha cap is already enforced by irrigated_area_limit.
        prev = baseline_area if y == YEARS[0] else m.total_irrigated_area[y - 1]
        return m.total_irrigated_area[y] <= prev * (1 + _AREA_RATE)

    model.area_annual_dn = Constraint(YEARS, rule=area_dn)
    model.area_annual_up = Constraint(YEARS, rule=area_up)

    # -- (10b) Terminal steady-state on irrigated area --
    # Finite-horizon boundary effect: in the last years, maintaining area has no
    # post-horizon payoff, so the discounted land anchor stops binding and area
    # winds down at ~the 3%/yr floor (an artefact, not a real result).  Require
    # non-decreasing area over the terminal window so the trajectory reaches a
    # steady state instead of an artificial end-of-horizon decline.  Tunable via
    # TERMINAL_SS_YEARS (0 disables).
    _TERM_SS = int(os.environ.get("TERMINAL_SS_YEARS", "7"))
    if _TERM_SS > 0:
        _term_years = YEARS[-_TERM_SS:]
        model.area_terminal_ss = Constraint(
            _term_years,
            rule=lambda m, y: m.total_irrigated_area[y] >= m.total_irrigated_area[y - 1])

    # -- (11) Calorie and protein demand --
    def calorie_demand(m, y):
        total_cal = sum(
            ((m.crop_prod[c, "Domestic", y] if (c, "Domestic", y) in m.crop_prod else 0)
             + (m.crop_prod_import[c, "Import", y] if (c, "Import", y) in m.crop_prod_import else 0))
            * crop_calories.get(c, 0)
            for c in crop_calories
        )
        return total_cal >= params[y]["calorie_demand"]

    def protein_demand(m, y):
        total_prot = sum(
            ((m.crop_prod[c, "Domestic", y] if (c, "Domestic", y) in m.crop_prod else 0)
             + (m.crop_prod_import[c, "Import", y] if (c, "Import", y) in m.crop_prod_import else 0))
            * crop_protein.get(c, 0)
            for c in crop_protein
        )
        return total_prot >= params[y]["protein_demand"]

    model.calorie_demand_con = Constraint(YEARS, rule=calorie_demand)
    model.protein_demand_con = Constraint(YEARS, rule=protein_demand)

    # -- (12) Energy allocation equality per (u, s, y) --
    def energy_alloc_rule(m, u, s, y):
        ef      = energy_req_fac.get((u, s), 0)
        req_mw  = m.production[u, s, y] * ef * 1e9 / (8_760 * 1000)
        alloc   = sum(m.energy_allocation[u, s, p, y] * pp_eff[p] for p in power_plants)
        return alloc == req_mw

    model.energy_allocation_con = Constraint(
        [(u, s, y) for (u, s) in _us_pairs for y in YEARS], rule=energy_alloc_rule)

    # -- (13) Power plant total production ≥ total energy required --
    def pp_production(m, y):
        total_alloc = sum(
            m.energy_allocation[u, s, p, y]
            for u in water_users for s in expanded_us[u] for p in power_plants
            if (u, s, p, y) in m.energy_allocation
        )
        total_req = sum(
            m.production[u, s, y] * energy_req_fac.get((u, s), 0) * 1e9 / (8_760 * 1000)
            for u in water_users for s in expanded_us[u]
        )
        return total_alloc >= total_req

    model.pp_production_con = Constraint(YEARS, rule=pp_production)

    # -- (14) Existing power plant capacity --
    existing_pp = [p for p in power_plants if p not in pp_new]

    def pp_cap_rule(m, p, y):
        total = sum(
            m.energy_allocation[u, s, p, y]
            for u in water_users for s in expanded_us[u]
            if (u, s, p, y) in m.energy_allocation
        )
        return total <= pp_eff[p] * pp_cap_year.get(y, {}).get(p, pp_cap[p])

    model.pp_capacity_con = Constraint(
        [(p, y) for p in existing_pp for y in YEARS], rule=pp_cap_rule)

    # -- (15) New power plant capacity (cumulative investments) --
    def new_pp_cap_rule(m, p, y):
        avail = sum(
            m.num_new_units[p, y_] * _NEW_PLANT_INVESTMENTS[p]["unit_size"] * pp_eff[p]
            for y_ in YEARS if y_ <= y and (y - y_) < PLANT_LIFETIME
        )
        total = sum(
            m.energy_allocation[u, s, p, y]
            for u in water_users for s in expanded_us[u]
            if (u, s, p, y) in m.energy_allocation
        )
        return total <= avail

    model.new_pp_cap_con = Constraint(
        [(p, y) for p in pp_new for y in YEARS], rule=new_pp_cap_rule)

    # -- (15b) Total new-power capacity ceiling ---------------------------------
    # The water-sector power fleet only needs to serve the water-sector energy
    # requirement.  Building capacity beyond dispatch costs only (below-precision)
    # capex, so num_new_units for power would otherwise float to arbitrary values.
    # Cap cumulative new nameplate capacity to a generous reserve multiple of the
    # actual water-sector energy requirement (MW), which eliminates the phantom
    # over-build while never binding on genuine demand / renewable-availability.
    _POWER_RESERVE = float(os.environ.get("POWER_RESERVE_MARGIN", "4.0"))

    def total_new_power_cap_rule(m, y):
        new_nameplate = sum(
            m.num_new_units[p, y_] * _NEW_PLANT_INVESTMENTS[p]["unit_size"]
            for p in pp_new for y_ in YEARS
            if y_ <= y and (y - y_) < PLANT_LIFETIME
        )
        req_mw = sum(
            m.production[u, s, y] * energy_req_fac.get((u, s), 0) * 1e9 / (8_760 * 1000)
            for u in water_users for s in expanded_us[u]
        )
        return new_nameplate <= _POWER_RESERVE * req_mw

    model.total_new_power_cap_con = Constraint(YEARS, rule=total_new_power_cap_rule)

    # -- (15c) Disable water-model power build (national fleet is exogenous) ------
    # Generation capacity grows exogenously along the national planning trajectory
    # (_CAP_BY_YEAR).  With the growing fleet the existing capacity already covers
    # all water-sector demand and satisfies the renewable share, so any new power
    # build is a below-precision-capex artifact.  Fix power new-build to zero.
    # Reversible: set DISABLE_POWER_BUILD=0 to re-enable endogenous power build.
    if os.environ.get("DISABLE_POWER_BUILD", "1") == "1":
        model.no_power_build = Constraint(
            [(p, y) for p in pp_new for y in YEARS],
            rule=lambda m, p, y: m.num_new_units[p, y] == 0)

    # -- (16) Renewable energy share --
    renewable_names = ["Renewable", "Solar", "Wind", "Nuclear", "NewSolar", "NewWind"]

    def ren_share_rule(m, y):
        ren_total = sum(
            m.energy_allocation[u, s, p, y]
            for u in water_users for s in expanded_us[u] for p in renewable_names
            if (u, s, p, y) in m.energy_allocation
        )
        all_total = sum(
            m.energy_allocation[u, s, p, y]
            for u in water_users for s in expanded_us[u] for p in power_plants
            if (u, s, p, y) in m.energy_allocation
        )
        return ren_total >= params[y]["ren_share"] * all_total

    model.ren_share_con = Constraint(YEARS, rule=ren_share_rule)

    # -- (17) Groundwater policy constraints --
    if scenario1:
        model.gw_limit = Constraint(
            YEARS,
            rule=lambda m, y: sum(
                m.production[u, s, y] for u in water_users for s in expanded_us[u]
                if s.startswith("Groundwater")
            ) <= 10,
        )
        print("  Applied: 10 BCM groundwater limit (all years)")
    elif scenario2:
        model.gw_limit = Constraint(
            YEARS,
            rule=lambda m, y: sum(
                m.production[u, s, y] for u in water_users for s in expanded_us[u]
                if s.startswith("Groundwater")
            ) <= 5,
        )
        print("  Applied: 5 BCM groundwater limit (all years)")
    else:
        # Technical upper bound (same 30 BCM reasoning as sequential model)
        model.technical_gw_max = Constraint(
            YEARS,
            rule=lambda m, y: sum(
                m.production[u, s, y] for u in water_users for s in expanded_us[u]
                if s.startswith("Groundwater")
            ) <= 30,
        )

    # -- (18) Surface water and TWW caps --
    model.max_sw = Constraint(
        YEARS,
        rule=lambda m, y: sum(
            m.production[u, s, y] for u in water_users for s in expanded_us[u]
            if s.startswith("SurfaceWater")
        ) <= 2,
    )

    model.max_tww = Constraint(
        YEARS,
        rule=lambda m, y: sum(
            m.production[u, s, y] for u in water_users for s in expanded_us[u]
            if s.startswith("TreatedWW")
        ) <= 0.8 * (params[y]["domestic_demand"] + params[y]["industry_demand"]),
    )

    # Total new TWW capacity ≤ headroom above existing (shrinks as existing fleet retires)
    def max_tww_new_cap(m, y):
        total_new_tww_cap = sum(
            m.num_new_units["TreatedWW_NewSaudiTWW", y_]
            * _NEW_PLANT_INVESTMENTS["TreatedWW_NewSaudiTWW"]["unit_size"]
            for y_ in YEARS if y_ <= y and (y - y_) < PLANT_LIFETIME
        )
        # Existing TWW cap is constant (no retirement in NLP); new capacity fills remaining headroom
        limit = max(0.0, 0.8 * (params[y]["domestic_demand"] + params[y]["industry_demand"]) - _TWW_EXISTING_CAP)
        return total_new_tww_cap <= limit

    model.max_tww_new_cap_con = Constraint(YEARS, rule=max_tww_new_cap)

    # -- Dam / surface-water cap: surface water is physically limited by rainfall/
    # runoff (not freely buildable). Total surface-water production and new dam
    # capacity are capped at the national physical maximum (~2.25 BCM), matching
    # the regional model (_SURFACE_MAX_BCM). Without this the cost objective — after
    # the scaling needed for the penalty-heavy formulation — leaves new-dam capacity
    # below solver precision, so it over-builds to physically impossible levels.
    _SURFACE_MAX_BCM = float(os.environ.get("SURFACE_MAX_BCM", "2.25"))
    model.max_dam = Constraint(
        YEARS,
        rule=lambda m, y: sum(
            m.production[u, s, y] for u in water_users for s in expanded_us[u]
            if s.startswith("SurfaceWater")
        ) <= _SURFACE_MAX_BCM,
    )

    # Dams may be BUILT, but nameplate capacity is bounded by what is needed to
    # produce the surface-water cap (production ≤ capacity × source_eff ≤ SURFACE_MAX),
    # plus a reserve margin.  This is a PRODUCTION-based bound (derived from max_dam),
    # not an arbitrary capacity cap: dams can grow until surface-water production
    # reaches 2.25 BCM.  It only prevents the (below-precision) investment cost from
    # over-building capacity to physically impossible levels.
    _DAM_EFF = float(os.environ.get("DAM_EFF", "0.425"))
    def max_dam_cap(m, y):
        total_dam_cap = _DAM_EXISTING_CAP + sum(
            m.num_new_units["SurfaceWater_NewSaudiDam", y_]
            * _NEW_PLANT_INVESTMENTS["SurfaceWater_NewSaudiDam"]["unit_size"]
            for y_ in YEARS if y_ <= y and (y - y_) < PLANT_LIFETIME
        )
        return total_dam_cap <= 1.1 * _SURFACE_MAX_BCM / _DAM_EFF
    model.max_dam_cap_con = Constraint(YEARS, rule=max_dam_cap)

    # No new dam investment permitted before DAM_START_YEAR (policy: 2035).
    _DAM_START_YEAR = int(os.environ.get("DAM_START_YEAR", "2035"))
    _pre_dam_years = [y for y in YEARS if y < _DAM_START_YEAR]
    if _pre_dam_years:
        model.no_dam_before = Constraint(
            _pre_dam_years,
            rule=lambda m, y: m.num_new_units["SurfaceWater_NewSaudiDam", y] == 0)

    # Dam construction-rate limit: large dams take ~8 yr to build (Ansar et al.,
    # 2014), so dam capacity cannot be added instantaneously.  Without this the
    # model front-loads the entire ~2.5 BCM of new dams into 2035 (a $4.6B one-year
    # capex spike).  Cap new dam capacity to DAM_BUILD_RATE BCM/yr, which spreads
    # the build over ~7 yr, consistent with the empirical construction duration.
    _DAM_BUILD_RATE = float(os.environ.get("DAM_BUILD_RATE", "0.35"))   # BCM/yr
    _dam_unit = _NEW_PLANT_INVESTMENTS["SurfaceWater_NewSaudiDam"]["unit_size"]
    model.dam_build_rate_con = Constraint(
        YEARS,
        rule=lambda m, y: m.num_new_units["SurfaceWater_NewSaudiDam", y] * _dam_unit
                          <= _DAM_BUILD_RATE)

    # -- Desalination capacity cap: desal serves only domestic + industry
    # (agriculture cannot use desal, see no_agr_desal), so total desal capacity
    # should never exceed those sectors' demand plus a small reserve margin.
    # Unlike dams/TWW, desal has no natural physical limit, so without this bound
    # the (below-precision) investment cost lets it over-build to absurd levels. --
    # Efficiency enters twice (capacity->production and production->delivered), so
    # delivered = nameplate * eff^2.  The cap is on NAMEPLATE, but demand is
    # DELIVERED water, so the ceiling must be (dom+ind)/eff^2 to let desal deliver
    # the full domestic+industry demand, times a small reserve margin.
    _DESAL_RESERVE = float(os.environ.get("DESAL_RESERVE", "1.15"))
    _DESAL_EFF     = float(os.environ.get("DESAL_EFF", "0.85"))   # RO/MED/MSF/NewRO
    def max_desal_cap(m, y):
        existing = sum(params[y]["desal_caps"].values())
        new = sum(
            m.num_new_units["Desalination_NewRO", y_]
            * _NEW_PLANT_INVESTMENTS["Desalination_NewRO"]["unit_size"]
            for y_ in YEARS if y_ <= y and (y - y_) < PLANT_LIFETIME
        )
        ceiling = _DESAL_RESERVE * (params[y]["domestic_demand"]
                                    + params[y]["industry_demand"]) / (_DESAL_EFF ** 2)
        return existing + new <= ceiling

    model.max_desal_cap_con = Constraint(YEARS, rule=max_desal_cap)

    # -- Desalination capacity vs. utilization (prevents building unused RO) ------
    # The demand ceiling above bounds absolute capacity, but in scenarios with
    # abundant groundwater desalination PRODUCTION is low, and the below-precision
    # investment cost otherwise lets the solver fill RO capacity toward that
    # ceiling even though little is produced (e.g. No-GW built ~12.5 BCM nameplate
    # to produce ~5.5 BCM, 44% utilisation).  Cap cumulative new-RO nameplate to a
    # small reserve margin over its own production, mirroring the power-fleet cap
    # (total_new_power_cap_con).  New plants use production <= nameplate directly.
    # RESERVE=1.15 (86.9% min utilisation, tried 2026-07-09/10) was NOT tight
    # enough: a 10-seed multistart check showed utilisation still wandering
    # 87-100%+ within a single scenario (as much variation within a scenario as
    # between 10 BCM and 5 BCM), so reported fleet size/investment cost depended
    # on which local optimum the solver happened to land in, not on genuine
    # scenario economics. Tightened to 95% min utilisation (1/0.95) so nameplate
    # tracks the (robust, seed-stable) production number directly instead of
    # floating in this flat objective direction.
    _DESAL_PROD_RESERVE = float(os.environ.get("DESAL_PROD_RESERVE", str(1 / 0.95)))
    def new_desal_cap_vs_prod(m, y):
        nameplate = sum(
            m.num_new_units["Desalination_NewRO", y_]
            * _NEW_PLANT_INVESTMENTS["Desalination_NewRO"]["unit_size"]
            for y_ in YEARS if y_ <= y and (y - y_) < PLANT_LIFETIME
        )
        prod = sum(
            m.production[u, "Desalination_NewRO", y]
            for u in water_users if "Desalination_NewRO" in expanded_us[u]
        )
        return nameplate <= _DESAL_PROD_RESERVE * prod

    model.new_desal_cap_vs_prod_con = Constraint(YEARS, rule=new_desal_cap_vs_prod)

    # -- New dam capacity vs. utilization (same fix as desal, 2026-07-11) --------
    # New dam nameplate had NO utilization-tying constraint at all (unlike desal):
    # only the fixed absolute ceiling (max_dam_cap) and the build-rate limit. A
    # 10-seed check showed utilization wandering 34-50% with no consistent pattern
    # between 10 BCM and 5 BCM scenarios — the reported "10 BCM builds more dams"
    # difference was this same flat-objective-direction noise, not real scenario
    # economics (nearly all seeds for BOTH scenarios actually converge to the
    # same ~5.65-5.68 BCM once the noise is filtered). RESERVE=1/0.9 matches new
    # dams' corrected efficiency (see _STATIC_SOURCE_EFF fix above) exactly, so
    # this pins nameplate to production/0.9 with no slack, same logic as desal.
    _DAM_PROD_RESERVE = float(os.environ.get("DAM_PROD_RESERVE", str(1 / 0.9)))
    def new_dam_cap_vs_prod(m, y):
        nameplate = sum(
            m.num_new_units["SurfaceWater_NewSaudiDam", y_]
            * _NEW_PLANT_INVESTMENTS["SurfaceWater_NewSaudiDam"]["unit_size"]
            for y_ in YEARS if y_ <= y and (y - y_) < PLANT_LIFETIME
        )
        prod = sum(
            m.production[u, "SurfaceWater_NewSaudiDam", y]
            for u in water_users if "SurfaceWater_NewSaudiDam" in expanded_us[u]
        )
        return nameplate <= _DAM_PROD_RESERVE * prod

    model.new_dam_cap_vs_prod_con = Constraint(YEARS, rule=new_dam_cap_vs_prod)

    # -- (19) Groundwater stock depletion (cross-year linking) --
    def gw_stock_rule(m, y):
        gw_prod_y = sum(
            m.production[u, s, y]
            for u in water_users for s in expanded_us[u]
            if s.startswith("Groundwater")
        )
        if y == YEARS[0]:
            return m.gw_stock[y] == GW_INITIAL_STOCK_BCM - gw_prod_y
        else:
            return m.gw_stock[y] == m.gw_stock[y - 1] - gw_prod_y

    model.gw_stock_con = Constraint(YEARS, rule=gw_stock_rule)
    # gw_stock domain=NonNegativeReals already enforces stock ≥ 0

    print(f"  Constraints added. Build time: {time.time()-t_build:.1f}s")

    # =========================================================================
    # WARM-START: initialise from sequential results (unconstrained only)
    # =========================================================================
    if not use_warmstart:
        print("  Cold-start — initialising from calibrated 2023 baseline (feasibility-consistent)")
    else:
        print("  Setting warm-start values from sequential results...")

    # Pre-compute calibrated cultivated areas so total_irrigated_area is consistent
    # with sum(cultivated_area) at the initial point.  Starting from a grossly
    # infeasible point (cultivated_area≈0 ha vs total_irrigated_area=787k ha) causes
    # MUMPS to fail factorising the KKT matrix on the very first IPOPT iteration
    # (internalSolverError), regardless of tolerance settings.
    _ca_init = {
        c: max(1.0, baseline_cprod.get(c, _Q_FLOOR) / yield_per_ha.get(c, 1.0))
        for c in domestic_crops
    }
    _tia_init = sum(_ca_init.values())

    # Estimate a representative groundwater production per year for gw_stock chaining.
    # Use 10 BCM/year (historical Saudi average) so gw_stock decreases plausibly.
    _GW_ANNUAL_ESTIMATE = 10.0  # BCM/year

    for y in YEARS:
        wd = ws_all.get(y, {})

        # -- Cultivated area & domestic crop production: use calibrated baseline --
        # These must be consistent with each other (prod_area equality) and with
        # total_irrigated_area (total_irrigated_area_def equality).
        for c in domestic_crops:
            ca = wd.get(("cultivated_area", c), _ca_init[c])
            cp = wd.get(("crop_prod", c), _ca_init[c] * yield_per_ha.get(c, 1.0))
            if (c, y) in model.cultivated_area:
                model.cultivated_area[c, y].value = max(0.01, ca)
                model.cultivated_area[c, y].stale = False
            if (c, "Domestic", y) in model.crop_prod:
                model.crop_prod[c, "Domestic", y].value = max(0.01, cp)
                model.crop_prod[c, "Domestic", y].stale = False

        # -- Total irrigated area: consistent with sum of cultivated areas --
        tia = wd.get(("total_irrigated_area",), _tia_init)
        model.total_irrigated_area[y].value = max(100_000, tia)
        model.total_irrigated_area[y].stale = False

        # -- Import crops --
        for c in import_crops:
            ci = wd.get(("crop_prod_import", c), max(0.1, baseline_cprimp.get(c, 0.1)))
            if (c, "Import", y) in model.crop_prod_import:
                model.crop_prod_import[c, "Import", y].value = max(0.0, ci)
                model.crop_prod_import[c, "Import", y].stale = False

        # -- Water production & allocation: load from sequential warm-start --
        for u in water_users:
            for s in expanded_us[u]:
                pv = wd.get(("production", u, s), 0.1)
                eff = source_eff.get(s, 1.0)
                av = wd.get(("allocation", u, s), pv * eff)

                # No TWW clamping: existing TWW capacity is constant (no retirement in NLP)

                if (u, s, y) in model.production:
                    model.production[u, s, y].value = max(0.01, pv)
                    model.production[u, s, y].stale = False
                if (u, s, y) in model.allocation:
                    model.allocation[u, s, y].value = max(0.01, av)
                    model.allocation[u, s, y].stale = False

        # -- GW stock: from warm-start or decreasing estimate --
        gws_default = max(0.0, GW_INITIAL_STOCK_BCM - (y - YEARS[0] + 1) * _GW_ANNUAL_ESTIMATE)
        gws = wd.get(("gw_stock",), gws_default)
        model.gw_stock[y].value = max(0.0, gws)
        model.gw_stock[y].stale = False

        # -- New plant investments: always zero at warm-start --
        # Sequential num_new_units = cumulative capacity; intertemporal = annual decision.
        # Loading cumulative values as annual investments would be semantically wrong.
        for pl in _ALL_NEW_PLANTS:
            if (pl, y) in model.num_new_units:
                model.num_new_units[pl, y].value = 0.0
                model.num_new_units[pl, y].stale = False

        # -- Energy allocation: small positive --
        for u in water_users:
            for s in expanded_us[u]:
                for p in power_plants:
                    if (u, s, p, y) in model.energy_allocation:
                        model.energy_allocation[u, s, p, y].value = 0.01
                        model.energy_allocation[u, s, p, y].stale = False

    # =========================================================================
    # MULTI-START perturbation.  The desalination cost is concave (economies of
    # scale) so the NLP is non-convex with multiple local optima.  MS_SEED varies
    # the starting point (multiplicative noise on crops + water production) so a
    # driver can sample many local optima and pick the best / plot the spread.
    # =========================================================================
    # Multi-start over SOLVER SETTINGS (not the primal start): perturbing the
    # primal warm-start crashes MUMPS on the ill-conditioned KKT (factorisation
    # failure at iter 0), so instead we start from the single factorisable
    # warm-start and vary IPOPT's barrier/scaling path.  Different paths converge
    # to different local optima of the non-convex (concave-desal) problem — we
    # already observe user-scaling -> 6.4e16 vs gradient-based -> 1.79e15.
    _ms_seed = os.environ.get("MS_SEED")
    _ms_opts = None
    if _ms_seed is not None:
        import random as _rnd
        _r = _rnd.Random(int(_ms_seed))
        # Bias toward the known-factorisable style (user-scaling, adaptive mu,
        # small bound_push); some combinations crash MUMPS, so the driver runs
        # each seed as its own subprocess (crash-isolated).
        # Restrict to the CONVERGING style (user-scaling + obj_scale 1e-15, which
        # reaches a genuine KKT point); vary only the barrier PATH (mu_init /
        # mu_strategy / bound_push).  Different barrier paths can still land in
        # different local optima of the non-convex (concave-desal) problem, but
        # none is doomed to non-convergence — so seeds that DON'T converge are a
        # real signal (filtered by relviol), not a self-inflicted bad config.
        # (gradient-based scaling and obj_scale 1e-14/1e-16 were removed: they
        # never converge here and only injected junk into the spread.)
        _ms_scaling = "user-scaling"
        _ms_mu_init = _r.choice([1e-2, 1e-3, 1e-4, 1e-5])
        _ms_mu_str  = _r.choice(["adaptive", "adaptive", "monotone"])
        _ms_bpush   = _r.choice([1e-6, 1e-6, 1e-4, 1e-2])
        _ms_objsc   = 1e-15
        model.scaling_factor[model.obj] = _ms_objsc
        _ms_opts = {"nlp_scaling_method": _ms_scaling,
                    "warm_start_init_point": "yes",
                    "warm_start_bound_push": _ms_bpush,
                    "mu_strategy": _ms_mu_str,
                    "mu_init": _ms_mu_init,
                    "max_soc": 20,
                    "max_iter": int(os.environ.get("A0_MAXITER", "300"))}
        print(f"  [MULTISTART seed={_ms_seed}] scaling={_ms_scaling} mu_init={_ms_mu_init} "
              f"mu={_ms_mu_str} bpush={_ms_bpush} objsc={_ms_objsc}", flush=True)

    # =========================================================================
    # SOLVE with retry sequence
    # =========================================================================

    _FEASIBLE_TC = {
        TerminationCondition.optimal,
        TerminationCondition.feasible,
        TerminationCondition.maxIterations,
        TerminationCondition.maxTimeLimit,
    }

    # Retry sequence: (tol, acceptable_tol, extra_opts, reinit_val)
    # obj_scaling_factor=1e-9 rescales the NPV objective (~$1T over 37 years) to O(1000)
    # so IPOPT gradient precision losses from large water-cost terms are avoided.
    _RETRY_SEQ = [
        # Attempt 0 (user-scaling): use the model.scaling_factor suffixes so IPOPT
        # solves the non-dimensionalised problem.  Fixes the ill-conditioning that
        # stalls the binding-GW (10/5 BCM) scenarios under auto/gradient scaling.
        # The dense PMP-penalty Hessian keeps dual infeasibility ~1e-4 even once
        # the point is fully feasible (constraint viol ~1e-10).  IPOPT then
        # mislabels this feasible point as "local infeasibility"; the feasible-
        # iterate fallback below detects the tiny constraint violation and
        # accepts it.
        # acc_tol=1e-6 + tight acceptable_* => IPOPT stops at the KKT-grade
        # acceptable level ("Solved To Acceptable Level") once the feasible
        # stationary point is reached, rather than continuing the barrier
        # reduction into the non-convex desal region and drifting to another
        # local optimum.  (acceptable_iter/acceptable_obj_change_tol are Unknown
        # keywords in this IPOPT build, so we rely on the default acceptable_iter.)
        (1e-6, 1e-6, {"nlp_scaling_method": "user-scaling",
                      "warm_start_init_point": "yes",
                      "warm_start_bound_push": 1e-6,
                      "mu_strategy": "adaptive",
                      "max_soc": 20,
                      "max_iter": int(os.environ.get("A0_MAXITER", "340"))},              None),
        # Attempt 1: warm-start-aware — designed for binding GW constraints.
        # warm_start_init_point=yes tells IPOPT not to disturb the primal warm
        # start (sequential result already sits at the binding GW limit).
        # warm_start_bound_push=1e-9 prevents the default 1% push that would
        # move GW production away from exactly 10/5 BCM before the first step.
        # mu_init=1e-4 starts with a tight barrier (close to the KKT point).
        # mu_strategy=adaptive + max_soc=20 help navigate the 37 binding dual
        # conditions simultaneously.
        (1e-3, 1e-3, {"nlp_scaling_method": "gradient-based",
                      "warm_start_init_point": "yes",
                      "warm_start_bound_push": 1e-9,
                      "warm_start_slack_bound_push": 1e-9,
                      "mu_strategy": "adaptive",
                      "mu_init": 1e-4,
                      "max_soc": 20,
                      "max_iter": 5_000},                                                 None),
        # Attempt 2: plain defaults — closest to the original working run.
        # No obj_scaling_factor, no extra scaling: let IPOPT auto-scale.
        (1e-4, None, {"max_iter": 3_000},                                                 None),
        # Attempt 3: looser tolerance, gradient-based scaling
        (1e-3, 1e-3, {"nlp_scaling_method": "gradient-based", "max_iter": 3_000},        None),
        # Attempt 3: even looser, adaptive mu, max_soc=20.
        # max_soc raises the second-order-correction budget (default 4) so IPOPT
        # has more chances to recover from a poor line-search step near the optimum,
        # preventing the "Restoration Failed" exit we observed in IPOPT 3.12.13.
        (1e-2, 1e-2, {"nlp_scaling_method": "gradient-based", "mu_strategy": "adaptive",
                      "max_soc": 20, "max_iter": 5_000},                                 None),
        # Attempt 4: reinit + obj scaling + max_soc
        (1e-2, 1e-2, {"nlp_scaling_method": "gradient-based", "mu_strategy": "adaptive",
                      "obj_scaling_factor": 1e-9, "bound_push": 0.1, "bound_frac": 0.1,
                      "max_soc": 20, "max_iter": 5_000},                                 1.0),
        # Attempt 5: monotone mu, reinit
        (1e-2, 1e-2, {"nlp_scaling_method": "none", "mu_strategy": "monotone",
                      "mu_init": 1.0, "bound_push": 0.01, "bound_frac": 0.01,
                      "max_iter": 10_000},                                                0.01),
        # Attempt 6: infeasible-problem mode as last resort
        (1e-1, 1e-1, {"expect_infeasible_problem": "yes",
                      "nlp_scaling_method": "gradient-based", "mu_strategy": "adaptive",
                      "mu_init": 10.0, "bound_push": 1e-6, "bound_frac": 1e-6,
                      "max_iter": 20_000},                                                0.1),
    ]

    # Multi-start: a single randomized-solver attempt from the warm-start (no
    # retry ladder — each start is one solver configuration, attributed cleanly).
    if _ms_opts is not None:
        _RETRY_SEQ = [(1e-6, 1e-4, _ms_opts, None)]

    _WS_PRESERVE = {"cultivated_area", "crop_prod_import", "total_irrigated_area"}

    def _try_solve(tol, acc_tol, extra_opts):
        solver = SolverFactory("ipopt")
        solver.options["tol"]         = tol
        solver.options["print_level"] = int(os.environ.get("DIAG_PRINT_LEVEL", "3"))
        # Hard wall-clock cap per attempt so a single ill-conditioned solve cannot
        # hang indefinitely (default: no cap; set DIAG_MAX_CPU to bound diagnostics).
        _mcpu = os.environ.get("DIAG_MAX_CPU")
        if _mcpu:
            solver.options["max_cpu_time"] = float(_mcpu)
        if acc_tol is not None:
            solver.options["acceptable_tol"] = acc_tol
        # extra_opts sets max_iter and all other options — no hidden 200k default
        for k, v in extra_opts.items():
            solver.options[k] = v
        results = solver.solve(model, tee=True, load_solutions=False)
        return results, results.solver.termination_condition

    def _max_constraint_violation(m):
        """Largest active-constraint violation in native units (0 = feasible)."""
        worst = 0.0
        for con in m.component_data_objects(Constraint, active=True):
            try:
                b = value(con.body)
            except Exception:
                continue
            if b is None:
                continue
            lb = value(con.lower) if con.lower is not None else None
            ub = value(con.upper) if con.upper is not None else None
            if lb is not None and b < lb:
                worst = max(worst, lb - b)
            if ub is not None and b > ub:
                worst = max(worst, b - ub)
        return worst

    def _feas_report(m):
        """Return (abs_worst, rel_worst, worst_con_name).  Relative violation
        normalizes each constraint by its own scale so a converged solve reads
        ~0 regardless of units (tonnage constraints otherwise dominate the
        absolute metric with a large constant)."""
        abs_w = 0.0; rel_w = 0.0; name = ""
        for con in m.component_data_objects(Constraint, active=True):
            try:
                b = value(con.body)
            except Exception:
                continue
            if b is None:
                continue
            lb = value(con.lower) if con.lower is not None else None
            ub = value(con.upper) if con.upper is not None else None
            v = 0.0
            if lb is not None and b < lb:
                v = lb - b
            if ub is not None and b > ub:
                v = max(v, b - ub)
            if v <= 0:
                continue
            scale = max(abs(lb or 0.0), abs(ub or 0.0), abs(b), 1.0)
            rel = v / scale
            if rel > rel_w:
                rel_w = rel; name = con.name
            abs_w = max(abs_w, v)
        return abs_w, rel_w, name

    # Terminations that may still hide a genuinely feasible point (see the
    # user-scaling note above): check the real constraint violation before retry.
    _INFEAS_TC = {TerminationCondition.infeasible, TerminationCondition.other,
                  TerminationCondition.noSolution, TerminationCondition.unknown}
    _loc_infeas = getattr(TerminationCondition, "locallyInfeasible", None)
    if _loc_infeas is not None:
        _INFEAS_TC.add(_loc_infeas)

    tc           = None
    results      = None
    solve_note   = ""
    _rf_accepted = False   # True if we accepted a Restoration-Failed iterate

    print(f"  Starting solve at {time.strftime('%H:%M:%S')} ...")
    for tol_val, acc_tol, extra_opts, reinit_val in _RETRY_SEQ:
        if reinit_val is not None:
            # Reinitialise non-warm-started variables, preserving crop/area values
            for var in model.component_objects(Var, active=True):
                if var.name in _WS_PRESERVE:
                    continue
                for idx in var:
                    if hasattr(var[idx].domain, "bounds"):
                        var[idx].value = reinit_val
                    else:
                        var[idx].value = reinit_val
                    var[idx].stale = False
            # Reconstruct crop_prod from cultivated_area
            for y in YEARS:
                for c in domestic_crops:
                    ca_val = model.cultivated_area[c, y].value
                    if ca_val is not None and (c, "Domestic", y) in model.crop_prod:
                        model.crop_prod[c, "Domestic", y].value = ca_val * yield_per_ha.get(c, 1.0)

        _sc  = extra_opts.get("nlp_scaling_method", "default")
        _mu  = extra_opts.get("mu_strategy", "")
        albl = f"tol={tol_val}, scale={_sc}" + (f", mu={_mu}" if _mu else "")
        if reinit_val is not None:
            albl += f", reinit={reinit_val}"
        if extra_opts.get("expect_infeasible_problem") == "yes":
            albl += ", infeas-mode"

        try:
            results, tc = _try_solve(tol_val, acc_tol, extra_opts)
            if tc in _FEASIBLE_TC:
                model.solutions.load_from(results)
                obj_val = value(model.obj)
                print(f"  [{albl}] Termination: {tc}   Objective (NPV): {obj_val:.4e}")
                solve_note = albl
                break
            elif tc in _INFEAS_TC:
                # May be a feasible point IPOPT mislabeled — verify actual
                # constraint violation before discarding it.  Use RELATIVE
                # violation (tonnage constraints dominate the absolute metric
                # with a large constant even when relatively feasible).
                model.solutions.load_from(results)
                _mv, _relv, _wc = _feas_report(model)
                obj_val = value(model.obj)
                if (obj_val is not None and not (math.isnan(obj_val) or math.isinf(obj_val))
                        and _relv < 1e-4):
                    print(f"  [{albl}] '{tc}' but point IS feasible "
                          f"(rel viol {_relv:.2e}) — accepting: NPV={obj_val:.4e}")
                    _rf_accepted = True
                    solve_note = albl + f" [feasible iterate, relviol={_relv:.1e}]"
                    break
                print(f"  [{albl}] {tc} (rel viol {_relv:.2e}, worst={_wc}) — retrying...")
                solve_note = f"{albl}: {tc}"
            else:
                print(f"  [{albl}] {tc} — retrying...")
                solve_note = f"{albl}: {tc}"
        except Exception as e:
            print(f"  [{albl}] EXCEPTION: {e} — retrying...")
            solve_note = f"{albl}: exception({e})"
            tc = None

    # -------------------------------------------------------------------------
    # Restoration-Failed fallback
    # IPOPT's "Restoration Failed" exit often means the solver reached a
    # near-optimal iterate but the subsequent line-search step violated
    # constraints.  The .sol file written by IPOPT still contains that iterate.
    # Load it and accept if all GW stock values are non-negative (basic feasibility).
    # -------------------------------------------------------------------------
    if tc not in _FEASIBLE_TC and tc == TerminationCondition.internalSolverError and results is not None:
        print("  All attempts ended in Restoration Failed — loading last IPOPT iterate for feasibility check...")
        try:
            model.solutions.load_from(results)
            obj_val = value(model.obj)
            if obj_val is not None and not (math.isnan(obj_val) or math.isinf(obj_val)):
                gw_ok = all(
                    value(model.gw_stock[y]) is not None and value(model.gw_stock[y]) >= -1e-3
                    for y in YEARS
                )
                if gw_ok:
                    print(f"  Iterate is feasible — accepting Restoration Failed solution: NPV={obj_val:.4e}")
                    _rf_accepted = True
                    solve_note += " [accepted Restoration-Failed iterate]"
                else:
                    print("  Iterate has negative GW stock — discarding, recording failure")
            else:
                print(f"  Iterate objective is NaN/inf ({obj_val}) — discarding")
        except Exception as e:
            print(f"  Could not load Restoration Failed iterate: {e}")

    # -------------------------------------------------------------------------
    # MULTI-START recording: log this start's objective + physical totals to a
    # shared CSV and exit (skip the per-year xlsx to keep each start cheap).
    # -------------------------------------------------------------------------
    if _ms_seed is not None:
        _accepted = _rf_accepted or (tc in _FEASIBLE_TC)
        try:
            if _accepted or results is not None:
                try:
                    model.solutions.load_from(results)
                except Exception:
                    pass
            _obj = value(model.obj)
        except Exception:
            _obj = float("nan")
        try:
            _real_npv = value(model._real_npv); _pen_npv = value(model._pen_npv)
            print(f"  [DECOMP] real_cost_NPV={_real_npv:.4e}  PMP_penalty_NPV={_pen_npv:.4e}  "
                  f"total={_real_npv + _pen_npv:.4e}", flush=True)
            _py = {y: value(v) for y, v in model._pen_by_year.items()}
            print(f"  [DECOMP-YR] penalty 2024={_py.get(2024,0):.4e} 2030={_py.get(2030,0):.4e} "
                  f"2045={_py.get(2045,0):.4e} 2060={_py.get(2060,0):.4e}", flush=True)
        except Exception as _e:
            print(f"  [DECOMP] unavailable: {_e}")
        try:
            _mv, _relv, _worstc = _feas_report(model)
        except Exception:
            _mv = _relv = float("nan"); _worstc = ""
        def _psum(pred):
            tot = 0.0
            for u in water_users:
                for s in expanded_us[u]:
                    if not pred(s):
                        continue
                    for y in YEARS:
                        if (u, s, y) in model.production:
                            v = value(model.production[u, s, y])
                            tot += v if v is not None else 0.0
            return tot
        try:
            gw_tot    = _psum(lambda s: s.startswith("Groundwater"))
            desal_tot = _psum(lambda s: s.startswith("Desalination"))
            dom_tot   = sum((value(model.crop_prod[c, "Domestic", y]) or 0.0)
                            for c in domestic_crops for y in YEARS if (c, "Domestic", y) in model.crop_prod)
            imp_tot   = sum((value(model.crop_prod_import[c, "Import", y]) or 0.0)
                            for c in import_crops for y in YEARS if (c, "Import", y) in model.crop_prod_import)
        except Exception:
            gw_tot = desal_tot = dom_tot = imp_tot = float("nan")
        import csv as _csv
        _csvpath = os.environ.get("MS_CSV", os.path.join(RESULTS_DIR_IT, f"multistart_{scenario_tag}.csv"))
        _newfile = not os.path.exists(_csvpath)
        with open(_csvpath, "a", newline="") as _f:
            _w = _csv.writer(_f)
            if _newfile:
                _w.writerow(["seed", "scenario", "status", "accepted", "objective",
                             "maxviol", "relviol", "worst_con", "gw_bcm_sum", "desal_bcm_sum",
                             "domcrop_tons_sum", "import_tons_sum"])
            _w.writerow([_ms_seed, scenario_tag, str(tc), _accepted, _obj,
                         _mv, _relv, _worstc, gw_tot, desal_tot, dom_tot, imp_tot])
        print(f"  [MULTISTART seed={_ms_seed}] obj={_obj:.6e} relviol={_relv:.2e} "
              f"accepted={_accepted} tc={tc}")
        # Production mode: reproduce a chosen best seed AND write the per-year
        # xlsx (fall through to the normal export instead of the cheap CSV-only
        # exit).  Used to materialise the selected multi-start optimum.
        if not os.environ.get("MS_WRITE_XLSX"):
            return

    if tc not in _FEASIBLE_TC and not _rf_accepted:
        status_str = str(tc) if tc is not None else "exception"
        print(f"  All attempts failed ({status_str}) — saving failure record")
        os.makedirs(RESULTS_DIR_IT, exist_ok=True)
        fail_path = os.path.join(RESULTS_DIR_IT, f"FAILED_{scenario_tag}.xlsx")
        pd.DataFrame([
            {"Metric": "Energy Scenario", "Value": energy_scenario},
            {"Metric": "GW Scenario",     "Value": gw_label},
            {"Metric": "Solver Status",   "Value": status_str},
            {"Metric": "Note",            "Value": solve_note},
        ]).to_excel(fail_path, index=False)
        return

    # =========================================================================
    # EXPORT RESULTS — one xlsx per year (identical format to sequential model)
    # =========================================================================
    print(f"  Saving year-by-year results to {RESULTS_DIR_IT}/")
    os.makedirs(RESULTS_DIR_IT, exist_ok=True)

    crop_water_reqs = _CROP_WATER_REQS_2023
    phase_product   = _PHASE_PRODUCT_2023
    food_crop_set   = set(food_crops)
    feed_crop_set   = set(feed_crops)
    result_crops    = sorted(food_crop_set | feed_crop_set)

    for y in YEARS:
        out_file = os.path.join(RESULTS_DIR_IT, f"{y}_{energy_scenario}_{gw_label}.xlsx")
        p        = params[y]

        # -- Water allocation & production --
        alloc_rows = []
        prod_rows  = []
        for u in water_users:
            for s in expanded_us[u]:
                av = value(model.allocation[u, s, y])  if (u, s, y) in model.allocation  else 0
                pv = value(model.production[u, s, y])  if (u, s, y) in model.production  else 0
                alloc_rows.append({"User": u, "Source": s,
                                   "Water Allocated (billion m3)": av,
                                   "Plant Type": "New" if s in _WATER_NEW_PLANTS else "Existing"})
                prod_rows.append({"User": u, "Source": s,
                                  "Water Produced (billion m3)": pv,
                                  "Plant Type": "New" if s in _WATER_NEW_PLANTS else "Existing"})

        # -- Remaining capacity --
        rem_rows = []
        for s in all_sources:
            parts = s.split("_", 1)
            st, sname = parts[0], parts[1] if len(parts) > 1 else ""
            if s in _WATER_NEW_PLANTS:
                cap = sum(
                    value(model.num_new_units[s, y_]) * _NEW_PLANT_INVESTMENTS[s]["unit_size"]
                    for y_ in YEARS if y_ <= y and (y - y_) < PLANT_LIFETIME
                )
            elif st == "TreatedWW":
                # Report constant capacity — matches what the model uses (no retirement in NLP)
                cap = _STATIC_SOURCE_CAPS.get(s, 0)
            elif st == "Desalination" and sname in ("RO", "MSF", "MED"):
                cap = p["desal_caps"].get(sname, 0)
            else:
                cap = _STATIC_SOURCE_CAPS.get(s, 0)
            used = sum(value(model.production[u, s, y]) for u in water_users if s in expanded_us[u])
            rem_rows.append({"Source": s, "Capacity (BCM)": cap,
                             "Used (BCM)": used, "Remaining (BCM)": cap - used})

        # -- Domestic crop production tab --
        dom_crop_list = sorted(domestic_crops)
        dom_crop_tab  = pd.DataFrame({
            "Crop": dom_crop_list,
            "Production (tons)":       [value(model.crop_prod[c, "Domestic", y]) if (c,"Domestic",y) in model.crop_prod else 0 for c in dom_crop_list],
            "Cultivated Area (ha)":    [value(model.cultivated_area[c, y])        if (c, y) in model.cultivated_area else 0   for c in dom_crop_list],
            "Yield (tons/ha)":         [yield_per_ha.get(c, 0)                                                                  for c in dom_crop_list],
            "Cultivation Cost ($/ton)":[p["culcost"].get(c, 0)                                                                  for c in dom_crop_list],
            "Total Cultivation Cost ($)":[(value(model.crop_prod[c,"Domestic",y]) if (c,"Domestic",y) in model.crop_prod else 0)*p["culcost"].get(c,0) for c in dom_crop_list],
            "Calories Provided":       [(value(model.crop_prod[c,"Domestic",y]) if (c,"Domestic",y) in model.crop_prod else 0)*crop_calories.get(c,0) for c in dom_crop_list],
            "Protein Provided":        [(value(model.crop_prod[c,"Domestic",y]) if (c,"Domestic",y) in model.crop_prod else 0)*crop_protein.get(c,0)  for c in dom_crop_list],
        })

        # -- Imported crops tab --
        imp_crop_list = sorted(import_crops)
        imp_crop_tab  = pd.DataFrame({
            "Crop": imp_crop_list,
            "Imported Quantity (tons)":  [value(model.crop_prod_import[c,"Import",y]) if (c,"Import",y) in model.crop_prod_import else 0 for c in imp_crop_list],
            "Import Cost ($/ton)":       [p["impcost"].get(c, 0) for c in imp_crop_list],
            "Total Import Cost ($)":     [(value(model.crop_prod_import[c,"Import",y]) if (c,"Import",y) in model.crop_prod_import else 0)*p["impcost"].get(c,0) for c in imp_crop_list],
            "Calories Provided":         [(value(model.crop_prod_import[c,"Import",y]) if (c,"Import",y) in model.crop_prod_import else 0)*crop_calories.get(c,0) for c in imp_crop_list],
            "Protein Provided":          [(value(model.crop_prod_import[c,"Import",y]) if (c,"Import",y) in model.crop_prod_import else 0)*crop_protein.get(c,0)  for c in imp_crop_list],
        })

        # -- Combined crop --
        dom_d = {c: (value(model.crop_prod[c,"Domestic",y]) if (c,"Domestic",y) in model.crop_prod else 0) for c in result_crops}
        imp_d = {c: (value(model.crop_prod_import[c,"Import",y]) if (c,"Import",y) in model.crop_prod_import else 0) for c in result_crops}
        crop_df = pd.DataFrame({
            "Crop": result_crops,
            "Domestic Production (tons)": [dom_d[c] for c in result_crops],
            "Imported (tons)":            [imp_d[c] for c in result_crops],
            "Calories Provided": [(dom_d[c]+imp_d[c])*crop_calories.get(c,0) for c in result_crops],
            "Protein Provided":  [(dom_d[c]+imp_d[c])*crop_protein.get(c,0)  for c in result_crops],
        })

        # -- Energy results --
        energy_rows = []
        for u in water_users:
            for s in expanded_us[u]:
                ef    = energy_req_fac.get((u,s), 0)
                pv    = value(model.production[u,s,y]) if (u,s,y) in model.production else 0
                req   = pv * ef * 1e9 / (8_760 * 1000)
                alloc = sum(value(model.energy_allocation[u,s,pp,y])
                            for pp in power_plants if (u,s,pp,y) in model.energy_allocation)
                energy_rows.append({"User": u, "Source": s,
                                    "Production (BCM)": pv, "Energy Factor (kWh/m3)": ef,
                                    "Energy Required (MW)": req, "Energy Allocated (MW)": alloc})

        # -- Power plant results --
        pp_rows = []
        for pp in power_plants:
            if pp in _WATER_NEW_PLANTS:
                continue
            total_alloc = sum(
                value(model.energy_allocation[u,s,pp,y])
                for u in water_users for s in expanded_us[u]
                if (u,s,pp,y) in model.energy_allocation
            )
            if pp in _NEW_PLANT_INVESTMENTS:
                cap = sum(
                    value(model.num_new_units[pp, y_]) * _NEW_PLANT_INVESTMENTS[pp]["unit_size"]
                    for y_ in YEARS if y_ <= y and (y-y_) < PLANT_LIFETIME
                )
            else:
                cap = pp_cap_year.get(y, {}).get(pp, pp_cap.get(pp, 0))
            pp_rows.append({"Power Plant": pp,
                            "Total Energy Allocated (MW)": total_alloc, "Capacity (MW)": cap,
                            "CO2 Emissions (t)": total_alloc * pp_co2.get(pp, 0) * 8_760_000})

        # -- Investment this year --
        _unit_counts = {
            pl: max(0.0, value(model.num_new_units[pl, y])) if (pl, y) in model.num_new_units else 0.0
            for pl in _NEW_PLANT_INVESTMENTS
        }
        inv_rows = [{"Plant": pl,
                     "Number of Units":  _unit_counts[pl],
                     "Unit Cost ($)":    _NEW_PLANT_INVESTMENTS[pl]["cost"],
                     "Total Cost ($)":   _unit_counts[pl] * _NEW_PLANT_INVESTMENTS[pl]["cost"]}
                    for pl in _NEW_PLANT_INVESTMENTS]

        # -- Cumulative investments across all years up to y --
        cum_rows = []
        for y_ in [_y for _y in YEARS if _y <= y]:
            for pl in _NEW_PLANT_INVESTMENTS:
                uv = value(model.num_new_units[pl, y_]) if (pl, y_) in model.num_new_units else 0
                if uv >= 0.01:
                    age    = y - y_
                    status = "Retired" if age >= PLANT_LIFETIME else "Active"
                    cum_rows.append({"Plant": pl, "Vintage Year": y_,
                                     "Units Invested": uv,
                                     "Unit Size (BCM or MW)": _NEW_PLANT_INVESTMENTS[pl]["unit_size"],
                                     "Total Capacity": uv * _NEW_PLANT_INVESTMENTS[pl]["unit_size"],
                                     "Age (years)": age,
                                     f"Status in {y}": status})

        # -- Summary metrics --
        agr_water = sum(value(model.allocation["Agriculture",s,y]) for s in expanded_us["Agriculture"])
        gw_prod   = sum(value(model.production[u,s,y]) for u in water_users for s in expanded_us[u] if s.startswith("Groundwater"))
        gw_stk    = value(model.gw_stock[y])
        total_brine = sum(
            value(model.production[u,s,y]) * 1e9 * _BRINE_FACTORS.get(f"{s.split('_')[0]}_{s.split('_')[1]}", 0)
            for u in water_users for s in expanded_us[u]
            if len(s.split("_")) >= 2 and f"{s.split('_')[0]}_{s.split('_')[1]}" in _BRINE_FACTORS
        )
        total_co2 = sum(
            value(model.energy_allocation[u,s,pp,y]) * pp_co2.get(pp,0) * 8_760_000
            for u in water_users for s in expanded_us[u] for pp in power_plants
            if (u,s,pp,y) in model.energy_allocation
        )
        # Cost breakdown (post-solve, same formula as sequential model)
        # Groundwater priced from total system-wide extraction that year — see
        # the matching comment in annual_cost_pyomo's water_cost block.
        _total_gw_alloc_v = sum(
            value(model.allocation[u2, s2, y]) for u2 in water_users for s2 in expanded_us[u2]
            if s2.startswith("Groundwater")
        )
        _gw_a, _gw_b, _gw_c, _gw_p, _gw_vmax = _DESAL_COST_COEFFS["Groundwater"]["coeffs"]
        _gw_V = _total_gw_alloc_v  # already in BCM
        _gw_unit_cost_v = (_gw_a * (_gw_V + 1e-6) ** _gw_b
                           + _gw_c * (_gw_V / _gw_vmax) ** _gw_p / max(1e-6, 1 - _gw_V / _gw_vmax))

        water_cost_v = 0
        for u in water_users:
            for s in expanded_us[u]:
                parts = s.split("_")
                if len(parts) < 2:
                    continue
                if parts[0].lower() == "desalination":
                    stype_key = f"{parts[0]}_{parts[1]}"
                else:
                    stype_key = None
                Q = value(model.allocation[u, s, y]) * 1e9
                if parts[0] == "Groundwater":
                    water_cost_v += Q * _gw_unit_cost_v
                    continue
                if stype_key and stype_key in _DESAL_COST_COEFFS:
                    a, b = _DESAL_COST_COEFFS[stype_key]["coeffs"]
                    if _DESAL_COST_COEFFS[stype_key]["type"] == "log":
                        uc = a * math.log(math.sqrt(Q**2 + 1e-12)) + b
                        if stype_key in _DESAL_COST_FLOOR:
                            uc = max(uc, _DESAL_COST_FLOOR[stype_key])
                    else:
                        uc = a * math.exp(b * Q)
                    water_cost_v += Q * uc
                else:
                    base_cost = get_source_base_cost(s)
                    cost = base_cost * p["price_esc"]
                    if parts[0] == "SurfaceWater" and u == "Domestic":
                        cost *= 1.10
                    water_cost_v += Q * cost
        energy_cost_v     = sum(value(model.energy_allocation[u,s,pp,y]) * p["pp_costs"].get(pp,0) * 8_760_000
                                for u in water_users for s in expanded_us[u] for pp in power_plants
                                if (u,s,pp,y) in model.energy_allocation)
        carbon_cost_v     = CARBON_DAMAGE_USD_PER_T * sum(
                                value(model.energy_allocation[u,s,pp,y]) * pp_co2.get(pp,0) * 8_760_000
                                for u in water_users for s in expanded_us[u] for pp in power_plants
                                if (u,s,pp,y) in model.energy_allocation)
        investment_cost_v = sum(_unit_counts[pl] * _NEW_PLANT_INVESTMENTS[pl]["cost"] for pl in _NEW_PLANT_INVESTMENTS)
        dom_crop_cost_v   = sum((value(model.crop_prod[c,"Domestic",y]) if (c,"Domestic",y) in model.crop_prod else 0) * p["culcost"].get(c,0) for c in domestic_crops)
        imp_crop_cost_v   = sum((value(model.crop_prod_import[c,"Import",y]) if (c,"Import",y) in model.crop_prod_import else 0) * p["impcost"].get(c,0) for c in import_crops)

        disc_y = (1.0 / (1.0 + DISCOUNT_RATE)) ** (y - YEARS[0])

        summary_data = [
            {"Metric": "Year",                      "Value": y},
            {"Metric": "Model Type",                "Value": "Intertemporal"},
            {"Metric": "Energy Scenario",           "Value": energy_scenario},
            {"Metric": "GW Scenario",               "Value": gw_label},
            {"Metric": "Population",                "Value": p["population"]},
            {"Metric": "Renewable Share",           "Value": p["ren_share"]},
            {"Metric": "Domestic Demand (BCM)",     "Value": p["domestic_demand"]},
            {"Metric": "Industry Demand (BCM)",     "Value": p["industry_demand"]},
            {"Metric": "Agr Water Use (BCM)",       "Value": agr_water},
            {"Metric": "GW Production (BCM)",       "Value": gw_prod},
            {"Metric": "GW Remaining Stock (BCM)",  "Value": gw_stk},
            {"Metric": "RO Fleet Capacity (BCM)",   "Value": p["desal_caps"]["RO"]},
            {"Metric": "MSF Fleet Capacity (BCM)",  "Value": p["desal_caps"]["MSF"]},
            {"Metric": "MED Fleet Capacity (BCM)",  "Value": p["desal_caps"]["MED"]},
            {"Metric": "TWW Fleet Capacity (BCM)",  "Value": _TWW_EXISTING_CAP},
            {"Metric": "Dam Fleet Capacity (BCM)",  "Value": _DAM_EXISTING_CAP},
            {"Metric": "Water Cost ($)",            "Value": water_cost_v},
            {"Metric": "Domestic Crop Cost ($)",    "Value": dom_crop_cost_v},
            {"Metric": "Imported Crop Cost ($)",    "Value": imp_crop_cost_v},
            {"Metric": "Energy Cost ($)",           "Value": energy_cost_v},
            {"Metric": "Carbon Damage Cost ($)",    "Value": carbon_cost_v},
            {"Metric": "Investment Cost ($)",       "Value": investment_cost_v},
            {"Metric": "Total Irrigated Area (ha)", "Value": value(model.total_irrigated_area[y])},
            {"Metric": "Total Brine Discharge (m3)","Value": total_brine},
            {"Metric": "Total CO2 Emissions (t)",   "Value": total_co2},
            {"Metric": "Discount Factor",           "Value": disc_y},
            {"Metric": "New RO Units",              "Value": _unit_counts["Desalination_NewRO"]},
            {"Metric": "New RO Capacity (BCM)",     "Value": _unit_counts["Desalination_NewRO"] * _NEW_PLANT_INVESTMENTS["Desalination_NewRO"]["unit_size"]},
            {"Metric": "New Dam Units",             "Value": _unit_counts["SurfaceWater_NewSaudiDam"]},
            {"Metric": "New TWW Units",             "Value": _unit_counts["TreatedWW_NewSaudiTWW"]},
            {"Metric": "NPV Objective (all yrs)",   "Value": value(model.obj)},
            {"Metric": "Solver Status",             "Value": str(tc)},
        ]

        with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
            pd.DataFrame(summary_data).to_excel(writer,         sheet_name="Summary",                    index=False)
            pd.DataFrame(alloc_rows).to_excel(writer,           sheet_name="Water Allocation",           index=False)
            pd.DataFrame(prod_rows).to_excel(writer,            sheet_name="Water Production",           index=False)
            pd.DataFrame(rem_rows).to_excel(writer,             sheet_name="Remaining Capacity",         index=False)
            dom_crop_tab.to_excel(writer,                        sheet_name="Domestic Crop Production",   index=False)
            imp_crop_tab.to_excel(writer,                        sheet_name="Imported Crops",             index=False)
            crop_df.to_excel(writer,                             sheet_name="Crop Production (Combined)", index=False)
            pd.DataFrame(energy_rows).to_excel(writer,          sheet_name="Energy Results",             index=False)
            pd.DataFrame(pp_rows).to_excel(writer,              sheet_name="Power Plant Results",        index=False)
            pd.DataFrame(inv_rows).to_excel(writer,             sheet_name="Investments",                index=False)
            if cum_rows:
                pd.DataFrame(cum_rows).to_excel(writer,         sheet_name="Cumulative Investments",     index=False)
            crop_water_reqs.to_excel(writer,                    sheet_name="Crop Water Req",             index=False)
            phase_product.to_excel(writer,                      sheet_name="Phase Product",              index=False)

        print(f"    Saved {y}_{energy_scenario}_{gw_label}.xlsx  "
              f"(GW={gw_prod:.2f} BCM, stock={gw_stk:.0f} BCM, area={value(model.total_irrigated_area[y]):.0f} ha)")

    print(f"\n  === SCENARIO COMPLETE: {scenario_tag} ===")
    print(f"  NPV objective: {value(model.obj):.4e}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR_IT, exist_ok=True)

    if TEST_SINGLE:
        scenarios_to_run = GW_SCENARIOS[:1]
        energy_to_run    = ENERGY_SCENARIOS[:1]
        scope_label = "TEST_SINGLE (STANDARD / No_GW_Limit)"
    else:
        scenarios_to_run = GW_SCENARIOS
        energy_to_run    = ENERGY_SCENARIOS
        scope_label = "FULL RUN (STANDARD x all 3 GW)"

    # DIAG: restrict to a single GW scenario by index (0=No,1=10BCM,2=5BCM)
    _diag_only = os.environ.get("DIAG_ONLY")
    if _diag_only is not None:
        scenarios_to_run = [GW_SCENARIOS[int(_diag_only)]]

    total = len(energy_to_run) * len(scenarios_to_run)
    print(f"\nIntertemporal model — {len(YEARS)} years × {total} scenarios")
    print(f"Discount rate: {DISCOUNT_RATE*100:.0f}%   |   GW stock: {GW_INITIAL_STOCK_BCM} BCM")
    print(f"Scope: {scope_label}")
    print(f"Results -> {RESULTS_DIR_IT}\n")

    run_count = 0
    for es in energy_to_run:
        for (s1, s2) in scenarios_to_run:
            run_count += 1
            print(f"\n[{run_count}/{total}]", end=" ")
            run_intertemporal(es, s1, s2)

    print(f"\nAll done. Results in: {RESULTS_DIR_IT}")
