"""
Multi-year, multi-scenario runner for the Saudi Water-Food-Energy-Nexus model.
Covers years 2023-2060 across all combinations of energy and groundwater scenarios.

Vintage/lifetime curve: new plant investments persist as inherited capacity in future
years. Each cohort (vintage) operates at full capacity for PLANT_LIFETIME years, then
retires. investment_history is rebuilt at startup from existing result files and updated
in-memory after each solve, so SKIP_EXISTING mode remains correct.

Each year run is independent except for inherited capacity passed in. Desalination
capacity is computed dynamically from the SWCC fleet sheet (plant-by-plant retirement
via Year Out). Renewable share and water demand are interpolated from known anchor-year
values. Population comes from Population_clean.xlsx.
"""

import math
import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from pyomo.environ import (
    ConcreteModel, sqrt, Var, Objective, Constraint, NonNegativeReals,
    SolverFactory, minimize, value, log, exp, Binary, Integers, Suffix
)
from pyomo.opt import TerminationCondition


# =============================================================================
# CONFIGURATION
# =============================================================================

YEARS_TO_RUN     = range(2023, 2061)
ENERGY_SCENARIOS = ["STANDARD"]
GW_SCENARIOS     = [
    (False, False),   # No groundwater limit
    (True,  False),   # 10 BCM limit
    (False, True),    # 5 BCM limit
]
SKIP_EXISTING = True   # Set False to re-run and overwrite existing output files

# Quick start: the full batch below is 38 years x 3 energy x 3 GW scenarios =
# 342 individual solves (hours). Set TEST_SINGLE = True (or env var
# TEST_SINGLE=1) to run only STANDARD / No_GW_Limit across all years -- a
# much faster way to confirm the pipeline works before committing to the
# full batch. This also produces exactly the 2023_STANDARD_No_GW_Limit.xlsx
# baseline that run_intertemporal_model.py needs regardless of which other
# scenarios you eventually run.
TEST_SINGLE = os.environ.get("TEST_SINGLE", "0") == "1"

PLANT_LIFETIME = 40  # years; flat curve — each vintage cohort runs at full capacity until retirement

# Estimated initial groundwater stock (BCM). Saudi Arabia's major aquifer systems
# (Saq, Wajid, Minjur) hold an estimated 2,500 BCM of non-renewable groundwater.
# Adjust this figure based on the most recent hydrological survey for your study.
GW_INITIAL_STOCK_BCM = 2500


# =============================================================================
# PATH CONSTANTS
# =============================================================================

# All paths are relative to this repository -- clone it anywhere and it runs.
# RESULTS_DIR must match run_intertemporal_model.py's SEQ_RESULTS_DIR: this
# script's output is that script's 2023-baseline + warm-start input.
MODEL_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(MODEL_DIR)
DATA_DIR    = os.path.join(REPO_ROOT, "data")
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(REPO_ROOT, "results", "calibration"))

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
# PRE-COMPUTED LOOKUPS  (loaded once at startup)
# =============================================================================

print("Loading lookup tables...")

# 1. Population (file in thousands; model needs persons)
_pop_df     = pd.read_excel(POP_FILE)
_pop_lookup = dict(zip(_pop_df["Year"].astype(int), _pop_df["Population"] * 1000))

def get_population(year):
    year = int(year)
    if year in _pop_lookup:
        return _pop_lookup[year]
    nearest = min(_pop_lookup, key=lambda y: abs(y - year))
    return _pop_lookup[nearest]


# 2. Domestic and industry water demand — piecewise linear interpolation.
#    Anchor values (BCM/yr) loaded from Water_Demand_Anchors.xlsx.
_demand_df = pd.read_excel(os.path.join(DATA_DIR, "Water_Demand_Anchors.xlsx")).sort_values("Year")
_dom_interp = interp1d(_demand_df["Year"], _demand_df["Domestic_BCM"],
                       kind="linear", fill_value="extrapolate")
_ind_interp = interp1d(_demand_df["Year"], _demand_df["Industry_BCM"],
                       kind="linear", fill_value="extrapolate")

def get_demand(year):
    return float(_dom_interp(year)), float(_ind_interp(year))


# 3. Renewable share — piecewise linear interpolation per scenario.
#    Computed from Wind+Solar capacity / Total capacity in each year's energy files.
_RENEWABLE_ANCHORS = {
    "STANDARD": {2023: 0.012, 2030: 0.225, 2040: 0.236, 2050: 0.230, 2060: 0.280},
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
    """Return {tech: BCM/yr} for each technology active in the given year."""
    year = int(year)
    if year not in _swcc.index:
        year = min(_swcc.index, key=lambda y: abs(y - year))
    row = _swcc.loc[year]
    return {
        tech: float(row[f"{tech}_Capacity_BCM"])
        for tech in ["RO", "MSF", "MED"]
    }


# 5. Year-to-data-file mapping (floor to nearest available data year)
_DATA_YEARS = [2023, 2030, 2040, 2050, 2060]

def map_data_year(year):
    floors = [y for y in _DATA_YEARS if y <= int(year)]
    return max(floors) if floors else _DATA_YEARS[0]


# 6. Energy base data — loaded once per scenario from the earliest available year.
#    Using a fixed energy file per scenario prevents the discrete jump in power-plant
#    capacities/costs that occurs when the file switches from the 2023 to the 2030
#    decade file at year 2030 (the 2030 file collapses 6 fuel types into 2 and has
#    29× more renewable capacity, distorting new-RO investment in years near 2029-2030).
#    Scenario differentiation is preserved through:
#      • the year-by-year renewable share target (already smoothly interpolated)
#      • the 2023 Investment-file investment parameters (_NEW_POWER_*)
#      • 2% per-year cost escalation on existing plant operational costs
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


# 7. Crop base prices from 2023 file (loaded once; applied with 2% escalation
#    inside run_model so prices grow smoothly without decade-boundary jumps).
_base_dom_df = pd.read_excel(f"{DATA_DIR}/2023/Crops_2023.xlsx", sheet_name="Domestic")
_base_dom_df.columns = _base_dom_df.columns.str.lower()
_base_imp_df = pd.read_excel(f"{DATA_DIR}/2023/Crops_2023.xlsx", sheet_name="Import")
_BASE_CULCOST = dict(zip(_base_dom_df["crop"], _base_dom_df["culcost ($/ton)"].astype(float)))
_BASE_IMPCOST = dict(zip(_base_imp_df["Crop"], _base_imp_df["Import Cost ($/ton)"].astype(float)))
del _base_dom_df, _base_imp_df

# 8. Calibration dual prices and logit coefficients (fixed to 2023 for all years).
_dp_water    = pd.read_excel(CALIB_FILE, sheet_name="Water_2023")
_dp_domestic = pd.read_excel(CALIB_FILE, sheet_name="Dom_2023")
_dp_import   = pd.read_excel(CALIB_FILE, sheet_name="Imp_2023")
_DUAL_PRICES_WATER    = dict(zip(_dp_water["Source"],  _dp_water["Dual"]))
_DUAL_PRICES_DOMESTIC = dict(zip(_dp_domestic["Crop"], _dp_domestic["Dual"]))
_DUAL_PRICES_IMPORT   = dict(zip(_dp_import["Crop"],   _dp_import["Dual"]))
del _dp_water, _dp_domestic, _dp_import

_logit_df = pd.read_excel(LOGIT_FILE, sheet_name="Dom_Production")
_logit_df.columns = _logit_df.columns.str.lower()
_logit_imp_df = pd.read_excel(LOGIT_FILE, sheet_name="Import_Production")
_logit_imp_df.columns = _logit_imp_df.columns.str.lower()
_LOGIT_COEF_1        = dict(zip(_logit_df["crop"],     _logit_df["coef_1"]))
_LOGIT_COEF_2        = dict(zip(_logit_df["crop"],     _logit_df["coef_2"]))
_LOGIT_COEF_1_IMPORT = dict(zip(_logit_imp_df["crop"], _logit_imp_df["coef_1"]))
_LOGIT_COEF_2_IMPORT = dict(zip(_logit_imp_df["crop"], _logit_imp_df["coef_2"]))
del _logit_df, _logit_imp_df

# 9. Agronomic parameters, user efficiencies, and investment costs — all fixed
#    to 2023 calibration data.  Using decade-specific files here causes discrete
#    jumps at the 2030→2040 and 2040→2050 data-year transitions that appear as
#    spikes in every result graph (except scenarios where water is unconstrained).
_crop_file_2023    = f"{DATA_DIR}/2023/Crops_2023.xlsx"
_climate_file_2023 = f"{DATA_DIR}/2023/Climate_2023.xlsx"
_wu_file_2023      = f"{DATA_DIR}/2023/WaterUsers_2023.xlsx"
_inv_file_2023     = f"{DATA_DIR}/2023/Investment.xlsx"

# Crop data
_dom_yield_2023 = pd.read_excel(_crop_file_2023, sheet_name="Domestic")
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

# Crop water requirements (from 2023 climate + crop coefficient files)
_crop_coef_2023   = pd.read_excel(_crop_file_2023, sheet_name="kc")
_phase_month_raw  = pd.read_excel(_crop_file_2023, sheet_name="PhaseMonth")
_et0_2023         = pd.read_excel(_crop_file_2023,    sheet_name="ET0")
_rain_2023        = pd.read_excel(_climate_file_2023, sheet_name="Rain")
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

# User efficiencies
_agr_2023 = pd.read_excel(_wu_file_2023, sheet_name="Agriculture"); _agr_2023.columns = _agr_2023.columns.str.lower()
_dom_2023 = pd.read_excel(_wu_file_2023, sheet_name="Domestic");    _dom_2023.columns = _dom_2023.columns.str.lower()
_ind_2023 = pd.read_excel(_wu_file_2023, sheet_name="Industrial");  _ind_2023.columns = _ind_2023.columns.str.lower()
_IRR_EFF_2023 = float(_agr_2023["irreff"].iloc[0])
_DOM_EFF_2023 = float(_dom_2023["domeff"].iloc[0])
_IND_EFF_2023 = float(_ind_2023["indeff"].iloc[0])
del _agr_2023, _dom_2023, _ind_2023

# Investment costs and parameters
_water_inv_2023 = pd.read_excel(_inv_file_2023, sheet_name="WaterSources")
_power_inv_2023 = pd.read_excel(_inv_file_2023, sheet_name="PowerPlant")
_power_inv_2023 = _power_inv_2023.rename(columns={"OppCost ($/kwh)": "Cost ($/kwh)", "Capacity": "Capacity (MW)"})
_NEW_WATER_COST      = dict(zip(_water_inv_2023["Name"], _water_inv_2023["Opex Cost ($/m3)"]))
_NEW_WATER_EFF       = dict(zip(_water_inv_2023["Name"], _water_inv_2023["Eff"]))
_NEW_WATER_ENERGY_D  = dict(zip(_water_inv_2023["Name"], _water_inv_2023["Energy (kwh/m3)"]))
# Full upfront CapEx per unit in dollars (Capex column is M$, convert to $)
_NEW_WATER_CAPEX     = {row["Name"]: float(row["Capex (Million USD)"]) * 1e6
                        for _, row in _water_inv_2023.iterrows()}
# Unit capacity in BCM (Capacity column in Excel is BCM)
_NEW_WATER_UNIT_SIZE = dict(zip(_water_inv_2023["Name"], _water_inv_2023["Capacity (BCM)"]))
_NEW_POWER_COST     = dict(zip(_power_inv_2023["Name"], _power_inv_2023["Cost ($/kwh)"]))
_NEW_POWER_EFF      = dict(zip(_power_inv_2023["Name"], _power_inv_2023["Eff"]))
_NEW_POWER_CO2      = dict(zip(_power_inv_2023["Name"], _power_inv_2023["CO2 (t/kwh)"]))
_NEW_POWER_CAP      = dict(zip(_power_inv_2023["Name"], _power_inv_2023["Capacity (MW)"]))
del _water_inv_2023, _power_inv_2023

# Module-level plant investments reference (used by vintage helpers before run_model builds its local copy)
_NEW_PLANT_INVESTMENTS_BASE = {
    "SurfaceWater_NewSaudiDam": {"unit_size": _NEW_WATER_UNIT_SIZE.get("NewSaudiDam", 0.10), "cost": _NEW_WATER_CAPEX.get("NewSaudiDam", 0.5e9)},
    "TreatedWW_NewSaudiTWW":    {"unit_size": _NEW_WATER_UNIT_SIZE.get("NewSaudiTWW", 0.20), "cost": _NEW_WATER_CAPEX.get("NewSaudiTWW", 0.3e9)},
    "Desalination_NewRO":       {"unit_size": _NEW_WATER_UNIT_SIZE.get("NewRO",       0.75), "cost": _NEW_WATER_CAPEX.get("NewRO",       2.0e9)},
    "NewNG":     {"unit_size": 500,  "cost": 0.4e9},
    "NewDiesel": {"unit_size": 500,  "cost": 0.5e9},
    "NewCrude":  {"unit_size": 500,  "cost": 0.5e9},
    "NewHFO":    {"unit_size": 1000, "cost": 1.0e9},
    "NewSolar":  {"unit_size": 1000, "cost": 0.5e9},
    "NewWind":   {"unit_size": 1000, "cost": 1.5e9},
}

# Water source capacities, energy factors, and efficiencies fixed to 2023.
# If the 2040/2050 files have different groundwater energy factors or surface-
# water capacities those cause a jump that only shows in water-constrained
# scenarios (the no-GW-limit case is immune because water isn't binding there).
_WATERSOURCES_2023_RAW = pd.read_excel(f"{DATA_DIR}/2023/WaterSources_2023.xlsx", sheet_name=None)

print("Lookup tables ready.")


# =============================================================================
# VINTAGE / LIFETIME TRACKING
# =============================================================================

def load_investment_history():
    """
    Scan RESULTS_DIR for solved result files and reconstruct the full investment
    history: {(energy_scenario, gw_label): {vintage_year: {plant: units}}}.
    Called once at startup so SKIP_EXISTING mode stays consistent.
    """
    history = {}
    if not os.path.exists(RESULTS_DIR):
        return history
    for fname in os.listdir(RESULTS_DIR):
        if not fname.endswith(".xlsx") or not fname[0].isdigit():
            continue
        stem = fname[:-5]  # strip .xlsx
        parts = stem.split("_", 1)
        if len(parts) < 2:
            continue
        try:
            vintage_year = int(parts[0])
        except ValueError:
            continue
        remainder = parts[1]
        energy_scenario = None
        for sc in ENERGY_SCENARIOS:
            if remainder.startswith(sc + "_"):
                energy_scenario = sc
                gw_label = remainder[len(sc) + 1:]
                break
        if energy_scenario is None:
            continue
        scenario_key = (energy_scenario, gw_label)
        fpath = os.path.join(RESULTS_DIR, fname)
        try:
            xl = pd.ExcelFile(fpath)
            if "Investments" not in xl.sheet_names:
                continue
            inv_df = pd.read_excel(fpath, sheet_name="Investments")
            year_entry = {}
            for _, row in inv_df.iterrows():
                plant = row["Plant"]
                units = float(row["Number of Units"])
                if units >= 0.01:
                    year_entry[plant] = units
            if year_entry:
                history.setdefault(scenario_key, {})[vintage_year] = year_entry
        except Exception as e:
            print(f"  [WARNING] Could not read investments from {fname}: {e}")
    return history


def compute_inherited_units(year, scenario_key, investment_history):
    """
    Return {plant: float_units} for all plants still within PLANT_LIFETIME in year.
    Only counts vintages strictly before `year` (current-year decisions are separate).
    """
    inherited = {}
    for plant in _NEW_PLANT_INVESTMENTS_BASE:
        total = 0.0
        for vintage_year, plants in investment_history.get(scenario_key, {}).items():
            if vintage_year < year and (year - vintage_year) < PLANT_LIFETIME:
                total += plants.get(plant, 0.0)
        inherited[plant] = total
    return inherited


# =============================================================================
# MODEL FUNCTION
# =============================================================================

def run_model(year, energy_scenario, scenario1, scenario2, investment_history=None, warm_start_state=None, hard_mode=False):
    """
    Build, solve, and export one scenario run.

    year               : int  -- simulation year (2023-2060)
    energy_scenario    : str  -- "STANDARD"
    scenario1          : bool -- True = 10 BCM groundwater limit
    scenario2          : bool -- True = 5  BCM groundwater limit
    investment_history : dict -- shared {(scenario, gw_label): {year: {plant: units}}}
                                 mutated in-place after each successful solve
    warm_start_state   : dict -- shared {scenario_key: {(var_name, idx): value}}
                                 previous year's solution values used as IPOPT starting point;
                                 mutated in-place after each successful solve
    hard_mode          : bool -- if True, append 2 extra ultra-aggressive attempts after the
                                 standard 7; used by run_until_clean.py for persistently failing years
    """
    if investment_history is None:
        investment_history = {}
    if warm_start_state is None:
        warm_start_state = {}

    year       = int(year)
    data_year  = map_data_year(year)
    ren_share  = get_renewable_share(year, energy_scenario)
    population = get_population(year)
    domestic_total_demand, industry_total_demand = get_demand(year)
    desal_caps = get_desal_capacity_bcm(year)

    gw_label     = "10_BCM_GW_Limit" if scenario1 else "5_BCM_GW_Limit" if scenario2 else "No_GW_Limit"
    scenario_key = (energy_scenario, gw_label)
    inherited_units = compute_inherited_units(year, scenario_key, investment_history)
    out_file  = os.path.join(RESULTS_DIR, f"{year}_{energy_scenario}_{gw_label}.xlsx")

    if SKIP_EXISTING and os.path.exists(out_file):
        try:
            _xl = pd.ExcelFile(out_file)
            if "Summary" in _xl.sheet_names:
                _df = pd.read_excel(out_file, sheet_name="Summary")
                _mask = _df["Metric"] == "Solver Status"
                if _mask.any():
                    _status = str(_df.loc[_mask, "Value"].values[0]).lower()
                    if _status in ["optimal", "feasible", "maxtimelimit"]:
                        print(f"  [SKIP] {year} | {energy_scenario} | {gw_label}")
                        return
        except Exception:
            pass  # unreadable file — rerun it
        print(f"  [RERUN] {year} | {energy_scenario} | {gw_label} (previous run failed)")

    print(f"\n{'='*72}")
    print(f"  Year={year}  Scenario={energy_scenario}  GW={gw_label}")
    print(f"  data_year={data_year}  pop={population:,.0f}")
    print(f"  domestic={domestic_total_demand:.3f} BCM  industry={industry_total_demand:.3f} BCM")
    print(f"  renewable_share={ren_share:.3f}")
    print(f"  Desal fleet (BCM): RO={desal_caps['RO']:.3f}  MSF={desal_caps['MSF']:.3f}  MED={desal_caps['MED']:.3f}")
    _inh_ro  = inherited_units.get("Desalination_NewRO", 0)
    _inh_dam = inherited_units.get("SurfaceWater_NewSaudiDam", 0)
    _inh_tww = inherited_units.get("TreatedWW_NewSaudiTWW", 0)
    print(f"  Inherited new plants: RO={_inh_ro:.0f} units  Dam={_inh_dam:.0f} units  TWW={_inh_tww:.0f} units")
    print(f"{'='*72}")

    # -------------------------------------------------------------------------
    # DATA LOADING
    # All behavioural parameters (crop, climate, efficiencies, investment costs)
    # come from 2023 module-level lookups so they never jump at decade boundaries.
    # Only water-source capacities and energy files are decade-specific (they
    # reflect real physical infrastructure changes over time).
    # -------------------------------------------------------------------------
    domestic_crops      = _DOMESTIC_CROPS_2023
    import_crops_data   = _IMP_CROPS_DF_2023
    crop_water_req_dict = _CROP_WATER_REQ_DICT
    yield_per_ha        = _YIELD_PER_HA_2023
    crop_categories     = _CROP_CATEGORIES
    crop_calories       = _CROP_CALORIES
    crop_protein        = _CROP_PROTEIN
    new_water_cost      = _NEW_WATER_COST
    new_water_eff       = _NEW_WATER_EFF
    new_water_energy_d  = _NEW_WATER_ENERGY_D
    new_power_cost      = _NEW_POWER_COST
    new_power_eff       = _NEW_POWER_EFF
    new_power_co2       = _NEW_POWER_CO2
    new_power_cap       = _NEW_POWER_CAP

    # Use pre-loaded globals (calibration prices, logit coefficients — all fixed to 2023)
    dual_prices_dict     = _DUAL_PRICES_WATER
    dual_prices_domestic = _DUAL_PRICES_DOMESTIC
    dual_prices_import   = _DUAL_PRICES_IMPORT
    logit_coef_1         = _LOGIT_COEF_1
    logit_coef_2         = _LOGIT_COEF_2
    logit_coef_1_import  = _LOGIT_COEF_1_IMPORT
    logit_coef_2_import  = _LOGIT_COEF_2_IMPORT

    # Crop prices: use pre-loaded 2023 baseline (see _BASE_CULCOST / _BASE_IMPCOST
    # at module level). Decade data files have independently calibrated prices
    # that don't follow a 2% path, causing composition jumps at transition years.
    crop_culcost = {crop: _BASE_CULCOST[crop] for crop in domestic_crops if crop in _BASE_CULCOST}
    crop_impcost = {crop: _BASE_IMPCOST[crop] for crop in import_crops_data["Crop"] if crop in _BASE_IMPCOST}

    # Escalate food prices by 2% per year from the 2023 base, regardless of
    # which data file is active. Using year-data_year would reset prices at
    # each anchor year (2030, 2040…) and could cause a downward jump if the
    # data file prices weren't calibrated with 2% compounding already built in.
    price_escalation = 1.02 ** (year - 2023)
    crop_culcost = {c: v * price_escalation for c, v in crop_culcost.items()}
    crop_impcost = {c: v * price_escalation for c, v in crop_impcost.items()}

    costs_sc = {c: crop_culcost[c] / 1000 for c in domestic_crops}
    exp_t    = {c: logit_coef_1[c] * np.exp(np.clip(logit_coef_2[c] * costs_sc[c], -100, 100)) for c in domestic_crops}
    exp_s    = sum(exp_t.values())
    min_crop_shares_dict = {c: exp_t[c] / exp_s for c in domestic_crops}

    costs_sc_imp = {c: crop_impcost[c] / 1000 for c in import_crops_data["Crop"]}
    exp_t_imp    = {c: logit_coef_1_import[c] * np.exp(np.clip(logit_coef_2_import[c] * costs_sc_imp[c], -100, 100)) for c in import_crops_data["Crop"]}
    exp_s_imp    = sum(exp_t_imp.values())
    min_crop_shares_import_dict = {c: exp_t_imp[c] / exp_s_imp for c in import_crops_data["Crop"]}

    # Water sources: use 2023 baseline capacities/energy-factors (fixed, no decade
    # jumps), then override Desalination capacity with the live SWCC fleet so
    # plant retirements are still reflected year-by-year.
    watersources_data = {k: df.copy() for k, df in _WATERSOURCES_2023_RAW.items()}
    desal_df = watersources_data["Desalination"].copy()
    desal_df["Capacity (m3)"] = desal_df["Capacity (m3)"].astype(float)
    desal_df["DesalEff (%)"]  = desal_df["DesalEff (%)"].astype(float)
    for idx, row in desal_df.iterrows():
        tech = row["Name"]
        if tech in desal_caps:
            desal_df.at[idx, "Capacity (m3)"] = desal_caps[tech] * 1e9
            desal_df.at[idx, "DesalEff (%)"]  = 1.0
    watersources_data["Desalination"] = desal_df

    energy_factors = {}
    for sheet_name, df in watersources_data.items():
        df.columns = df.columns.str.lower()
        energy_col = next((c for c in df.columns if "energy (kwh/m3)" in c.lower()), None)
        if energy_col is None:
            raise ValueError(f"No energy factor column in WaterSources sheet: {sheet_name}")
        for _, row in df.iterrows():
            energy_factors.setdefault(sheet_name, {})[row["name"]] = float(row[energy_col])

    water_supply = {}
    for source_type, df in watersources_data.items():
        df.columns = df.columns.str.lower()
        water_supply[source_type] = []
        for _, row in df.iterrows():
            entry = {"name": row["name"], "capacity (BCM)": row["capacity (m3)"] / 1e9}
            if source_type in ["TreatedWW", "SurfaceWater"]:
                entry["cost"] = row["cost ($/m3)"]
            water_supply[source_type].append(entry)

    # Apply 2% annual cost escalation to surface water and treated WW (same
    # basis as food price escalation — keeps all cost signals consistent).
    for _st in ["TreatedWW", "SurfaceWater"]:
        for _item in water_supply.get(_st, []):
            if "cost" in _item:
                _item["cost"] *= price_escalation

    source_efficiencies = {}
    eff_col_map = {"Desalination": "desaleff (%)", "SurfaceWater": "dameff (%)", "TreatedWW": "twweff (%)"}
    for source_type, df in watersources_data.items():
        df.columns = df.columns.str.lower()
        eff_col = eff_col_map.get(source_type)
        if eff_col and eff_col in df.columns:
            for _, row in df.iterrows():
                source_efficiencies[f"{source_type}_{row['name']}"] = float(row[eff_col])

    # Crop water requirements and nutritional lookup already precomputed at module level.
    crop_water_reqs = _CROP_WATER_REQS_2023   # kept for export to "Crop Water Req" sheet
    phase_product   = _PHASE_PRODUCT_2023      # kept for export to "Phase Product" sheet
    food_crops = [c for c, cat in crop_categories.items() if isinstance(cat, str) and cat.lower() == "food"]
    feed_crops = [c for c, cat in crop_categories.items() if isinstance(cat, str) and cat.lower() == "feed"]
    total_calorie_demand = population * 1970 * 365
    total_protein_demand = population * 107  * 365 / 1000

    power_plants = {k: dict(v) for k, v in _POWER_PLANTS_BASE[energy_scenario].items()}
    for name in new_power_cost:
        if name in power_plants:
            power_plants[name]["Cost ($/kWh)"] = new_power_cost[name]
            power_plants[name]["Efficiency"]   = new_power_eff.get(name, power_plants[name]["Efficiency"])
            power_plants[name]["CO2 (t/kwh)"] = new_power_co2.get(name, power_plants[name]["CO2 (t/kwh)"])
            if name in new_power_cap:
                power_plants[name]["Capacity (MW)"] = new_power_cap[name]
        else:
            power_plants[name] = {
                "Cost ($/kWh)":  new_power_cost[name],
                "Efficiency":    new_power_eff.get(name, 1.0),
                "CO2 (t/kwh)":  new_power_co2.get(name, 0),
                "Capacity (MW)": new_power_cap.get(name, 0),
            }

    # Apply 2% annual cost escalation to all power plants (same basis as food
    # and water cost escalation). Capacities and CO2 factors are not escalated.
    for _p in power_plants:
        power_plants[_p]["Cost ($/kWh)"] *= price_escalation

    # New power plants are excluded from power_plant_capacity_constraint below;
    # new_power_plant_capacity_constraint (with inherited + current units) is their
    # sole capacity limit, avoiding any fixed-capacity ceiling that would prevent
    # multiple vintage cohorts from being usable simultaneously.
    power_plants_new = ["NewNG", "NewDiesel", "NewCrude", "NewHFO", "NewSolar", "NewWind"]

    water_users = {
        "Agriculture": 10,
        "Industry":    float(industry_total_demand),
        "Domestic":    float(domestic_total_demand),
    }
    user_efficiencies = {
        "Agriculture": _IRR_EFF_2023,
        "Domestic":    _DOM_EFF_2023,
        "Industry":    _IND_EFF_2023,
    }
    user_sources = {
        "Agriculture": ["Groundwater", "SurfaceWater", "Desalination", "TreatedWW"],
        "Industry":    ["Groundwater", "Desalination", "TreatedWW"],
        "Domestic":    ["Groundwater", "Desalination"],
    }
    new_plants = ["SurfaceWater_NewSaudiDam", "TreatedWW_NewSaudiTWW", "Desalination_NewRO"]

    expanded_user_sources = {}
    for user, types in user_sources.items():
        expanded_user_sources[user] = []
        for source_type in types:
            if source_type in water_supply:
                expanded_user_sources[user].extend(
                    [f"{source_type}_{s['name']}" for s in water_supply[source_type]]
                )
        for np_ in new_plants:
            if np_ not in expanded_user_sources[user]:
                expanded_user_sources[user].append(np_)
    expanded_user_sources["Domestic"] = [
        s for s in expanded_user_sources["Domestic"]
        if not (s.startswith("TreatedWW") or s.startswith("SurfaceWater"))
    ]
    expanded_user_sources["Industry"] = [
        s for s in expanded_user_sources["Industry"] if not s.startswith("SurfaceWater")
    ]

    # Only RO allowed as new desalination investment (no new MSF or MED).
    # unit_size from Investment.xlsx Capacity column (BCM); cost = full upfront CapEx (Capex column, M$).
    new_plant_investments = {
        "SurfaceWater_NewSaudiDam": {"unit_size": _NEW_WATER_UNIT_SIZE.get("NewSaudiDam", 0.10), "cost": _NEW_WATER_CAPEX.get("NewSaudiDam", 0.5e9)},
        "TreatedWW_NewSaudiTWW":    {"unit_size": _NEW_WATER_UNIT_SIZE.get("NewSaudiTWW", 0.20), "cost": _NEW_WATER_CAPEX.get("NewSaudiTWW", 0.3e9)},
        "Desalination_NewRO":       {"unit_size": _NEW_WATER_UNIT_SIZE.get("NewRO",       0.75), "cost": _NEW_WATER_CAPEX.get("NewRO",       2.0e9)},
        "NewNG":                    {"unit_size": 500,  "cost": 0.4e9},
        "NewDiesel":                {"unit_size": 500,  "cost": 0.5e9},
        "NewCrude":                 {"unit_size": 500,  "cost": 0.5e9},
        "NewHFO":                   {"unit_size": 1000, "cost": 1.0e9},
        "NewSolar":                 {"unit_size": 1000, "cost": 0.5e9},
        "NewWind":                  {"unit_size": 1000, "cost": 1.5e9},
    }

    for name in new_water_cost:
        source_efficiencies[name] = new_water_eff.get(name, 1.0)

    user_energy = {}
    for user, types in user_sources.items():
        user_energy[user] = []
        for source_type in types:
            if source_type in water_supply:
                for source in water_supply[source_type]:
                    sname = f"{source_type}_{source['name']}"
                    user_energy[user].append({
                        "source_name":  sname,
                        "energy_factor": energy_factors[source_type].get(source["name"], None),
                    })
    for name in new_water_cost:
        for user in water_users:
            user_energy.setdefault(user, []).append(
                {"source_name": name, "energy_factor": new_water_energy_d.get(name, 0)}
            )

    total_supply = {
        f"{st}_{s['name']}": s["capacity (BCM)"]
        for st, sources in water_supply.items() for s in sources
    }

    desal_cost_coeffs = {
        "Desalination_MSF": {"type": "log", "coeffs": [-0.087, 2.3055]},
        "Desalination_MED": {"type": "log", "coeffs": [-0.137, 2.792]},
        "Desalination_RO":  {"type": "log", "coeffs": [-0.125, 2.3882]},
        "Groundwater":      {"type": "exp", "coeffs": [0.8359, 3e-11]},
    }
    brine_factors = {
        "Desalination_RO": 0.59, "Desalination_MSF": 0.72,
        "Desalination_MED": 0.74, "Desalination_NewRO": 0.59,
    }

    def log_unit_cost(Q, coeffs):
        a, b = coeffs
        return a * log(sqrt(Q**2 + 1e-12)) + b

    def exp_unit_cost(Q, coeffs):
        a, b = coeffs
        return a * exp(b * Q)

    def get_desal_unit_cost(source_type, Q):
        info = desal_cost_coeffs[source_type]
        return log_unit_cost(Q, info["coeffs"]) if info["type"] == "log" else exp_unit_cost(Q, info["coeffs"])

    # -------------------------------------------------------------------------
    # BUILD PYOMO MODEL
    # -------------------------------------------------------------------------
    model = ConcreteModel()

    model.allocation = Var(
        [(u, s) for u in water_users for s in expanded_user_sources[u]], domain=NonNegativeReals
    )
    model.production = Var(
        [(u, s) for u in water_users for s in expanded_user_sources[u]], domain=NonNegativeReals
    )
    energy_requirements = {
        (u, s): (
            model.production[u, s] * (
                next((info["energy_factor"] for info in user_energy[u] if info["source_name"] == s), None) or 0
            ) * 1e9 / (8760 * 1000)
            if any(info["source_name"] == s for info in user_energy[u]) else 0
        )
        for u in water_users for s in expanded_user_sources[u]
    }
    model.dual              = Suffix(direction=Suffix.IMPORT)
    model.crop_prod         = Var(list(domestic_crops), ["Domestic"], domain=NonNegativeReals)
    model.crop_prod_import  = Var(list(import_crops_data["Crop"]), ["Import"], domain=NonNegativeReals)
    model.total_irrigated_area = Var(domain=NonNegativeReals)
    model.cultivated_area   = Var(domestic_crops, domain=NonNegativeReals)
    model.energy_allocation = Var(
        [(u, s, p) for u in water_users for s in expanded_user_sources[u] for p in power_plants],
        domain=NonNegativeReals
    )
    model.num_new_units   = Var(new_plant_investments.keys(), domain=Integers, bounds=(0, None))
    model.build_new_desal = Var(domain=Binary)

    # 2023 is the calibration base year — no greenfield investment, only existing capacity.
    if year == 2023:
        for _plant in new_plant_investments:
            model.num_new_units[_plant].setub(0)

    def total_cost_rule(model):
        water_cost = 0
        for u in water_users:
            for s in expanded_user_sources[u]:
                parts = s.split("_")
                if len(parts) < 2:
                    continue
                if parts[0].lower() == "desalination":
                    source_type = f"{parts[0]}_{parts[1]}"
                elif parts[0] == "Groundwater":
                    source_type = "Groundwater"
                else:
                    source_type = parts[0]
                Q_yearly = model.allocation[u, s] * 1e9
                if source_type in desal_cost_coeffs:
                    water_cost += Q_yearly * get_desal_unit_cost(source_type, Q_yearly)
                else:
                    sname = parts[1]
                    cost  = next((item.get("cost", 1) for item in water_supply.get(source_type, []) if item["name"] == sname), 1)
                    if source_type == "SurfaceWater" and u == "Domestic":
                        cost *= 1.10
                    water_cost += Q_yearly * cost

        crop_cost = (
            sum(model.crop_prod[c, "Domestic"] * crop_culcost[c]
                for c in domestic_crops if (c, "Domestic") in model.crop_prod) +
            sum(model.crop_prod_import[c, "Import"] * crop_impcost[c]
                for c in import_crops_data["Crop"] if (c, "Import") in model.crop_prod_import)
        )
        energy_cost = sum(
            model.energy_allocation[u, s, p] * power_plants[p]["Cost ($/kWh)"] * 8760000
            for u in water_users for s in expanded_user_sources[u] for p in power_plants
        )

        crop_pen_dom = {c: abs(dual_prices_domestic.get(c, 0)) for c in domestic_crops}
        total_dom_prod = sum(model.crop_prod[c, "Domestic"] for c in domestic_crops)
        soft_pen_dom = sum(
            crop_pen_dom[c] * (model.crop_prod[c, "Domestic"] - min_crop_shares_dict.get(c, 0) * total_dom_prod)**2
            for c in crop_pen_dom
        )
        crop_pen_imp = {c: abs(dual_prices_import.get(c, 0)) for c in import_crops_data["Crop"]}
        total_imp_prod = sum(
            model.crop_prod_import[c, "Import"] for c in import_crops_data["Crop"]
            if (c, "Import") in model.crop_prod_import
        )
        soft_pen_imp = sum(
            crop_pen_imp.get(c, 0) * (
                model.crop_prod_import[c, "Import"] - min_crop_shares_import_dict.get(c, 0) * total_imp_prod
            )**2
            for c in import_crops_data["Crop"]
            if (c, "Import") in model.crop_prod_import and c in min_crop_shares_import_dict
        )

        total_water_prod   = sum(model.production[u, s] for u in water_users for s in expanded_user_sources[u])
        total_dom_prod_w   = sum(model.production["Domestic", s] for s in expanded_user_sources["Domestic"])
        total_dom_desal    = sum(model.production["Domestic", s] for s in expanded_user_sources["Domestic"] if s.startswith("Desalination"))
        total_desal        = sum(model.production[u, s] for u in water_users for s in expanded_user_sources[u] if s.startswith("Desalination"))
        total_treatedww    = sum(model.production[u, s] for u in water_users for s in expanded_user_sources[u] if s.startswith("TreatedWW"))
        total_surfacewater = sum(model.production[u, s] for u in water_users for s in expanded_user_sources[u] if s.startswith("SurfaceWater"))

        # dom_desal_pen replaced by hard constraint model.domestic_desal_min below
        # dom_desal_pen  = abs(dual_prices_dict.get("Domestic_Desalination_Min", 0)) * (0.6 * total_dom_prod_w - total_dom_desal)**2
        desal_pen        = abs(dual_prices_dict.get("Total_Desalination_Min",    0)) * (0.18 * total_water_prod - total_desal)**2
        tww_pen          = abs(dual_prices_dict.get("TreatedWW_Min",             0)) * (0.025 * total_water_prod - total_treatedww)**2
        sw_pen           = abs(dual_prices_dict.get("SurfaceWater_Min",          0)) * (0.05  * total_water_prod - total_surfacewater)**2
        unused_land_pen  = 1e9 * (787600 - model.total_irrigated_area)

        total_pen       = desal_pen + tww_pen + sw_pen + unused_land_pen
        investment_cost = sum(model.num_new_units[p] * new_plant_investments[p]["cost"] for p in new_plant_investments)

        return water_cost + crop_cost + energy_cost + soft_pen_dom + soft_pen_imp + total_pen + investment_cost

    model.obj = Objective(rule=total_cost_rule, sense=minimize)

    def demand_rule_lower(model, user):
        total = sum(model.allocation[user, s] for s in expanded_user_sources[user])
        return total >= 0 if user == "Agriculture" else total >= water_users[user]

    def demand_rule_upper(model, user):
        if user == "Agriculture":
            return Constraint.Skip
        return sum(model.allocation[user, s] for s in expanded_user_sources[user]) <= 1.1 * water_users[user]

    model.demand_rule_lower = Constraint(water_users.keys(), rule=demand_rule_lower)
    model.demand_rule_upper = Constraint(water_users.keys(), rule=demand_rule_upper)

    # HARD CONSTRAINT: at least 58% of domestic water production must come from desalination.
    # To revert to soft penalty: comment out this block and uncomment dom_desal_pen above.
    model.domestic_desal_min = Constraint(rule=lambda m:
        sum(m.production["Domestic", s] for s in expanded_user_sources["Domestic"]
            if s.startswith("Desalination"))
        >= 0.58 * sum(m.production["Domestic", s] for s in expanded_user_sources["Domestic"])
    )

    # HARD CONSTRAINT: agriculture cannot use any desalination source.
    agr_desal_sources = [s for s in expanded_user_sources["Agriculture"] if s.startswith("Desalination")]
    model.no_agr_desal = Constraint(agr_desal_sources,
        rule=lambda m, s: m.allocation["Agriculture", s] == 0
    )

    def capacity_rule(model, source):
        source_type, source_name = source.split("_", 1)
        if source in new_plant_investments:
            capacity = (inherited_units.get(source, 0) + model.num_new_units[source]) * new_plant_investments[source]["unit_size"]
        else:
            capacity = next(
                (item["capacity (BCM)"] for item in water_supply.get(source_type, []) if item["name"] == source_name), 0
            )
        eff = source_efficiencies.get(source, 1.0)
        return sum(model.production[u, source] for u in water_users if source in expanded_user_sources[u]) <= capacity * eff

    model.capacity_constraint = Constraint(
        list({s for u in water_users for s in expanded_user_sources[u]}), rule=capacity_rule
    )
    def new_water_cap_rule(model, plant):
        return (
            sum(model.production[u, plant] for u in water_users if plant in expanded_user_sources[u])
            <= (inherited_units.get(plant, 0) + model.num_new_units[plant]) * new_plant_investments[plant]["unit_size"]
        )
    model.new_water_plant_capacity_constraint = Constraint(
        ["SurfaceWater_NewSaudiDam", "TreatedWW_NewSaudiTWW", "Desalination_NewRO"], rule=new_water_cap_rule
    )
    model.production_constraint = Constraint(
        [(u, s) for u in water_users for s in expanded_user_sources[u]],
        rule=lambda model, u, s: model.allocation[u, s] == model.production[u, s] * user_efficiencies.get(u, 1.0)
    )

    def prod_area_rule(model, crop):
        if (crop, "Domestic") in model.crop_prod:
            return model.cultivated_area[crop] == model.crop_prod[crop, "Domestic"] / yield_per_ha[crop]
        return Constraint.Skip

    model.prod_area = Constraint(domestic_crops, rule=prod_area_rule)
    model.agr_water_demand = Constraint(rule=lambda m:
        sum(m.allocation["Agriculture", s] for s in expanded_user_sources["Agriculture"])
        == sum(m.cultivated_area[c] * 10000 * crop_water_req_dict[c] for c in domestic_crops) / 1e9
    )
    model.total_cultivated_area_constraint = Constraint(rule=lambda m:
        m.total_irrigated_area == sum(m.cultivated_area[c] for c in domestic_crops)
    )
    model.total_irrigated_area_limit = Constraint(rule=lambda m: m.total_irrigated_area <= 787600)
    model.calorie_demand_constraint  = Constraint(rule=lambda m: sum(
        ((m.crop_prod[c, "Domestic"] if (c, "Domestic") in m.crop_prod else 0) +
         (m.crop_prod_import[c, "Import"] if (c, "Import") in m.crop_prod_import else 0)) * crop_calories.get(c, 0)
        for c in crop_calories) >= total_calorie_demand)
    model.protein_demand_constraint  = Constraint(rule=lambda m: sum(
        ((m.crop_prod[c, "Domestic"] if (c, "Domestic") in m.crop_prod else 0) +
         (m.crop_prod_import[c, "Import"] if (c, "Import") in m.crop_prod_import else 0)) * crop_protein.get(c, 0)
        for c in crop_protein) >= total_protein_demand)

    def energy_alloc_rule(model, u, s):
        return (
            sum(model.energy_allocation[u, s, p] * power_plants[p]["Efficiency"] for p in power_plants)
            >= energy_requirements[(u, s)]
        )
    model.energy_allocation_constraint = Constraint(energy_requirements.keys(), rule=energy_alloc_rule)
    model.power_plant_production_constraint = Constraint(rule=lambda m:
        sum(m.energy_allocation[u, s, p] for u in water_users for s in expanded_user_sources[u] for p in power_plants)
        >= sum(energy_requirements[k] for k in energy_requirements)
    )
    model.power_plant_capacity_constraint = Constraint(
        [p for p in power_plants if p not in power_plants_new],
        rule=lambda model, plant: (
            sum(model.energy_allocation[u, s, plant] for u in water_users for s in expanded_user_sources[u])
            <= power_plants[plant]["Efficiency"] * power_plants[plant]["Capacity (MW)"]
        )
    )
    model.new_power_plant_capacity_constraint = Constraint(
        power_plants_new,
        rule=lambda model, plant: (
            sum(model.energy_allocation[u, s, plant]
                for u in water_users for s in expanded_user_sources[u]
                if (u, s, plant) in model.energy_allocation)
            <= (inherited_units.get(plant, 0) + model.num_new_units[plant]) * new_plant_investments[plant]["unit_size"] * power_plants[plant]["Efficiency"]
        )
    )

    if scenario1:
        model.total_groundwater_limit = Constraint(rule=lambda m:
            sum(m.production[u, s] for u in water_users for s in expanded_user_sources[u]
                if s.startswith("Groundwater")) <= 10
        )
        print("  Applied: 10 BCM groundwater limit")
    if scenario2:
        model.total_groundwater_limit = Constraint(rule=lambda m:
            sum(m.production[u, s] for u in water_users for s in expanded_user_sources[u]
                if s.startswith("Groundwater")) <= 5
        )
        print("  Applied: 5 BCM groundwater limit")

    model.max_surfacewater_production_constraint = Constraint(rule=lambda m:
        sum(m.production[u, s] for u in water_users for s in expanded_user_sources[u]
            if s.startswith("SurfaceWater")) <= 2
    )
    # Cap TWW production at 80% of domestic demand.
    model.max_treatedww_production_constraint = Constraint(rule=lambda m:
        sum(m.production[u, s] for u in water_users for s in expanded_user_sources[u] if s.startswith("TreatedWW"))
        <= 0.8 * domestic_total_demand
    )
    # Cap new TWW investment so installed capacity cannot exceed the production cap.
    # Without this, the model can build idle TWW units and still satisfy the
    # production constraint — meaning excess capacity accumulates unchecked.
    _tww_existing_cap  = sum(item["capacity (BCM)"] for item in water_supply.get("TreatedWW", []))
    _dam_existing_cap  = sum(item["capacity (BCM)"] for item in water_supply.get("SurfaceWater", []))
    _tww_inherited_cap = inherited_units.get("TreatedWW_NewSaudiTWW", 0) * new_plant_investments["TreatedWW_NewSaudiTWW"]["unit_size"]
    _tww_new_cap_limit = max(0.0, 0.8 * domestic_total_demand - _tww_existing_cap - _tww_inherited_cap)
    model.max_tww_new_capacity_constraint = Constraint(rule=lambda m:
        m.num_new_units["TreatedWW_NewSaudiTWW"] * new_plant_investments["TreatedWW_NewSaudiTWW"]["unit_size"]
        <= _tww_new_cap_limit
    )

    renewable_plant_names = ["Renewable", "Solar", "Wind", "Nuclear", "NewSolar", "NewWind"]
    model.min_renewable_energy_share_constraint = Constraint(rule=lambda m:
        sum(m.energy_allocation[u, s, p]
            for u in water_users for s in expanded_user_sources[u]
            for p in renewable_plant_names if (u, s, p) in m.energy_allocation)
        >= ren_share * sum(m.energy_allocation[u, s, p]
            for u in water_users for s in expanded_user_sources[u]
            for p in power_plants if (u, s, p) in m.energy_allocation)
    )

    # -------------------------------------------------------------------------
    # SOLVE
    # -------------------------------------------------------------------------
    # Warm-start: use the previous year's solution as IPOPT's starting point so
    # the solver stays near the prior allocation rather than jumping to an
    # arbitrary near-equivalent solution.  Falls back to 0.1 for any variable
    # not present in the stored state (e.g. first year, or after a skipped year).
    _prior = warm_start_state.get(scenario_key)
    for var in model.component_objects(Var, active=True):
        for idx in var:
            key = (var.name, idx)
            if _prior is not None and key in _prior:
                var[idx].value = _prior[key]
            elif var[idx].domain is Binary:
                var[idx].value = 0
            elif var[idx].domain is Integers:
                var[idx].value = 0
            else:
                var[idx].value = 0.1
            var[idx].stale = False

    _FEASIBLE_TC = {
        TerminationCondition.optimal,
        TerminationCondition.feasible,
        TerminationCondition.maxIterations,
        TerminationCondition.maxTimeLimit,
    }

    # Retry sequence: (tol, acceptable_tol, extra_options_dict, reinit_val)
    # reinit_val: if not None, reset all continuous vars to this value before solving.
    # The 4th attempt uses a fresh reinit + options that help escape restoration failures:
    #   bound_push/frac  — start interior point further from variable bounds
    #   mu_init          — larger initial barrier parameter
    _RETRY_SEQUENCE = [
        (1e-4, None, {},                                                                          None),
        (1e-3, 1e-3, {},                                                                          None),
        (1e-2, 1e-2, {"nlp_scaling_method": "gradient-based", "mu_strategy": "adaptive"},        None),
        (1e-2, 1e-2, {"nlp_scaling_method": "gradient-based", "mu_strategy": "adaptive",
                      "bound_push": 0.1, "bound_frac": 0.1, "mu_init": 0.1},                     1.0),
        # 5th attempt: disable all scaling (avoids NaN/Inf that cause "did not exit normally"),
        # use monotone barrier updates for stability, tiny reinit to stay well inside bounds.
        (1e-2, 1e-2, {"nlp_scaling_method": "none", "mu_strategy": "monotone",
                      "mu_init": 1.0, "bound_push": 0.01, "bound_frac": 0.01},                   0.01),
        # 6th attempt: tell IPOPT the problem may be infeasible — switches to a more robust
        # restoration strategy. Very loose tolerance (0.1) + large mu_init + tiny bound push
        # to approach the feasible region from well inside the bound box.
        # max_iter doubled so restoration has more room to work.
        (1e-1, 1e-1, {"expect_infeasible_problem": "yes",
                      "nlp_scaling_method": "gradient-based", "mu_strategy": "adaptive",
                      "mu_init": 10.0, "bound_push": 1e-6, "bound_frac": 1e-6,
                      "max_iter": 80000},                                                          0.1),
        # 7th attempt: heavy linear-system recovery.
        # slack-based linear system scaling conditions the KKT matrix differently
        # from NLP scaling; combined with 30 refinement steps this is the most
        # numerically conservative option available without HSL/mumps_pivtol.
        (1e-1, 1e-1, {"nlp_scaling_method": "none", "mu_strategy": "monotone",
                      "mu_init": 1e-3, "bound_push": 1e-4, "bound_frac": 1e-4,
                      "linear_system_scaling": "slack-based",
                      "max_refinement_steps": 30, "min_refinement_steps": 5,
                      "bound_relax_factor": 1e-6,
                      "max_iter": 120000},                                                        0.5),
    ]

    if hard_mode:
        # 8th: extreme tolerance + very high iter budget + maximum refinement + relaxed bounds.
        # Used only by run_until_clean.py for years that failed the standard 7 attempts twice.
        _RETRY_SEQUENCE += [
            (1.0, 1.0, {"nlp_scaling_method": "none", "mu_strategy": "monotone",
                        "mu_init": 1e-4, "bound_push": 1e-6, "bound_frac": 1e-6,
                        "linear_system_scaling": "slack-based",
                        "max_refinement_steps": 50, "min_refinement_steps": 10,
                        "bound_relax_factor": 1e-3,
                        "max_iter": 300000},                                                      0.1),
            # 9th: last resort — maximum relaxation, no reinit (keep current iterate).
            (2.0, 2.0, {"nlp_scaling_method": "none", "mu_strategy": "monotone",
                        "mu_init": 1e-6, "bound_push": 1e-8, "bound_frac": 1e-8,
                        "linear_system_scaling": "slack-based",
                        "max_refinement_steps": 100, "min_refinement_steps": 20,
                        "bound_relax_factor": 1e-2,
                        "max_iter": 500000},                                                      None),
        ]

    def _try_solve(tol, acceptable_tol, extra_opts):
        solver = SolverFactory("ipopt")
        solver.options["max_iter"]    = 100000
        solver.options["tol"]         = tol
        solver.options["print_level"] = 3
        # Extra iterative refinement after each linear solve: improves accuracy
        # when the MUMPS factorization is ill-conditioned without rejecting it outright.
        # This is the only available lever for internalSolverError on this build
        # (mumps_pivtol is not registered in this binary despite MUMPS being active).
        solver.options["max_refinement_steps"] = 5
        solver.options["min_refinement_steps"] = 1
        if acceptable_tol is not None:
            solver.options["acceptable_tol"] = acceptable_tol
        for k, v in extra_opts.items():
            solver.options[k] = v
        results = solver.solve(model, tee=False, load_solutions=False)
        tc = results.solver.termination_condition
        return results, tc

    tc = None
    results = None
    solve_note = ""

    for tol_val, acc_tol, extra_opts, reinit_val in _RETRY_SEQUENCE:
        if reinit_val is not None:
            for var in model.component_objects(Var, active=True):
                for idx in var:
                    if var[idx].domain not in (Binary, Integers):
                        var[idx].value = reinit_val
                        var[idx].stale = False
        _sc = extra_opts.get("nlp_scaling_method", "default")
        _mu = extra_opts.get("mu_strategy", "")
        attempt_label = f"tol={tol_val}, scale={_sc}" + (f", mu={_mu}" if _mu else "")
        if reinit_val is not None:
            attempt_label += f", reinit={reinit_val}"
        if extra_opts.get("expect_infeasible_problem") == "yes":
            attempt_label += ", infeas-mode"
        try:
            results, tc = _try_solve(tol_val, acc_tol, extra_opts)
            if tc in _FEASIBLE_TC:
                model.solutions.load_from(results)
                print(f"  [{attempt_label}] Termination: {tc}   Objective: {value(model.obj):.2f}")
                solve_note = attempt_label
                # Save crop decision variables as warm-start for the next year.
                # Only cultivated_area and crop_prod_import are stored — these
                # are what causes year-to-year oscillation and are not bounded
                # by the changing fleet.  Production/allocation variables are
                # intentionally excluded: they are tightly constrained by
                # plant capacities that change (retirements, new investments),
                # and warm-starting them with stale capacity values causes
                # IPOPT to start from a severely infeasible point and crash.
                _WARMSTART_VARS = {"cultivated_area", "crop_prod_import"}
                _ws = {}
                for _var in model.component_objects(Var, active=True):
                    if _var.name not in _WARMSTART_VARS:
                        continue
                    for _idx in _var:
                        try:
                            _ws[(_var.name, _idx)] = value(_var[_idx])
                        except Exception:
                            pass
                warm_start_state[scenario_key] = _ws
                break
            else:
                print(f"  [{attempt_label}] {tc} — retrying...")
                solve_note = f"{attempt_label}: {tc}"
        except Exception as e:
            print(f"  [{attempt_label}] EXCEPTION: {e} — retrying...")
            solve_note = f"{attempt_label}: exception({e})"
            tc = None

    if tc not in _FEASIBLE_TC:
        status_str = str(tc) if tc is not None else "exception"
        print(f"  All attempts failed ({status_str}) — saving status file and continuing")
        with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
            pd.DataFrame([
                {"Metric": "Year",            "Value": year},
                {"Metric": "Energy Scenario", "Value": energy_scenario},
                {"Metric": "GW Scenario",     "Value": gw_label},
                {"Metric": "Solver Status",   "Value": status_str},
                {"Metric": "Note",            "Value": solve_note},
            ]).to_excel(writer, sheet_name="Summary", index=False)
        return

    # -------------------------------------------------------------------------
    # RESULTS EXPORT
    # -------------------------------------------------------------------------
    allocation_results = []
    production_results = []
    for u in water_users:
        for s in expanded_user_sources[u]:
            allocation_results.append({
                "User": u, "Source": s,
                "Water Allocated (billion m3)": value(model.allocation[u, s]) if (u, s) in model.allocation else 0,
                "Plant Type": "New" if s in new_plants else "Existing",
            })
            production_results.append({
                "User": u, "Source": s,
                "Water Produced (billion m3)": value(model.production[u, s]) if (u, s) in model.production else 0,
                "Plant Type": "New" if s in new_plants else "Existing",
            })

    remaining_cap_results = []
    for source, base_cap in total_supply.items():
        if source in new_plant_investments:
            cap = (inherited_units.get(source, 0) + value(model.num_new_units[source])) * new_plant_investments[source]["unit_size"]
        else:
            cap = base_cap
        total_used = sum(value(model.production[u, source]) for u in water_users if source in expanded_user_sources[u])
        remaining_cap_results.append({
            "Source": source, "Capacity (BCM)": cap,
            "Used (BCM)": total_used, "Remaining (BCM)": cap - total_used,
        })

    result_crops = sorted(set(food_crops + feed_crops))
    dom_prod = {c: value(model.crop_prod[c, "Domestic"]) if (c, "Domestic") in model.crop_prod else 0 for c in result_crops}
    imp_prod = {c: value(model.crop_prod_import[c, "Import"]) if (c, "Import") in model.crop_prod_import else 0 for c in result_crops}
    crop_df = pd.DataFrame({
        "Crop": result_crops,
        "Domestic Production (tons)": [dom_prod[c] for c in result_crops],
        "Imported (tons)": [imp_prod[c] for c in result_crops],
    })
    crop_df["Calories Provided"] = [(dom_prod[c] + imp_prod[c]) * crop_calories.get(c, 0) for c in result_crops]
    crop_df["Protein Provided"]  = [(dom_prod[c] + imp_prod[c]) * crop_protein.get(c, 0)  for c in result_crops]

    # Domestic crop production tab (one row per domestically produced crop)
    dom_crop_list = sorted(domestic_crops)
    dom_crop_tab = pd.DataFrame({
        "Crop":                       dom_crop_list,
        "Production (tons)":          [value(model.crop_prod[c, "Domestic"]) if (c, "Domestic") in model.crop_prod else 0 for c in dom_crop_list],
        "Cultivated Area (ha)":       [value(model.cultivated_area[c]) if c in model.cultivated_area else 0 for c in dom_crop_list],
        "Yield (tons/ha)":            [yield_per_ha.get(c, 0) for c in dom_crop_list],
        "Cultivation Cost ($/ton)":   [crop_culcost.get(c, 0) for c in dom_crop_list],
        "Total Cultivation Cost ($)": [(value(model.crop_prod[c, "Domestic"]) if (c, "Domestic") in model.crop_prod else 0) * crop_culcost.get(c, 0) for c in dom_crop_list],
        "Calories Provided":          [(value(model.crop_prod[c, "Domestic"]) if (c, "Domestic") in model.crop_prod else 0) * crop_calories.get(c, 0) for c in dom_crop_list],
        "Protein Provided":           [(value(model.crop_prod[c, "Domestic"]) if (c, "Domestic") in model.crop_prod else 0) * crop_protein.get(c, 0) for c in dom_crop_list],
    })

    # Imported crops tab (one row per importable crop)
    imp_crop_list = sorted(import_crops_data["Crop"].tolist())
    imp_crop_tab = pd.DataFrame({
        "Crop":                   imp_crop_list,
        "Imported Quantity (tons)": [value(model.crop_prod_import[c, "Import"]) if (c, "Import") in model.crop_prod_import else 0 for c in imp_crop_list],
        "Import Cost ($/ton)":    [crop_impcost.get(c, 0) for c in imp_crop_list],
        "Total Import Cost ($)":  [(value(model.crop_prod_import[c, "Import"]) if (c, "Import") in model.crop_prod_import else 0) * crop_impcost.get(c, 0) for c in imp_crop_list],
        "Calories Provided":      [(value(model.crop_prod_import[c, "Import"]) if (c, "Import") in model.crop_prod_import else 0) * crop_calories.get(c, 0) for c in imp_crop_list],
        "Protein Provided":       [(value(model.crop_prod_import[c, "Import"]) if (c, "Import") in model.crop_prod_import else 0) * crop_protein.get(c, 0) for c in imp_crop_list],
    })

    energy_results = []
    for u, sources in user_energy.items():
        for info in sources:
            sname    = info["source_name"]
            prod_v   = value(model.production[u, sname]) if (u, sname) in model.production else 0
            req_mw   = value(energy_requirements[(u, sname)]) if (u, sname) in energy_requirements else 0
            alloc_mw = sum(value(model.energy_allocation[u, sname, p]) for p in power_plants if (u, sname, p) in model.energy_allocation)
            energy_results.append({
                "User": u, "Source": sname,
                "Production (BCM)": prod_v,
                "Energy Factor (kWh/m3)": info["energy_factor"],
                "Energy Required (MW)": req_mw,
                "Energy Allocated (MW)": alloc_mw,
            })

    # Clamp fractional unit counts below 0.01 to zero (IPOPT interior-point noise)
    _unit_counts = {
        p: (v if (v := value(model.num_new_units[p])) >= 0.01 else 0.0)
        for p in new_plant_investments
    }

    power_plant_results = []
    for plant in power_plants:
        if plant in new_plants:
            continue
        total_alloc = sum(
            value(model.energy_allocation[u, s, plant])
            for u in water_users for s in expanded_user_sources[u]
            if (u, s, plant) in model.energy_allocation
        )
        cap = (
            (inherited_units.get(plant, 0) + _unit_counts.get(plant, 0)) * new_plant_investments[plant]["unit_size"]
            if plant in new_plant_investments else power_plants[plant]["Capacity (MW)"]
        )
        power_plant_results.append({
            "Power Plant": plant,
            "Total Energy Allocated (MW)": total_alloc,
            "Capacity (MW)": cap,
            "CO2 Emissions (t)": total_alloc * power_plants[plant]["CO2 (t/kwh)"] * 8760000,
        })

    # Update in-memory history so subsequent years in this run see this year's investments
    current_year_investments = {p: _unit_counts[p] for p in new_plant_investments if _unit_counts[p] >= 0.01}
    if current_year_investments:
        investment_history.setdefault(scenario_key, {})[year] = current_year_investments

    investment_results = [
        {
            "Plant": plant,
            "Number of Units": _unit_counts[plant],
            "Unit Cost ($)": new_plant_investments[plant]["cost"],
            "Total Cost ($)": _unit_counts[plant] * new_plant_investments[plant]["cost"],
        }
        for plant in new_plant_investments
    ]

    # Cumulative vintage table: one row per (plant, vintage_year) cohort still in history
    cumulative_inv_rows = []
    all_vintages = dict(investment_history.get(scenario_key, {}))
    for vintage_year in sorted(all_vintages.keys()):
        for plant, units in all_vintages[vintage_year].items():
            if units < 0.01:
                continue
            age    = year - vintage_year
            status = "Retired" if age >= PLANT_LIFETIME else "Active"
            cumulative_inv_rows.append({
                "Plant":            plant,
                "Vintage Year":     vintage_year,
                "Units Invested":   units,
                "Unit Size (BCM or MW)": new_plant_investments[plant]["unit_size"],
                "Total Capacity":   units * new_plant_investments[plant]["unit_size"],
                "Age (years)":      age,
                "Status in {}".format(year): status,
            })

    water_cost_v = 0
    for u in water_users:
        for s in expanded_user_sources[u]:
            parts = s.split("_")
            if len(parts) < 2:
                continue
            if parts[0].lower() == "desalination":
                st = f"{parts[0]}_{parts[1]}"
            elif parts[0] == "Groundwater":
                st = "Groundwater"
            else:
                st = parts[0]
            Q = value(model.allocation[u, s]) * 1e9
            if st in desal_cost_coeffs:
                a, b = desal_cost_coeffs[st]["coeffs"]
                uc   = a * math.log(math.sqrt(Q**2 + 1e-12)) + b if desal_cost_coeffs[st]["type"] == "log" else a * math.exp(b * Q)
                water_cost_v += Q * uc
            else:
                sname = parts[1]
                if sname in new_water_cost:
                    # New plant: unit cost is O&M only (CapEx captured in investment_results)
                    cost = new_water_cost[sname] * price_escalation
                else:
                    cost = next((item.get("cost", 1) for item in water_supply.get(st, []) if item["name"] == sname), 1)
                if st == "SurfaceWater" and u == "Domestic":
                    cost *= 1.10
                water_cost_v += Q * cost

    energy_cost_v     = sum(
        value(model.energy_allocation[u, s, p]) * power_plants[p]["Cost ($/kWh)"] * 8760000
        for u in water_users for s in expanded_user_sources[u] for p in power_plants
        if (u, s, p) in model.energy_allocation
    )
    investment_cost_v = sum(_unit_counts[p] * new_plant_investments[p]["cost"] for p in new_plant_investments)
    dom_crop_cost_v   = sum(value(model.crop_prod[c, "Domestic"]) * crop_culcost[c] for c in domestic_crops)
    imp_crop_cost_v   = sum(value(model.crop_prod_import[c, "Import"]) * crop_impcost[c] for c in import_crops_data["Crop"])

    total_brine = 0
    for u in water_users:
        for s in expanded_user_sources[u]:
            parts = s.split("_")
            if len(parts) >= 2:
                st = f"{parts[0]}_{parts[1]}"
                if st in brine_factors:
                    total_brine += value(model.production[u, s]) * 1e9 * brine_factors[st]

    total_co2 = sum(
        value(model.energy_allocation[u, s, p]) * power_plants[p]["CO2 (t/kwh)"] * 8760000
        for u in water_users for s in expanded_user_sources[u] for p in power_plants
        if (u, s, p) in model.energy_allocation
    )

    agr_water_use_v = sum(
        value(model.allocation["Agriculture", s]) for s in expanded_user_sources["Agriculture"]
    )

    total_gw_production_v = sum(
        value(model.production[u, s])
        for u in water_users for s in expanded_user_sources[u]
        if s.startswith("Groundwater")
    )

    summary_data = [
        {"Metric": "Year",                     "Value": year},
        {"Metric": "Energy Scenario",          "Value": energy_scenario},
        {"Metric": "GW Scenario",              "Value": gw_label},
        {"Metric": "Data File Year",           "Value": data_year},
        {"Metric": "Population",               "Value": population},
        {"Metric": "Renewable Share",          "Value": ren_share},
        {"Metric": "Domestic Demand (BCM)",    "Value": domestic_total_demand},
        {"Metric": "Industry Demand (BCM)",    "Value": industry_total_demand},
        {"Metric": "Agr Water Use (BCM)",      "Value": agr_water_use_v},
        {"Metric": "GW Production (BCM)",      "Value": total_gw_production_v},
        {"Metric": "RO Fleet Capacity (BCM)",  "Value": desal_caps["RO"]},
        {"Metric": "MSF Fleet Capacity (BCM)", "Value": desal_caps["MSF"]},
        {"Metric": "MED Fleet Capacity (BCM)", "Value": desal_caps["MED"]},
        {"Metric": "TWW Fleet Capacity (BCM)", "Value": _tww_existing_cap},
        {"Metric": "Dam Fleet Capacity (BCM)", "Value": _dam_existing_cap},
        {"Metric": "Water Cost ($)",           "Value": water_cost_v},
        {"Metric": "Domestic Crop Cost ($)",   "Value": dom_crop_cost_v},
        {"Metric": "Imported Crop Cost ($)",   "Value": imp_crop_cost_v},
        {"Metric": "Energy Cost ($)",          "Value": energy_cost_v},
        {"Metric": "Investment Cost ($)",      "Value": investment_cost_v},
        {"Metric": "Total Irrigated Area (ha)","Value": value(model.total_irrigated_area)},
        {"Metric": "Total Brine Discharge (m3)","Value": total_brine},
        {"Metric": "Total CO2 Emissions (t)",  "Value": total_co2},
        {"Metric": "New RO Units",                "Value": _unit_counts["Desalination_NewRO"]},
        {"Metric": "New RO Capacity (BCM)",       "Value": _unit_counts["Desalination_NewRO"] * new_plant_investments["Desalination_NewRO"]["unit_size"]},
        {"Metric": "Inherited RO Units",          "Value": inherited_units.get("Desalination_NewRO", 0)},
        {"Metric": "Inherited RO Capacity (BCM)", "Value": inherited_units.get("Desalination_NewRO", 0) * new_plant_investments["Desalination_NewRO"]["unit_size"]},
        {"Metric": "Total RO Capacity (BCM)",     "Value": (inherited_units.get("Desalination_NewRO", 0) + _unit_counts["Desalination_NewRO"]) * new_plant_investments["Desalination_NewRO"]["unit_size"]},
        {"Metric": "New Dam Units",               "Value": _unit_counts["SurfaceWater_NewSaudiDam"]},
        {"Metric": "New Dam Capacity (BCM)",      "Value": _unit_counts["SurfaceWater_NewSaudiDam"] * new_plant_investments["SurfaceWater_NewSaudiDam"]["unit_size"]},
        {"Metric": "Inherited Dam Units",         "Value": inherited_units.get("SurfaceWater_NewSaudiDam", 0)},
        {"Metric": "Inherited Dam Capacity (BCM)","Value": inherited_units.get("SurfaceWater_NewSaudiDam", 0) * new_plant_investments["SurfaceWater_NewSaudiDam"]["unit_size"]},
        {"Metric": "Total Dam Capacity (BCM)",    "Value": (inherited_units.get("SurfaceWater_NewSaudiDam", 0) + _unit_counts["SurfaceWater_NewSaudiDam"]) * new_plant_investments["SurfaceWater_NewSaudiDam"]["unit_size"]},
        {"Metric": "New TWW Units",               "Value": _unit_counts["TreatedWW_NewSaudiTWW"]},
        {"Metric": "New TWW Capacity (BCM)",      "Value": _unit_counts["TreatedWW_NewSaudiTWW"] * new_plant_investments["TreatedWW_NewSaudiTWW"]["unit_size"]},
        {"Metric": "Inherited TWW Units",         "Value": inherited_units.get("TreatedWW_NewSaudiTWW", 0)},
        {"Metric": "Inherited TWW Capacity (BCM)","Value": inherited_units.get("TreatedWW_NewSaudiTWW", 0) * new_plant_investments["TreatedWW_NewSaudiTWW"]["unit_size"]},
        {"Metric": "Total TWW Capacity (BCM)",    "Value": (inherited_units.get("TreatedWW_NewSaudiTWW", 0) + _unit_counts["TreatedWW_NewSaudiTWW"]) * new_plant_investments["TreatedWW_NewSaudiTWW"]["unit_size"]},
        {"Metric": "Total Cost (Objective)",   "Value": value(model.obj)},
        {"Metric": "Solver Status",            "Value": str(tc)},
    ]

    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        pd.DataFrame(summary_data).to_excel(writer,       sheet_name="Summary",                   index=False)
        pd.DataFrame(allocation_results).to_excel(writer, sheet_name="Water Allocation",          index=False)
        pd.DataFrame(production_results).to_excel(writer, sheet_name="Water Production",          index=False)
        pd.DataFrame(remaining_cap_results).to_excel(writer,sheet_name="Remaining Capacity",      index=False)
        dom_crop_tab.to_excel(writer,                      sheet_name="Domestic Crop Production",  index=False)
        imp_crop_tab.to_excel(writer,                      sheet_name="Imported Crops",            index=False)
        crop_df.to_excel(writer,                           sheet_name="Crop Production (Combined)",index=False)
        pd.DataFrame(energy_results).to_excel(writer,     sheet_name="Energy Results",            index=False)
        pd.DataFrame(power_plant_results).to_excel(writer,sheet_name="Power Plant Results",       index=False)
        pd.DataFrame(investment_results).to_excel(writer, sheet_name="Investments",               index=False)
        if cumulative_inv_rows:
            pd.DataFrame(cumulative_inv_rows).to_excel(writer, sheet_name="Cumulative Investments", index=False)
        crop_water_reqs.to_excel(writer,                  sheet_name="Crop Water Req",            index=False)
        phase_product.to_excel(writer,                    sheet_name="Phase Product",             index=False)

    print(f"  Saved -> {out_file}")


# =============================================================================
# MAIN LOOP
# =============================================================================

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if TEST_SINGLE:
        _energy_scenarios = ["STANDARD"]
        _gw_scenarios     = [(False, False)]
        _scope_label = "TEST_SINGLE (STANDARD / No_GW_Limit only)"
    else:
        _energy_scenarios = ENERGY_SCENARIOS
        _gw_scenarios     = GW_SCENARIOS
        _scope_label = "FULL BATCH (all scenarios)"

    total_runs = len(list(YEARS_TO_RUN)) * len(_energy_scenarios) * len(_gw_scenarios)
    print(f"Scope: {_scope_label}")
    print(f"Total runs planned: {total_runs}")
    print(f"Years: {list(YEARS_TO_RUN)[0]}-{list(YEARS_TO_RUN)[-1]}")
    print(f"Energy scenarios: {_energy_scenarios}")
    print(f"GW scenarios: {len(_gw_scenarios)} variants")
    print(f"Skip existing: {SKIP_EXISTING}")
    print(f"Plant lifetime: {PLANT_LIFETIME} years\n")

    # Rebuild investment history from any existing result files (supports SKIP_EXISTING resume)
    print("Scanning for existing investment history...")
    investment_history = load_investment_history()
    total_vintages = sum(len(v) for v in investment_history.values())
    print(f"  Found {total_vintages} existing vintage-year entries across {len(investment_history)} scenario(s)\n")

    # warm_start_state accumulates the previous solved year's variable values per scenario,
    # so IPOPT starts near the prior solution rather than from scratch each year.
    warm_start_state = {}

    run_count = 0
    for year in YEARS_TO_RUN:
        for energy_scenario in _energy_scenarios:
            for (s1, s2) in _gw_scenarios:
                run_count += 1
                print(f"[{run_count}/{total_runs}]", end=" ")
                run_model(year, energy_scenario, s1, s2,
                          investment_history=investment_history,
                          warm_start_state=warm_start_state)

    print(f"\nAll runs complete. Results in: {RESULTS_DIR}")

    # -------------------------------------------------------------------------
    # CONSOLIDATED BATCH SUMMARY
    # -------------------------------------------------------------------------
    print("\nBuilding consolidated batch summary...")
    summary_rows  = []
    dom_crop_rows = []
    imp_crop_rows = []
    _ID_COLS = ["Year", "Energy Scenario", "GW Scenario"]

    for fname in sorted(os.listdir(RESULTS_DIR)):
        if not fname.endswith(".xlsx") or not fname[0].isdigit():
            continue
        fpath = os.path.join(RESULTS_DIR, fname)
        try:
            xl = pd.ExcelFile(fpath)
            if "Summary" not in xl.sheet_names:
                continue
            df_sum = pd.read_excel(fpath, sheet_name="Summary")
            row = {r["Metric"]: r["Value"] for _, r in df_sum.iterrows() if pd.notna(r["Metric"])}
            row["File"] = fname
            summary_rows.append(row)

            meta = {k: row[k] for k in _ID_COLS if k in row}

            if "Domestic Crop Production" in xl.sheet_names:
                df_dom = pd.read_excel(fpath, sheet_name="Domestic Crop Production")
                dom_row = dict(meta)
                for _, r in df_dom.iterrows():
                    dom_row[r["Crop"]] = r["Production (tons)"]
                dom_crop_rows.append(dom_row)

            if "Imported Crops" in xl.sheet_names:
                df_imp = pd.read_excel(fpath, sheet_name="Imported Crops")
                imp_row = dict(meta)
                for _, r in df_imp.iterrows():
                    imp_row[r["Crop"]] = r["Imported Quantity (tons)"]
                imp_crop_rows.append(imp_row)

        except Exception as e:
            summary_rows.append({"File": fname, "Note": f"read error: {e}"})

    if summary_rows:
        batch_df = pd.DataFrame(summary_rows)
        col_order = ["File", "Year", "Energy Scenario", "GW Scenario", "Solver Status",
                     "Population", "Renewable Share", "Domestic Demand (BCM)", "Industry Demand (BCM)",
                     "Agr Water Use (BCM)", "GW Production (BCM)",
                     "Cumulative GW Extraction (BCM)", "Remaining GW Stock (BCM)",
                     "RO Fleet Capacity (BCM)", "MSF Fleet Capacity (BCM)", "MED Fleet Capacity (BCM)",
                     "New RO Units", "New RO Capacity (BCM)",
                     "New Dam Units", "New Dam Capacity (BCM)",
                     "New TWW Units", "New TWW Capacity (BCM)",
                     "Water Cost ($)", "Energy Cost ($)", "Investment Cost ($)",
                     "Domestic Crop Cost ($)", "Imported Crop Cost ($)",
                     "Total Cost (Objective)", "Total Irrigated Area (ha)", "Total Brine Discharge (m3)",
                     "Total CO2 Emissions (t)",
                     "Data File Year", "Note"]
        existing_cols = [c for c in col_order if c in batch_df.columns]
        extra_cols    = [c for c in batch_df.columns if c not in col_order]
        batch_df = batch_df[existing_cols + extra_cols]

        # Compute cumulative GW depletion per (Energy Scenario, GW Scenario) group.
        # Rows where GW Production is missing (failed runs) are treated as 0 extraction.
        if "GW Production (BCM)" in batch_df.columns and "Year" in batch_df.columns:
            batch_df["Year"] = pd.to_numeric(batch_df["Year"], errors="coerce")
            batch_df = batch_df.sort_values(["Energy Scenario", "GW Scenario", "Year"]).reset_index(drop=True)
            batch_df["GW Production (BCM)"] = pd.to_numeric(batch_df["GW Production (BCM)"], errors="coerce").fillna(0)
            batch_df["Cumulative GW Extraction (BCM)"] = (
                batch_df.groupby(["Energy Scenario", "GW Scenario"])["GW Production (BCM)"]
                .cumsum()
            )
            batch_df["Remaining GW Stock (BCM)"] = GW_INITIAL_STOCK_BCM - batch_df["Cumulative GW Extraction (BCM)"]

        dom_crop_df = pd.DataFrame(dom_crop_rows).sort_values(_ID_COLS).reset_index(drop=True) if dom_crop_rows else pd.DataFrame()
        imp_crop_df = pd.DataFrame(imp_crop_rows).sort_values(_ID_COLS).reset_index(drop=True) if imp_crop_rows else pd.DataFrame()

        summary_path = os.path.join(RESULTS_DIR, "_batch_summary.xlsx")
        with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
            batch_df.to_excel(writer,    sheet_name="Summary",                  index=False)
            if not dom_crop_df.empty:
                dom_crop_df.to_excel(writer, sheet_name="Domestic Crop Production", index=False)
            if not imp_crop_df.empty:
                imp_crop_df.to_excel(writer, sheet_name="Imported Crops",           index=False)

        feasible_n   = (batch_df.get("Solver Status", pd.Series(dtype=str))
                        .isin(["optimal", "feasible", "maxIterations", "maxTimeLimit"])).sum()
        infeasible_n = len(batch_df) - feasible_n
        print(f"  Summary: {len(batch_df)} runs  |  {feasible_n} feasible  |  {infeasible_n} infeasible/error")
        print(f"  Saved -> {summary_path}")

    # -------------------------------------------------------------------------
    # SCENARIO INPUTS EXPORT
    # -------------------------------------------------------------------------
    print("\nExporting scenario input parameters...")
    input_rows = []
    for yr in YEARS_TO_RUN:
        dom_d, ind_d = get_demand(yr)
        input_rows.append({
            "Year":                         yr,
            "Domestic Demand (BCM)":        round(dom_d, 4),
            "Industry Demand (BCM)":        round(ind_d, 4),
            "Renewable Share STANDARD (%)": round(get_renewable_share(yr, "STANDARD") * 100, 2),
        })
    inputs_df = pd.DataFrame(input_rows)
    inputs_path = os.path.join(RESULTS_DIR, "_scenario_inputs.xlsx")
    inputs_df.to_excel(inputs_path, index=False)
    print(f"  Saved -> {inputs_path}")
