"""
results_numbers.py
==================
Prints every headline number the paper's Projection Results text quotes (Figures 4-8), per
GW scenario, from a results directory, so the text can be checked/rewritten against the
final result set.  Usage:  python results_numbers.py <results_dir>
"""
import os, sys, io
import numpy as np, pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Results", "intertemporal")
YEARS = list(range(2024, 2061)); DISC = 0.05
SC = [("No_GW_Limit", "No limit"), ("10_BCM_GW_Limit", "10 BCM"), ("5_BCM_GW_Limit", "5 BCM")]


def load(scen):
    d = {}
    for y in YEARS:
        f = os.path.join(RES, f"{y}_STANDARD_{scen}.xlsx")
        if not os.path.exists(f):
            return None
        x = pd.read_excel(f, sheet_name=None)
        d[y] = x
    return d


def summ(x):
    return {r.Metric: r.Value for r in x["Summary"].itertuples() if pd.notna(r.Metric)}


def prod_table(x):
    p = x["Water Production"].copy(); p.columns = ["User", "Source", "BCM", "Type"]
    p["st"] = p["Source"].str.split("_").str[0]
    return p


def m(v, n=1):
    return f"{v:.{n}f}"


for scen, lab in SC:
    D = load(scen)
    print("\n" + "=" * 78 + f"\n{lab}  ({scen})\n" + "=" * 78)
    if D is None:
        print("  [results incomplete]"); continue
    S = {y: summ(D[y]) for y in YEARS}
    P = {y: prod_table(D[y]) for y in YEARS}
    dom = {y: D[y]["Domestic Crop Production"].set_index("Crop")["Production (tons)"] for y in YEARS}
    imp = {y: D[y]["Imported Crops"].set_index("Crop")["Imported Quantity (tons)"] for y in YEARS}

    def src(y, st, user=None):
        p = P[y]; p = p[p.st == st]
        return p[p.User == user].BCM.sum() if user else p.BCM.sum()
    tot = lambda y: P[y].BCM.sum()

    print("-- Fig 4: crops (million tons)")
    for y in (2024, 2030, 2035, 2060):
        print(f"  {y}: domestic {m(dom[y].sum()/1e6,2)}  imports {m(imp[y].sum()/1e6,2)}")
    for y in (2024, 2060):
        top = dom[y].sort_values(ascending=False).head(5)
        print(f"  top domestic {y}: " + ", ".join(f"{k} {v/1e6:.2f}" for k, v in top.items()))
    print("  2060 imports: " + ", ".join(f"{k} {imp[2060][k]/1e6:.1f}" for k in ("MaizeFeed", "Rice", "Citrus") if k in imp[2060].index))
    print("  Other Grain 2024 (Mt):", m(dom[2024].get("Other Grain", 0) / 1e6, 2), "| Dates 2024/2060:", m(dom[2024].get("Dates", 0) / 1e6, 2), m(dom[2060].get("Dates", 0) / 1e6, 2),
          "| Fodder 2060:", m(dom[2060].get("Fodder", 0) / 1e6, 2))
    print("  domestic path 5-yr:", ", ".join(f"{y}:{dom[y].sum()/1e6:.2f}" for y in range(2024, 2061, 4)))

    print("-- Fig 5: water production (BCM)")
    for y in (2024, 2060):
        print(f"  {y}: total {m(tot(y))}  GW {m(src(y,'Groundwater'))} ({100*src(y,'Groundwater')/tot(y):.0f}%)  desal {m(src(y,'Desalination'))}  TWW {m(src(y,'TreatedWW'))}  SW {m(src(y,'SurfaceWater'))}")
    for y in (2024, 2060):
        print(f"  agr {y}: GW {m(src(y,'Groundwater','Agriculture'))} TWW {m(src(y,'TreatedWW','Agriculture'))} SW {m(src(y,'SurfaceWater','Agriculture'))} | dom+ind desal {m(src(y,'Desalination','Domestic')+src(y,'Desalination','Industry'))} | dom+ind GW {m(src(y,'Groundwater','Domestic')+src(y,'Groundwater','Industry'))}")
    swy = next((y for y in YEARS if src(y, 'SurfaceWater') >= 1.99), None)
    print("  surface water: 2024", m(src(2024, 'SurfaceWater'), 2), "2034", m(src(2034, 'SurfaceWater'), 2), "2040", m(src(2040, 'SurfaceWater'), 2), "2060", m(src(2060, 'SurfaceWater'), 2), "| first year >=1.99:", swy)

    print("-- Fig 6: fleet (BCM capacity)")
    for y in (2024, 2060):
        rc = D[y]["Remaining Capacity"]
        ex = rc[rc.Source.str.startswith("Desalination") & ~rc.Source.str.contains("NewRO")]["Capacity (BCM)"].sum()
        nr = rc[rc.Source == "Desalination_NewRO"]["Capacity (BCM)"].sum()
        print(f"  {y}: existing desal {m(ex,2)} new RO {m(nr,2)} | TWW fleet {S[y].get('TWW Fleet Capacity (BCM)')} | dam fleet {S[y].get('Dam Fleet Capacity (BCM)')}")
    ex24 = D[2024]["Remaining Capacity"]; ex24 = ex24[ex24.Source.str.startswith("Desalination") & ~ex24.Source.str.contains("NewRO")]
    print(f"  2024 existing desal used {m(ex24['Used (BCM)'].sum(),2)} of {m(ex24['Capacity (BCM)'].sum(),2)} BCM  | 2024 invest ${S[2024]['Investment Cost ($)']/1e9:.2f}B (new RO prod {m(src(2024,'Desalination') - ex24['Used (BCM)'].sum(),2)})")

    print("-- Fig 7: electricity / CO2")
    el = {y: D[y]["Energy Results"]["Energy Allocated (MW)"].sum() * 8760 / 1e6 for y in YEARS}
    co2 = {y: S[y]["Total CO2 Emissions (t)"] / 1e6 for y in YEARS}
    for y in (2024, 2035, 2060):
        print(f"  {y}: {m(el[y])} TWh, {m(co2[y])} Mt CO2 ({co2[y]/el[y]:.3f} t/MWh)")
    print(f"  electricity change 2024->2060: {100*(el[2060]/el[2024]-1):.0f}% | CO2 min {m(min(co2.values()))} in {min(co2, key=co2.get)}")

    print("-- Fig 8: discounted costs (billion $, 5%, 2024 base)")
    d = {y: 1 / (1 + DISC) ** (y - 2024) for y in YEARS}
    comp = {"water supply": "Water Cost ($)", "energy": "Energy Cost ($)", "domestic crops": "Domestic Crop Cost ($)", "imported crops": "Imported Crop Cost ($)"}
    c = {k: sum(S[y][v] * d[y] for y in YEARS) / 1e9 for k, v in comp.items()}
    c["investment"] = sum(S[y]["Investment Cost ($)"] * d[y] for y in YEARS) / 1e9
    T = sum(c.values())
    print("  " + "  ".join(f"{k} {v:.1f}" for k, v in c.items()) + f"  | TOTAL {T:.0f}")
    print(f"  crops share {100*(c['domestic crops']+c['imported crops'])/T:.0f}% | water sector share {100*(c['water supply']+c['energy']+c['investment'])/T:.0f}%")
    types = {"Desal (new RO)": "Desalination_NewRO", "TWW": "TreatedWW_NewSaudiTWW", "Dams": "SurfaceWater_NewSaudiDam", "Loss reduction": "Domestic_NRW_LossReduction"}
    inv = {}
    for k, pl in types.items():
        inv[k] = 0.0
        for y in YEARS:
            iv = D[y]["Investments"]; inv[k] += iv.loc[iv["Plant"].astype(str) == pl, "Total Cost ($)"].sum() * d[y] / 1e9
    print("  investment by type: " + "  ".join(f"{k} {v:.1f}" for k, v in inv.items()) + f"  | total {sum(inv.values()):.1f}")
    print(f"  solver: {S[2024].get('Solver Status')} | feasibility {S[2024].get('Feasibility Check')} | NPV obj {S[2024].get('NPV Objective (all yrs)'):.3e}")
