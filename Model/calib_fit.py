import sys, os, pandas as pd, numpy as np
f=sys.argv[1]
x=pd.read_excel(f,sheet_name=None)
s={r.Metric:r.Value for r in x["Summary"].itertuples() if pd.notna(r.Metric)}
print("status",s["Solver Status"],"| area",round(s["Total Irrigated Area (ha)"]),"| CO2 Mt",round(s["Total CO2 Emissions (t)"]/1e6,2))
p=x["Water Production"]; p.columns=["User","Source","BCM","Type"]; p["st"]=p.Source.str.split("_").str[0]
src=p.groupby("st").BCM.sum(); sec=p.groupby("User").BCM.sum()
obs_src={"Desalination":3.15,"Groundwater":13.364509,"SurfaceWater":1.02,"TreatedWW":0.505}
obs_sec={"Agriculture":13.011177,"Domestic":3.616568,"Industry":1.411765}
M=[];O=[]
for k,v in obs_src.items(): print(f"  {k:13s}{src.get(k,0):8.3f} {v:8.3f} {100*(src.get(k,0)-v)/v:+6.1f}%"); M.append(src.get(k,0));O.append(v)
for k,v in obs_sec.items(): print(f"  {k:13s}{sec.get(k,0):8.3f} {v:8.3f} {100*(sec.get(k,0)-v)/v:+6.1f}%"); M.append(sec.get(k,0));O.append(v)
M=np.array(M);O=np.array(O)
print(f"WATER  R2 {1-((M-O)**2).sum()/((O-O.mean())**2).sum():.4f}  MAE% {100*np.mean(np.abs(M-O)/O):.1f}  RMSE% {100*np.sqrt(np.mean(((M-O)/O)**2)):.1f}")
old=pd.read_excel(os.path.join(os.path.dirname(os.path.abspath(__file__)),"Results","2023 Calibration Results.xlsx"),sheet_name="Crop Production").dropna(subset=["Crop"]).set_index("Crop")
dm=x["Domestic Crop Production"].set_index("Crop")["Production (tons)"]; imm=x["Imported Crops"].set_index("Crop")["Imported Quantity (tons)"]
od=old["Observed_Domestic"]; oi=old["Observed_Import"]
od=od.drop(od.index[od.index.isna()],errors="ignore")
c=[i for i in od.index if isinstance(i,str)]
dm=dm.reindex(c); imm=imm.reindex(c); od=od.reindex(c); oi=oi.reindex(c)
for nm,m,o in (("DOM",dm,od),("IMP",imm,oi)):
    r2=1-((m-o)**2).sum()/((o-o.mean())**2).sum()
    err=np.abs(m-o)[o>0]/o[o>0]
    print(f"{nm} total model {m.sum()/1e6:.2f} Mt obs {o.sum()/1e6:.2f} Mt | R2(tons) {r2:.4f} | MAE% {100*err.mean():.1f} | worst {err.idxmax()} {100*err.max():.0f}%")
sm=dm/dm.sum(); so=od/od.sum(); print(f"DOM share MAE {100*np.mean(np.abs(sm-so)):.2f} pp")
