import os, pandas as pd, numpy as np
BASE=r"C:\Users\TulikaChoudhary\Desktop\c502"
CSV =os.path.join(BASE,"FEATURES","features_with_genetic_rowwise.csv")
PRED=os.path.join(BASE,"FEATURES","mm_predictions.csv")

pred=pd.read_csv(PRED)
df=pd.read_csv(CSV)

dx_col=next((c for c in df.columns if c.upper()=="DX_BL"), None) or next((c for c in df.columns if str(c).upper().startswith("DX")), None)
df=df[[ "PTID", dx_col ]].copy(); df["PTID"]=df["PTID"].astype(str)
pred["PTID"]=pred["PTID"].astype(str)

m=pred.merge(df.drop_duplicates("PTID"), on="PTID", how="left").rename(columns={dx_col:"DX_true"})
m=m[m["DX_true"].notna()].copy()
m["y_true"]=(m["DX_true"].astype(str).str.upper()=="AD").astype(int)
m["y_hat"] =(m["Prob_AD"]>=0.5).astype(int)

tp=int(((m["y_true"]==1)&(m["y_hat"]==1)).sum())
tn=int(((m["y_true"]==0)&(m["y_hat"]==0)).sum())
fp=int(((m["y_true"]==0)&(m["y_hat"]==1)).sum())
fn=int(((m["y_true"]==1)&(m["y_hat"]==0)).sum())

acc=(tp+tn)/max(1,len(m))
print("[ROW CONFUSION] N=",len(m), " ACC=",round(acc,3), " TP=",tp," TN=",tn," FP=",fp," FN=",fn)
