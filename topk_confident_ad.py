import os, pandas as pd
BASE=r"C:\Users\TulikaChoudhary\Desktop\c502"
PRED=os.path.join(BASE,"FEATURES","mm_predictions.csv")
OUT =os.path.join(BASE,"FEATURES","topK_confident_AD.csv")
K=25  # adjust for how many top subjects you want

pred=pd.read_csv(PRED).dropna(subset=["Prob_AD"])
top=pred.sort_values("Prob_AD", ascending=False).head(K)
top.to_csv(OUT, index=False)
print(f"[DONE] Wrote -> {OUT}")
print(top.head(10).to_string(index=False))
