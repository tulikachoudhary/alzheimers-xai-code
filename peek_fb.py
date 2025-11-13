# peek_pending.py
import requests, pandas as pd

r = requests.get("http://127.0.0.1:8000/pending_for_review", timeout=10).json()
items = r.get("items", [])
if not items:
    print("[INFO] No pending cases. Run score_all_ad.py first.")
    raise SystemExit

df = pd.DataFrame(items)[["id","ptid","y_hat_prob"]].rename(
    columns={"id":"prediction_id","ptid":"PTID","y_hat_prob":"Prob_AD"}
)

print("\nTop 5 AD-like (highest prob):")
print(df.sort_values("Prob_AD", ascending=False).head(5).to_string(index=False))

print("\nTop 5 CN-like (lowest prob):")
print(df.sort_values("Prob_AD", ascending=True).head(5).to_string(index=False))
