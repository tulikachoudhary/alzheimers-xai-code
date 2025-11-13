import os, pandas as pd, numpy as np, nibabel as nib
import matplotlib.pyplot as plt

BASE=r"C:\Users\TulikaChoudhary\Desktop\c502"
TOPK=os.path.join(BASE,"FEATURES","topK_confident_AD.csv")
OUT =os.path.join(BASE,"VIS")
os.makedirs(OUT, exist_ok=True)

def mid_slices(vol):
    # vol shape (D,H,W) in your pipeline
    d,h,w = vol.shape
    return vol[d//2,:,:], vol[:,h//2,:], vol[:,:,w//2]  # axial, coronal, sagittal

def normalize(img):
    img = np.nan_to_num(img.astype(np.float32))
    p1, p99 = np.percentile(img, 1), np.percentile(img, 99)
    img = np.clip((img - p1) / max(1e-5, (p99 - p1)), 0, 1)
    return img

def main():
    df = pd.read_csv(TOPK)
    print(f"[INFO] Visualizing {len(df)} rows from {TOPK}")
    for i,row in df.iterrows():
        path = str(row.get("MRI_PATH",""))
        ptid = str(row.get("PTID",""))
        prob = float(row.get("Prob_AD", np.nan))
        if not os.path.isfile(path):
            print("[WARN] missing file:", path); continue
        try:
            vol = nib.load(path).get_fdata()
        except Exception as e:
            print("[WARN] failed to load:", path, e); continue

        # keep it simple: percentile-normalize and pick mid slices
        vol = np.nan_to_num(vol.astype(np.float32))
        A,C,S = mid_slices(vol)
        A,C,S = normalize(A), normalize(C), normalize(S)

        fig = plt.figure(figsize=(10,4))
        fig.suptitle(f"PTID={ptid}  Prob_AD={prob:.3f}", fontsize=12)
        ax1 = plt.subplot(1,3,1); ax1.imshow(A.T, cmap="gray", origin="lower"); ax1.set_title("Axial"); ax1.axis("off")
        ax2 = plt.subplot(1,3,2); ax2.imshow(C.T, cmap="gray", origin="lower"); ax2.set_title("Coronal"); ax2.axis("off")
        ax3 = plt.subplot(1,3,3); ax3.imshow(S.T, cmap="gray", origin="lower"); ax3.set_title("Sagittal"); ax3.axis("off")

        bn = f"{i:03d}_{ptid.replace('/','-')}_Prob{prob:.3f}.png"
        out_path = os.path.join(OUT, bn)
        plt.tight_layout(rect=[0,0,1,0.92])
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print("[OK] wrote", out_path)

if __name__ == "__main__":
    main()
