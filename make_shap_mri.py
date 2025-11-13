# make_ig_mri.py — Integrated Gradients overlays for MRI (tabular stream zeroed)

import os, numpy as np, pandas as pd, nibabel as nib
import torch, torch.nn as nn
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
BASE = r"C:\Users\TulikaChoudhary\Desktop\c502"
MODEL_PT   = os.path.join(BASE, "MODELS", "mm_cnn_lstm_fallback.pt")
MERGED_CSV = os.path.join(BASE, "FEATURES", "features_with_genetic_rowwise.csv")
TOPK_CSV   = os.path.join(BASE, "FEATURES", "topK_confident_AD.csv")
OUT_DIR    = os.path.join(BASE, "VIS", "ig_mri_overlay")
os.makedirs(OUT_DIR, exist_ok=True)

VOL = (64, 64, 64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] Using device:", DEVICE)

# --------------- MODEL (same as train) ---------------
class MRI_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv3d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool3d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8*8, 256), nn.ReLU(), nn.Dropout(0.3),
        )
    def forward(self, x):
        return self.head(self.f(x))

class Clin_LSTM(nn.Module):
    def __init__(self, in_dim, hid=128):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hid, batch_first=True)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return h[-1]

class MMNet(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.cnn = MRI_CNN()
        self.rnn = Clin_LSTM(feat_dim, 128)
        self.cls = nn.Sequential(
            nn.Linear(256+128,128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128,1),
        )
    def forward(self, mri, clin):
        fmri = self.cnn(mri)
        fcli = self.rnn(clin)
        return self.cls(torch.cat([fmri, fcli], dim=1))  # (B,1) logits

# MRI-only wrapper returning (B,1)
class MRIOnlyWrapper(nn.Module):
    def __init__(self, mm_model, feat_dim):
        super().__init__()
        self.mm = mm_model
        self.feat_dim = feat_dim
    def forward(self, mri):  # single tensor input
        B = mri.shape[0]
        clin_zeros = torch.zeros((B,1,self.feat_dim), device=mri.device, dtype=mri.dtype)
        return self.mm(mri, clin_zeros)  # (B,1)

# --------------- HELPERS ---------------
def build_feat_dim(csv_path):
    df = pd.read_csv(csv_path)
    dx_col = next((c for c in df.columns if c.upper()=="DX_BL"),
                  next((c for c in df.columns if str(c).upper().startswith("DX")), None))
    ban = ["DX","DIAG","PATH","PTID","RID","SUBJECT","SCANDATE","EXAMDATE","VISCODE"]
    X = df.select_dtypes(include=["number","bool"]).copy()
    drop_cols = [c for c in X.columns if any(k in c.upper() for k in ban)]
    if dx_col and dx_col in X.columns: drop_cols.append(dx_col)
    X = X.drop(columns=list(set(drop_cols)), errors="ignore")
    return X.shape[1]

def load_norm_nii(path, target=VOL):
    try:
        img = nib.load(path); data = img.get_fdata().astype(np.float32)
    except Exception:
        data = np.zeros(target, dtype=np.float32)
    data = np.nan_to_num(data)
    pads = [(0, max(0, target[i]-data.shape[i])) for i in range(3)]
    data = np.pad(data, pads, mode="constant")[:target[0],:target[1],:target[2]]
    m, s = float(np.mean(data)), float(np.std(data))
    return (np.zeros_like(data) if s<1e-6 else (data-m)/s).astype(np.float32)

def mid_slices(v):
    d,h,w = v.shape
    return v[d//2,:,:], v[:,h//2,:], v[:,:,w//2]

def overlay_three(title, vols, cams, out_path):
    (A,C,S) = vols; (a,c,s) = cams
    fig = plt.figure(figsize=(12,4)); fig.suptitle(title, fontsize=20)
    for i,(bg,hm,name) in enumerate([(A,a,"Axial"),(C,c,"Coronal"),(S,s,"Sagittal")],1):
        ax = plt.subplot(1,3,i)
        bg = (bg - np.percentile(bg,1))
        denom = max(1e-6, np.percentile(bg,99))
        bg = np.clip(bg/denom, 0, 1)
        ax.imshow(bg.T, cmap="gray", origin="lower")
        ax.imshow(np.clip(hm,0,1).T, cmap="magma", origin="lower", alpha=0.45)
        ax.set_title(name); ax.axis("off")
    plt.tight_layout(rect=[0,0,1,0.88]); plt.savefig(out_path, dpi=160); plt.close(fig)

def normalize01(x):
    x = x - x.min()
    mx = x.max()
    return x if mx < 1e-6 else x / mx

# --------------- MAIN ---------------
def main():
    feat_dim = build_feat_dim(MERGED_CSV)
    print(f"[INFO] Tabular feat_dim = {feat_dim}")

    mm = MMNet(feat_dim).to(DEVICE)
    mm.load_state_dict(torch.load(MODEL_PT, map_location=DEVICE))
    mm.eval()

    wrapped = MRIOnlyWrapper(mm, feat_dim).to(DEVICE)
    wrapped.eval()

    # target=0 since model returns logits (B,1)
    ig = IntegratedGradients(wrapped)

    rows = pd.read_csv(TOPK_CSV)
    rows = rows[pd.to_numeric(rows.get("Prob_AD", 0), errors="coerce").notna()].reset_index(drop=True)
    if rows.empty:
        raise SystemExit("[FATAL] TOPK CSV has no explainable rows.")

    for i, r in rows.iterrows():
        nii = str(r.get("MRI_PATH","")); pt = str(r.get("PTID","")); prob = float(r.get("Prob_AD", np.nan))
        if not os.path.isfile(nii):
            print("[WARN] missing MRI file:", nii); continue

        vol = load_norm_nii(nii)  # (D,H,W) normalized
        Xt  = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,D,H,W)

        # zero baseline (same shape)
        baseline = torch.zeros_like(Xt, device=DEVICE)

        # compute IG attributions on the logit
        at = ig.attribute(inputs=Xt, baselines=baseline, target=0, n_steps=32, internal_batch_size=4)
        # at shape: (1,1,D,H,W)
        at = at.detach().cpu().numpy()[0,0]  # (D,H,W)
        at = np.nan_to_num(np.abs(at).astype(np.float32))  # magnitude

        # scale to [0,1]
        cam = normalize01(at)

        A,C,S = mid_slices(vol); a,c,s = mid_slices(cam)
        title = f"PTID={pt}  Prob_AD={prob:.3f}" if np.isfinite(prob) else f"PTID={pt}"
        safe_pt = (pt or f"row{i}").replace("/","-").replace("\\","-")
        outp = os.path.join(OUT_DIR, f"ig_overlay_{i:03d}_{safe_pt}.png")
        overlay_three(title, (A,C,S), (a,c,s), outp)
        print("[OK] wrote", outp)

    print("[DONE] IG MRI overlays →", OUT_DIR)

if __name__ == "__main__":
    main()
