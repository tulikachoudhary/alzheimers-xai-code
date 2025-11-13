import os, json, numpy as np, pandas as pd, nibabel as nib
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from glob import glob

BASE = r"C:\Users\TulikaChoudhary\Desktop\c502"
PRED = os.path.join(BASE,"FEATURES","topK_confident_AD.csv")
MODEL= os.path.join(BASE,"MODELS","mm_cnn_lstm_fallback.pt")
OUT  = os.path.join(BASE,"VIS","gradcam")
os.makedirs(OUT, exist_ok=True)

VOL=(64,64,64); DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- same nets as training ---
class MRI_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv3d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool3d(2),     # -> 32^3
            nn.Conv3d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool3d(2),    # -> 16^3
            nn.Conv3d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool3d(2),    # -> 8^3
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64*8*8*8,256), nn.ReLU(), nn.Dropout(0.3))
    def forward(self, x):
        x = self.f(x)
        return self.head(x)

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
        self.cls = nn.Sequential(nn.Linear(256+128,128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128,1))
    def forward(self, mri, clin):
        fmri = self.cnn(mri)
        fcli = self.rnn(clin)
        z = torch.cat([fmri, fcli], dim=1)
        return self.cls(z)

def load_norm_nii(path):
    img=nib.load(path); data=img.get_fdata().astype(np.float32); data=np.nan_to_num(data)
    # center crop/pad to VOL
    pads=[(0,max(0,VOL[i]-data.shape[i])) for i in range(3)]
    data=np.pad(data,pads,mode="constant")[:VOL[0],:VOL[1],:VOL[2]]
    m=float(np.mean(data)); s=float(np.std(data))
    return (np.zeros_like(data) if s<1e-6 else (data-m)/s).astype(np.float32)

def build_features(df):
    from sklearn.preprocessing import StandardScaler
    ban=["DX","DIAG","PATH","PTID","RID","SUBJECT","SCANDATE","EXAMDATE","VISCODE"]
    X=df.select_dtypes(include=["number","bool"]).copy()
    X=X.drop(columns=[c for c in X.columns if any(k in c.upper() for k in ban)], errors="ignore").fillna(0).astype(np.float32)
    sc=StandardScaler(); Xs=sc.fit_transform(X)
    return pd.DataFrame(Xs,columns=X.columns,index=df.index)

def mid_slices(vol):
    d,h,w=vol.shape
    return vol[d//2,:,:], vol[:,h//2,:], vol[:,:,w//2]

def overlay_slice(bg, cam2d, alpha=0.45):
    # normalize both; cam2d in [0,1]
    bg=(bg - np.percentile(bg,1)); bg=np.clip(bg/ max(1e-6, np.percentile(bg,99)),0,1)
    cam2d=np.clip(cam2d,0,1)
    # build overlay using matplotlib
    fig=plt.figure(figsize=(3.6,3.6))
    plt.imshow(bg.T, cmap="gray", origin="lower")
    plt.imshow(cam2d.T, cmap="jet", origin="lower", alpha=alpha)
    plt.axis("off")
    fig.tight_layout(pad=0)
    return fig

def gradcam_volume(model, mri_t):
    """ Grad-CAM on last conv block output (shape Bx64x8x8x8) """
    activ=None; grads=None
    def fwd_hook(module, inp, out):
        nonlocal activ; activ=out.detach()           # B,C,D,H,W
    def bwd_hook(module, gin, gout):
        nonlocal grads; grads=gout[0].detach()

    last_conv = model.cnn.f[6]  # nn.Conv3d(32,64,3,padding=1)
    h1 = last_conv.register_forward_hook(fwd_hook)
    h2 = last_conv.register_backward_hook(bwd_hook)

    try:
        model.zero_grad()
        out = model.cls( torch.cat([ model.cnn.head(model.cnn.f(mri_t)), torch.zeros((1,128), device=mri_t.device) ], dim=1) )
        # We want Grad-CAM for AD (logit). Use the scalar logit
        logit = out.view(-1)[0]
        logit.backward(retain_graph=True)

        # weights = global-average-pool of grads over D/H/W
        w = grads.mean(dim=(2,3,4), keepdim=True)          # (B,C,1,1,1)
        cam = (w * activ).sum(dim=1, keepdim=False)        # (B,D,H,W)
        cam = torch.relu(cam)[0].cpu().numpy()             # (D,H,W), ReLU
        # normalize to [0,1]
        cam = cam - cam.min(); 
        if cam.max() > 1e-6: cam = cam / cam.max()
        return cam
    finally:
        h1.remove(); h2.remove()

def main():
    # load prediction rows
    top = pd.read_csv(PRED)
    if not os.path.exists(MODEL):
        raise SystemExit(f"[FATAL] Missing model: {MODEL}")

    # also need features for shape; we can re-read from the big CSV to get numeric feature count
    FULL=os.path.join(BASE,"FEATURES","features_with_genetic_rowwise.csv")
    full=pd.read_csv(FULL)
    dx=next((c for c in full.columns if c.upper()=="DX_BL"), None) or next((c for c in full.columns if str(c).upper().startswith("DX")), None)
    full=full[full[dx].astype(str).str.upper().isin(["AD","MCI","CN"])].copy()
    X=build_features(full)
    feat_dim=X.shape[1]

    # build & load model (same weights)
    model=MMNet(feat_dim).to(DEVICE)
    model.load_state_dict(torch.load(MODEL, map_location=DEVICE))
    model.eval()

    # for each top row
    for i,row in top.iterrows():
        nii = str(row.get("MRI_PATH",""))
        pt  = str(row.get("PTID",""))
        pr  = float(row.get("Prob_AD", np.nan))
        if not os.path.isfile(nii):
            print("[WARN] missing:", nii); continue

        vol = load_norm_nii(nii)                # (D,H,W)
        mri_t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,D,H,W)

        # Grad-CAM on MRI stream only (clin set to zeros just to pass shape)
        cam = gradcam_volume(model, mri_t)      # (D,H,W) in [0,1]

        # middle slices
        A,C,S = mid_slices(vol)
        a,c,s = mid_slices(cam)

        # save 3-panel overlay
        fig = plt.figure(figsize=(10,4))
        fig.suptitle(f"PTID={pt}  Prob_AD={pr:.3f}", fontsize=12)

        ax1 = plt.subplot(1,3,1); ax1.imshow(A.T, cmap="gray", origin="lower"); ax1.imshow(a.T, cmap="jet", alpha=0.45, origin="lower"); ax1.set_title("Axial CAM"); ax1.axis("off")
        ax2 = plt.subplot(1,3,2); ax2.imshow(C.T, cmap="gray", origin="lower"); ax2.imshow(c.T, cmap="jet", alpha=0.45, origin="lower"); ax2.set_title("Coronal CAM"); ax2.axis("off")
        ax3 = plt.subplot(1,3,3); ax3.imshow(S.T, cmap="gray", origin="lower"); ax3.imshow(s.T, cmap="jet", alpha=0.45, origin="lower"); ax3.set_title("Sagittal CAM"); ax3.axis("off")

        bn = f"cam_{i:03d}_{pt.replace('/','-')}_Prob{pr:.3f}.png"
        outp=os.path.join(OUT,bn)
        plt.tight_layout(rect=[0,0,1,0.92])
        plt.savefig(outp, dpi=150)
        plt.close(fig)
        print("[OK] wrote", outp)

if __name__=="__main__":
    main()
