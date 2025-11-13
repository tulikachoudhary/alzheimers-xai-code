import os, re, json, random
from glob import glob
from collections import Counter
import numpy as np, pandas as pd, nibabel as nib

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, classification_report

# --------- CONFIG ----------
BASE = r"C:\Users\TulikaChoudhary\Desktop\c502"
CSV  = os.path.join(BASE, "FEATURES", "features_with_genetic_rowwise.csv")  # <- your merged file
MRI_DIRS = [
    os.path.join(BASE, "ALL_NIFTI"),            # RAW first
    os.path.join(BASE, "ALL_NIFTI_PROCESSED"),
    os.path.join(BASE, "ALL_NFTI_PROCESSED"),
]
MODELS_DIR = os.path.join(BASE, "MODELS")
os.makedirs(MODELS_DIR, exist_ok=True)

VOL = (64, 64, 64)
BATCH_SIZE = 4
EPOCHS = 1
LR = 1e-4
WD = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RNG = 42
print("[INFO] Using device:", DEVICE)

def set_seed(s=RNG):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
set_seed()

# --------- HELPERS ----------
def first_dir(cands):
    for p in cands:
        if os.path.isdir(p): return p
    return cands[0]

def load_norm_nii(path, target=VOL):
    try:
        img = nib.load(path); data = img.get_fdata()
    except Exception:
        data = np.zeros(target, dtype=np.float32)
        return data
    data = np.nan_to_num(data)
    pads = [(0, max(0, target[i]-data.shape[i])) for i in range(3)]
    data = np.pad(data, pads, mode="constant")
    data = data[:target[0], :target[1], :target[2]]
    std = float(np.std(data)); mean = float(np.mean(data))
    if std < 1e-6: data = np.zeros_like(data, dtype=np.float32)
    else:         data = (data - mean) / std
    return data.astype(np.float32)

def find_mri_for_row(row, mri_dir):
    # 1) direct PATH if present
    p = row.get("PATH", None)
    if isinstance(p, str) and os.path.isfile(p): return p
    # 2) try tokens (PTID/RID/SUBJECT) if present
    for key in ("PTID","SUBJECT"):
        v = str(row.get(key,"")).strip()
        if v:
            hits = glob(os.path.join(mri_dir, f"*{v}*.nii*"))
            if hits: return hits[0]
    rid = row.get("RID", None)
    try:
        if pd.notna(rid):
            rid_int = int(float(rid))
            hits = glob(os.path.join(mri_dir, f"*{rid_int}*.nii*"))
            if hits: return hits[0]
    except Exception:
        pass
    # 3) BEST-EFFORT: try a YYYYMMDD if a date-like column exists
    for c in row.index:
        if "DATE" in c.upper():
            s = str(row[c])
            m = re.search(r"(\d{4})[-_/]?(\d{2})[-_/]?(\d{2})", s)
            if m:
                ymd = "".join(m.groups())
                hits = glob(os.path.join(mri_dir, f"*{ymd}*.nii*"))
                if hits: return hits[0]
    return None

def build_feature_matrix(df):
    # numeric only; drop obvious leakage/IDs if they exist
    ban = ["DX","DIAG","PATH","PTID","RID","SUBJECT","SCANDATE","EXAMDATE","VISCODE"]
    X = df.select_dtypes(include=["number","bool"]).copy()
    drop_cols = [c for c in X.columns if any(k in c.upper() for k in ban)]
    X = X.drop(columns=drop_cols, errors="ignore").fillna(0).astype(np.float32)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    df_feat = pd.DataFrame(Xs, columns=X.columns, index=df.index)
    return df_feat, scaler

# --------- DATASET ----------
class MMDS(Dataset):
    def __init__(self, df, feats, label_col, mri_dir, augment=False):
        self.df = df.reset_index(drop=True)
        self.feats = feats.astype(np.float32).reset_index(drop=True)
        self.label_col = label_col
        self.mri_dir = mri_dir
        self.augment = augment

    def __len__(self): return len(self.df)

    def _maybe_aug(self, vol):
        if not self.augment: return vol
        if random.random()<0.5: vol = vol[::-1,:,:].copy()
        if random.random()<0.5: vol = vol[:,::-1,:].copy()
        if random.random()<0.5: vol = vol[:,:,::-1].copy()
        return vol

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = 1.0 if str(row[self.label_col]).upper()=="AD" else 0.0
        nii = find_mri_for_row(row, self.mri_dir)
        if nii is None:
            vol = np.zeros(VOL, dtype=np.float32)
        else:
            vol = load_norm_nii(nii, VOL)
        vol = self._maybe_aug(vol)
        mri_t  = torch.from_numpy(vol).unsqueeze(0)             # (1,D,H,W)
        feat_v = self.feats.iloc[idx].to_numpy(dtype=np.float32)
        clin_t = torch.from_numpy(feat_v).unsqueeze(0)          # (1,F) as 1-step seq
        y_t    = torch.tensor([label], dtype=torch.float32)
        return mri_t, clin_t, y_t

# --------- MODELS ----------
class MRI_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv3d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool3d(2),  # 64->32
            nn.Conv3d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool3d(2), # 32->16
            nn.Conv3d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool3d(2), # 16->8
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64*8*8*8, 256), nn.ReLU(), nn.Dropout(0.3))
    def forward(self, x): return self.head(self.f(x))

class Clin_LSTM(nn.Module):
    def __init__(self, in_dim, hid=128):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hid, batch_first=True)
    def forward(self, x):
        _, (h, _) = self.lstm(x)  # x: (B,1,F)
        return h[-1]               # (B,hid)

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

# --------- TRAIN/EVAL ----------
def train_one(model, loader, crit, opt):
    model.train(); tot=0; n=0
    for mri, clin, y in loader:
        mri, clin, y = mri.to(DEVICE), clin.to(DEVICE), y.to(DEVICE).view(-1)
        logit = model(mri, clin).view(-1)
        loss = crit(logit, y)
        opt.zero_grad(); loss.backward(); opt.step()
        b = y.size(0); tot += loss.item()*b; n += b
    return tot/max(1,n)

def eval_one(model, loader):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for mri, clin, y in loader:
            mri, clin = mri.to(DEVICE), clin.to(DEVICE)
            p = torch.sigmoid(model(mri, clin).view(-1)).cpu().numpy()
            ys.extend(y.view(-1).cpu().numpy().tolist()); ps.extend(p.tolist())
    ys = np.array(ys, dtype=np.int32); ps = np.array(ps, dtype=np.float32)
    preds = (ps>=0.5).astype(int)
    acc = (preds==ys).mean() if len(ys) else 0.0
    try: auc = roc_auc_score(ys, ps)
    except: auc = float("nan")
    p,r,f1,_ = precision_recall_fscore_support(ys, preds, average="binary", zero_division=0)
    return acc, auc, p, r, f1

# --------- MAIN ----------
def main():
    mri_dir = first_dir(MRI_DIRS)
    print("[INFO] MRI dir:", mri_dir)
    df = pd.read_csv(CSV)
    # label column
    dx_col = next((c for c in df.columns if c.upper()=="DX_BL"), None) \
          or next((c for c in df.columns if str(c).upper().startswith("DX")), None)
    if dx_col is None: raise SystemExit("[FATAL] Need DX or DX_BL column in merged file.")
    # limit to AD/MCI/CN just in case
    keep = df[dx_col].astype(str).str.upper().isin({"AD","MCI","CN"})
    df = df.loc[keep].copy()
    y = (df[dx_col].astype(str).str.upper()=="AD").astype(int).values

    feats, scaler = build_feature_matrix(df)

    # diagnostics: how many rows resolve an MRI?
    hits = 0
    for _, row in df.iterrows():
        if find_mri_for_row(row, mri_dir): hits += 1
    print(f"[INFO] Rows with a resolvable MRI: {hits} / {len(df)}")

    # split (row-level, stratified)
    strat = y if len(np.unique(y))>1 else None
    train_idx, val_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=RNG, stratify=strat)

    tr_df, va_df = df.iloc[train_idx].copy(), df.iloc[val_idx].copy()
    tr_X,  va_X  = feats.iloc[train_idx].copy(), feats.iloc[val_idx].copy()

    train_ds = MMDS(tr_df, tr_X, dx_col, mri_dir, augment=True)
    val_ds   = MMDS(va_df, va_X, dx_col, mri_dir, augment=False)
    tr_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    va_loader = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = MMNet(feat_dim=feats.shape[1]).to(DEVICE)

    # class imbalance handling
    pos = int(y.sum()); neg = int(len(y)-pos)
    pos_w = torch.tensor([ (neg/max(1,pos)) if pos>0 else 1.0 ], dtype=torch.float32).to(DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    best_auc=-1.0; best_path=os.path.join(MODELS_DIR,"mm_cnn_lstm_fallback.pt")
    for ep in range(1, EPOCHS+1):
        tl = train_one(model, tr_loader, crit, opt)
        acc, auc, p, r, f1 = eval_one(model, va_loader)
        print(f"Epoch {ep:02d}/{EPOCHS} | loss={tl:.4f} | val_acc={acc:.3f} | val_auc={auc:.3f} | P={p:.3f} R={r:.3f} F1={f1:.3f}")
        if np.nan_to_num(auc) > best_auc:
            best_auc=float(auc); torch.save(model.state_dict(), best_path)
            print(f"  [*] saved -> {best_path} (best_auc={best_auc:.3f})")

    print("[DONE] Best AUC:", round(best_auc,3))
    meta = {"vol": VOL, "feat_dim": feats.shape[1], "dx_col": dx_col}
    with open(os.path.join(MODELS_DIR,"mm_cnn_lstm_fallback.meta.json"),"w") as f: json.dump(meta,f,indent=2)

if __name__=="__main__":
    main()
