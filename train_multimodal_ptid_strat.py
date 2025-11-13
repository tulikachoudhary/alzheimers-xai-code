import os, re, json, random
from glob import glob
import numpy as np, pandas as pd, nibabel as nib
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support

BASE = r"C:\Users\TulikaChoudhary\Desktop\c502"
CSV  = os.path.join(BASE, "FEATURES", "features_with_genetic_rowwise.csv")
MRI_DIRS = [os.path.join(BASE,"ALL_NIFTI"), os.path.join(BASE,"ALL_NIFTI_PROCESSED"), os.path.join(BASE,"ALL_NFTI_PROCESSED")]
MODELS_DIR = os.path.join(BASE, "MODELS"); os.makedirs(MODELS_DIR, exist_ok=True)
VOL=(64,64,64); BATCH=4; EPOCHS=6; LR=1e-4; WD=1e-4
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
RNG=42
random.seed(RNG); np.random.seed(RNG); torch.manual_seed(RNG)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(RNG)

PTID_RE = re.compile(r"(\d{3}_S_\d{4})")

def first_dir(xs): 
    for p in xs:
        if os.path.isdir(p): return p
    return xs[0]

def canon_ptid(x:str):
    s=str(x) if x is not None else ""
    m=PTID_RE.search(s)
    return m.group(1) if m else s.strip()

def load_norm(path):
    try: d = nib.load(path).get_fdata()
    except: return np.zeros(VOL, np.float32)
    d = np.nan_to_num(d)
    pads=[(0,max(0,VOL[i]-d.shape[i])) for i in range(3)]
    d=np.pad(d,pads); d=d[:VOL[0],:VOL[1],:VOL[2]]
    s=float(np.std(d)); m=float(np.mean(d))
    return (np.zeros_like(d) if s<1e-6 else (d-m)/s).astype(np.float32)

def find_nii(row, mdir):
    p=row.get("PATH")
    if isinstance(p,str) and os.path.isfile(p): return p
    for key in ("PTID","SUBJECT"):
        v=str(row.get(key,"")).strip()
        if v:
            hits=glob(os.path.join(mdir,f"*{v}*.nii*"))
            if hits: return hits[0]
    rid=row.get("RID")
    try:
        if pd.notna(rid):
            hits=glob(os.path.join(mdir,f"*{int(float(rid))}*.nii*"))
            if hits: return hits[0]
    except: pass
    return None

def build_X(df):
    ban=["DX","DIAG","PATH","PTID","RID","SUBJECT","SCANDATE","EXAMDATE","VISCODE"]
    X=df.select_dtypes(include=["number","bool"]).copy()
    X=X.drop(columns=[c for c in X.columns if any(k in c.upper() for k in ban)], errors="ignore").fillna(0).astype(np.float32)
    sc=StandardScaler(); Xs=sc.fit_transform(X)
    return pd.DataFrame(Xs,columns=X.columns,index=df.index), sc

class DS(torch.utils.data.Dataset):
    def __init__(self, df, feats, dx_col, mdir, aug=False):
        self.df=df.reset_index(drop=True); self.feats=feats.reset_index(drop=True)
        self.dx_col=dx_col; self.mdir=mdir; self.aug=aug
    def __len__(self): return len(self.df)
    def _aug(self,v):
        if not self.aug: return v
        import random
        if random.random()<0.5: v=v[::-1,:,:].copy()
        if random.random()<0.5: v=v[:,::-1,:].copy()
        if random.random()<0.5: v=v[:,:,::-1].copy()
        return v
    def __getitem__(self,i):
        r=self.df.iloc[i]
        y=1.0 if str(r[self.dx_col]).upper()=="AD" else 0.0
        nii=find_nii(r,self.mdir); vol=load_norm(nii) if nii else np.zeros(VOL,np.float32)
        vol=self._aug(vol)
        return torch.from_numpy(vol).unsqueeze(0), torch.from_numpy(self.feats.iloc[i].to_numpy(np.float32)).unsqueeze(0), torch.tensor([y],dtype=torch.float32)

class CNN(nn.Module):
    def __init__(self): 
        super().__init__()
        self.f=nn.Sequential(
            nn.Conv3d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool3d(2))
        self.h=nn.Sequential(nn.Flatten(), nn.Linear(64*8*8*8,256), nn.ReLU(), nn.Dropout(0.3))
    def forward(self,x): return self.h(self.f(x))
class LSTM(nn.Module):
    def __init__(self,dim,h=128): 
        super().__init__(); self.l=nn.LSTM(dim,h,batch_first=True)
    def forward(self,x): _,(h,_)=self.l(x); return h[-1]
class Net(nn.Module):
    def __init__(self,feat_dim):
        super().__init__(); self.c=CNN(); self.r=LSTM(feat_dim,128)
        self.cls=nn.Sequential(nn.Linear(256+128,128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128,1))
    def forward(self,mri,clin): return self.cls(torch.cat([self.c(mri), self.r(clin)],1))

def ptid_stratified_row_split(df, dx_col, test_size=0.2, seed=RNG):
    # Build a group key from PTID if available; else derive from PATH; else each row is its own group
    if "PTID" in df.columns and df["PTID"].notna().any():
        g = df["PTID"].astype(str).apply(canon_ptid)
    elif "PATH" in df.columns and df["PATH"].notna().any():
        g = df["PATH"].astype(str).apply(canon_ptid)
    else:
        g = pd.Series([f"ROW_{i}" for i in range(len(df))], index=df.index)

    # Per-group label: 1 if ANY AD in that PTID
    y_group = df.groupby(g)[dx_col].apply(lambda s: int((s.astype(str).str.upper()=="AD").any()))
    groups = y_group.index.to_numpy()
    ylab   = y_group.to_numpy()

    # Stratified split on GROUPS (not rows)
    if len(np.unique(ylab)) < 2:
        # fallback: random split on groups if only one class exists (edge case)
        perm = np.random.RandomState(seed).permutation(len(groups))
        cut = int(len(groups)*(1-test_size))
        tr_groups = set(groups[perm[:cut]])
        va_groups = set(groups[perm[cut:]])
    else:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        idx_tr, idx_va = next(sss.split(groups, ylab))
        tr_groups = set(groups[idx_tr]); va_groups = set(groups[idx_va])

    # Safety: ensure both sets have at least one AD. If not, move one AD group.
    def ensure_both(tr, va):
        def has_pos(gs):
            return int(y_group.loc[list(gs)].sum()) > 0
        if has_pos(tr) and has_pos(va): return tr, va
        # Move one AD group from the richer side to the poorer
        ad_groups = set(y_group[y_group==1].index)
        if not has_pos(tr) and ad_groups & va:
            move = next(iter(ad_groups & va))
            va.remove(move); tr.add(move)
        elif not has_pos(va) and ad_groups & tr:
            move = next(iter(ad_groups & tr))
            tr.remove(move); va.add(move)
        return tr, va

    tr_groups, va_groups = ensure_both(tr_groups, va_groups)

    tr_idx = df.index[g.isin(tr_groups)].to_numpy()
    va_idx = df.index[g.isin(va_groups)].to_numpy()
    return tr_idx, va_idx, g

def main():
    mdir=first_dir(MRI_DIRS); print("[INFO] MRI dir:", mdir)
    df=pd.read_csv(CSV)

    dx=next((c for c in df.columns if c.upper()=="DX_BL"), None) or next((c for c in df.columns if str(c).upper().startswith("DX")), None)
    if dx is None: raise SystemExit("[FATAL] Need DX/DX_BL in CSV.")
    df=df[df[dx].astype(str).str.upper().isin({"AD","MCI","CN"})].copy()

    # Only keep rows where an MRI can be resolved (so we actually use the CNN)
    mask = []
    for _, r in df.iterrows():
        mask.append(find_nii(r, mdir) is not None)
    df = df.loc[mask].copy()
    print(f"[INFO] Rows with resolvable MRI kept: {len(df)}")

    X, _ = build_X(df)

    tr_idx, va_idx, g_series = ptid_stratified_row_split(df, dx, test_size=0.2, seed=RNG)
    tr_df, va_df = df.loc[tr_idx].copy(), df.loc[va_idx].copy()
    tr_X,  va_X  = X.loc[tr_idx].copy(),  X.loc[va_idx].copy()

    def cnts(d):
        y = (d[dx].astype(str).str.upper()=="AD").astype(int)
        return int(y.sum()), int(len(y)-y.sum())
    ad_tr, non_tr = cnts(tr_df); ad_va, non_va = cnts(va_df)
    print(f"[INFO] Train AD={ad_tr}, Non-AD={non_tr} | Val AD={ad_va}, Non-AD={non_va}")

    tr_ds=DS(tr_df,tr_X,dx,mdir,aug=True); va_ds=DS(va_df,va_X,dx,mdir,aug=False)
    tr_ld=DataLoader(tr_ds,batch_size=BATCH,shuffle=True,num_workers=0)
    va_ld=DataLoader(va_ds,batch_size=BATCH,shuffle=False,num_workers=0)

    model=Net(feat_dim=X.shape[1]).to(DEVICE)
    pos=ad_tr; neg=non_tr
    pos_w=torch.tensor([ (neg/max(1,pos)) if pos>0 else 1.0 ], dtype=torch.float32).to(DEVICE)
    crit=nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt=torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    def train_one():
        model.train(); tot=0;n=0
        for mri,clin,yb in tr_ld:
            mri,clin,yb = mri.to(DEVICE),clin.to(DEVICE),yb.to(DEVICE).view(-1)
            logit=model(mri,clin).view(-1); loss=crit(logit,yb)
            opt.zero_grad(); loss.backward(); opt.step()
            b=yb.size(0); tot+=loss.item()*b; n+=b
        return tot/max(1,n)

    def eval_one():
        model.eval(); ys=[]; ps=[]
        with torch.no_grad():
            for mri,clin,yb in va_ld:
                mri,clin = mri.to(DEVICE),clin.to(DEVICE)
                prob=torch.sigmoid(model(mri,clin).view(-1)).cpu().numpy()
                ys.extend(yb.view(-1).numpy().tolist()); ps.extend(prob.tolist())
        ys=np.array(ys,dtype=np.int32); ps=np.array(ps,dtype=np.float32)
        pred=(ps>=0.5).astype(int)
        acc=(pred==ys).mean() if len(ys) else 0.0
        try: auc=roc_auc_score(ys,ps)
        except: auc=float("nan")
        p,r,f1,_=precision_recall_fscore_support(ys,pred,average="binary",zero_division=0)
        return acc,auc,p,r,f1

    best_auc=-1.0; best=os.path.join(MODELS_DIR,"mm_cnn_lstm_ptidstrat.pt")
    for ep in range(1,EPOCHS+1):
        tl=train_one()
        acc,auc,p,r,f1=eval_one()
        print(f"Epoch {ep:02d}/{EPOCHS} | loss={tl:.4f} | val_acc={acc:.3f} | val_auc={auc:.3f} | P={p:.3f} R={r:.3f} F1={f1:.3f}")
        if np.nan_to_num(auc)>best_auc:
            best_auc=float(auc); torch.save(model.state_dict(),best)
            print(f"  [*] saved -> {best} (auc={best_auc:.3f})")

if __name__=="__main__":
    main()
