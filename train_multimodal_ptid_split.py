import os, re, json, random
from glob import glob
import numpy as np, pandas as pd, nibabel as nib
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, classification_report

BASE = r"C:\Users\TulikaChoudhary\Desktop\c502"
CSV  = os.path.join(BASE, "FEATURES", "features_with_genetic_rowwise.csv")
MRI_DIRS = [os.path.join(BASE,"ALL_NIFTI"), os.path.join(BASE,"ALL_NIFTI_PROCESSED"), os.path.join(BASE,"ALL_NFTI_PROCESSED")]
MODELS_DIR = os.path.join(BASE, "MODELS"); os.makedirs(MODELS_DIR, exist_ok=True)
VOL=(64,64,64); BATCH=4; EPOCHS=6; LR=1e-4; WD=1e-4
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
RNG=42
random.seed(RNG); np.random.seed(RNG); torch.manual_seed(RNG)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(RNG)

def first_dir(xs): 
    for p in xs:
        if os.path.isdir(p): return p
    return xs[0]

def load_norm(path):
    try:
        d = nib.load(path).get_fdata()
    except Exception:
        return np.zeros(VOL, np.float32)
    d = np.nan_to_num(d)
    pads=[(0,max(0,VOL[i]-d.shape[i])) for i in range(3)]
    d=np.pad(d,pads); d=d[:VOL[0],:VOL[1],:VOL[2]]
    s=float(np.std(d)); m=float(np.mean(d))
    return (np.zeros_like(d) if s<1e-6 else (d-m)/s).astype(np.float32)

def find_nii(row, mdir):
    p=row.get("PATH"); 
    if isinstance(p,str) and os.path.isfile(p): return p
    for key in ("PTID","SUBJECT"):
        v=str(row.get(key,"")).strip()
        if v:
            hits=glob(os.path.join(mdir,f"*{v}*.nii*")); 
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

def main():
    mdir=first_dir(MRI_DIRS); print("[INFO] MRI dir:", mdir)
    df=pd.read_csv(CSV)
    dx=next((c for c in df.columns if c.upper()=="DX_BL"), None) or next((c for c in df.columns if str(c).upper().startswith("DX")), None)
    if dx is None: raise SystemExit("[FATAL] Need DX/DX_BL in CSV.")
    df=df[df[dx].astype(str).str.upper().isin({"AD","MCI","CN"})].copy()

    # group key: prefer PTID, else PATH, else row index
    if "PTID" in df.columns and df["PTID"].notna().any():
        groups=df["PTID"].astype(str)
    elif "PATH" in df.columns and df["PATH"].notna().any():
        groups=df["PATH"].astype(str)
    else:
        groups=pd.Series(np.arange(len(df)), index=df.index)

    X, scaler = build_X(df)
    y=(df[dx].astype(str).str.upper()=="AD").astype(int).values

    # group split (80/20)
    gss=GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RNG)
    tr_idx, va_idx = next(gss.split(df.index.values, y, groups=groups.values))
    tr_df, va_df = df.iloc[tr_idx].copy(), df.iloc[va_idx].copy()
    tr_X,  va_X  = X.iloc[tr_idx].copy(),  X.iloc[va_idx].copy()

    # report class counts
    def counts(d): 
        yy=(d[dx].astype(str).str.upper()=="AD").astype(int)
        return int(yy.sum()), int(len(yy)-yy.sum())
    ad_tr, non_tr = counts(tr_df); ad_va, non_va = counts(va_df)
    print(f"[INFO] Train AD={ad_tr}, Non-AD={non_tr} | Val AD={ad_va}, Non-AD={non_va}")

    # datasets / loaders
    tr_ds=DS(tr_df,tr_X,dx,mdir,aug=True); va_ds=DS(va_df,va_X,dx,mdir,aug=False)
    tr_ld=DataLoader(tr_ds,batch_size=BATCH,shuffle=True,num_workers=0)
    va_ld=DataLoader(va_ds,batch_size=BATCH,shuffle=False,num_workers=0)

    model=Net(feat_dim=X.shape[1]).to(DEVICE)
    pos=int((tr_df[dx].astype(str).str.upper()=="AD").sum()); neg=len(tr_df)-pos
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
        return acc,auc,p,r,f1,ys,pred,ps

    best_auc=-1.0; best=os.path.join(MODELS_DIR,"mm_cnn_lstm_ptidsplit.pt")
    for ep in range(1,EPOCHS+1):
        tl=train_one()
        acc,auc,p,r,f1,ys,pred,ps=eval_one()
        print(f"Epoch {ep:02d}/{EPOCHS} | loss={tl:.4f} | val_acc={acc:.3f} | val_auc={auc:.3f} | P={p:.3f} R={r:.3f} F1={f1:.3f}")
        if np.nan_to_num(auc)>best_auc:
            best_auc=float(auc); torch.save(model.state_dict(),best); 
            with open(os.path.join(MODELS_DIR,"mm_ptidsplit_val_report.json"),"w") as f:
                json.dump({"acc":float(acc),"auc":float(auc),"p":float(p),"r":float(r),"f1":float(f1)},f,indent=2)
            print(f"  [*] saved -> {best} (auc={best_auc:.3f})")

if __name__=="__main__":
    main()
