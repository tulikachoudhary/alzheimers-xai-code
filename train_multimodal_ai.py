"""
train_multimodal_ad_binary.py
JSON-aware + ADNIMERGE labels + PTID canonicalization + SITE+DATE resolver

Fixes:
• Labels come from ADNIMERGE (any AD for a PTID ⇒ AD=1) using canonical PTIDs.
• MRI resolution now also uses SITE+EXAMDATE nearest match (works with names like
  '002_Accelerated_Sagittal_MPRAGE_20220708.nii.gz' that lack PTIDs).
"""

import os, json, random, re
from glob import glob
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, classification_report

# =========================
# CONFIG
# =========================
BASE = r"C:\Users\TulikaChoudhary\Desktop\c502"

MRI_FALLBACKS = [
    os.path.join(BASE, "ALL_NIFTI"),            # RAW first
    os.path.join(BASE, "ALL_NIFTI_PROCESSED"),
    os.path.join(BASE, "ALL_NFTI_PROCESSED"),
]

CLINICAL_PATH   = os.path.join(BASE, "FEATURES", "features_with_genetic.csv")
ADNIMERGE_PATH  = os.path.join(BASE, "ADNIMERGE.csv")
MODELS_DIR      = os.path.join(BASE, "MODELS")
os.makedirs(MODELS_DIR, exist_ok=True)

EPOCHS = 8
BATCH_SIZE = 4
LR = 1e-4
WEIGHT_DECAY = 1e-4
RANDOM_STATE = 42
VOL_SIZE = (64, 64, 64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] Using device:", DEVICE)

def set_seed(seed=RANDOM_STATE):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
set_seed()

# =========================
# HELPERS
# =========================
_PTID_RE  = re.compile(r"(\d{3}_S_\d{4})")       # canonical PTID
_SITE_RE  = re.compile(r"^(\d{3})_")             # leading site code in filename
_DATE_RE8 = re.compile(r"(\d{8})")               # YYYYMMDD in filename

def canon_ptid(x: str) -> str:
    s = str(x) if x is not None else ""
    m = _PTID_RE.search(s)
    return m.group(1) if m else s.strip()

def first_existing_folder(candidates):
    for p in candidates:
        if os.path.isdir(p):
            return p
    return candidates[0]

def load_and_normalize_mri(nii_path, target_shape=VOL_SIZE):
    img = nib.load(nii_path)
    data = img.get_fdata()
    data = np.nan_to_num(data)
    pads = [(0, max(0, target_shape[i] - data.shape[i])) for i in range(3)]
    data = np.pad(data, pads, mode="constant")
    data = data[:target_shape[0], :target_shape[1], :target_shape[2]]
    std = float(np.std(data)); mean = float(np.mean(data))
    if std < 1e-6: data = np.zeros_like(data, dtype=np.float32)
    else:         data = (data - mean) / std
    return data.astype(np.float32)

def _pair_json_to_nii(stem):
    if os.path.exists(stem + ".nii.gz"): return stem + ".nii.gz"
    if os.path.exists(stem + ".nii"):    return stem + ".nii"
    return None

def build_json_index(mri_dir):
    """PTID -> [nii] using JSON sidecars (if any)."""
    json_paths = glob(os.path.join(mri_dir, "**", "*.json"), recursive=True)
    index = defaultdict(list)
    for jp in json_paths:
        try:
            with open(jp, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        joined = " ".join(str(d.get(k, "")) for k in (
            "PatientName","PatientID","SeriesDescription","ProtocolName","AcquisitionDate","AcquisitionDateTime"
        ))
        m = _PTID_RE.search(joined)
        if not m: continue
        ptid = m.group(1).strip()
        nii = _pair_json_to_nii(os.path.splitext(jp)[0])
        if nii: index[ptid].append(nii)
    return index

def to_date_any(s):
    try:
        d = pd.to_datetime(s, errors="coerce")
        return d.date() if pd.notna(d) else None
    except Exception:
        return None

def to_date_ymd(s8):
    try:
        return datetime.strptime(s8, "%Y%m%d").date()
    except Exception:
        return None

def build_site_date_index(mri_dir):
    """
    Build index: (site 'NNN', date) -> [paths], choosing largest file first per (site,date).
    Works for filenames like '002_Accelerated_Sagittal_MPRAGE_20220708.nii.gz'.
    """
    index = {}
    nii_paths = [str(p) for p in Path(mri_dir).rglob("*.nii")] + [str(p) for p in Path(mri_dir).rglob("*.nii.gz")]
    for p in nii_paths:
        name = Path(p).name
        msite = _SITE_RE.match(name)
        mdate = _DATE_RE8.search(name)
        if not msite or not mdate:   # skip files that don't have both
            continue
        site = msite.group(1)
        dt   = to_date_ymd(mdate.group(1))
        if not dt: continue
        index.setdefault((site, dt), []).append(p)
    # sort desc by size per key
    for k, paths in index.items():
        uniq = {}
        for path in paths:
            try: sz = os.path.getsize(path)
            except Exception: sz = 0
            uniq[path] = sz
        index[k] = [p for p,_ in sorted(uniq.items(), key=lambda kv: kv[1], reverse=True)]
    return index

def nearest_site_date_path(site, exam_dates, site_date_index, windows=(0,7,30,90,365,1095)):
    """Find best NIfTI path for given site and any of the subject's exam dates within expanding windows."""
    if not exam_dates: return None
    # Precollect available dates for this site
    site_dates = sorted({dt for (s, dt) in site_date_index.keys() if s == site})
    if not site_dates: return None
    for W in windows:
        best_dt = None
        best_gap = None
        for ed in exam_dates:
            for dt in site_dates:
                gap = abs((dt - ed).days)
                if gap <= W and (best_gap is None or gap < best_gap):
                    best_dt, best_gap = dt, gap
        if best_dt is not None:
            return site_date_index[(site, best_dt)][0]  # largest file on that day
    return None

def build_feature_matrix(df):
    block_substrings = [
        "PTID","PTID_CANON","RID","SUBJECT","PATH","__SOURCE_FILE","KEY","USED",
        "DX","DIAG",
        "PROB_",
        "SCANDATE","EXAMDATE",
        "VISCODE","VISCODE2","VISCODE_CANON","VISCODE_PARSED","VISCODE_ORIG",
        "RID_ADNI","VISCODE_ADNI","EXAMDATE_ADNI","DX_ADNI","DX_BL_ADNI",
    ]
    upper_blocks = [s.upper() for s in block_substrings]
    drop_cols = [c for c in df.columns if any(k in c.upper() for k in upper_blocks)]
    X = df.drop(columns=drop_cols, errors="ignore").select_dtypes(include=["number","bool"]).copy()
    X = X.fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    df[X.columns] = X_scaled
    return df, list(X.columns)

# =========================
# DATASET
# =========================
class ADNI_MultimodalDataset(Dataset):
    """ returns (MRI_volume, clinical_seq(1,F), label_float) """
    def __init__(self, df, mri_dir, feature_cols, json_index, ptid2dates, site_date_index, label_col="label_bin", augment=False):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.mri_dir = mri_dir
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.augment = augment
        self.json_index = json_index
        self.ptid2dates = ptid2dates          # {PTID_CANON: [dates]}
        self.site_date_index = site_date_index

    def __len__(self): return len(self.df)

    def _resolve_mri_path(self, row):
        # 1) direct PATH
        p = row.get("PATH", None)
        if isinstance(p, str) and os.path.isfile(p):
            return p

        # Prefer canonical PTID
        ptid = (str(row.get("PTID_CANON","")).strip()
                or str(row.get("PTID","")).strip())

        # 2) PTID filename match
        if ptid:
            hits = glob(os.path.join(self.mri_dir, f"*{ptid}*.nii*"))
            if hits: return hits[0]

        # 3) RID filename match
        rid = row.get("RID", None)
        try:
            rid_int = int(float(rid)) if pd.notna(rid) else None
        except Exception:
            rid_int = None
        if rid_int is not None:
            hits = glob(os.path.join(self.mri_dir, f"*{rid_int}*.nii*"))
            if hits: return hits[0]

        # 4) SUBJECT filename match
        subj = str(row.get("SUBJECT","")).strip()
        if subj:
            hits = glob(os.path.join(self.mri_dir, f"*{subj}*.nii*"))
            if hits: return hits[0]

        # 5) JSON PTID index
        if ptid and ptid in self.json_index and len(self.json_index[ptid]) > 0:
            return self.json_index[ptid][0]

        # 6) SITE+DATE nearest (ADNIMERGE exam dates)
        # site = first 3 digits of canonical PTID
        m = re.match(r"^(\d{3})_S_", ptid)
        if m:
            site = m.group(1)
            dates = self.ptid2dates.get(ptid, [])
            hit = nearest_site_date_path(site, dates, self.site_date_index)
            if hit:
                return hit

        return None

    def _maybe_augment(self, vol):
        if self.augment:
            if random.random() < 0.5: vol = vol[::-1, :, :].copy()
            if random.random() < 0.5: vol = vol[:, ::-1, :].copy()
            if random.random() < 0.5: vol = vol[:, :, ::-1].copy()
        return vol

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = float(row[self.label_col])  # 0.0 or 1.0
        nii = self._resolve_mri_path(row)
        if nii is None:
            mri = np.zeros(VOL_SIZE, dtype=np.float32)
        else:
            try:    mri = load_and_normalize_mri(nii, VOL_SIZE)
            except: mri = np.zeros(VOL_SIZE, dtype=np.float32)
        mri = self._maybe_augment(mri)
        feat = row[self.feature_cols].to_numpy(dtype=np.float32)
        clin_seq = np.expand_dims(feat, axis=0)              # (1,F)
        mri_t = torch.from_numpy(mri).unsqueeze(0)           # (1,D,H,W)
        clin_t = torch.from_numpy(clin_seq)                  # (1,F)
        lab_t  = torch.tensor(label, dtype=torch.float32).view(1)
        return mri_t, clin_t, lab_t

# =========================
# MODELS (same as before)
# =========================
class MRI_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool3d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8*8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
    def forward(self, x):
        x = self.features(x)
        return self.fc(x)

class ClinGen_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return h[-1]

class MultiModalNet(nn.Module):
    def __init__(self, cnn, lstm):
        super().__init__()
        self.cnn = cnn
        self.lstm = lstm
        self.classifier = nn.Sequential(
            nn.Linear(256+128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
    def forward(self, mri, clin):
        fmri = self.cnn(mri)
        fclin = self.lstm(clin)
        fused = torch.cat([fmri, fclin], dim=1)
        return self.classifier(fused)

# =========================
# TRAIN / EVAL
# =========================
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, total = 0.0, 0
    for mri, clin, y in loader:
        mri, clin, y = mri.to(DEVICE), clin.to(DEVICE), y.to(DEVICE)
        logits = model(mri, clin).view(-1)
        loss = criterion(logits, y.view(-1))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        b = y.size(0); total_loss += loss.item() * b; total += b
    return total_loss / max(1, total)

def eval_epoch(model, loader):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for mri, clin, y in loader:
            mri, clin = mri.to(DEVICE), clin.to(DEVICE)
            logits = model(mri, clin).view(-1)
            prob = torch.sigmoid(logits).cpu().numpy()
            ys.extend(y.view(-1).cpu().numpy().tolist())
            ps.extend(prob.tolist())
    ys = np.array(ys, dtype=np.int32)
    ps = np.array(ps, dtype=np.float32)
    preds = (ps >= 0.5).astype(np.int32)
    if len(np.unique(ys)) < 2:
        acc = (preds == ys).mean() if len(ys) else 0.0
        return acc, float("nan"), 0.0, 0.0, 0.0
    acc = accuracy_score(ys, preds)
    try:    auc = roc_auc_score(ys, ps)
    except: auc = float("nan")
    p, r, f1, _ = precision_recall_fscore_support(ys, preds, average="binary", zero_division=0)
    return acc, auc, p, r, f1

# =========================
# MAIN
# =========================
def main():
    # Choose MRI folder + indices
    mri_folder = first_existing_folder(MRI_FALLBACKS)
    print("[INFO] MRI folder:", mri_folder)
    print("[1/9] Building JSON index (PTID → NIfTI)…")
    json_index = build_json_index(mri_folder)
    print(f"[INFO] JSON-indexed PTIDs: {len(json_index)}")

    print("[2/9] Load CSV:", CLINICAL_PATH)
    df = pd.read_csv(CLINICAL_PATH)

    # Detect diagnosis column (for filtering only; labels from ADNIMERGE)
    dx_col = next((c for c in df.columns if c.upper()=="DX_BL"), None)
    if dx_col is None:
        dx_col = next((c for c in df.columns if str(c).upper().startswith("DX")), None)
    if dx_col is None: raise RuntimeError("No diagnosis column found.")
    print(f"[INFO] Using diagnosis column (for filtering): {dx_col}")

    # Canonicalize PTID in features CSV
    if "PTID" not in df.columns:
        raise RuntimeError("features_with_genetic.csv must include a PTID column.")
    df["PTID"] = df["PTID"].astype(str)
    df["PTID_CANON"] = df["PTID"].apply(canon_ptid)

    # ADNIMERGE-powered patient-level labels (any-AD ⇒ AD=1) + PTID→dates
    am = pd.read_csv(ADNIMERGE_PATH, low_memory=False)
    am_ptid = next((c for c in am.columns if c.upper()=="PTID"), None)
    am_dx   = next((c for c in am.columns if c.upper().startswith("DX")), None)
    am_exam = next((c for c in am.columns if "EXAM" in c.upper() and "DATE" in c.upper()), None)
    if am_ptid is None or am_dx is None or am_exam is None:
        raise RuntimeError("ADNIMERGE must have PTID, DX*, and EXAMDATE columns.")
    am["_PTID"] = am[am_ptid].astype(str).apply(canon_ptid)
    am["_DX"]   = am[am_dx].astype(str).str.upper().str.strip()
    am["_EXAM"] = am[am_exam].apply(to_date_any)
    am = am.dropna(subset=["_EXAM"])
    _AD_NAMES = {"AD","ALZHEIMER'S DISEASE","ALZHEIMER DISEASE","ALZHEIMERS DISEASE"}
    ptids_ad = set(am.loc[am["_DX"].isin(_AD_NAMES), "_PTID"])

    # Mapping PTID->list of EXAM dates
    ptid2dates = am.groupby("_PTID")["_EXAM"].apply(lambda s: sorted(list({d for d in s if d is not None}))).to_dict()

    # Keep relevant rows but label from ADNIMERGE membership (via PTID_CANON)
    _valid_labels = {"AD","MCI","CN","EMCI","LMCI"}
    df[dx_col] = df[dx_col].astype(str).str.upper().str.strip()
    df = df[df[dx_col].isin(_valid_labels)].copy()
    df["label_bin"] = df["PTID_CANON"].isin(ptids_ad).astype(np.float32)

    # Remap PATH base to active MRI folder
    if "PATH" in df.columns:
        df["PATH"] = df["PATH"].astype(str)
        old_bases = [
            r"C:\Users\TulikaChoudhary\Desktop\c502\PREPROC",
            r"C:\Users\TulikaChoudhary\Desktop\c502\ALL_NIFTI",
            r"C:\Users\TulikaChoudhary\Desktop\c502\ALL_NFTI_PROCESSED",
            r"C:\Users\TulikaChoudhary\Desktop\c502\ALL_NIFTI_PROCESSED",
        ]
        for ob in old_bases:
            df["PATH"] = df["PATH"].str.replace(ob, mri_folder, regex=False)

    # Build SITE+DATE index from filenames
    site_date_index = build_site_date_index(mri_folder)

    # Keep rows with resolvable MRI (now includes JSON + SITE+DATE)
    print("[3/9] Resolving MRI files…")
    def row_resolvable(row):
        # 1) direct PATH
        p = row.get("PATH", None)
        if isinstance(p, str) and os.path.isfile(p): return True
        # 2) PTID/RID/SUBJECT tokens
        ptid = (str(row.get("PTID_CANON","")).strip() or str(row.get("PTID","")).strip())
        if ptid and glob(os.path.join(mri_folder, f"*{ptid}*.nii*")): return True
        rid = row.get("RID", None)
        try:
            rid_int = int(float(rid)) if pd.notna(rid) else None
        except Exception:
            rid_int = None
        if rid_int is not None and glob(os.path.join(mri_folder, f"*{rid_int}*.nii*")): return True
        subj = str(row.get("SUBJECT","")).strip()
        if subj and glob(os.path.join(mri_folder, f"*{subj}*.nii*")): return True
        # 3) JSON
        if ptid in json_index and len(json_index[ptid]) > 0: return True
        # 4) SITE+DATE
        m = re.match(r"^(\d{3})_S_", ptid)
        if m:
            site = m.group(1)
            dates = ptid2dates.get(ptid, [])
            hit = nearest_site_date_path(site, dates, site_date_index)
            if hit: return True
        return False

    keep = df.apply(row_resolvable, axis=1).to_numpy(dtype=bool)
    print(f"[INFO] Found MRIs for {keep.sum()} / {len(df)} rows.")
    df = df.loc[keep].copy()
    if len(df) == 0:
        raise RuntimeError("No rows with resolvable MRIs. Check MRI folder/JSON/PATHs/filenames.")

    # DEBUG
    uniq_ptids = df["PTID_CANON"].dropna().unique().tolist()
    print(f"[DEBUG] PTIDs (post-mask) sample: {uniq_ptids[:10]}")

    # Subject counts by PTID_CANON
    subj_labels = df.groupby("PTID_CANON")["label_bin"].max().astype(int)
    n_ad_subj = int((subj_labels == 1).sum())
    n_nonad_subj = int((subj_labels == 0).sum())
    print(f"[INFO] Subject-level counts after MRI filter → AD: {n_ad_subj}, Non-AD: {n_nonad_subj}")
    if n_ad_subj < 2:
        print("[FATAL] Not enough AD subjects with resolvable MRIs (need ≥2).")
        print("        This trainer now checks PATH/filename/JSON and SITE+DATE. If still 1,")
        print("        your RAW set may lack AD scans for additional PTIDs at this time.")
        return

    # Build numeric feature matrix
    print("[4/9] Building feature matrix…")
    df, feature_cols = build_feature_matrix(df)
    print(f"[INFO] Using {len(feature_cols)} tabular features.")

    # Patient-level split using PTID_CANON
    print("[5/9] Patient-level split (binary)…")
    g = df.groupby("PTID_CANON")["label_bin"].max().astype(int)
    ptids = g.index.astype(str).tolist()
    ptid_labels = g.values.astype(int).tolist()

    def build_split(ptids, ptid_labels, test_size=0.2, seed=RANDOM_STATE):
        n_pos = int(sum(ptid_labels))
        n_neg = len(ptid_labels) - n_pos
        if n_pos >= 2 and n_neg >= 2:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            tr_idx, va_idx = next(sss.split(ptids, ptid_labels))
            mode = "stratified"
        else:
            X_dummy = np.zeros(len(ptids)); y_dummy = np.zeros(len(ptids))
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            tr_idx, va_idx = next(gss.split(X_dummy, y_dummy, groups=ptids))
            mode = "group-only"
        tr = set(np.array(ptids)[tr_idx]); va = set(np.array(ptids)[va_idx])
        return tr, va, mode

    train_ptids, val_ptids, mode = build_split(ptids, ptid_labels)

    # Ensure both classes in VAL
    lab_map = dict(zip(ptids, ptid_labels))
    def ensure_val_has_both(train_ptids, val_ptids):
        vlabs = [lab_map[p] for p in val_ptids]
        if 0 in vlabs and 1 in vlabs: return train_ptids, val_ptids, False
        need = 1 if 1 not in vlabs else 0
        for p in list(train_ptids):
            if lab_map[p] == need:
                train_ptids.remove(p); val_ptids.add(p)
                return train_ptids, val_ptids, True
        return train_ptids, val_ptids, False

    train_ptids, val_ptids, adjusted = ensure_val_has_both(train_ptids, val_ptids)

    train_df = df[df["PTID_CANON"].astype(str).isin(train_ptids)].copy()
    val_df   = df[df["PTID_CANON"].astype(str).isin(val_ptids)].copy()

    def cls_counts(x):
        c_ad = int((x["label_bin"]==1).sum())
        return {"AD": c_ad, "Non-AD": len(x)-c_ad}

    print(f"[INFO] Split mode: {mode} | adjusted_for_val_class_balance={adjusted}")
    print(f"[INFO] Train size: {len(train_df)} {cls_counts(train_df)} | Val size: {len(val_df)} {cls_counts(val_df)}")

    # Datasets/loaders
    print("[6/9] Building datasets/loaders…")
    train_ds = ADNI_MultimodalDataset(train_df, mri_folder, feature_cols, json_index, ptid2dates, site_date_index, label_col="label_bin", augment=True)
    val_ds   = ADNI_MultimodalDataset(val_df,   mri_folder, feature_cols, json_index, ptid2dates, site_date_index, label_col="label_bin", augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model / loss / opt
    print("[7/9] Building model…")
    cnn  = MRI_CNN()
    lstm = ClinGen_LSTM(input_dim=len(feature_cols), hidden_dim=128)
    model = MultiModalNet(cnn, lstm).to(DEVICE)

    n_pos_tr = int((train_df["label_bin"] == 1).sum())
    n_neg_tr = int((train_df["label_bin"] == 0).sum())
    pos_weight = torch.tensor([ (n_neg_tr / max(1, n_pos_tr)) if n_pos_tr > 0 else 1.0 ],
                              dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Train
    print("[8/9] Training…")
    best_auc = -1.0
    best_path = os.path.join(MODELS_DIR, "cnn_lstm_binary.pt")
    for epoch in range(1, EPOCHS+1):
        tr_loss = train_epoch(model, train_loader, criterion, optimizer)
        acc, auc, p, r, f1 = eval_epoch(model, val_loader)
        print(f"Epoch {epoch:02d}/{EPOCHS} | train_loss={tr_loss:.4f} | val_acc={acc:.3f} | val_auc={auc:.3f} | P={p:.3f} R={r:.3f} F1={f1:.3f}")
        if np.nan_to_num(auc) > best_auc:
            best_auc = float(auc)
            torch.save(model.state_dict(), best_path)
            print(f"  [*] Saved best model → {best_path} (val_auc={best_auc:.3f})")

    print("[9/9] Done. Best val_auc:", round(best_auc, 3))
    print("Model path:", best_path)

    meta = {"feature_cols": feature_cols, "label_mapping": {"AD":1, "NON_AD":0}, "vol_size": VOL_SIZE}
    with open(os.path.join(MODELS_DIR, "cnn_lstm_binary.meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved meta →", os.path.join(MODELS_DIR, "cnn_lstm_binary.meta.json"))

    # Final detailed report on val
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for mri, clin, y in val_loader:
            mri, clin = mri.to(DEVICE), clin.to(DEVICE)
            prob = torch.sigmoid(model(mri, clin).view(-1)).cpu().numpy()
            ys.extend(y.view(-1).cpu().numpy().tolist())
            ps.extend(prob.tolist())
    ys = np.array(ys, dtype=np.int32); ps = np.array(ps, dtype=np.float32); preds = (ps >= 0.5).astype(int)
    print("\n[VAL] Confusion/Report")
    print(" Accuracy:", accuracy_score(ys, preds))
    try: print(" AUROC:", roc_auc_score(ys, ps))
    except: print(" AUROC: nan")
    print(classification_report(ys, preds, digits=3))

if __name__ == "__main__":
    main()
