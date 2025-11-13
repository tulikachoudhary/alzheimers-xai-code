"""
Late-Fusion BiLSTM + (optional) MRI embeddings for AD diagnosis (ADNI)
----------------------------------------------------------------------
- Reads FEATURES/features_with_adnimerge.csv
- Builds longitudinal sequences per SEQ_KEY (RID recommended if PTID is sparse)
- Resolves column name variants (ADAS/education/APOE), deduplicates cols
- BiLSTM encodes sequences; optional concat with MRI embeddings -> MLP head
- Saves best model + validation probabilities

Run:
  python lstm_fusion.py
"""

import os
import random
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================ CONFIG ============================ #
BASE = os.path.dirname(os.path.abspath(__file__))
FEATURES_CSV = os.path.join(BASE, "FEATURES", "features_with_adnimerge.csv")
MRI_EMB_CSV = os.path.join(BASE, "FEATURES", "mri_embeddings.csv")  # optional
OUT_DIR = os.path.join(BASE, "MODELS")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-3
HIDDEN = 64
NUM_LAYERS = 1
BIDIRECTIONAL = True
DROPOUT = 0.2
USE_MRI = False        # set True if MRI_EMB_CSV is available
VAL_SPLIT = 0.2

# Choose sequence key. If PTID coverage is low, use RID.
SEQ_KEY = "RID"        # "PTID" or "RID"

# Label column (binary AD vs non-AD). If absent, derive from DX/DX_bl.
LABEL_COL = "DX_AD"

# Date + subject cols
SCAN_DATE_COL = "ScanDate"   # parsed to datetime
SUBJECT_COL = "subject"      # align MRI embeddings if available

# Canonical feature wishes (we'll resolve to what's present)
CANON_FEATURES = [
    # cognition
    "MMSE", "CDRSB", "ADAS11", "ADAS13", "RAVLT_immediate", "RAVLT_learning",
    # demographics
    "AGE", "EDUCATION_YEARS",
    # genetics
    "APOE4",
]

# ============================ UTILS ============================= #
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _series_from_maybe_dataframe(col):
    """Ensure we have a 1D Series even if duplicate column names created a 2D frame."""
    import pandas as pd
    if isinstance(col, pd.DataFrame):
        return col.iloc[:, 0]
    return col

def simple_impute(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
        s = _series_from_maybe_dataframe(out[c])
        # Try numeric; if fails, leave as object & fillna with 0 later
        sn = pd.to_numeric(s, errors="coerce")
        if sn.dtype.kind in "if":
            med = sn.median()
            sn = sn.fillna(med)
        else:
            sn = sn.fillna(0)
        out[c] = sn
    return out

def resolve_feature_synonyms(df: pd.DataFrame, desired: List[str]) -> List[str]:
    """Map desired names to actual df columns; drop missing safely."""
    synonyms = {
        # cognition
        "MMSE": ["MMSE", "MMSE_total", "MMSE Total Score", "MMSE_bl"],
        "CDRSB": ["CDRSB", "CDRSB_bl", "CDRSB_total"],
        "ADAS11": ["ADAS11", "ADAS11_bl", "ADAS11_SCORE"],
        "ADAS13": ["ADAS13", "ADAS13_bl", "ADAS13_SCORE"],
        "RAVLT_immediate": ["RAVLT_immediate", "RAVLT.immediate", "RAVLT_immediate_bl", "RAVLT Immediate"],
        "RAVLT_learning": ["RAVLT_learning", "RAVLT.learning", "RAVLT_learning_bl", "RAVLT Learning"],
        # demographics
        "AGE": ["AGE", "AGE_bl", "AGE_AT_SCAN"],
        "EDUCATION_YEARS": ["EDUCATION_YEARS", "PTEDUCAT", "Years_Education", "EDUCATION"],
        # genetics
        "APOE4": ["APOE4", "APOE_E4", "APOE-e4", "APOE-ε4"],
        # genotype strings we can convert
        "APOE": ["APOE", "APOE Genotype"],
    }
    resolved = []
    for want in desired:
        if want in df.columns:
            resolved.append(want)
            continue
        cands = synonyms.get(want, [])
        hit = next((c for c in cands if c in df.columns), None)
        if hit is not None:
            resolved.append(hit)
        else:
            if want == "APOE4":
                ghit = next((c for c in synonyms["APOE"] if c in df.columns), None)
                if ghit is not None:
                    resolved.append(ghit)  # placeholder; later convert to APOE4 count
                else:
                    print(f"[WARN] '{want}' not found and no APOE genotype available; dropping.")
            else:
                print(f"[WARN] '{want}' not found; dropping.")
    # de-dup
    final = []
    seen = set()
    for c in resolved:
        if c not in seen:
            seen.add(c); final.append(c)
    print(f"[INFO] Using sequence features (resolved): {final}")
    return final

def derive_columns(df: pd.DataFrame, seq_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Derive standard columns (APOE4, EDUCATION_YEARS) if needed."""
    out = df.copy()
    # APOE4 from genotype strings
    if "APOE4" not in out.columns:
        apoe_text_col = next((c for c in seq_cols if c.upper().startswith("APOE") and c != "APOE4"), None)
        if apoe_text_col and apoe_text_col in out.columns:
            s = _series_from_maybe_dataframe(out[apoe_text_col]).astype(str)
            out["APOE4"] = (
                s.str.extractall(r"(4)").groupby(level=0).size().reindex(out.index, fill_value=0)
            )
            seq_cols = ["APOE4" if c == apoe_text_col else c for c in seq_cols]
    # EDUCATION_YEARS from alternatives
    if "EDUCATION_YEARS" not in out.columns:
        for cand in ["PTEDUCAT", "Years_Education", "EDUCATION"]:
            if cand in out.columns:
                out["EDUCATION_YEARS"] = pd.to_numeric(_series_from_maybe_dataframe(out[cand]), errors="coerce")
                break
    return out, seq_cols

# ======================= DATA PREPARATION ======================= #
def load_feature_table() -> Tuple[pd.DataFrame, List[str]]:
    if not os.path.exists(FEATURES_CSV):
        raise FileNotFoundError(f"Missing file: {FEATURES_CSV}")
    df = pd.read_csv(FEATURES_CSV)

    # Deduplicate any repeated column names from merges
    if df.columns.duplicated().any():
        dup_count = df.columns[df.columns.duplicated()].size
        print(f"[WARN] Detected duplicate column names ({dup_count}); keeping first occurrence.")
        df = df.loc[:, ~df.columns.duplicated()]

    # Dates
    if SCAN_DATE_COL in df.columns:
        df[SCAN_DATE_COL] = pd.to_datetime(_series_from_maybe_dataframe(df[SCAN_DATE_COL]), errors="coerce")

    # Label
    if LABEL_COL not in df.columns:
        guess_col = "DX" if "DX" in df.columns else ("DX_bl" if "DX_bl" in df.columns else None)
        if guess_col is None:
            raise ValueError(f"Cannot find {LABEL_COL} or DX/DX_bl to derive labels.")
        s = _series_from_maybe_dataframe(df[guess_col]).astype(str).str.upper()
        df[LABEL_COL] = s.str.contains("AD").astype(int)

    # Resolve and derive
    seq_resolved = resolve_feature_synonyms(df, CANON_FEATURES)
    df, seq_resolved = derive_columns(df, seq_resolved)

    # Sequence key
    key = SEQ_KEY
    if key not in df.columns:
        if "RID" in df.columns:
            key = "RID"; print("[WARN] SEQ_KEY not found; falling back to RID.")
        elif "PTID" in df.columns:
            key = "PTID"; print("[WARN] SEQ_KEY not found; falling back to PTID.")
        else:
            raise ValueError(f"Sequence key '{SEQ_KEY}' not found and no RID/PTID available.")

    # Keep subset
    keep = [key, SCAN_DATE_COL, LABEL_COL, SUBJECT_COL] + seq_resolved + ["APOE4", "EDUCATION_YEARS"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    # Impute numeric features (exclude meta)
    meta = [key, SCAN_DATE_COL, LABEL_COL, SUBJECT_COL]
    use_cols = [c for c in df.columns if c not in meta]
    df = simple_impute(df, use_cols)

    # Sort
    if SCAN_DATE_COL in df.columns:
        df = df.sort_values([key, SCAN_DATE_COL])

    # Rename to unified key and build final feature list present
    df.rename(columns={key: "SEQ_KEY"}, inplace=True)
    seq_features = [c for c in df.columns if c not in ["SEQ_KEY", SCAN_DATE_COL, LABEL_COL, SUBJECT_COL]]

    return df, seq_features

def train_val_split_keys(df: pd.DataFrame, val_split=VAL_SPLIT) -> Tuple[List[str], List[str]]:
    keys = df["SEQ_KEY"].dropna().unique().tolist()
    rng = np.random.RandomState(SEED)
    rng.shuffle(keys)
    n_val = max(1, int(len(keys) * val_split))
    val = set(keys[:n_val])
    train = [k for k in keys if k not in val]
    return train, list(val)

# ===================== DATASET / COLLATE ======================== #
class SeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, keys: List[str], seq_features: List[str],
                 use_mri: bool=False, mri_map: Optional[Dict[str, np.ndarray]]=None):
        self.df = df[df["SEQ_KEY"].isin(keys)].copy()
        self.seq_features = seq_features
        self.use_mri = use_mri
        self.mri_map = mri_map or {}
        self.groups = []
        for k, g in self.df.groupby("SEQ_KEY"):
            X = g[self.seq_features].to_numpy(dtype=np.float32)
            y = int(g[LABEL_COL].iloc[-1])  # label from last visit
            subj = str(g[SUBJECT_COL].iloc[-1]) if SUBJECT_COL in g.columns else str(k)
            mri_vec = self.mri_map.get(subj, None)
            self.groups.append((k, X, y, subj, mri_vec))

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        k, X, y, subj, mri_vec = self.groups[idx]
        return {
            "key": k,
            "seq": torch.from_numpy(X),
            "y": torch.tensor(y, dtype=torch.long),
            "subj": subj,
            "mri": None if mri_vec is None else torch.from_numpy(mri_vec.astype(np.float32)),
        }

def pad_collate(batch):
    seqs = [b["seq"] for b in batch]
    lens = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    feats = seqs[0].shape[1]
    maxT = int(lens.max())
    padded = torch.zeros(len(batch), maxT, feats, dtype=torch.float32)
    for i, s in enumerate(seqs):
        padded[i, :s.shape[0]] = s
    y = torch.stack([b["y"] for b in batch])
    mri = None
    if batch[0]["mri"] is not None:
        mdim = batch[0]["mri"].shape[0]
        mri = torch.stack([b["mri"] if b["mri"] is not None else torch.zeros(mdim) for b in batch])
    return {"x": padded, "lens": lens, "y": y, "mri": mri}

# ============================ MODEL ============================= #
class BiLSTMEncoder(nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN, num_layers=NUM_LAYERS,
                 bidirectional=BIDIRECTIONAL, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.out_dim = hidden * (2 if bidirectional else 1)

    def forward(self, x, lens):
        packed = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        if self.lstm.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h_last = h_n[-1]
        return h_last

class FusionClassifier(nn.Module):
    def __init__(self, seq_dim, mri_dim=None):
        super().__init__()
        self.use_mri = mri_dim is not None
        in_dim = seq_dim + (mri_dim or 0)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

    def forward(self, h_seq, h_mri=None):
        if self.use_mri and h_mri is not None:
            x = torch.cat([h_seq, h_mri], dim=1)
        else:
            x = h_seq
        return self.mlp(x)

# ========================== TRAIN / EVAL ======================== #
def accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

def train_epoch(model, enc, loader, optimizer, device):
    model.train(); enc.train()
    ce = nn.CrossEntropyLoss()
    total_loss = total_acc = n = 0
    for batch in loader:
        x, lens, y = batch["x"].to(device), batch["lens"].to(device), batch["y"].to(device)
        h_seq = enc(x, lens)
        h_mri = batch["mri"].to(device) if (batch["mri"] is not None) else None
        logits = model(h_seq, h_mri)
        loss = ce(logits, y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item() * y.size(0)
        total_acc  += accuracy(logits.detach(), y) * y.size(0)
        n += y.size(0)
    return total_loss / n, total_acc / n

def eval_epoch(model, enc, loader, device):
    model.eval(); enc.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = total_acc = n = 0
    y_true = []; y_prob = []
    with torch.no_grad():
        for batch in loader:
            x, lens, y = batch["x"].to(device), batch["lens"].to(device), batch["y"].to(device)
            h_seq = enc(x, lens)
            h_mri = batch["mri"].to(device) if (batch["mri"] is not None) else None
            logits = model(h_seq, h_mri)
            loss = ce(logits, y)
            prob_ad = torch.softmax(logits, dim=1)[:, 1]
            total_loss += loss.item() * y.size(0)
            total_acc  += accuracy(logits, y) * y.size(0)
            n += y.size(0)
            y_true.extend(y.cpu().tolist())
            y_prob.extend(prob_ad.cpu().tolist())
    return total_loss / n, total_acc / n, y_true, y_prob

def load_mri_map(path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(path):
        return {}
    emb = pd.read_csv(path)
    subj_col = SUBJECT_COL if SUBJECT_COL in emb.columns else "subject"
    feat_cols = [c for c in emb.columns if c != subj_col]
    m = {}
    for _, r in emb.iterrows():
        m[str(r[subj_col])] = r[feat_cols].to_numpy(dtype=np.float32)
    return m

# =============================== MAIN =========================== #
def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    df, seq_features = load_feature_table()
    n_keys = df["SEQ_KEY"].nunique()
    print(f"[INFO] Rows: {len(df)} | SEQ_KEYs: {n_keys} | Seq features: {len(seq_features)}")
    if n_keys < 30 and SEQ_KEY != "RID" and "RID" in pd.read_csv(FEATURES_CSV, nrows=5).columns:
        print("[HINT] Few unique sequence keys; consider setting SEQ_KEY='RID' at top of file.")

    train_keys, val_keys = train_val_split_keys(df, VAL_SPLIT)

    mri_map = load_mri_map(MRI_EMB_CSV) if USE_MRI else {}
    mri_dim = None if (not USE_MRI or not mri_map) else len(next(iter(mri_map.values())))

    ds_tr = SeqDataset(df, train_keys, seq_features, use_mri=USE_MRI, mri_map=mri_map)
    ds_va = SeqDataset(df, val_keys,   seq_features, use_mri=USE_MRI, mri_map=mri_map)

    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=pad_collate)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate)

    in_dim = len(seq_features)
    enc = BiLSTMEncoder(in_dim).to(device)
    model = FusionClassifier(enc.out_dim, mri_dim=mri_dim).to(device)

    optimizer = torch.optim.Adam(list(enc.parameters()) + list(model.parameters()), lr=LR)

    best_val = 0.0
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, enc, dl_tr, optimizer, device)
        va_loss, va_acc, y_true, y_prob = eval_epoch(model, enc, dl_va, device)
        print(f"[E{epoch:02d}] train loss={tr_loss:.4f} acc={tr_acc:.3f} | val loss={va_loss:.4f} acc={va_acc:.3f}")
        if va_acc > best_val:
            best_val = va_acc
            torch.save({
                "encoder": enc.state_dict(),
                "classifier": model.state_dict(),
                "config": {
                    "USE_MRI": USE_MRI,
                    "HIDDEN": HIDDEN,
                    "BIDIRECTIONAL": BIDIRECTIONAL,
                    "NUM_LAYERS": NUM_LAYERS,
                    "SEQ_FEATURES": seq_features,
                    "SEQ_KEY": SEQ_KEY,
                }
            }, os.path.join(OUT_DIR, "bilstm_fusion_best.pt"))
            pd.DataFrame({"y_true": y_true, "prob_ad": y_prob}).to_csv(
                os.path.join(OUT_DIR, "bilstm_val_predictions.csv"), index=False
            )
            print(f"[INFO] Saved best model (acc={best_val:.3f}) → {OUT_DIR}")

if __name__ == "__main__":
    main()
