# merge_by_row.py
# Row-wise (positional) merge of two CSVs, ignoring IDs.
# Options for unequal lengths: strict (error), clip (use min length), pad (pad with NaN).

import os
import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser(description="Row-wise merge of two CSVs (ignore IDs).")
    ap.add_argument("--left",  required=True, help="Path to LEFT CSV (base/features file).")
    ap.add_argument("--right", required=True, help="Path to RIGHT CSV (genetic file).")
    ap.add_argument("--out",   required=True, help="Path to output CSV.")
    ap.add_argument("--mode",  choices=["strict","clip","pad"], default="strict",
                    help="How to handle unequal row counts: "
                         "'strict' (error), 'clip' (truncate to min), 'pad' (pad shorter with NaN).")
    ap.add_argument("--right-prefix", default="GEN_",
                    help="Prefix to add to RIGHT (genetic) columns to avoid name clashes.")
    ap.add_argument("--low-memory", action="store_true",
                    help="Pass low_memory=True to pandas.read_csv (default False).")
    args = ap.parse_args()

    # Read CSVs
    left  = pd.read_csv(args.left, low_memory=args.low_memory)
    right = pd.read_csv(args.right, low_memory=args.low_memory)

    # Report shapes
    print(f"[INFO] LEFT  rows={len(left):,} cols={left.shape[1]}")
    print(f"[INFO] RIGHT rows={len(right):,} cols={right.shape[1]}")

    # Add prefix to RIGHT columns (except duplicates we’ll handle next)
    right_cols = list(right.columns)
    # If both have unnamed index columns written to CSV, drop them first
    for df in (left, right):
        drop_these = [c for c in df.columns if str(c).lower().startswith("unnamed:")]
        if drop_these:
            df.drop(columns=drop_these, inplace=True, errors="ignore")

    # Recompute in case we dropped cols
    right_cols = list(right.columns)

    # Avoid collisions: if a column name exists on the left, prefix it on the right
    new_right_cols = []
    for c in right_cols:
        new_c = c
        if c in left.columns or c == "":
            new_c = f"{args.right_prefix}{c or 'col'}"
        new_right_cols.append(new_c)
    right.columns = new_right_cols

    # Handle unequal lengths
    nL, nR = len(left), len(right)
    if nL != nR:
        print(f"[WARN] Row counts differ (LEFT={nL}, RIGHT={nR}). mode={args.mode}")
        if args.mode == "strict":
            raise SystemExit("[FATAL] Row counts differ and mode=strict. Use --mode clip or --mode pad.")
        elif args.mode == "clip":
            n = min(nL, nR)
            left  = left.iloc[:n].reset_index(drop=True)
            right = right.iloc[:n].reset_index(drop=True)
            print(f"[INFO] Clipped both to {n} rows.")
        elif args.mode == "pad":
            # pad the shorter with NaNs
            if nL < nR:
                pad_rows = nR - nL
                pad_df = pd.DataFrame(index=range(pad_rows), columns=left.columns)
                left = pd.concat([left, pad_df], ignore_index=True)
                print(f"[INFO] Padded LEFT by {pad_rows} rows → {len(left)}")
            elif nR < nL:
                pad_rows = nL - nR
                pad_df = pd.DataFrame(index=range(pad_rows), columns=right.columns)
                right = pd.concat([right, pad_df], ignore_index=True)
                print(f"[INFO] Padded RIGHT by {pad_rows} rows → {len(right)}")

    # Align indices and concat columns
    left  = left.reset_index(drop=True)
    right = right.reset_index(drop=True)
    merged = pd.concat([left, right], axis=1)

    # Write out
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f"[DONE] Wrote → {args.out}")
    print(f"[REPORT] Final shape: rows={len(merged):,}, cols={merged.shape[1]}")

if __name__ == "__main__":
    main()
