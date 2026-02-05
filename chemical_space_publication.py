import os
import joblib
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors
import umap


BIG_FILE = "enamine_alcohols.csv"
TESTED_FILE = "substrates.csv"
NEW_FILE = None  # optionally set to "new.csv"

UMAP_MODEL_FILE = "umap_model_bigfit.joblib"

OUT_BASEMAP = "chemical_space_basemap.csv"
OUT_NEW = "chemical_space_new.csv"
OUT_REPORT = "chemical_space_report.txt"

FP_RADIUS = 2
FP_NBITS = 2048  

UMAP_N_NEIGHBORS = 25
UMAP_MIN_DIST = 0.15
UMAP_METRIC = "jaccard"
UMAP_RANDOM_STATE = 42

VENDI_EPS = 1e-12

AXIS_Q_LOW = 0.01
AXIS_Q_HIGH = 0.99
AXIS_PAD = 0.05


def mol_from_smiles(smi):
    if smi is None:
        return None
    smi = str(smi).strip()
    if not smi:
        return None
    return Chem.MolFromSmiles(smi)


def compute_fsp3(smiles: pd.Series) -> np.ndarray:
    vals = []
    for s in smiles.astype(str).fillna(""):
        m = mol_from_smiles(s)
        vals.append(Descriptors.FractionCSP3(m) if m is not None else np.nan)
    return np.array(vals, dtype=float)


def compute_morgan_bitvectors(smiles: pd.Series, radius: int = FP_RADIUS, nbits: int = FP_NBITS) -> np.ndarray:
    X = np.zeros((len(smiles), nbits), dtype=np.uint8)
    for i, s in enumerate(smiles.astype(str).fillna("")):
        m = mol_from_smiles(s)
        if m is None:
            continue
        bv = rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)
        arr = np.zeros((nbits,), dtype=np.int8)
        Chem.DataStructs.ConvertToNumpyArray(bv, arr)
        X[i, :] = arr.astype(np.uint8)
    return X


def tanimoto_similarity_matrix_from_bitmatrix(X: np.ndarray) -> np.ndarray:
    X = (X > 0).astype(np.uint8)
    inter = (X @ X.T).astype(np.float64)
    pop = X.sum(axis=1).astype(np.float64)
    union = pop[:, None] + pop[None, :] - inter

    S = np.zeros_like(inter, dtype=np.float64)
    mask = union > 0
    S[mask] = inter[mask] / union[mask]
    np.fill_diagonal(S, 1.0)
    return S


def vendi_score_from_similarity(S: np.ndarray, eps: float = VENDI_EPS) -> float:
    S = np.asarray(S, dtype=np.float64)
    S = 0.5 * (S + S.T)

    tr = float(np.trace(S))
    if not np.isfinite(tr) or tr <= 0:
        return float("nan")

    K = S / tr
    w = np.linalg.eigvalsh(K)
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s <= 0:
        return float("nan")

    p = w / s
    p = p[p > eps]
    return float(np.exp(-(p * np.log(p)).sum()))


def vendi_score_from_smiles(smiles: pd.Series) -> float:
    X = compute_morgan_bitvectors(smiles, radius=FP_RADIUS, nbits=FP_NBITS)
    S = tanimoto_similarity_matrix_from_bitmatrix(X)
    return vendi_score_from_similarity(S)


def robust_square_ranges(df_xy: pd.DataFrame):
    x = df_xy["UMAP1"].to_numpy()
    y = df_xy["UMAP2"].to_numpy()
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return {"x_range": [-1, 1], "y_range": [-1, 1]}

    xmin, xmax = np.quantile(x, [AXIS_Q_LOW, AXIS_Q_HIGH])
    ymin, ymax = np.quantile(y, [AXIS_Q_LOW, AXIS_Q_HIGH])

    span = max(xmax - xmin, ymax - ymin)
    if not np.isfinite(span) or span <= 0:
        span = 1.0

    xmid = (xmin + xmax) / 2
    ymid = (ymin + ymax) / 2
    span = span * (1 + AXIS_PAD)
    half = span / 2

    return {"x_range": [xmid - half, xmid + half], "y_range": [ymid - half, ymid + half]}


def ensure_cols(df: pd.DataFrame):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    if "id" not in df.columns or "smiles" not in df.columns:
        raise ValueError("CSV must contain columns: id, smiles")
    df["id"] = df["id"].astype(str)
    df["smiles"] = df["smiles"].astype(str)
    return df


def main():
    df_big = ensure_cols(pd.read_csv(BIG_FILE))
    df_tested = ensure_cols(pd.read_csv(TESTED_FILE))

    df_big["set"] = "big"
    df_tested["set"] = "tested"

    if os.path.exists(UMAP_MODEL_FILE):
        umap_model = joblib.load(UMAP_MODEL_FILE)
    else:
        X_big = compute_morgan_bitvectors(df_big["smiles"], nbits=FP_NBITS)
        umap_model = umap.UMAP(
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            metric=UMAP_METRIC,
            random_state=UMAP_RANDOM_STATE,
        ).fit(X_big)
        joblib.dump(umap_model, UMAP_MODEL_FILE)

    def project(df: pd.DataFrame, set_name: str):
        X = compute_morgan_bitvectors(df["smiles"], nbits=FP_NBITS)
        emb = umap_model.transform(X)
        out = df.copy()
        out["set"] = set_name
        out["UMAP1"] = emb[:, 0]
        out["UMAP2"] = emb[:, 1]
        out["Fsp3"] = compute_fsp3(out["smiles"])
        return out

    df_big_emb = project(df_big, "big")
    df_tested_emb = project(df_tested, "tested")

    df_base = pd.concat([df_big_emb, df_tested_emb], ignore_index=True)
    df_base.to_csv(OUT_BASEMAP, index=False)

    vendi_tested = vendi_score_from_smiles(df_tested["smiles"])

    vendi_combined = None
    df_new_emb = None
    if NEW_FILE:
        df_new = ensure_cols(pd.read_csv(NEW_FILE))
        df_new_emb = project(df_new, "new")
        df_new_emb.to_csv(OUT_NEW, index=False)

        combined_smiles = pd.concat([df_tested["smiles"], df_new["smiles"]], ignore_index=True)
        vendi_combined = vendi_score_from_smiles(combined_smiles)

    ranges = robust_square_ranges(df_big_emb)

    lines = []
    lines.append(f"Vendi (tested; 2048-bit Morgan, Tanimoto): {vendi_tested:.6f}")
    if vendi_combined is not None:
        lines.append(f"Vendi (tested + new; 2048-bit Morgan, Tanimoto): {vendi_combined:.6f}")
    lines.append(f"Axis clip percentiles: {AXIS_Q_LOW}–{AXIS_Q_HIGH}, pad={AXIS_PAD}")
    lines.append(f"UMAP x_range: {ranges['x_range']}")
    lines.append(f"UMAP y_range: {ranges['y_range']}")
    lines.append(f"Saved basemap: {OUT_BASEMAP}")
    if df_new_emb is not None:
        lines.append(f"Saved new: {OUT_NEW}")

    report = "\n".join(lines)
    print(report)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
