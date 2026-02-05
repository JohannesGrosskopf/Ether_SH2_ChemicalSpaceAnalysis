import os
import io
import base64
import joblib

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors

import umap
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State


BIG_FILE = "enamine_alcohols.csv"   
TESTED_FILE = "substrates.csv"      

UMAP_MODEL_FILE = "umap_model_bigfit.joblib"
BASEMAP_FILE = "umap_coordinates_bigfit.csv"

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


def mol_from_smiles(smi: str):
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


def load_reference_sets():
    cwd = os.getcwd()
    for f in [BIG_FILE, TESTED_FILE]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Could not find '{f}' in {cwd}")

    df_big = pd.read_csv(BIG_FILE)
    df_tested = pd.read_csv(TESTED_FILE)

    for name, df in [("BIG", df_big), ("TESTED", df_tested)]:
        df.columns = [c.strip() for c in df.columns]
        if "id" not in df.columns or "smiles" not in df.columns:
            raise ValueError(f"{name} file must contain columns: id, smiles")
        df["id"] = df["id"].astype(str)
        df["smiles"] = df["smiles"].astype(str)

    df_big["set"] = "big"
    df_tested["set"] = "tested"
    return df_big, df_tested


def train_or_load_bigfit_umap(df_big: pd.DataFrame):
    if os.path.exists(UMAP_MODEL_FILE) and os.path.exists(BASEMAP_FILE):
        umap_model = joblib.load(UMAP_MODEL_FILE)
        df_basemap = pd.read_csv(BASEMAP_FILE)
        df_big_emb = df_basemap[df_basemap["set"] == "big"].copy()
        return umap_model, df_big_emb

    X_big = compute_morgan_bitvectors(df_big["smiles"], nbits=FP_NBITS)
    f_big = compute_fsp3(df_big["smiles"])

    umap_model = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=UMAP_RANDOM_STATE,
        verbose=True,
    )

    emb_big = umap_model.fit_transform(X_big)

    df_big_emb = df_big.copy()
    df_big_emb["UMAP1"] = emb_big[:, 0]
    df_big_emb["UMAP2"] = emb_big[:, 1]
    df_big_emb["Fsp3"] = f_big
    df_big_emb["set"] = "big"

    joblib.dump(umap_model, UMAP_MODEL_FILE)
    return umap_model, df_big_emb


def project_into_big_space(umap_model, df_any: pd.DataFrame, set_name: str) -> pd.DataFrame:
    df_any = df_any.copy()
    df_any.columns = [c.strip() for c in df_any.columns]
    if "id" not in df_any.columns or "smiles" not in df_any.columns:
        raise ValueError(f"{set_name} CSV must contain columns: id, smiles")

    df_any["id"] = df_any["id"].astype(str)
    df_any["smiles"] = df_any["smiles"].astype(str)
    df_any["set"] = set_name

    X = compute_morgan_bitvectors(df_any["smiles"], nbits=FP_NBITS)
    df_any["Fsp3"] = compute_fsp3(df_any["smiles"])

    if not hasattr(umap_model, "transform"):
        raise RuntimeError("UMAP transform() not available. Upgrade umap-learn (>=0.5).")

    emb = umap_model.transform(X)
    df_any["UMAP1"] = emb[:, 0]
    df_any["UMAP2"] = emb[:, 1]
    return df_any


def square_ranges_quantile(df_for_range: pd.DataFrame, pad: float = AXIS_PAD, q_low: float = AXIS_Q_LOW, q_high: float = AXIS_Q_HIGH):
    x = df_for_range["UMAP1"].to_numpy()
    y = df_for_range["UMAP2"].to_numpy()
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return [-1, 1], [-1, 1]

    xmin, xmax = np.quantile(x, [q_low, q_high])
    ymin, ymax = np.quantile(y, [q_low, q_high])

    span = max(xmax - xmin, ymax - ymin)
    if not np.isfinite(span) or span <= 0:
        span = 1.0

    xmid = (xmin + xmax) / 2
    ymid = (ymin + ymax) / 2
    span = span * (1 + pad)
    half = span / 2
    return [xmid - half, xmid + half], [ymid - half, ymid + half]


def make_figure(df_base: pd.DataFrame, df_new: pd.DataFrame | None, view_mode: str, highlight_id: str | None):
    df_big = df_base[df_base["set"] == "big"].copy()
    df_tested = df_base[df_base["set"] == "tested"].copy()
    df_new = df_new.copy() if df_new is not None and len(df_new) else pd.DataFrame()

    all_f = pd.concat([df_big["Fsp3"], df_tested["Fsp3"]], ignore_index=True)
    cmin = float(np.nanmin(all_f.values)) if all_f.notna().any() else 0.0
    cmax = float(np.nanmax(all_f.values)) if all_f.notna().any() else 1.0

    x_range, y_range = square_ranges_quantile(df_big)

    fig = go.Figure()

    def _colorbar_dict():
        return dict(title="Fsp³", thickness=18, len=0.75)

    big_opacity = 0.08 if view_mode == "all" else 0.12

    def add_big(show_colorbar: bool):
        if len(df_big) == 0:
            return
        marker = dict(
            size=4,
            color=df_big["Fsp3"],
            colorscale="Viridis",
            cmin=cmin,
            cmax=cmax,
            opacity=big_opacity,
            line=dict(width=0),
            showscale=show_colorbar,
        )
        if show_colorbar:
            marker["colorbar"] = _colorbar_dict()
        fig.add_trace(
            go.Scattergl(
                x=df_big["UMAP1"],
                y=df_big["UMAP2"],
                mode="markers",
                name="big",
                marker=marker,
                customdata=df_big["id"].to_numpy().reshape(-1, 1),
                hovertemplate="ID: %{customdata[0]}<extra></extra>",
            )
        )

    def add_tested(show_colorbar: bool):
        if len(df_tested) == 0:
            return
        marker = dict(
            size=12,
            color=df_tested["Fsp3"],
            colorscale="Viridis",
            cmin=cmin,
            cmax=cmax,
            opacity=1.0,
            line=dict(width=1.5, color="black"),
            showscale=show_colorbar,
        )
        if show_colorbar:
            marker["colorbar"] = _colorbar_dict()
        fig.add_trace(
            go.Scattergl(
                x=df_tested["UMAP1"],
                y=df_tested["UMAP2"],
                mode="markers",
                name="tested",
                marker=marker,
                customdata=np.stack([df_tested["id"], df_tested["smiles"]], axis=1),
                hovertemplate="ID: %{customdata[0]}<br>SMILES: %{customdata[1]}<extra></extra>",
            )
        )

    def add_new():
        if len(df_new) == 0:
            return
        marker = dict(
            size=13,
            color="rgb(200,200,200)",
            opacity=1.0,
            line=dict(width=2.2, color="rgb(90,90,90)"),
            showscale=False,
        )
        fig.add_trace(
            go.Scattergl(
                x=df_new["UMAP1"],
                y=df_new["UMAP2"],
                mode="markers",
                name="new",
                marker=marker,
                customdata=np.stack([df_new["id"], df_new["smiles"]], axis=1),
                hovertemplate="ID: %{customdata[0]}<br>SMILES: %{customdata[1]}<extra></extra>",
            )
        )

    if view_mode == "big":
        add_big(show_colorbar=True)
    elif view_mode == "tested":
        add_tested(show_colorbar=True)
    elif view_mode == "big+tested":
        add_big(show_colorbar=True)
        add_tested(show_colorbar=False)
    elif view_mode == "new":
        add_new()
    else:
        add_big(show_colorbar=True)
        add_tested(show_colorbar=False)
        add_new()

    if highlight_id:
        hid = str(highlight_id).strip()
        hit = None
        hit_base = df_base[df_base["id"] == hid]
        if len(hit_base):
            hit = hit_base.iloc[0]
        elif len(df_new):
            hit_new = df_new[df_new["id"] == hid]
            if len(hit_new):
                hit = hit_new.iloc[0]

        if hit is not None:
            ok = True
            if view_mode == "big" and hit.get("set") != "big":
                ok = False
            if view_mode == "tested" and hit.get("set") != "tested":
                ok = False
            if view_mode == "new" and hit.get("set") != "new":
                ok = False
            if view_mode == "big+tested" and hit.get("set") not in ("big", "tested"):
                ok = False

            if ok:
                fig.add_trace(
                    go.Scatter(
                        x=[hit["UMAP1"]],
                        y=[hit["UMAP2"]],
                        mode="markers",
                        name="highlight",
                        marker=dict(
                            size=20,
                            symbol="circle-open",
                            line=dict(width=4, color="yellow"),
                            color="rgba(0,0,0,0)",
                        ),
                        hovertemplate=f"Highlighted ID: {hit['id']}<extra></extra>",
                    )
                )

    fig.update_layout(
        width=850,
        height=850,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title="UMAP 1", showgrid=False, zeroline=False, mirror=True, linecolor="black", showticklabels=False, range=x_range),
        yaxis=dict(title="UMAP 2", showgrid=False, zeroline=False, mirror=True, linecolor="black", showticklabels=False, range=y_range, scaleanchor="x", scaleratio=1),
        margin=dict(l=30, r=40, t=50, b=30),
    )
    return fig


def parse_upload(contents: str) -> pd.DataFrame:
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    except Exception:
        df = pd.read_csv(io.StringIO(decoded.decode("latin-1")))
    df.columns = [c.strip() for c in df.columns]
    return df


def main():
    df_big, df_tested = load_reference_sets()

    vendi_tested = vendi_score_from_smiles(df_tested["smiles"])

    umap_model, df_big_emb = train_or_load_bigfit_umap(df_big)
    df_tested_emb = project_into_big_space(umap_model, df_tested, "tested")

    df_base = pd.concat([df_big_emb, df_tested_emb], ignore_index=True)
    df_base.to_csv(BASEMAP_FILE, index=False)

    app = Dash(__name__)
    app.title = "Chemical Space (UMAP bigfit + Vendi 2048)"

    app.layout = html.Div(
        style={"fontFamily": "Arial", "maxWidth": "1200px", "margin": "0 auto"},
        children=[
            html.H2("Chemical Space (fit UMAP on big; project tested + new)"),
            html.Div(
                style={"display": "flex", "gap": "20px", "alignItems": "flex-start"},
                children=[
                    html.Div(
                        style={"width": "340px"},
                        children=[
                            dcc.Dropdown(
                                id="view_mode",
                                options=[
                                    {"label": "Big dataset only", "value": "big"},
                                    {"label": "Tested only", "value": "tested"},
                                    {"label": "Big + tested", "value": "big+tested"},
                                    {"label": "New only", "value": "new"},
                                    {"label": "All", "value": "all"},
                                ],
                                value="all",
                                clearable=False,
                            ),
                            html.Br(),
                            dcc.Input(
                                id="search_id",
                                type="text",
                                placeholder="Search ID (exact) + Enter",
                                debounce=True,
                                style={"width": "100%"},
                            ),
                            html.Br(),
                            html.Br(),
                            dcc.Upload(
                                id="upload_new",
                                children=html.Div(["Drag & drop or ", html.A("select CSV")]),
                                style={
                                    "width": "100%",
                                    "height": "60px",
                                    "lineHeight": "60px",
                                    "borderWidth": "1px",
                                    "borderStyle": "dashed",
                                    "borderRadius": "8px",
                                    "textAlign": "center",
                                },
                                multiple=False,
                            ),
                            html.Div(id="upload_status", style={"whiteSpace": "pre-wrap", "marginTop": "10px", "fontSize": "12px"}),
                            dcc.Store(id="store_new_df"),
                            html.Hr(),
                            html.Div(
                                id="vendi_scores",
                                style={
                                    "whiteSpace": "pre-wrap",
                                    "fontSize": "12px",
                                    "padding": "10px",
                                    "border": "1px solid rgba(0,0,0,0.15)",
                                    "borderRadius": "8px",
                                    "background": "rgba(250,250,250,0.9)",
                                },
                                children=(
                                    f"Vendi (tested; 2048-bit Morgan, Tanimoto): {vendi_tested:.3f}\n"
                                    f"Vendi (tested + new): (upload a file)\n"
                                    f"Axis clip: {int(AXIS_Q_LOW*100)}–{int(AXIS_Q_HIGH*100)}th pct"
                                ),
                            ),
                        ],
                    ),
                    html.Div(
                        style={"flex": "1"},
                        children=[
                            dcc.Graph(id="umap_plot", figure=make_figure(df_base, None, "all", None)),
                            html.Pre(id="selected_info", style={"fontSize": "12px"}),
                        ],
                    ),
                ],
            ),
        ],
    )

    @app.callback(
        Output("store_new_df", "data"),
        Output("upload_status", "children"),
        Input("upload_new", "contents"),
        State("upload_new", "filename"),
        prevent_initial_call=True,
    )
    def handle_upload(contents, filename):
        if not contents:
            return None, "No file uploaded."
        try:
            df_new_raw = parse_upload(contents)
            df_new = project_into_big_space(umap_model, df_new_raw, "new")
            msg = f"Loaded + projected: {filename}\nRows: {len(df_new)}\n"
            return df_new.to_json(date_format="iso", orient="split"), msg
        except Exception as e:
            return None, f"Upload error:\n{e}"

    @app.callback(
        Output("umap_plot", "figure"),
        Output("selected_info", "children"),
        Output("vendi_scores", "children"),
        Input("view_mode", "value"),
        Input("search_id", "value"),
        Input("store_new_df", "data"),
    )
    def update_plot(view_mode, search_id, new_json):
        df_new = pd.read_json(new_json, orient="split") if new_json else None
        fig = make_figure(df_base=df_base, df_new=df_new, view_mode=view_mode, highlight_id=search_id)

        if df_new is not None and len(df_new):
            combined_smiles = pd.concat([df_tested["smiles"], df_new["smiles"]], ignore_index=True)
            vendi_combined = vendi_score_from_smiles(combined_smiles)
            vendi_text = (
                f"Vendi (tested; 2048-bit Morgan, Tanimoto): {vendi_tested:.3f}\n"
                f"Vendi (tested + new; 2048-bit Morgan, Tanimoto): {vendi_combined:.3f}\n"
                f"Axis clip: {int(AXIS_Q_LOW*100)}–{int(AXIS_Q_HIGH*100)}th pct"
            )
        else:
            vendi_text = (
                f"Vendi (tested; 2048-bit Morgan, Tanimoto): {vendi_tested:.3f}\n"
                f"Vendi (tested + new): (upload a file)\n"
                f"Axis clip: {int(AXIS_Q_LOW*100)}–{int(AXIS_Q_HIGH*100)}th pct"
            )

        info = ""
        if search_id:
            sid = str(search_id).strip()
            row = None
            hit_base = df_base[df_base["id"] == sid]
            if len(hit_base):
                row = hit_base.iloc[0].to_dict()
            elif df_new is not None and len(df_new):
                hit_new = df_new[df_new["id"] == sid]
                if len(hit_new):
                    row = hit_new.iloc[0].to_dict()

            if row:
                info = (
                    f"ID: {row.get('id')}\n"
                    f"Set: {row.get('set')}\n"
                    f"SMILES: {row.get('smiles')}\n"
                    f"UMAP1: {row.get('UMAP1')}\n"
                    f"UMAP2: {row.get('UMAP2')}\n"
                    f"Fsp3: {row.get('Fsp3')}\n"
                )
            else:
                info = f"No compound found with ID: {sid}"

        return fig, info, vendi_text

    app.run(debug=False, host="127.0.0.1", port=8050)


if __name__ == "__main__":
    main()
