
# -*- coding: utf-8 -*-
"""
Random Search for Tracking Optimization 


Pairs GT/TRK CSV files by base name (ignoring _gt/_trk suffixes).
Splits data into train/validation sets (reproducible).
Searches for a single global combination of parameters using RANDOM SEARCH (faster than grid search), minimizing the pooled RMSE across the entire TRAIN set.
Evaluates on the VALID set, exports CSVs, and displays GT / raw TRK / optimized TRK.

"""
import os
import json, random
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, medfilt
import matplotlib.pyplot as plt

try:
    from mplsoccer import Pitch
    USE_MPLSOCCER = True
except Exception:
    USE_MPLSOCCER = False

# =============================
#          PARAMÈTRES
# =============================

BASE_DIR = Path(__file__).resolve().parent

# folders
GT_DIR  = BASE_DIR / "datasets" / "gt_data"
TRK_DIR = BASE_DIR / "datasets" / "trck_data"
OUT_DIR = BASE_DIR / "outputs" / "random_search"
PLOTS_POS_DIR = OUT_DIR / "plots" / "pos"
PLOTS_AS_DIR = OUT_DIR / "plots" / "AS"
# create files if not existing
OUT_DIR.mkdir(exist_ok=True)
PLOTS_POS_DIR.mkdir(exist_ok=True)
PLOTS_AS_DIR.mkdir(exist_ok=True)

TRAIN_RATIO = 0.6       # 60% train / 40% valid
SEED = 42               # reproductible

# ----------------- random search -----------------
SG_WINDOW_CHOICES = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 
                     35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 
                     67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99]
# order of polynomial
SG_POLY_CHOICES   = [2, 3, 4, 5]
# median (0 = disabled)
MEDIAN_CHOICES    = [0, 3, 5, 7]
# Biais x/y 
BIAS_RANGE        = (-0.50, 0.50)   # ex: +/- 25 cm if coordinates meters

N_RANDOM_ITERS    = 100000             # number of random combinations to try
PLOT_N_VALID      = 10              # number of valid files to plot

# names of expected columns (adapt here if needed)
GT_IMAGE_COL  = "image_id"
GT_X_COL      = "x_bottom_middle"
GT_Y_COL      = "y_bottom_middle"
TRK_IMAGE_COL = "image_id"
TRK_X_COL     = "x"
TRK_Y_COL     = "y"

# Pitch (mètres)
L, W = 105.0, 68.0

# =============================
#           fonctions
# =============================

def rmse_xy(x_true, y_true, x_pred, y_pred):
    dx = x_pred - x_true
    dy = y_pred - y_true
    return float(np.sqrt(np.mean(dx*dx + dy*dy)))

def apply_smoothing(x_raw, y_raw, median_window, sg_window, sg_poly, bias_x, bias_y):
    """
    Lissage = (filtre médian optionnel) -> (Savitzky–Golay) -> (biais statique).
    Retourne x_best, y_best.
    """
    x = x_raw.astype(float).copy()
    y = y_raw.astype(float).copy()

    # 1) Médian (anti-pics)
    if median_window and median_window > 1:
        k = median_window if median_window % 2 == 1 else median_window - 1
        if k < 1: k = 1
        x = medfilt(x, kernel_size=k)
        y = medfilt(y, kernel_size=k)

    # 2) Savitzky–Golay (lissage qui conserve la forme)
    n = len(x)
    w = sg_window if sg_window % 2 == 1 else sg_window - 1
    if w < 3: w = 3
    w = min(w, n if n % 2 == 1 else n - 1)
    if w <= sg_poly:
        w = max(sg_poly + 1 + (sg_poly % 2 == 0), 3)
        if w > n:
            return x + bias_x, y + bias_y

    try:
        xs = savgol_filter(x, window_length=w, polyorder=sg_poly, mode="interp")
        ys = savgol_filter(y, window_length=w, polyorder=sg_poly, mode="interp")
    except Exception:
        xs, ys = x, y

    # 3) Biais (offset)
    return xs + bias_x, ys + bias_y

def load_and_merge(gt_path: Path, trk_path: Path):
    """Loads a GT/TRK pair, merges on image_id, returns DataFrame with standardized columns."""
    gt  = pd.read_csv(gt_path)
    trk = pd.read_csv(trk_path)

    gt_use = gt[[GT_IMAGE_COL, GT_X_COL, GT_Y_COL]].rename(
        columns={GT_IMAGE_COL:"image_id", GT_X_COL:"x_gt", GT_Y_COL:"y_gt"}
    )
    trk_use = trk[[TRK_IMAGE_COL, TRK_X_COL, TRK_Y_COL]].rename(
        columns={TRK_IMAGE_COL:"image_id", TRK_X_COL:"x_trk", TRK_Y_COL:"y_trk"}
    )
    m = (pd.merge(gt_use, trk_use, on="image_id", how="inner")
           .dropna(subset=["x_gt","y_gt","x_trk","y_trk"])
           .sort_values("image_id").reset_index(drop=True))
    return m

def standardize_key(p: Path):
    """key for pairing: base name without usual suffixes."""
    s = p.stem.lower()
    for suf in ["_gt","-gt","_groundtruth","_ground_truth","_trk","_trk_clean","_track","_tracking"]:
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s

def plot_pitch(gtx, gty, trk_x_raw, trk_y_raw, trk_x_best, trk_y_best, title, PLOTS_DIR):
    """display positions on a soccer pitch."""
    
    gtxp = gtx + L/2.0; gtyp = (-gty) + W/2.0
    rxp  = trk_x_raw + L/2.0; ryp  = (-trk_y_raw) + W/2.0
    bxp  = trk_x_best + L/2.0; byp = (-trk_y_best) + W/2.0
    if USE_MPLSOCCER:
        try:
            pitch = Pitch(pitch_type='custom', pitch_length=L, pitch_width=W)
            fig, ax = plt.subplots(figsize=(13.33, 7.5), dpi=150)
            pitch.draw(ax=ax)
            ax.scatter(gtxp, gtyp,  s=8, alpha=0.9, label="GT")
            ax.scatter(rxp,  ryp,   s=7, alpha=0.6, label="TRK brut")
            ax.scatter(bxp,  byp,   s=8, alpha=0.9, label="TRK optimisé")
            ax.set_title(title)
            ax.legend()
            plt.tight_layout()  # Rend le plot plus lisible
            file_path = os.path.join(PLOTS_DIR, f"{title.replace(' ', '_')}.png")
            plt.savefig(file_path, dpi=150, bbox_inches='tight')
            plt.show()
            # Fermeture pour éviter de surcharger la mémoire
            plt.close(fig)
            return
        except Exception as e:
            print(f"mplsoccer unaivailable ({e}) → fallback Matplotlib.")

    fig, ax = plt.subplots(figsize=(8,5))
    ax.set_xlim(0, L); ax.set_ylim(0, W); ax.invert_yaxis()
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_title(title)
    ax.scatter(gtxp, gtyp,  s=8, alpha=0.9, label="GT")
    ax.scatter(rxp,  ryp,   s=7, alpha=0.6, label="TRK brut")
    ax.scatter(bxp,  byp,   s=8, alpha=0.9, label="TRK optimisé")
    ax.legend(); plt.show()

# =============================
#        RANDOM SEARCH
# =============================
def random_search(train_data, n_iter=800, seed=42):
    """
    Tries n_iter randomly sampled combinations and returns (best_params, best_rmse_pooled).
    Pooled RMSE = sqrt(total SSE across the entire train set / total number of points).
    """
    rnd = random.Random(seed)
    best, best_rmse = None, float("inf")

    # Compteur total de points pour pondérer correctement
    total_points = sum(len(m) for m in train_data)
    if total_points == 0:
        return None, float("inf")

    for i in range(n_iter):
        mw = rnd.choice(MEDIAN_CHOICES)
        w  = rnd.choice(SG_WINDOW_CHOICES)
        p  = rnd.choice(SG_POLY_CHOICES)
        bx = rnd.uniform(*BIAS_RANGE)
        by = rnd.uniform(*BIAS_RANGE)

        sse = 0.0
        for m in train_data:
            x_gt = m["x_gt"].to_numpy(float); y_gt = m["y_gt"].to_numpy(float)
            x0   = m["x_trk"].to_numpy(float); y0   = m["y_trk"].to_numpy(float)
            xb, yb = apply_smoothing(x0, y0, mw, w, p, bx, by)
            dx = xb - x_gt; dy = yb - y_gt
            sse += float(np.sum(dx*dx + dy*dy))

        rmse_pooled = float(np.sqrt(sse / total_points))

        if rmse_pooled < best_rmse:
            best_rmse = rmse_pooled
            best = {"median_window": mw, "sg_window": w, "sg_poly": p, "bias_x": bx, "bias_y": by}

        # petit log pour suivre l'avancement
        if (i+1) % max(1, n_iter // 10) == 0:
            print(f"[RandomSearch] {i+1}/{n_iter}  best_rmse={best_rmse:.4f}  best={best}")

    return best, best_rmse

# =============================
#            MAIN
# =============================
def main():
    out_dir = Path(OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)
    gt_dir  = Path(GT_DIR);  trk_dir = Path(TRK_DIR)

    gt_map  = {standardize_key(p): p for p in gt_dir.glob("*.csv")}
    trk_map = {standardize_key(p): p for p in trk_dir.glob("*.csv")}
    keys    = sorted(set(gt_map) & set(trk_map))
    if not keys:
        raise RuntimeError("No paired GT/TRK files found.")

    # Split train/valid reproductible
    random.seed(SEED)
    random.shuffle(keys)
    n_train = int(len(keys) * TRAIN_RATIO)
    train_keys = sorted(keys[:n_train])
    valid_keys = sorted(keys[n_train:])

    pd.DataFrame({"pair_key": train_keys}).to_csv(out_dir / "train_pairs.csv", index=False)
    pd.DataFrame({"pair_key": valid_keys}).to_csv(out_dir / "valid_pairs.csv", index=False)

    # Charge TRAIN
    train_data = []
    for k in train_keys:
        m = load_and_merge(gt_map[k], trk_map[k])
        if len(m) >= 5:
            train_data.append(m)

    # ---------- RANDOM SEARCH SUR TRAIN ----------
    best_params, best_rmse_train = random_search(train_data, n_iter=N_RANDOM_ITERS, seed=SEED)
    with open(out_dir / "best_params_random.json", "w", encoding="utf-8") as f:
        json.dump({"best_params": best_params, "rmse_train_pooled": best_rmse_train}, f, indent=2)
    print("[BEST PARAMS - RANDOM]", json.dumps({"best_params": best_params, "rmse_train_pooled": best_rmse_train}, indent=2))

    # ---------- ÉVALUATION SUR VALID (+ export traj + plots) ----------
    valid_rows = []
    sse = 0.0; cnt = 0
    shown = 0
    for k in valid_keys:
        m = load_and_merge(gt_map[k], trk_map[k])
        if len(m) < 5:
            continue
        x_gt = m["x_gt"].to_numpy(float); y_gt = m["y_gt"].to_numpy(float)
        x0   = m["x_trk"].to_numpy(float); y0   = m["y_trk"].to_numpy(float)
        bp   = best_params
        xb, yb = apply_smoothing(x0, y0, bp["median_window"], bp["sg_window"], bp["sg_poly"], bp["bias_x"], bp["bias_y"])
        rmse = rmse_xy(x_gt, y_gt, xb, yb)
        valid_rows.append({"pair_key": k, "rmse": rmse})

        # export trajectoire
        df = pd.DataFrame({
            "image_id": m["image_id"],
            "x_gt": x_gt, "y_gt": y_gt,
            "x_trk_raw": x0, "y_trk_raw": y0,
            "x_trk_best": xb, "y_trk_best": yb
        })
        df.to_csv(out_dir / f"valid_{k}_best_trajectory.csv", index=False)

        # accumulate global RMSE valid
        dx = xb - x_gt; dy = yb - y_gt
        sse += float(np.sum(dx*dx + dy*dy))
        cnt += len(m)

        # plot quelques fichiers
        if shown < PLOT_N_VALID:
            plot_pitch(x_gt, y_gt, x0, y0, xb, yb, f"Valid {k}",PLOTS_POS_DIR)
            shown += 1

    if valid_rows:
        valid_df = pd.DataFrame(valid_rows).sort_values("rmse").reset_index(drop=True)
        rmse_valid_global = float(np.sqrt(sse / cnt)) if cnt else float("nan")
        valid_df.loc[len(valid_df)] = {"pair_key": "__GLOBAL_POOLED__", "rmse": rmse_valid_global}
        valid_df.to_csv(out_dir / "valid_summary_random.csv", index=False)
        print(f"[VALID GLOBAL RMSE] {rmse_valid_global:.4f}")

if __name__ == "__main__":
    main()
