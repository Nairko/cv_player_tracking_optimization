# -*- coding: utf-8 -*-
"""
Optuna + PSO for Tracking Optimization 

Pipeline:
1) Pair GT/TRK csv by basename.
2) Split train/valid (reproducible).
3) Optimize global smoothing params to minimize pooled RMSE over TRAIN:
   - Stage A: Optuna (TPE)
   - Stage B: PSO (Particle Swarm), initialized near Optuna best
4) Evaluate best on VALID, export CSVs + plots.

Outputs are written under: outputs/Optuna_PSO/
"""

import os, json, random, math
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, medfilt
import matplotlib.pyplot as plt

try:
    from mplsoccer import Pitch
    USE_MPLSOCCER = True
except Exception:
    USE_MPLSOCCER = False

try:
    import optuna
    HAVE_OPTUNA = True
except Exception:
    HAVE_OPTUNA = False
    print("⚠️ Optuna unavailable. Install with 'pip install optuna' to enable optimization.")

# =============================
#            PATHS
# =============================
BASE_DIR = Path(__file__).resolve().parent

GT_DIR  = BASE_DIR / "datasets" / "gt_data"
TRK_DIR = BASE_DIR / "datasets" / "trck_data"
OUT_DIR = BASE_DIR / "outputs" / "Optuna_PSO"
PLOTS_POS_DIR = OUT_DIR / "plots" / "pos"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_POS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO = 0.6
SEED = 42
PLOT_N_VALID = 10


#      SEARCH SPACES
# Discrete spaces for smoothing
MEDIAN_CHOICES = [0, 3, 5, 7]                         # 0 disables median
SG_WINDOW_CHOICES = [w for w in range(5, 100, 2)]     # odd 5..99
SG_POLY_CHOICES   = [2, 3, 4]                         # poly order

# Continuous biases (meters)
BIAS_MIN, BIAS_MAX = -1.00, 1.00

# ---------- Optuna config ----------
N_TRIALS  = 400           # increase if you want more exploration
PRUNING   = False         # set True to enable MedianPruner (early stop trials)

# ---------- PSO config (simple implementation, no external libs) ----------
PSO_PARTICLES = 30
PSO_ITERS     = 150
W_INERTIA     = 0.6       # inertia weight
C_COGNITIVE   = 1.6       # particle best attraction
C_SOCIAL      = 1.6       # swarm best attraction


#     DATA COLUMN NAMES
GT_IMAGE_COL  = "image_id"
GT_X_COL      = "x_bottom_middle"
GT_Y_COL      = "y_bottom_middle"
TRK_IMAGE_COL = "image_id"
TRK_X_COL     = "x"
TRK_Y_COL     = "y"


#      PITCH DIMENSIONS
L, W = 105.0, 68.0

#       Fonctions

def rmse_xy(x_true, y_true, x_pred, y_pred):
    dx = x_pred - x_true
    dy = y_pred - y_true
    return float(np.sqrt(np.mean(dx*dx + dy*dy)))

def apply_smoothing(x_raw, y_raw, median_window, sg_window, sg_poly, bias_x, bias_y):
    """Median (optional) -> Savitzky-Golay -> + bias."""
    x = x_raw.astype(float).copy()
    y = y_raw.astype(float).copy()

    if median_window and median_window > 1:
        k = median_window if median_window % 2 == 1 else median_window - 1
        if k < 1: k = 1
        x = medfilt(x, kernel_size=k)
        y = medfilt(y, kernel_size=k)

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

    return xs + bias_x, ys + bias_y

def load_and_merge(gt_path: Path, trk_path: Path) -> pd.DataFrame:
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
    s = p.stem.lower()
    for suf in ["_gt","-gt","_groundtruth","_ground_truth","_trk","_trk_clean","_track","_tracking"]:
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s

def pooled_rmse_over_pairs(pairs: List[pd.DataFrame], params: Dict[str, float]) -> float:
    """Compute pooled RMSE over all rows of all train files."""
    mw = int(params["median_window"])
    w  = int(params["sg_window"])
    p  = int(params["sg_poly"])
    bx = float(params["bias_x"])
    by = float(params["bias_y"])

    sse_total = 0.0
    total_pts = 0
    for m in pairs:
        x_gt = m["x_gt"].to_numpy(float); y_gt = m["y_gt"].to_numpy(float)
        x0   = m["x_trk"].to_numpy(float); y0   = m["y_trk"].to_numpy(float)
        xb, yb = apply_smoothing(x0, y0, mw, w, p, bx, by)
        dx = xb - x_gt; dy = yb - y_gt
        sse_total += float(np.sum(dx*dx + dy*dy))
        total_pts += len(m)
    if total_pts == 0:
        return float("inf")
    return float(np.sqrt(sse_total / total_pts))

def plot_pitch(gtx, gty, trk_x_raw, trk_y_raw, trk_x_best, trk_y_best, title, out_dir: Path):
    """Show & save GT + raw TRK + best TRK as PNG."""
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
            ax.set_title(title); ax.legend(); plt.tight_layout()
            f = out_dir / f"{title.replace(' ', '_')}.png"
            plt.savefig(f, dpi=150, bbox_inches='tight'); plt.show(); plt.close(fig); return
        except Exception as e:
            print(f"mplsoccer unvailable ({e}) → fallback Matplotlib.")

    fig, ax = plt.subplots(figsize=(8,5))
    ax.set_xlim(0,L); ax.set_ylim(0,W); ax.invert_yaxis()
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_title(title)
    ax.scatter(gtxp, gtyp,  s=8, alpha=0.9, label="GT")
    ax.scatter(rxp,  ryp,   s=7, alpha=0.6, label="TRK brut")
    ax.scatter(bxp,  byp,   s=8, alpha=0.9, label="TRK optimisé")
    ax.legend()
    f = out_dir / f"{title.replace(' ', '_')}.png"
    plt.savefig(f, dpi=150, bbox_inches='tight'); plt.show(); plt.close(fig)

# =============================
#          OPTIMIZERS
# =============================
def optuna_optimize(train_pairs: List[pd.DataFrame], n_trials: int = N_TRIALS, seed: int = SEED):
    """
    Optuna objective: minimize pooled RMSE across TRAIN.
    - median_window, sg_window, sg_poly are categorical (discrete)
    - bias_x, bias_y are uniform continuous in [BIAS_MIN, BIAS_MAX]
    """
    if not HAVE_OPTUNA:
        raise RuntimeError("Optuna unvailable.")

    def objective(trial: "optuna.trial.Trial") -> float:
        mw = trial.suggest_categorical("median_window", MEDIAN_CHOICES)
        w  = trial.suggest_categorical("sg_window", SG_WINDOW_CHOICES)
        p  = trial.suggest_categorical("sg_poly", SG_POLY_CHOICES)
        bx = trial.suggest_float("bias_x", BIAS_MIN, BIAS_MAX)
        by = trial.suggest_float("bias_y", BIAS_MIN, BIAS_MAX)

        params = {"median_window": mw, "sg_window": w, "sg_poly": p, "bias_x": bx, "bias_y": by}
        return pooled_rmse_over_pairs(train_pairs, params)

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner  = optuna.pruners.MedianPruner() if PRUNING else optuna.pruners.NopPruner()
    study   = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_trial.params
    best_value  = study.best_value

    # Save study summary
    with open(OUT_DIR / "optuna_best.json", "w", encoding="utf-8") as f:
        json.dump({"best_params": best_params, "best_rmse_train": best_value,
                   "n_trials": n_trials}, f, indent=2)

    return best_params, best_value, study

def pso_optimize(train_pairs: List[pd.DataFrame],
                 init_params: Dict[str, float],
                 n_particles: int = PSO_PARTICLES,
                 n_iters: int = PSO_ITERS,
                 seed: int = SEED):
    """
    Simple PSO (no external dependency) on a 5D vector:
      [mw_idx, w_idx, p_idx, bias_x, bias_y]
    mw_idx ∈ [0, len(MEDIAN_CHOICES)-1], etc.  (rounded when evaluated)
    bias_x, bias_y ∈ [BIAS_MIN, BIAS_MAX]
    """
    rnd = random.Random(seed)
    np.random.seed(seed)

    # Map init_params to index space for discrete dims
    def to_idx(val, choices):
        # nearest index in choices
        arr = np.array(choices)
        return int(np.argmin(np.abs(arr - val)))

    def decode_particle(pos_vec):
        """Convert continuous particle vector to actual params dict."""
        mw_idx = int(np.clip(round(pos_vec[0]), 0, len(MEDIAN_CHOICES)-1))
        w_idx  = int(np.clip(round(pos_vec[1]), 0, len(SG_WINDOW_CHOICES)-1))
        p_idx  = int(np.clip(round(pos_vec[2]), 0, len(SG_POLY_CHOICES)-1))
        bx     = float(np.clip(pos_vec[3], BIAS_MIN, BIAS_MAX))
        by     = float(np.clip(pos_vec[4], BIAS_MIN, BIAS_MAX))
        return {
            "median_window": MEDIAN_CHOICES[mw_idx],
            "sg_window":     SG_WINDOW_CHOICES[w_idx],
            "sg_poly":       SG_POLY_CHOICES[p_idx],
            "bias_x":        bx,
            "bias_y":        by,
        }

    def fitness(pos_vec):
        params = decode_particle(pos_vec)
        return pooled_rmse_over_pairs(train_pairs, params)

    # Bounds for each dim
    lb = np.array([0, 0, 0, BIAS_MIN, BIAS_MIN], dtype=float)
    ub = np.array([len(MEDIAN_CHOICES)-1,
                   len(SG_WINDOW_CHOICES)-1,
                   len(SG_POLY_CHOICES)-1,
                   BIAS_MAX, BIAS_MAX], dtype=float)

    # Initialize swarm around init_params (with small noise)
    mw0 = to_idx(init_params["median_window"], MEDIAN_CHOICES)
    w0  = to_idx(init_params["sg_window"], SG_WINDOW_CHOICES)
    p0  = to_idx(init_params["sg_poly"], SG_POLY_CHOICES)
    bx0 = float(init_params["bias_x"])
    by0 = float(init_params["bias_y"])
    center = np.array([mw0, w0, p0, bx0, by0], dtype=float)

    # particles positions & velocities
    X = np.tile(center, (n_particles, 1)) + np.random.normal(0, [0.7, 1.5, 0.5, 0.05, 0.05], size=(n_particles,5))
    V = np.zeros_like(X)
    # clamp to bounds
    X = np.minimum(np.maximum(X, lb), ub)

    pbest_X = X.copy()
    pbest_f = np.array([fitness(x) for x in X])
    gbest_i = int(np.argmin(pbest_f))
    gbest_X = pbest_X[gbest_i].copy()
    gbest_f = pbest_f[gbest_i]

    for it in range(1, n_iters+1):
        r1 = np.random.rand(*X.shape)
        r2 = np.random.rand(*X.shape)
        V = (W_INERTIA*V
             + C_COGNITIVE*r1*(pbest_X - X)
             + C_SOCIAL*r2*(gbest_X - X))
        X = X + V
        X = np.minimum(np.maximum(X, lb), ub)

        fvals = np.array([fitness(x) for x in X])

        improved = fvals < pbest_f
        pbest_X[improved] = X[improved]
        pbest_f[improved] = fvals[improved]

        gi = int(np.argmin(pbest_f))
        if pbest_f[gi] < gbest_f:
            gbest_f = pbest_f[gi]
            gbest_X = pbest_X[gi].copy()

        if it % max(1, n_iters//10) == 0:
            print(f"[PSO] iter {it}/{n_iters}  best_rmse={gbest_f:.4f}  best_params={decode_particle(gbest_X)}")

    best_params = decode_particle(gbest_X)
    return best_params, float(gbest_f)

# =============================
#              MAIN
# =============================
def main():
    # Pair files
    gt_map  = {standardize_key(p): p for p in GT_DIR.glob("*.csv")}
    trk_map = {standardize_key(p): p for p in TRK_DIR.glob("*.csv")}
    keys    = sorted(set(gt_map) & set(trk_map))
    if not keys:
        raise RuntimeError("Aucune paire trouvée. Vérifie datasets/gt_data et datasets/trck_data")

    # Split train/valid (repro)
    random.seed(SEED)
    random.shuffle(keys)
    n_train = int(len(keys) * TRAIN_RATIO)
    train_keys = sorted(keys[:n_train])
    valid_keys = sorted(keys[n_train:])

    pd.DataFrame({"pair_key": train_keys}).to_csv(OUT_DIR / "train_pairs.csv", index=False)
    pd.DataFrame({"pair_key": valid_keys}).to_csv(OUT_DIR / "valid_pairs.csv", index=False)

    # Load train
    train_pairs = []
    for k in train_keys:
        m = load_and_merge(gt_map[k], trk_map[k])
        if len(m) >= 5:
            train_pairs.append(m)

    # ---------- Stage A: OPTUNA ----------
    if not HAVE_OPTUNA:
        raise RuntimeError("Install optuna to run optimization: pip install optuna")
    bestA_params, bestA_rmse, study = optuna_optimize(train_pairs, n_trials=N_TRIALS, seed=SEED)
    print("[OPTUNA BEST]", bestA_params, bestA_rmse)

    # ---------- Stage B: PSO (refinement) ----------
    bestB_params, bestB_rmse = pso_optimize(train_pairs, init_params=bestA_params,
                                            n_particles=PSO_PARTICLES, n_iters=PSO_ITERS, seed=SEED)
    print("[PSO BEST]", bestB_params, bestB_rmse)

    # Choose the best of both worlds
    if bestB_rmse < bestA_rmse:
        best_params = bestB_params
        best_rmse   = bestB_rmse
        best_stage  = "PSO"
    else:
        best_params = bestA_params
        best_rmse   = bestA_rmse
        best_stage  = "Optuna"

    with open(OUT_DIR / "best_params_final.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_stage": best_stage,
            "best_params": best_params,
            "best_rmse_train": best_rmse,
            "optuna_best": {"params": bestA_params, "rmse_train": bestA_rmse, "n_trials": N_TRIALS},
            "pso_best":    {"params": bestB_params, "rmse_train": bestB_rmse,
                            "particles": PSO_PARTICLES, "iters": PSO_ITERS}
        }, f, indent=2)

    # ---------- Evaluate on VALID (+ exports + plots) ----------
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

        df = pd.DataFrame({
            "image_id": m["image_id"],
            "x_gt": x_gt, "y_gt": y_gt,
            "x_trk_raw": x0, "y_trk_raw": y0,
            "x_trk_best": xb, "y_trk_best": yb
        })
        df.to_csv(OUT_DIR / f"valid_{k}_best_trajectory.csv", index=False)

        dx = xb - x_gt; dy = yb - y_gt
        sse += float(np.sum(dx*dx + dy*dy))
        cnt += len(m)

        if shown < PLOT_N_VALID:
            plot_pitch(x_gt, y_gt, x0, y0, xb, yb, f"Valid_{k}", PLOTS_POS_DIR)
            shown += 1

    if valid_rows:
        valid_df = pd.DataFrame(valid_rows).sort_values("rmse").reset_index(drop=True)
        rmse_valid_global = float(np.sqrt(sse / cnt)) if cnt else float("nan")
        valid_df.loc[len(valid_df)] = {"pair_key": "__GLOBAL_POOLED__", "rmse": rmse_valid_global}
        valid_df.to_csv(OUT_DIR / "valid_summary_optuna_pso.csv", index=False)
        print(f"[VALID GLOBAL RMSE] {rmse_valid_global:.4f}")

if __name__ == "__main__":
    main()
