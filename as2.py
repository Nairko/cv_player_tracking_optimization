# kinematics.py (version sans SG)
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- Utils temps ----------
def _time_vector(df: pd.DataFrame, fps_default: float = 25.0) -> np.ndarray:
    """
    Renvoie t (en secondes) à partir de 'timestamp_ms' si présent,
    sinon à partir de 'image_id' en supposant un pas ~ 1/fps_default.
    """
    if "timestamp_ms" in df.columns:
        t = (df["timestamp_ms"].to_numpy(float) - float(df["timestamp_ms"].iloc[0])) / 1000.0
    else:
        img = df["image_id"].to_numpy(float)
        if len(img) >= 2 and np.all(np.diff(img) > 0):
            t = (img - img[0]) / fps_default
        else:
            t = np.arange(len(df), dtype=float) / fps_default
    return t


# ---------- Dérivées + cinématique (brute) ----------
def _kinematics_from_xy_raw(x: np.ndarray, y: np.ndarray, t: np.ndarray):
    """
    Calcule vitesse et accélération à partir des positions (sans SG).
    """
    vx = np.gradient(np.asarray(x, float), t)
    vy = np.gradient(np.asarray(y, float), t)
    ax = np.gradient(vx, t)
    ay = np.gradient(vy, t)
    speed = np.sqrt(vx * vx + vy * vy)
    accel = np.sqrt(ax * ax + ay * ay)
    return speed, accel


def compute_kinematics_from_csv(csv_path: Path, fps_default: float = 25.0):
    """
    Charge un CSV 'valid_*_best_trajectory.csv' et calcule vitesses/accélérations
    pour GT, TRK brut et TRK optimisé — sans aucun Savitzky–Golay.
    """
    df = pd.read_csv(csv_path)
    t = _time_vector(df, fps_default=fps_default)

    speed, accel = {}, {}

    # GT — brut
    speed["GT"], accel["GT"] = _kinematics_from_xy_raw(
        df["x_gt"].to_numpy(float), df["y_gt"].to_numpy(float), t
    )

    # TRK brut — brut
    speed["TRK brut"], accel["TRK brut"] = _kinematics_from_xy_raw(
        df["x_trk_raw"].to_numpy(float), df["y_trk_raw"].to_numpy(float), t
    )

    # TRK optimisé — brut (pas de SG)
    speed["TRK optimisé"], accel["TRK optimisé"] = _kinematics_from_xy_raw(
        df["x_trk_best"].to_numpy(float), df["y_trk_best"].to_numpy(float), t
    )

    return t, speed, accel


# ---------- Plots ----------
def plot_kinematics(t, speed: dict, accel: dict, title: str,
                    save_dir: Path, dpi: int = 150):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Vitesse
    fig1, ax1 = plt.subplots(figsize=(13.33, 7.5), dpi=dpi)
    for k, v in speed.items():
        ax1.plot(t, v, label=k)
    ax1.set_title(f"{title} – Vitesse (m/s)")
    ax1.set_xlabel("Temps (s)")
    ax1.set_ylabel("Vitesse (m/s)")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()
    ax1.set_ylim(0, 30)
    plt.tight_layout()
    f1 = save_dir / f"{title.replace(' ', '_')}_speed.png"
    plt.savefig(f1, dpi=dpi, bbox_inches="tight")
    plt.close(fig1)

    # Accélération
    fig2, ax2 = plt.subplots(figsize=(13.33, 7.5), dpi=dpi)
    for k, a in accel.items():
        ax2.plot(t, a, label=k)
    ax2.set_title(f"{title} – Accélération (m/s²)")
    ax2.set_xlabel("Temps (s)")
    ax2.set_ylabel("Accélération (m/s²)")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend()
    ax2.set_ylim(0, 30)
    plt.tight_layout()
    f2 = save_dir / f"{title.replace(' ', '_')}_accel.png"
    plt.savefig(f2, dpi=dpi, bbox_inches="tight")
    plt.close(fig2)

    print(f"[SAVE] {f1}")
    print(f"[SAVE] {f2}")


# ---------- Exemple d’utilisation ----------
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    OUT_DIR = BASE_DIR / "outputs" / "grid_search"
    KIN_DIR = OUT_DIR / "plots" / "AS"

    csv_files = sorted(OUT_DIR.glob("valid_*_best_trajectory.csv"))
    if not csv_files:
        print("Aucun fichier 'valid_*_best_trajectory.csv' trouvé.")

    for csv_path in csv_files:
        pair_key = csv_path.stem.replace("valid_", "").replace("_best_trajectory", "")
        print(f"\n[PROCESS] {pair_key}")
        t, speed, accel = compute_kinematics_from_csv(csv_path, fps_default=25)
        plot_kinematics(t, speed, accel, f"Valid_{pair_key}", KIN_DIR)
