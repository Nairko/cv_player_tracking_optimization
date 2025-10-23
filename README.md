# ⚽ Player Tracking Optimization – SoccerNet GSR

This repository presents a method to **optimize player tracking positions** extracted from the **SoccerNet Game State Reconstruction (GSR)** pipeline.

---

## 📁 Repository Structure

### `extract_data_from_sn/`
Contains a **basic script** for extracting a single player's tracking data based on their `tracking_id`.

### `datasets/`
Includes the **datasets** used to optimize the parameters of the **Savitzky–Golay smoothing filter**.  
The data is automatically split into **60% for training** and **40% for validation**.

### Optimization Methods
Three optimization methods were tested to minimize the RMSE between **Ground Truth (GT)** and **raw tracking** data:

- `grid_search.py` – exhaustive parameter search  
- `random_search.py` – stochastic parameter sampling  
- `optuna_pso.py` – hybrid optimization using **Optuna** (TPE-Gaussian sampler with PSO-inspired exploration)

Each optimizer exports its results to:
```
outputs/<optimizer_name>/
```

### `as2.py`
Computes the **player’s speed and acceleration** based on:
- Ground Truth positions (GT)
- Raw tracking positions
- Optimized tracking positions

This allows quantitative evaluation of smoothing and bias correction effects.

---

## 🎯 Objective

The main objective is to **find the optimal smoothing and bias parameters** that best align tracking data with ground truth, improving positional accuracy and preserving realistic player dynamics.

---

## ⚙️ Dependencies

- Python ≥ 3.8  
- NumPy  
- Pandas  
- SciPy  
- Matplotlib  
- (Optional) `mplsoccer` for soccer-pitch visualization  

Install all dependencies with:
```bash
pip install numpy pandas scipy matplotlib mplsoccer optuna
```

---

## 🚀 Usage

1. Place GT and TRK CSVs into the appropriate dataset folder.  
2. Run one of the optimization scripts:
   ```bash
   python grid_search.py
   # or
   python random_search.py
   # or
   python optuna_pso.py
   ```
3. Results (optimized parameters, RMSE metrics, and plots) are automatically saved in:
   ```
   outputs/<optimizer_name>/
   ```

---

## 📊 Results

The following table shows the **Root Mean Squared Error (RMSE)** obtained on both the **training** and **validation** sets:

| Optimization method | RMSE pooled (train) | RMSE (valid) |
|----------------------|--------------------:|--------------:|
| Without optimization | 1.4775 | 1.4020 |
| Grid search | **1.2407** | **1.1912** |
| Random search | **1.2353** | **1.1908** |
| Optuna + PSO | **1.2350** | **1.1912** |

All optimization methods significantly reduce the RMSE compared to unoptimized tracking data.  
**Optuna+PSO** provides the most stable and consistent performance across datasets.

---

### 🔧 Best Parameters Found

| Optimization | Median window | SG window | SG poly | Bias X | Bias Y |
|---------------|:--------------:|:----------:|:--------:|:--------:|:--------:|
| Grid search | 7 | 83 | 3 | -0.09 | -0.25 |
| Random search | 7 | 81 | 3 | -0.09 | -0.34 |
| Optuna + PSO | 7 | 83 | 3 | -0.08 | -0.36 |

---
<img width="1984" height="1110" alt="image" src="https://github.com/user-attachments/assets/6d011d29-376e-49eb-aa7b-2272cdbde861" />

## 🧩 Interpretation

- The **Savitzky–Golay filter** with a large window (~80) effectively smooths noise while preserving player dynamics.  
- A **small positional bias correction** (~–0.1 m on X and ~–0.3 m on Y) helps align tracking with ground truth.  
- The **RMSE improvement (~15%)** demonstrates the impact of optimized smoothing and bias correction in refining player position tracking.

---


