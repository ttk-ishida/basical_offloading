# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np

EPS = 1e-12

# t_loc = C / f_loc
def t_loc(C: np.ndarray, f_loc: np.ndarray) -> np.ndarray:
    return C / np.maximum(f_loc, EPS)

# t_mec = C / f_mec + D / gamma
def t_mec(C: np.ndarray, f_mec: np.ndarray, D: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    return C / np.maximum(f_mec, EPS) + D / np.maximum(gamma, EPS)

# u_loc = ln(1 + max(t_sen - t_loc, 0))
def u_loc(t_sen: np.ndarray, t_loc_vals: np.ndarray) -> np.ndarray:
    gap = np.maximum(t_sen - t_loc_vals, 0.0)
    return np.log1p(gap)

# u_mec = θ ln(1 + max(t_sen - t_mec, 0)) - (1-θ) ρ_m f_mec
def u_mec(t_sen: np.ndarray, t_mec_vals: np.ndarray, f_mec: np.ndarray, theta: float = 0.6, rho_m: float = 1e-3) -> np.ndarray:
    gap = np.maximum(t_sen - t_mec_vals, 0.0)
    return theta * np.log1p(gap) - (1.0 - theta) * rho_m * f_mec