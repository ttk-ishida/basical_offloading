# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from models import t_loc, t_mec, u_loc, u_mec
from bmacr import bmacr_allocate
from wireless import gamma_vector_for_users, DEFAULTS as WLS_DEFAULTS

BIG_NEG = -1e12

def run_best_response(C: np.ndarray, D: np.ndarray, t_sen: np.ndarray, f_loc: np.ndarray,
                      F_total: float = 5e3, theta: float = 0.6, rho_m: float = 1e-3,
                      wl_params: dict | None = None,
                      max_iter: int = 50, verbose: bool = False):
    """
    CORAO の外側ループ：
      1) MEC候補集合に BMACR で f_mec を更新
      2) 各 i が MEC/Local の効用＋制約で選択
      3) 収束まで繰返し
    """
    N = len(C)
    offload = np.zeros(N, dtype=bool)  # 初期はローカル
    wl_p = WLS_DEFAULTS if wl_params is None else {**WLS_DEFAULTS, **wl_params}

    # ローカル側（固定）
    t_loc_vals_all = t_loc(C, f_loc)
    u_loc_vals_all = u_loc(t_sen, t_loc_vals_all)

    for it in range(max_iter):
        idx = np.where(offload)[0]
        n_mec = len(idx)

        if n_mec > 0:
            gamma = gamma_vector_for_users(n_mec, params=wl_p)
            f_mec = bmacr_allocate(C[idx], D[idx], gamma, t_sen[idx], F_total, theta, rho_m)
            t_mec_vals = t_mec(C[idx], f_mec, D[idx], gamma)
            u_mec_vals = u_mec(t_sen[idx], t_mec_vals, f_mec, theta, rho_m)
        else:
            f_mec = np.array([])
            t_mec_vals = np.array([])
            u_mec_vals = np.array([])

        changed = False
        for i in range(N):
            # Local
            u_loc_i = u_loc_vals_all[i] if t_loc_vals_all[i] <= t_sen[i] else BIG_NEG

            # MEC
            if offload[i]:
                k = np.where(idx == i)[0][0]
                u_mec_i = u_mec_vals[k] if t_mec_vals[k] <= t_sen[i] else BIG_NEG
            else:
                # 暫定評価（単独 MEC 仮定）
                gamma_i = gamma_vector_for_users(1, params=wl_p)[0]
                f_guess = max(F_total / max(n_mec + 1, 1), 1.0)
                t_mec_i = t_mec(C[i], f_guess, D[i], gamma_i)
                u_mec_i = u_mec(np.array([t_sen[i]]), np.array([t_mec_i]),
                                np.array([f_guess]), theta, rho_m)[0]
                if t_mec_i > t_sen[i]:
                    u_mec_i = BIG_NEG

            new_offload = (u_mec_i > u_loc_i)
            if new_offload != offload[i]:
                offload[i] = new_offload
                changed = True

        if verbose:
            print(f"iter={it:02d} MEC={offload.sum()} users")
        if not changed:
            break

    # 最終の MEC 割当（出力用再計算）
    idx = np.where(offload)[0]
    n_mec = len(idx)
    if n_mec > 0:
        gamma = gamma_vector_for_users(n_mec, params=wl_p)
        f_mec = bmacr_allocate(C[idx], D[idx], gamma, t_sen[idx], F_total, theta, rho_m)
        t_mec_vals = t_mec(C[idx], f_mec, D[idx], gamma)
    else:
        f_mec = np.array([])
        t_mec_vals = np.array([])

    return offload, idx, f_mec, t_mec_vals, t_loc_vals_all
