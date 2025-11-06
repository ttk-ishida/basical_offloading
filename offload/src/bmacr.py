# -*- coding: utf-8 -*-
"""
BMACR: MEC 資源割り当て（λの二分探索）
- 目的: \sum_i u_i^{mec}(f_i) を最大化, s.t. \sum_i f_i <= F_total, f_i >= 0,
および t_mec_i = C_i/f_i + D_i/γ_i <= t_sen_i
- 手法: KKT 条件に基づいて各 i の一変数方程式 dL/df_i = 0 を満たす f_i(λ) を
二分探索で解き、全体で \sum f_i(λ) が F_total になるよう λ を二分探索。

備考:
厳密な式(22)が閉形式を与える前提でも、gap = t_sen - (C/f + D/γ) が f に依存するため、
ここでは KKT の一階条件を満たす f を「各ユーザごとに一変数の二分探索」で解く。
これにより KKT を満たす点を機械的・安定に求められる。
"""

from __future__ import annotations
import numpy as np


EPS = 1e-12


def _t_mec(C: float, f: float, D: float, gamma: float) -> float:
    return C / max(f, EPS) + D / max(gamma, EPS)


def _u_prime_wrt_f(C: float, D: float, gamma: float, t_sen: float, f: float, theta: float, rho_m: float, lam: float) -> float:
    """d/df of Lagrangian for user i: θ * (C / (f^2 * (1+gap))) - (1-θ)ρ_m - λ
    where gap = max(t_sen - t_mec, 0).
    ただし gap<=0 の領域では効用の増分が消えるので、最小限の f で打切る扱いにする。
    """
    tm = _t_mec(C, f, D, gamma)
    gap = max(t_sen - tm, 0.0)
    term1 = theta * (C / (f*f)) / (1.0 + gap) if gap > 0.0 else 0.0
    return term1 - (1.0 - theta) * rho_m - lam

def _f_min_for_deadline(C: float, D: float, gamma: float, t_sen: float) -> float:
    """t_mec = C/f + D/γ <= t_sen を満たす最小 f。
    t_sen <= D/γ の場合は実行不可能（非常に大きな f を要求）なので np.inf を返す。
    """
    slack = t_sen - D / max(gamma, EPS)
    if slack <= EPS:
        return np.inf
    return C / max(slack, EPS)

def _solve_f_i(C: float, D: float, gamma: float, t_sen: float, theta: float, rho_m: float, lam: float, f_upper: float = 1e6) -> float:
    """各 i について dL/df = 0 を満たす f を二分探索で解く。
    - 下限は「締切を満たす最小 f」: f_min
    - 上限は f_upper（十分大きな値）
    - 導関数が下限で既に負なら f_min が最適（増やす意味がない）
    - 一方、上限でまだ正なら上限を拡大
    """
    f_min = _f_min_for_deadline(C, D, gamma, t_sen)
    if not np.isfinite(f_min):
        return 0.0 # 不可なら MEC に割当しない（外側でローカルに回る想定）


    # 下限時点の導関数
    der_lo = _u_prime_wrt_f(C, D, gamma, t_sen, max(f_min, EPS), theta, rho_m, lam)
    if der_lo <= 0.0:
        return max(f_min, 0.0)

    f_hi = max(f_min * 2.0, 1.0)
    for _ in range(64):
        der_hi = _u_prime_wrt_f(C, D, gamma, t_sen, f_hi, theta, rho_m, lam)
        if der_hi < 0.0:
            break
        f_hi *= 2.0
        if f_hi > f_upper:
            break
    
    # 二分探索
    lo, hi = max(f_min, EPS), max(f_hi, f_min * 1.1)
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        der = _u_prime_wrt_f(C, D, gamma, t_sen, mid, theta, rho_m, lam)
        if der > 0.0:
            lo = mid
        else:
            hi = mid
    return lo

def bmacr_allocate(C: np.ndarray, D: np.ndarray, gamma: np.ndarray, t_sen: np.ndarray, F_total: float, theta: float = 0.6, rho_m: float = 1e-3, lam_max: float = 1e6) -> np.ndarray:
    """与えられた MEC候補集合に対して λ を二分探索し、\sum f_i = F_total を満たす f を返す。
    """
    n = len(C)
    if n == 0:
        return np.zeros(0, dtype=float)
    # λ の二分探索
    lam_lo, lam_hi = 0.0, lam_max
    f = np.zeros(n, dtype=float)

    for _ in range(80):
        lam = 0.5 * (lam_lo + lam_hi)
        for i in range(n):
            f[i] = _solve_f_i(C[i], D[i], gamma[i], t_sen[i], theta, rho_m, lam)
        s = f.sum()
        if abs(s - F_total) / max(F_total, 1.0) < 1e-6:
            break
        if s > F_total:
            lam_lo = lam
        else:
            lam_hi = lam

    # 和がぴったりでない場合は軽くスケール
    total = f.sum()
    if total > EPS:
        scale = min(1.0, F_total / total)
        f *= scale
    return f