# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np

# 切断正規からのサンプリング：範囲外は引き直し（クリップではなく再試行）
def sample_speed_truncnorm(n: int, mu: float = 20.0, sigma: float = 5.0, vmin: float = 5.0, vmax: float = 35.0, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    v = np.empty(n, dtype=float)
    i = 0
    while i < n:
        draw = rng.normal(mu, sigma, size=n - i)
        ok = (draw >= vmin) & (draw <= vmax)
        k = ok.sum()
        if k == 0:
            continue
        v[i:i+k] = draw[ok]
        i += k
    return v

# RSU 半径 R[m]、速度 v[km/h] → 滞在時間 t_m [s]
def dwell_time_seconds(v: np.ndarray, R: float = 150.0) -> np.ndarray:
    v_mps = v / 3.6
    return 2.0 * R / np.maximum(v_mps, 0.1)


# 実効締切 t_sen = min(t_max, t_m)
def effective_deadline(t_max: np.ndarray, t_m: np.ndarray) -> np.ndarray:
    return np.minimum(t_max, t_m)