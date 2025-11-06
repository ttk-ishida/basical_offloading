# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np


# 典型的な既定値（必要に応じて main.py から上書き）
DEFAULTS = dict(
    R_phy=27e6,             # 物理層ビットレート [bps]
    payload_bits=1500*8,    # 平均ペイロード [bit]
    sigma=13e-6,            # 空きスロット [s]
    Ts=110e-6,              # 成功スロット [s]
    Tc=110e-6,              # 衝突スロット [s]
    CWmin=15,               # CWmin=15 → W=16
    m=5,                    # 最大リトライ段数
    gamma_const=18e6        # RSU方針などでの実用上限[bps]（ここを上げ下げ可）
)



def fixed_point_tau(n: int, CWmin: int = 15, m: int = 5, tol: float = 1e-9, iters: int = 200) -> tuple[float, float]:
    """Bianchi 固定点：Pc = 1-(1-tau)^(n-1),
    tau = 2(1-2Pc) / ((1-2Pc)(CWmin+1) + Pc*CWmin*(1-(2Pc)^m))
    """
    if n <= 0:
        return 0.0, 0.0
    Pc = 0.1
    for _ in range(iters):
        num = 2 * (1 - 2*Pc)
        den = (1 - 2*Pc) * (CWmin + 1) + Pc * CWmin * (1 - (2*Pc)**m)
        tau = num / max(den, 1e-12)
        tau = float(np.clip(tau, 1e-9, 1-1e-9))
        Pc_new = 1 - (1 - tau)**(n - 1)
        if abs(Pc_new - Pc) < tol:
            Pc = Pc_new
        break
    Pc = Pc_new
    return tau, Pc


def mac_throughput_bps(n: int, R_phy: float, payload_bits: int, sigma: float, Ts: float, Tc: float, CWmin: int, m: int) -> float:
    if n <= 0:
        return 0.0
    tau, Pc = fixed_point_tau(n, CWmin=CWmin, m=m)
    Ptr = 1 - (1 - tau)**n
    Ps = (n * tau * (1 - tau)**(n-1)) / max(Ptr, 1e-12)
    Tbar = (1-Ptr)*sigma + Ptr*Ps*Ts + Ptr*(1-Ps)*Tc
    E_payload = Ptr * Ps * payload_bits
    S = E_payload / max(Tbar, 1e-12)
    return min(S, R_phy)



def gamma_vector_for_users(n: int, params: dict | None = None, gamma_const: float = 12e6) -> np.ndarray:
    p = {**DEFAULTS, **(params or {})}
    S = mac_throughput_bps(n, p['R_phy'], p['payload_bits'], p['sigma'], p['Ts'], p['Tc'], p['CWmin'], p['m'])
    g = min(p['gamma_const'], S)
    return np.full(n, g, dtype=float)