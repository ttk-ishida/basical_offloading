# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from traffic import sample_speed_truncnorm, dwell_time_seconds, effective_deadline
from decide import run_best_response

RNG = np.random.default_rng(42)

# ===== 実験パラメータ（Table II を参照しつつ調整） =====
N = 60                      # 車両数
R = 150.0                   # RSU 半径 [m]
mu_v, sigma_v = 30.0, 7.0   # 速度[km/h]の切断正規 μ, σ
vmin, vmax = 10.0, 50.0
# =======================================================


# タスク：計算量C[cycles], 送信データ D[Byte], 端末CPU f_loc[cycles/s], 締切 t_max[s]
C = RNG.uniform(2e8, 8e8, size=N)
# D = RNG.uniform(1e6, 5e6, size=N)
# 送信データ量は 50〜200 KB をビットに変換（x8）
D = RNG.uniform(50_000, 200_000, size=N) * 8  # bits
# 端末CPU
f_loc = RNG.uniform(0.8e9, 2.0e9, size=N)  # 0.8〜2.0 GHz 相当
# 締切に余裕帯（0.25〜1.0秒）— 元論文がミリ秒級でない限り、この帯が現実的
t_max = RNG.uniform(0.25, 1.0, size=N)


# 速度→RSU滞在→実効締切 t_sen
v = sample_speed_truncnorm(N, mu_v, sigma_v, vmin, vmax, RNG)
t_m = dwell_time_seconds(v, R)
t_sen = effective_deadline(t_max, t_m)


# MEC 総演算資源・重み
F_total = 5.0e10     # 50 GHz 相当
THETA = 0.7       # MEC優先度をやや上げる
RHO_M = 1e-4      # MECコストをやや下げる


# 802.11p/Bianchi パラメータ（必要に応じて Table II に合わせて調整）
wl_params = dict(
    R_phy=27e6,               # PHY上限
    payload_bits=1500*8,      # 平均フレームサイズ
    sigma=13e-6,
    Ts=110e-6,
    Tc=110e-6,
    CWmin=15,
    m=5,
    gamma_const=18e6          # RSUポリシー上限（必要なら 24e6 などへ）
)


# ===== 実行 =====
offload, idx, f_mec, t_mec_vals, t_loc_vals = run_best_response(
    C, D, t_sen, f_loc,
    F_total=F_total, theta=THETA, rho_m=RHO_M,
    wl_params=wl_params,
    max_iter=60, verbose=True,
)


# ===== レポート =====
N_mec = offload.sum()
print("\n=== RESULT ===")
print(f"Users: N={len(C)} MEC offloaded: {N_mec} ({N_mec/len(C):.1%})")
if N_mec > 0:
    print(f"MEC f_avg={f_mec.mean():.2f}, t_mec_avg={t_mec_vals.mean():.4f}s")
print(f"Local t_loc_avg={t_loc_vals.mean():.4f}s")

ok_local = (~offload) & (t_loc_vals <= t_sen)
ok_mec = np.zeros(len(C), dtype=bool)
if N_mec > 0:
    ok_mec[idx] = (t_mec_vals <= t_sen[idx])
finish_rate = (ok_local | ok_mec).mean()
print(f"Finish within deadline: {finish_rate:.1%}")