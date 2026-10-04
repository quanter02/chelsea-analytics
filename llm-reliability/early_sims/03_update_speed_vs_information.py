"""업데이트 빈도 vs 정보량 vs 변화 감지형 적응 망각 (CUSUM).

실행: early_sims 폴더 안에서 python 03_update_speed_vs_information.py  (결과 그림은 같은 폴더에 저장)
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in fm.findSystemFonts():
    if "NanumGothic.ttf" in f: fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": "NanumGothic", "axes.unicode_minus": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
    "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
COL = {"주 1회 업데이트 (기준)": "#2a78d6", "하루 1회 업데이트 (같은 데이터량)": "#eb6834",
       "하루 1회 + 빠른 망각": "#eda100", "데이터 2배 수집": "#1baf7a", "변화 감지형 적응 망각": "#4a3aa7"}
INK, MUTED = "#0b0b0b", "#8a8984"
A, X0, DAYS, CHG = 10.0, 8.0, 52*7, 26*7
au = lambda b: A * X0**(b-1)
truth = np.array([au(0.55 if d < CHG else 0.40) for d in range(DAYS)])
g = np.array([1, np.log(X0)])

def run(seed, per_week=100, every=7, half_life_w=6.6, adaptive=False):
    rng = np.random.default_rng(seed)
    lam = 0.5 ** (every / (half_life_w * 7))          # 업데이트 1회당 감쇠
    S = np.zeros((2, 2)); v = np.zeros(2); beta = None; est = np.full(DAYS, np.nan)
    cus_p = cus_n = 0.0; buf = []
    for d in range(DAYS):
        b = 0.55 if d < CHG else 0.40
        n = rng.poisson(per_week / 7)
        x = np.clip(np.exp(rng.normal(np.log(4), 0.45, n)), 1, 12)
        y = A * x**(b-1) * np.exp(rng.normal(0, 0.25, n))
        buf.append((np.log(x), np.log(y)))
        if (d + 1) % every == 0:                       # 업데이트 시점
            lx = np.concatenate([a for a, _ in buf]); ly = np.concatenate([c for _, c in buf]); buf = []
            X = np.column_stack([np.ones_like(lx), lx])
            if adaptive and beta is not None and len(ly) > 0:   # CUSUM: 새 데이터의 예측 오차 감시
                z = (ly - X @ beta).mean() / (0.25 / np.sqrt(len(ly)))
                cus_p = max(0, cus_p + z - 0.5); cus_n = max(0, cus_n - z - 0.5)
                if max(cus_p, cus_n) > 5:              # 경보 → 과거 데이터 대부분 폐기
                    S *= 0.02; v *= 0.02; cus_p = cus_n = 0.0
            S = lam * S + X.T @ X; v = lam * v + X.T @ ly
            if np.linalg.det(S) > 1e-9: beta = np.linalg.solve(S, v)
        if beta is not None: est[d] = np.exp(g @ beta)
    return est

STRATS = {
    "주 1회 업데이트 (기준)":            dict(every=7),
    "하루 1회 업데이트 (같은 데이터량)": dict(every=1),
    "하루 1회 + 빠른 망각":             dict(every=1, half_life_w=1.5),
    "데이터 2배 수집":                  dict(every=7, per_week=200),
    "변화 감지형 적응 망각":            dict(every=7, half_life_w=30, adaptive=True),
}
REPS = 300; res = {}
for name, kw in STRATS.items():
    E = np.array([run(s, **kw) for s in range(REPS)])
    err = np.abs(E - truth)
    pre = np.nanmean(err[:, 15*7:CHG]) / truth[0] * 100
    lag = []
    for e in E:
        ok = np.abs(e[CHG:] - truth[-1]) < 0.1 * truth[-1]
        # 처음으로 7일 연속 10% 이내에 들어온 시점
        run_ = np.convolve(ok, np.ones(7, int), "valid") == 7
        idx = np.where(run_)[0]; lag.append(idx[0] / 7 if len(idx) else np.nan)
    res[name] = dict(curve=np.nanmean(err, 0) / truth * 100, pre=pre, lag=np.nanmedian(lag), never=np.isnan(lag).mean())
    print(f"{name:24s} 변화전 오차 {pre:5.2f}%  적응 {np.nanmedian(lag):5.1f}주  미도달 {np.isnan(lag).mean()*100:4.1f}%  52주 오차 {res[name]['curve'][-1]:.2f}%")

fig, ax = plt.subplots(1, 2, figsize=(15, 6.2), dpi=150, gridspec_kw={"width_ratios": [1.35, 1]})
fig.suptitle("업데이트를 자주 돌리는 것 vs 정보를 더 얻는 것 (26주차 실제 선호 변화, 300회 반복)", fontsize=15, color=INK, x=0.01, ha="left")
a = ax[0]; wk = np.arange(DAYS) / 7
for name, r in res.items():
    a.plot(wk, r["curve"], color=COL[name], lw=2, label=name)
a.axvline(26, color=MUTED, ls=":", lw=1); a.set_xlim(14, 52); a.set_ylim(0, 30)
a.set_title("① 진짜 값 대비 평균 오차 (%)", loc="left", color=INK); a.set_xlabel("주차"); a.set_ylabel("평균 절대 오차 (%)")
a.legend(frameon=False, fontsize=9.5)
a = ax[1]
for name, r in res.items():
    a.scatter(r["pre"], r["lag"], s=150 if name.startswith("주 1회") else 70, color=COL[name], edgecolor="#fcfcfb", linewidth=2, zorder=3)
    lab, off = {"주 1회 업데이트 (기준)": ("주 1회 = 하루 1회 (같은 데이터량)\n두 점이 겹침", (10, -26)),
                "하루 1회 업데이트 (같은 데이터량)": ("", (0, 0)),
                "데이터 2배 수집": ("데이터 2배 수집", (-10, 10))}.get(name, (name, (10, 6)))
    if lab: a.annotate(lab, (r["pre"], r["lag"]), xytext=off, textcoords="offset points", fontsize=9.5, color=INK,
                       ha="right" if off[0] < 0 else "left")
a.set_title("② 트레이드오프: 왼쪽 아래일수록 좋음", loc="left", color=INK)
a.set_xlabel("변화 전 평소 오차 (%) - 노이즈에 흔들리는 정도"); a.set_ylabel("변화 후 적응 시간 (주, 중앙값)")
a.set_xlim(0, max(r["pre"] for r in res.values()) * 1.6); a.set_ylim(0, max(r["lag"] for r in res.values()) * 1.25)
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig("utility_speed_sim.png", facecolor=fig.get_facecolor())
