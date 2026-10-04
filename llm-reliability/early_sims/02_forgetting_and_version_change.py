"""52주 업데이트: 실제 선호 변화(망각 계수 vs 누적)와 텍스트 추출 버전 교체(버전 보정).

실행: early_sims 폴더 안에서 python 02_forgetting_and_version_change.py  (결과 그림은 같은 폴더에 저장)
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
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#8a8984"

W, CHANGE, X0, A = 52, 26, 8.0, 10.0        # 52주, 26주차에 변화, 관심 지점 x=8의 평균효용
LAM = 0.9                                   # 망각 계수 (반감기 약 6.6주)
au = lambda b: A * X0**(b-1)

def draw(n, b, rng, sd):                    # 관측: 평균효용 = A x^(b-1) * 잡음
    x = np.clip(np.exp(rng.normal(np.log(4), 0.45, n)), 1, 12)
    return x, A * x**(b-1) * np.exp(rng.normal(0, sd, n))

def wls_au(lx, ly, w):                      # 가중 최소제곱 → x=8에서의 평균효용과 표준오차
    X = np.column_stack([np.ones_like(lx), lx]); Wx = X * w[:, None]
    beta = np.linalg.solve(X.T @ Wx, Wx.T @ ly)
    r = ly - X @ beta; neff = w.sum()**2 / (w**2).sum()
    s2 = (w * r**2).sum() / w.sum() * neff / max(neff - 2, 1)
    V = s2 * np.linalg.inv(X.T @ Wx) * (w**2).sum() / w.sum()
    g = np.array([1, np.log(X0)]); m = g @ beta
    return np.exp(m), np.exp(m) * np.sqrt(g @ V @ g)

# ─── 시뮬레이션 A: 26주차에 실제 선호가 바뀜 (b 0.55 → 0.40) ───
def run_A(seed):
    rng = np.random.default_rng(seed); obs = []
    out = {k: [] for k in ("cum", "lam")}
    for t in range(W):
        b = 0.55 if t < CHANGE else 0.40
        x, y = draw(100, b, rng, 0.25); obs.append((np.full(100, t), x, y))
        tt, xx, yy = map(np.concatenate, zip(*obs))
        out["cum"].append(wls_au(np.log(xx), np.log(yy), np.ones_like(xx)))
        out["lam"].append(wls_au(np.log(xx), np.log(yy), LAM**(t - tt)))
    return {k: np.array(v) for k, v in out.items()}
truthA = np.array([au(0.55 if t < CHANGE else 0.40) for t in range(W)])

# ─── 시뮬레이션 B: 선호는 그대로, 26주차에 텍스트 추출 버전 교체 (v2가 15% 높게 채점) ───
BIAS2 = 1.15
def run_B(seed):
    rng = np.random.default_rng(seed); obs = []; ratio = 1.0
    out = {k: [] for k in ("cum", "lam", "cal")}
    for t in range(W):
        xs, ys = draw(30, 0.55, rng, 0.15)                 # 설문 (정확, 소량)
        xt, yt = draw(150, 0.55, rng, 0.35)                # 텍스트 추출 (부정확, 대량)
        v = 1 if t < CHANGE else 2
        if v == 2: yt = yt * BIAS2
        if t == CHANGE:                                     # 보정: 같은 기준 텍스트 200개를 v1, v2로 모두 처리
            xc, yc = draw(200, 0.55, rng, 0.35)
            ratio = np.exp(np.mean(np.log(yc * BIAS2 * np.exp(rng.normal(0, .05, 200))) - np.log(yc)))
        obs.append((t, xs, ys, xt, yt, v))
        T, LX, LY, Wq, LYc = [], [], [], [], []
        for (s, a, b_, c, d, vv) in obs:
            T += [np.full(len(a), s), np.full(len(c), s)]; LX += [np.log(a), np.log(c)]
            LY += [np.log(b_), np.log(d)]; LYc += [np.log(b_), np.log(d / (ratio if vv == 2 else 1))]
            Wq += [np.full(len(a), 1/.15**2), np.full(len(c), 1/.35**2)]   # 신뢰도 가중치
        T, LX, LY, Wq, LYc = map(np.concatenate, (T, LX, LY, Wq, LYc))
        out["cum"].append(wls_au(LX, LY, Wq))
        out["lam"].append(wls_au(LX, LY, Wq * LAM**(t - T)))
        out["cal"].append(wls_au(LX, LYc, Wq * LAM**(t - T)))
    return {k: np.array(v) for k, v in out.items()}
truthB = np.full(W, au(0.55))

REPS = 200
RA = [run_A(s) for s in range(REPS)]; RB = [run_B(1000 + s) for s in range(REPS)]
weeks = np.arange(1, W + 1)

fig, ax = plt.subplots(2, 2, figsize=(14, 10.5), dpi=150)
fig.suptitle("데이터가 매주 쌓일 때, x=8에서의 평균효용 추정치가 어떻게 움직이는가 (52주)", fontsize=16, color=INK, x=0.02, ha="left")

def band(a, r, key, col, lab):
    m, s = r[key][:, 0], r[key][:, 1]
    a.fill_between(weeks, m - 1.96*s, m + 1.96*s, color=col, alpha=0.15, linewidth=0)
    a.plot(weeks, m, color=col, lw=2, label=lab)
def mark(a):
    a.axvline(CHANGE + 0.5, color=MUTED, lw=1, ls=":"); a.set_xlim(1, W)

a = ax[0,0]; band(a, RA[0], "cum", ORANGE, "그냥 누적 (모든 데이터 동일 가중)"); band(a, RA[0], "lam", BLUE, f"망각 계수 λ={LAM}")
a.step(weeks, truthA, where="mid", color=INK, ls="--", lw=1.8, label="진짜 값"); mark(a)
a.text(CHANGE + 1, a.get_ylim()[1]*0.98 if False else 4.25, "26주차: 실제 선호 변화", color=INK, fontsize=10, va="top")
a.set_title("A-1. 실제 선호가 바뀌었을 때 (1회 실행, 95% 구간)", loc="left", color=INK)
a.set_ylabel("평균효용 추정치 AU(8)"); a.set_xlabel("주차"); a.legend(frameon=False, fontsize=9, loc="lower left")

a = ax[0,1]
for k, col, lab in (("cum", ORANGE, "그냥 누적"), ("lam", BLUE, f"망각 계수 λ={LAM}")):
    err = np.mean([np.abs(r[k][:, 0] - truthA) for r in RA], 0)
    a.plot(weeks, err, color=col, lw=2, label=lab)
mark(a); a.set_title(f"A-2. 평균 오차 |추정 - 진짜| ({REPS}회 반복 평균)", loc="left", color=INK)
a.set_ylabel("평균 절대 오차"); a.set_xlabel("주차"); a.legend(frameon=False, fontsize=9)

a = ax[1,0]; band(a, RB[0], "cum", ORANGE, "그냥 누적 (버전 무시)"); band(a, RB[0], "lam", BLUE, f"망각 계수 λ={LAM} (버전 무시)")
band(a, RB[0], "cal", AQUA, f"망각 계수 + 버전 보정")
a.plot(weeks, truthB, color=INK, ls="--", lw=1.8, label="진짜 값 (변화 없음)"); mark(a)
a.text(CHANGE + 1, a.get_ylim()[1], "26주차: 텍스트 추출 v1에서 v2로 교체", color=INK, fontsize=10, va="top")
a.set_title("B-1. 측정 도구만 바뀌었을 때 (1회 실행, 95% 구간)", loc="left", color=INK)
a.set_ylabel("평균효용 추정치 AU(8)"); a.set_xlabel("주차"); a.legend(frameon=False, fontsize=9, loc="lower left")

a = ax[1,1]
for k, col, lab in (("cum", ORANGE, "그냥 누적 (버전 무시)"), ("lam", BLUE, "망각 계수 (버전 무시)"), ("cal", AQUA, "망각 계수 + 버전 보정")):
    bias = np.mean([(r[k][:, 0] / truthB - 1) * 100 for r in RB], 0)
    a.plot(weeks, bias, color=col, lw=2, label=lab)
a.axhline(0, color=INK, lw=1); mark(a)
a.set_title(f"B-2. 가짜 변화의 크기: 진짜 대비 편향 % ({REPS}회 반복 평균)", loc="left", color=INK)
a.set_ylabel("편향 (%)"); a.set_xlabel("주차"); a.legend(frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig("utility_time_sim.png", facecolor=fig.get_facecolor())

# 요약 수치
print("truth A pre/post", truthA[0].round(3), truthA[-1].round(3))
for k in ("cum", "lam"):
    E = np.array([np.abs(r[k][:, 0] - truthA) for r in RA]); M = np.array([r[k][:, 0] for r in RA])
    lag = []
    for m in M:
        idx = np.where(np.abs(m[CHANGE:] - truthA[-1]) < 0.1 * truthA[-1])[0]
        lag.append(idx[0] + 1 if len(idx) else np.nan)
    print(k, "err wk20-26 %.3f  wk30 %.3f  wk40 %.3f  wk52 %.3f" % (E[:, 19:26].mean(), E[:, 29].mean(), E[:, 39].mean(), E[:, 51].mean()),
          "lag-to-within-10%% median %s, never %d/%d" % (np.nanmedian(lag), np.isnan(lag).sum(), REPS),
          "final est %.3f" % M[:, -1].mean())
for k in ("cum", "lam", "cal"):
    Bb = np.array([(r[k][:, 0] / truthB - 1) * 100 for r in RB])
    print(k, "bias%% wk25 %.1f wk28 %.1f wk32 %.1f wk40 %.1f wk52 %.1f  sd wk52 %.1f" % tuple(list(Bb.mean(0)[[24, 27, 31, 39, 51]]) + [Bb[:, 51].std()]))
