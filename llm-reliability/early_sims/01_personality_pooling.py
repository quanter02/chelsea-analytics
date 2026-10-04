"""16유형별 분리 추정 vs 연속 성격 점수 통합 모형, 텍스트 추출 점수의 감쇠 편향 (가상 1,600명).

실행: early_sims 폴더 안에서 python 01_personality_pooling.py  (결과 그림은 같은 폴더에 저장)
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
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8, "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb"})
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#8a8984"

A, B0, BE, BN = 10.0, 0.55, 0.12, 0.05   # 진짜 모형: AU = A * x^(b-1), b = B0 + BE*E + BN*N
NOISE = 0.15
def simulate(n, rng):
    z = rng.standard_normal((n, 4))                      # 성격 4축 (진짜 연속 점수)
    x = np.clip(np.exp(rng.normal(np.log(4), 0.45, n)), 1, 12)   # 소비량: 가운데 몰림, 양 끝 희소
    b = B0 + BE*z[:,0] + BN*z[:,1]
    y = A * x**(b-1) * np.exp(rng.normal(0, NOISE, n))
    return z, x, y

def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X@beta; s2 = r@r/(len(y)-X.shape[1])
    return beta, s2*np.linalg.pinv(X.T@X)

def design(lx, Z):  # 절편 + 성격별 절편 이동 + 기울기(log x) * (1 + 성격)
    return np.column_stack([np.ones_like(lx), Z, lx[:,None]*np.column_stack([np.ones_like(lx), Z])]) if Z.ndim==2 else None

rng = np.random.default_rng(7)
N = 1600
z, x, y = simulate(N, rng)
types = (z > 0).astype(int) @ np.array([8,4,2,1])
grid = np.linspace(1, 12, 200); lg = np.log(grid)

# 모형 1: 16유형별 따로
# 모형 2: 연속 성격 점수를 넣은 통합 모형 (전체 데이터 공유)
Xp = design(np.log(x), z); bp, Vp = ols(Xp, np.log(y))
wid_sep, wid_pool, res = [], [], {}
for t in range(16):
    m = types == t; zt = z[m].mean(0)
    Xs = np.column_stack([np.ones(m.sum()), np.log(x[m])]); bs, Vs = ols(Xs, np.log(y[m]))
    Gs = np.column_stack([np.ones_like(lg), lg]); ms = Gs@bs; ss = np.sqrt(np.einsum("ij,jk,ik->i", Gs, Vs, Gs))
    Gp = design(lg, np.tile(zt, (len(lg),1))); mp = Gp@bp; sp = np.sqrt(np.einsum("ij,jk,ik->i", Gp, Vp, Gp))
    truth = A*grid**(B0+BE*zt[0]+BN*zt[1]-1)
    f = lambda mu, s, k: np.exp(mu + k*1.96*s)
    wid_sep.append(f(ms,ss,1)-f(ms,ss,-1)); wid_pool.append(f(mp,sp,1)-f(mp,sp,-1))
    res[t] = dict(n=m.sum(), m=m, truth=truth, sep=(f(ms,ss,-1),np.exp(ms),f(ms,ss,1)), pool=(f(mp,sp,-1),np.exp(mp),f(mp,sp,1)))
wid_sep, wid_pool = np.median(wid_sep,0), np.median(wid_pool,0)

# 감쇠 편향: 외향성 효과(BE) 추정 — 설문 점수(r≈0.9) vs 텍스트 추출 점수(r≈0.35)
def noisy(zc, r, rng): return r*zc + np.sqrt(1-r**2)*rng.standard_normal(zc.shape)
est = {"설문 점수\n(r≈0.90)": [], "텍스트 추출 점수\n(r≈0.35)": [], "설문 + 텍스트\n(보조 변수)": []}
for k in range(300):
    r2 = np.random.default_rng(100+k); zz, xx, yy = simulate(N, r2); lx = np.log(xx); ly = np.log(yy)
    s = noisy(zz, 0.90, r2); t = noisy(zz, 0.35, r2)
    est["설문 점수\n(r≈0.90)"].append(ols(design(lx, s), ly)[0][6])
    est["텍스트 추출 점수\n(r≈0.35)"].append(ols(design(lx, t), ly)[0][6])
    both = np.column_stack([s, t[:, :1]])     # 설문 4축 + 텍스트 외향성 1개 추가
    est["설문 + 텍스트\n(보조 변수)"].append(ols(design(lx, both), ly)[0][7])

fig, ax = plt.subplots(2, 2, figsize=(13, 10), dpi=150)
fig.suptitle("시뮬레이션: 16유형별 분리 추정 vs 연속 성격 점수 통합 모형", fontsize=16, color=INK, x=0.02, ha="left", y=0.995)

a = ax[0,0]
a.hist(x, bins=np.linspace(1,12,23), color=MUTED, edgecolor="#fcfcfb", linewidth=2)
a.set_title("① 소비량(x) 분포 — 가운데 몰리고 양 끝은 희소", loc="left", color=INK)
a.set_xlabel("소비량 x"); a.set_ylabel("관측 수 (명)")

tmin = min(res, key=lambda t: res[t]["n"]); R = res[tmin]
a = ax[0,1]
a.scatter(x[R["m"]], y[R["m"]], s=14, color=MUTED, alpha=0.6, label=f"이 유형의 관측치 (n={R['n']})", zorder=2)
a.fill_between(grid, R["sep"][0], R["sep"][2], color=ORANGE, alpha=0.18, linewidth=0)
a.plot(grid, R["sep"][1], color=ORANGE, lw=2, label="유형별 분리 추정 (95% 구간)")
a.fill_between(grid, R["pool"][0], R["pool"][2], color=BLUE, alpha=0.25, linewidth=0)
a.plot(grid, R["pool"][1], color=BLUE, lw=2, label="연속 점수 통합 모형 (95% 구간)")
a.plot(grid, R["truth"], color=INK, lw=2, ls="--", label="진짜 평균효용 곡선")
a.set_ylim(0, np.percentile(R["sep"][2], 97)*1.05)
a.set_title("② 표본이 가장 적은 유형의 평균효용 곡선", loc="left", color=INK)
a.set_xlabel("소비량 x"); a.set_ylabel("평균효용 AU(x)"); a.legend(frameon=False, fontsize=9)

a = ax[1,0]
a.plot(grid, wid_sep, color=ORANGE, lw=2); a.plot(grid, wid_pool, color=BLUE, lw=2)
a.text(grid[-1], wid_sep[-1], " 유형별 분리", color=INK, va="center", fontsize=10)
a.text(grid[-1], wid_pool[-1], " 통합 모형", color=INK, va="center", fontsize=10)
a.set_yscale("log"); a.set_xlim(1, 14)
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullLocator
a.yaxis.set_major_locator(FixedLocator([0.1,0.2,0.5,1,2,5])); a.yaxis.set_major_formatter(FormatStrFormatter("%g")); a.yaxis.set_minor_locator(NullLocator())
a.set_title("③ 95% 신뢰구간 폭 (16유형 중앙값, 로그축) — 작을수록 안정", loc="left", color=INK)
a.set_xlabel("소비량 x"); a.set_ylabel("구간 폭 (평균효용 단위)")

a = ax[1,1]
labels = list(est); cols = [BLUE, ORANGE, BLUE]
for i, k in enumerate(labels):
    v = np.array(est[k]); lo, md, hi = np.percentile(v, [2.5, 50, 97.5])
    a.plot([i, i], [lo, hi], color=cols[i], lw=2); a.scatter([i], [md], s=70, color=cols[i], zorder=3, edgecolor="#fcfcfb", linewidth=2)
    a.text(i+0.08, md, f"{md:.3f}", va="center", fontsize=10, color=INK)
a.axhline(BE, color=INK, ls="--", lw=1.5); a.text(1.5, BE+0.004, "진짜 값 0.12", va="bottom", ha="center", fontsize=10, color=INK)
a.set_xticks(range(3)); a.set_xticklabels(labels); a.set_xlim(-0.5, 2.5); a.set_ylim(0, 0.15)
a.set_title("④ '외향성 → 체감 속도' 효과 추정 (300회 반복, 95% 범위)", loc="left", color=INK)
a.set_ylabel("추정된 효과 크기")
fig.tight_layout(rect=(0,0,1,0.97))
fig.savefig("utility_sim.png", facecolor=fig.get_facecolor())

i_mid = np.argmin(abs(grid-4)); 
print("smallest type n:", R["n"], "type counts:", sorted(res[t]["n"] for t in res))
for xv in (1, 4, 11):
    i = np.argmin(abs(grid-xv)); print(f"x={xv}: sep {wid_sep[i]:.3f} pool {wid_pool[i]:.3f} ratio {wid_sep[i]/wid_pool[i]:.1f}")
print("end/mid ratio sep", wid_sep[-1]/wid_sep[i_mid], "pool", wid_pool[-1]/wid_pool[i_mid])
for k,v in est.items(): print(k.replace("\n"," "), np.percentile(v,[2.5,50,97.5]).round(3))
print("x share <2:", (x<2).mean(), ">8:", (x>8).mean())
