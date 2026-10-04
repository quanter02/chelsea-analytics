"""04 결과(llm_res.pkl)를 그래프로 그림.

실행: early_sims 폴더 안에서 python 04b_plot_llm_breakeven.py  (결과 그림은 같은 폴더에 저장)
"""
import numpy as np, pickle, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in fm.findSystemFonts():
    if "NanumGothic.ttf" in f: fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": "NanumGothic", "axes.unicode_minus": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
    "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
INK, MUTED = "#0b0b0b", "#8a8984"
D = pickle.load(open("llm_res.pkl", "rb")); out, PS = D["out"], D["PS"]
E = lambda k: out[k][0][:, 4:].mean(); C = lambda k: out[k][1]
POL = [("llm_gated", "애매할 때만 호출", "#eb6834"), ("llm_alarm", "경보 시 호출", "#2a78d6"),
       ("llm_event", "사건 기록 시 + 경보 시 호출", "#1baf7a"), ("llm_weekly", "매주 호출", "#4a3aa7")]
stat, oracle = E(("stat", 1.0)), E(("oracle", 1.0))

fig, ax = plt.subplots(1, 3, figsize=(19, 6.6), dpi=150, gridspec_kw={"width_ratios": [1.25, 1, 1]})
fig.suptitle("LLM 원인 분류의 손익분기점과 호출 타이밍 (104주, 실제 변화 2회 + 측정 변화 2회, 200개 가상 세계)", fontsize=15, color=INK, x=0.01, ha="left")

a = ax[0]
for k, lab, col in POL:
    y = [E((k, p)) for p in PS]; a.plot(PS, y, color=col, lw=2, marker="o", ms=5, label=lab)
    i = next(i for i in range(1, len(PS)) if y[i] <= stat)                  # 손익분기점: 통계 분류선과 교차
    be = PS[i-1] + (stat - y[i-1]) * (PS[i] - PS[i-1]) / (y[i] - y[i-1]) if y[0] > stat else PS[0]
    if y[0] > stat: a.scatter([be], [stat], s=90, color=col, edgecolor="#fcfcfb", linewidth=2, zorder=4)
    print(f"{lab}: 손익분기 정확도 {'≤' if y[0] <= stat else ''}{be:.2f}")
a.axhline(stat, color=INK, ls="--", lw=1.5); a.text(0.62, stat + 0.06, f"LLM 없이 통계 분류만: {stat:.2f}%", color=INK, fontsize=10)
a.axhline(oracle, color=MUTED, ls=":", lw=1.5); a.text(0.405, oracle - 0.17, f"원인을 항상 맞히는 경우(경보 시): {oracle:.2f}%", color=INK, fontsize=10)
a.set_title("① 손익분기점: LLM 분류 정확도가 몇 % 넘어야 이득인가", loc="left", color=INK)
a.set_xlabel("LLM 원인 분류 정확도"); a.set_ylabel("평균 오차 (%, 낮을수록 좋음)")
a.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); a.legend(frameon=False, fontsize=9.5, loc="upper right")

P = 0.8
a = ax[1]
pts = [("통계 분류만", 0, stat, INK)] + [(lab, C((k, P)), E((k, P)), col) for k, lab, col in POL]
for lab, c, e, col in pts:
    a.scatter([c], [e], s=110, color=col, edgecolor="#fcfcfb", linewidth=2, zorder=3)
    a.annotate(f"{lab}\n{c:.1f}회, {e:.2f}%", (c, e), xytext=(10, 4) if c < 90 else (-10, 12), textcoords="offset points",
               fontsize=9.5, color=INK, ha="left" if c < 90 else "right")
a.set_xlim(-5, 115); a.set_ylim(1.7, 3.3)
a.set_title(f"② 호출 횟수 vs 오차 (LLM 정확도 {P:.0%})", loc="left", color=INK)
a.set_xlabel("104주 동안 LLM 호출 횟수"); a.set_ylabel("평균 오차 (%)")

a = ax[2]
seq = sorted(pts, key=lambda r: r[1]); labs, vals = [], []
for (l0, c0, e0, _), (l1, c1, e1, col) in zip(seq, seq[1:]):
    labs.append(f"{l1}\n(+{c1-c0:.1f}회)"); vals.append((e0 - e1) / (c1 - c0))
bars = a.bar(range(len(vals)), vals, color=[r[3] for r in seq[1:]], width=0.6)
for i, v in enumerate(vals): a.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=10, color=INK)
a.set_yscale("log"); a.set_xticks(range(len(vals))); a.set_xticklabels(labs, fontsize=9)
a.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
a.set_title("③ 호출 1회를 더할 때 줄어드는 오차 (한계효용, 로그축)", loc="left", color=INK)
a.set_ylabel("추가 호출 1회당 오차 감소 (%p)")
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig("utility_llm_sim.png", facecolor=fig.get_facecolor())
for l, v in zip(labs, vals): print(l.replace("\n", " "), round(v, 4))
