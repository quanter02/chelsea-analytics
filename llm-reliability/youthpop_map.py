"""청년 인구 신호(v5)를 첫 지도(regional_demo.draw_map)와 같은 방식으로 그린다.

    python youthpop_map.py   # → output/model_upgrade/youthpop_map.png, youthpop_map.md

1) 2026년 1~7월 25~34세 인구 증가율 − 시도 평균  2) v5가 보는 2026 혼인 변화(2025 대비)
3) 2013~2018 설계 구간에서 신호가 줄인 오차(%p, 시군구 평균)
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Polygon

import challenger_youthpop as Y
from challenger_size import frame
from llmrel import ledger, regional_tune as RT, report  # noqa: F401  (report: 한글 글꼴 설정)
from model_upgrade import OUT

CMAP = LinearSegmentedColormap.from_list("div", ["#1f7a8c", "#7fb5bf", "#e9edf0", "#f2b48a", "#d9622b"])


def panel(ax, geo, values: dict, lim: float, title: str, unit: str, fig):
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    for f in geo["features"]:
        v = values.get(f["properties"]["code"])
        col = CMAP(norm(np.clip(v, -lim, lim))) if v is not None and np.isfinite(v) else "#cccccc"
        g = f["geometry"]
        for poly in ([g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]):
            ax.add_patch(Polygon(np.array(poly[0]), closed=True, facecolor=col, edgecolor="white", linewidth=0.25))
    ax.set_xlim(124.5, 131.0); ax.set_ylim(33.0, 38.7); ax.set_aspect(1.2); ax.axis("off")
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), ax=ax, shrink=0.45, pad=0.01, orientation="horizontal")
    cb.set_label(unit, fontsize=8); cb.ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=10)


def main():
    geo = json.load(open("data_regional/kr_regions_map.json", encoding="utf-8"))
    sgg, _ = RT.load()
    mr = pd.read_csv("data_regional/kr_marriages_by_region.csv", dtype={"code": str})
    names = mr.drop_duplicates("code").set_index("code").name.str.strip().to_dict()
    sig = Y.pop_signal(sgg.columns, names)
    s26 = (sig.loc[2026] * 100).to_dict()

    v5 = {x["key"]: x["payload"]["pred_2026"] for x in ledger.read() if x["kind"] == "nowcast_2026_v5"}
    chg = {c: (v / np.exp(sgg.loc[2025, c]) - 1) * 100 for c, v in v5.items() if c in sgg}

    gamma = [x["payload"]["gamma"] for x in ledger.read() if x["kind"] == "model_candidate" and x["key"] == "marriage_upgrade_v5_youthpop"][-1]
    rhos = [x["payload"]["rho_by_size"] for x in ledger.read() if x["kind"] == "model_candidate" and x["key"] == "marriage_upgrade_v3_size"][-1]
    d = frame(sgg, Y.DESIGN)
    d["v3"] = d.base - d["size"].map(rhos).astype(float) * d.idio
    d["sig"] = [sig[c].get(y, np.nan) if c in sig else np.nan for c, y in zip(d.code, d.year)]
    d = d.dropna(subset=["sig"])
    d["gain"] = (np.expm1((d.truth - d.v3).abs()) - np.expm1((d.truth - (d.v3 + gamma * d.sig)).abs())) * 100
    gain = d.groupby("code").gain.mean().to_dict()

    fig, axes = plt.subplots(1, 3, figsize=(15, 6.4))
    panel(axes[0], geo, s26, 3, "① 2026년 1~7월 청년(25~34) 인구\n증가율 − 시도 평균", "%p (주황 = 시도보다 더 늘어남)", fig)
    panel(axes[1], geo, chg, 15, "② v5가 보는 2026 혼인 건수\n변화 (2025 대비)", "% (주황 = 늘어남)", fig)
    panel(axes[2], geo, gain, 2, "③ 신호가 줄인 오차\n(2013~2018 설계 구간 평균)", "%p (주황 = 신호가 도움)", fig)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.08, wspace=0.03)
    fig.text(0.5, 0.01, "월별 청년 인구 신호 (v5, 사후 설계 도전자 · 판정 2027년 3월) — 회색 = 자료 없음", ha="center", fontsize=9, color="#555")
    fig.savefig(OUT / "youthpop_map.png", dpi=140); plt.close(fig)

    nm = lambda c: f"{names.get(c[:2], '')} {names.get(c, c)}".strip()
    top = lambda dct, k, rev: ", ".join(f"{nm(c)} {v:+.1f}" for c, v in sorted(((c, v) for c, v in dct.items() if np.isfinite(v)), key=lambda kv: kv[1], reverse=rev)[:k])
    sido_chg = pd.Series(chg).groupby(lambda c: c[:2]).median()
    help_share = float(np.mean([v > 0 for v in gain.values()]))
    (OUT / "youthpop_map.md").write_text(f"""# 청년 인구 신호 지도 (v5)

![지도](youthpop_map.png)

- ① 청년이 시도보다 많이 늘어난 곳: {top(s26, 6, True)}
- ① 청년이 시도보다 많이 줄어든 곳: {top(s26, 6, False)}
- ② v5가 2026 혼인 증가를 가장 크게 보는 곳: {top(chg, 6, True)}
- ③ 설계 구간에서 신호가 오차를 줄인 시군구 비율: {help_share:.0%}. 가장 많이 줄인 곳: {top(gain, 5, True)}
- 시도별 v5 2026 혼인 변화 중앙값(%): {", ".join(f"{names.get(s, s)} {v:+.1f}" for s, v in sido_chg.sort_values(ascending=False).items())}
""", encoding="utf-8")
    print((OUT / "youthpop_map.md").read_text())


if __name__ == "__main__":
    main()
