"""아파트 입주 물량(v6 자료)을 첫 지도와 같은 방식으로 그린다.

    python housing_map.py   # → output/model_upgrade/housing_map.png, housing_map.md

1) 2013~2018 입주 아파트 ÷ 2025 전체 주택  2) 2021~2025 같은 비율  3) 설계 구간에서 '올해 입주' 신호가 v3 오차를 줄인 정도
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.patches import Polygon

import challenger_youthpop as Y
from challenger_housing import housing_signal
from challenger_size import frame
from llmrel import ledger, regional_tune as RT, report  # noqa: F401  (report: 한글 글꼴 설정)
from model_upgrade import OUT
from youthpop_map import panel

SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f1ec", "#f2b48a", "#d9622b", "#8f2d0f"])


def seq_panel(ax, geo, values: dict, vmax: float, title: str, unit: str, fig):
    norm = PowerNorm(0.5, vmin=0, vmax=vmax)
    for f in geo["features"]:
        v = values.get(f["properties"]["code"])
        col = SEQ(norm(min(v, vmax))) if v is not None and np.isfinite(v) else "#cccccc"
        g = f["geometry"]
        for poly in ([g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]):
            ax.add_patch(Polygon(np.array(poly[0]), closed=True, facecolor=col, edgecolor="white", linewidth=0.25))
    ax.set_xlim(124.5, 131.0); ax.set_ylim(33.0, 38.7); ax.set_aspect(1.2); ax.axis("off")
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=SEQ), ax=ax, shrink=0.45, pad=0.01, orientation="horizontal")
    cb.set_label(unit, fontsize=8); cb.ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=10)


def main():
    geo = json.load(open("data_regional/kr_regions_map.json", encoding="utf-8"))
    sgg, _ = RT.load()
    mr = pd.read_csv("data_regional/kr_marriages_by_region.csv", dtype={"code": str})
    names = mr.drop_duplicates("code").set_index("code").name.str.strip().to_dict()
    h = pd.read_csv("data_housing/kr_housing_by_built_year_2025.csv", dtype={"code": str})
    apt = h[h.kind == "아파트"].pivot(index="code", columns="built", values="value").fillna(0)
    stock = h[(h.kind == "주택_계") & (h.built == "T000")].set_index("code").value
    share = lambda ys: (apt[[f"T{y}" for y in ys]].sum(axis=1) / stock * 100).to_dict()
    early, late = share(range(2013, 2019)), share(range(2021, 2026))

    hs = housing_signal(sgg.columns)
    rhos = [x["payload"]["rho_by_size"] for x in ledger.read() if x["kind"] == "model_candidate" and x["key"] == "marriage_upgrade_v3_size"][-1]
    d = frame(sgg, Y.DESIGN)
    d["v3"] = d.base - d["size"].map(rhos).astype(float) * d.idio
    d["s"] = [hs[c].get(y, np.nan) if c in hs else np.nan for c, y in zip(d.code, d.year)]
    d = d.dropna(subset=["s"])
    # v3 단독 기준 δ(1.75)로 그린다: 입주 신호 자체가 어디서 맞는지 보려는 그림
    d1 = 1.75
    d["gain"] = (np.expm1((d.truth - d.v3).abs()) - np.expm1((d.truth - (d.v3 + d1 * d.s)).abs())) * 100
    gain = d.groupby("code").gain.mean().to_dict()

    fig, axes = plt.subplots(1, 3, figsize=(15, 6.4))
    seq_panel(axes[0], geo, early, 40, "① 2013~2018 입주 아파트\n÷ 전체 주택", "% (진할수록 신축 많음)", fig)
    seq_panel(axes[1], geo, late, 40, "② 2021~2025 입주 아파트\n÷ 전체 주택", "% (진할수록 신축 많음)", fig)
    panel(axes[2], geo, gain, 2, "③ '올해 입주' 신호가 줄인 오차\n(v3 기준, 2013~2018 평균)", "%p (주황 = 신호가 도움)", fig)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.08, wspace=0.03)
    fig.text(0.5, 0.01, "아파트 입주 물량 (2025 주택총조사 건축연도별, KOSIS DT_1JU1520) — 회색 = 자료 없음", ha="center", fontsize=9, color="#555")
    fig.savefig(OUT / "housing_map.png", dpi=140); plt.close(fig)

    nm = lambda c: f"{names.get(c[:2], '')} {names.get(c, c)}"
    top = lambda dct, k: ", ".join(f"{nm(c)} {v:.1f}" for c, v in sorted(((c, v) for c, v in dct.items() if c in names and np.isfinite(v)), key=lambda kv: -kv[1])[:k])
    (OUT / "housing_map.md").write_text(f"""# 아파트 입주 물량 지도

![지도](housing_map.png)

- ① 2013~2018 신축 비율 상위(%): {top(early, 8)}
- ② 2021~2025 신축 비율 상위(%): {top(late, 8)}
- ③ '올해 입주' 신호가 오차를 가장 많이 줄인 곳(%p): {top(gain, 6)}. 오차를 줄인 시군구 비율 {np.mean([v > 0 for v in gain.values()]):.0%}
""", encoding="utf-8")
    print((OUT / "housing_map.md").read_text())


if __name__ == "__main__":
    main()
