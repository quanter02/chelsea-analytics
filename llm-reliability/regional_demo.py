"""시군구별 결혼 시장 성비: 미혼 남성이 미혼 여성의 몇 배인가, 왜 그런가.

    python regional_demo.py            # KOSIS에서 새로 받음 → data_regional/, output/regional/report.md
    python regional_demo.py --cached   # 저장된 CSV로 다시 계산
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Polygon

from llmrel import regional as R
from llmrel.report import md_table

ROOT = Path(__file__).parent
D = ROOT / "data_regional"
OUT = ROOT / "output" / "regional"
YEAR = 2025
CMAP = LinearSegmentedColormap.from_list("div", ["#1f7a8c", "#e9edf0", "#f2b48a", "#d9622b", "#8f2d0f"])
NORM = LogNorm(vmin=0.8, vmax=3.2)


def draw_map(geo: dict, values: dict, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 7.4))
    for f in geo["features"]:
        v = values.get(f["properties"]["code"])
        col = CMAP(NORM(v)) if v is not None and not np.isnan(v) else "#cccccc"
        g = f["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        for poly in polys:
            ax.add_patch(Polygon(np.array(poly[0]), closed=True, facecolor=col, edgecolor="white", linewidth=0.25))
    ax.set_xlim(124.5, 131.0); ax.set_ylim(33.0, 38.7); ax.set_aspect(1.2); ax.axis("off")
    sm = plt.cm.ScalarMappable(norm=NORM, cmap=CMAP)
    cb = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.01, ticks=[0.8, 1, 1.5, 2, 3])
    cb.ax.set_yticklabels(["0.8", "1.0 (같음)", "1.5", "2.0", "3.0"]); cb.set_label("미혼 남성 ÷ 미혼 여성")
    ax.set_title(title, fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def draw_decomp(r: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    col = {"광역시 구·군": "#2a63d4", "도 지역 시": "#9aa3ac", "도 지역 군": "#d9622b"}
    for k, g in r.groupby("kind"):
        ax.scatter(g.pop_ratio, g.mrate_ratio, s=np.sqrt(g["미혼 남"] + g["미혼 여"]) / 4, color=col[k], alpha=.7, label=k, edgecolor="white", lw=.5)
    for _, x in r.nlargest(6, "unmarried_ratio").iterrows():
        ax.annotate(x.label.split(" ")[1], (x.pop_ratio, x.mrate_ratio), xytext=(4, 2), textcoords="offset points", fontsize=8)
    for _, x in r.nsmallest(1, "unmarried_ratio").iterrows():
        ax.annotate(x.label.split(" ")[1], (x.pop_ratio, x.mrate_ratio), xytext=(4, -9), textcoords="offset points", fontsize=8)
    xs = np.linspace(0.8, 2.4, 50)
    for k in (1, 1.5, 2, 3):
        ax.plot(xs, k / xs, ":", color="#999", lw=.8)
        xl = min(2.3, max(0.84, k / 1.85))
        ax.text(xl, k / xl, f"미혼 성비 {k}", fontsize=7.5, color="#666", va="bottom", ha="left")
    ax.set_xlabel("인구 성비 (남 ÷ 여, 25~39세) · 오른쪽일수록 젊은 여성이 적음"); ax.set_ylabel("미혼율 비 (남 ÷ 여) · 위일수록 남성이 결혼을 덜 함")
    ax.set_xlim(0.8, 2.4); ax.set_ylim(0.95, 1.95); ax.grid(alpha=.3); ax.legend(loc="lower right", fontsize=9)
    ax.set_title("미혼 성비 = 인구 성비 × 미혼율 비 (점 크기 = 미혼 인구)")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); D.mkdir(exist_ok=True)
    if "--cached" in sys.argv:
        u = pd.read_csv(D / "kr_unmarried_by_region.csv", dtype={"code": str})
        m = pd.read_csv(D / "kr_migration_by_region.csv", dtype={"mcode": str})
    else:
        u, m = R.fetch_unmarried(), R.fetch_migration()
        u.to_csv(D / "kr_unmarried_by_region.csv", index=False); m.to_csv(D / "kr_migration_by_region.csv", index=False)
    geo = json.load(open(D / "kr_regions_map.json", encoding="utf-8"))

    t = R.attach_migration(R.region_table(u, YEAR), m)
    nat = t[t.code == "00"].iloc[0]
    r = t[t.code != "00"].copy()
    dec = R.decompose(t)
    t22 = R.region_table(u, 2022).set_index("code")
    r["change"] = r.code.map(t.set_index("code").unmarried_ratio - t22.unmarried_ratio)
    ages = pd.DataFrame([{"연령": a, "미혼 남 ÷ 미혼 여": x.unmarried_ratio, "남는 미혼 남성 (명)": x.excess_men,
                          "여성 미혼율": x.f_unmarried_pct, "남성 미혼율": x.m_unmarried_pct}
                         for a in R.AGES.values() for x in [R.region_table(u, YEAR, [a]).query("code == '00'").iloc[0]]])
    kinds = r.groupby("kind").agg(지역수=("code", "size"), 미혼성비=("unmarried_ratio", "median"), 인구성비=("pop_ratio", "median"),
                                  미혼율비=("mrate_ratio", "median"), 여성미혼율=("f_unmarried_pct", "median"),
                                  남성미혼율=("m_unmarried_pct", "median"), 여성순이동=("mig_F", "median"), 남성순이동=("mig_M", "median")).reset_index()
    lp, lu = np.log(r.pop_ratio), np.log(r.unmarried_ratio)
    corr = {k: (np.corrcoef(lp, v)[0, 1], np.corrcoef(lu, v)[0, 1]) for k, v in
            [("20~34세 여성 순이동률", r.mig_F), ("20~34세 남성 순이동률", r.mig_M), ("남성 − 여성 순이동률", r.mig_M - r.mig_F)]}

    draw_map(geo, dict(zip(r.code, r.unmarried_ratio)), OUT / "map.png", f"25~39세 미혼 남성 ÷ 미혼 여성 ({YEAR})")
    draw_decomp(r, OUT / "decomp.png")

    cols = {"label": "지역", "unmarried_ratio": "미혼 성비", "미혼 남": "미혼 남", "미혼 여": "미혼 여", "pop_ratio": "인구 성비",
            "mrate_ratio": "미혼율 비", "f_unmarried_pct": "여성 미혼율", "m_unmarried_pct": "남성 미혼율"}
    fmt = {"미혼 성비": "{:.2f}", "인구 성비": "{:.2f}", "미혼율 비": "{:.2f}", "여성 미혼율": "{:.0f}%", "남성 미혼율": "{:.0f}%",
           "미혼 남": "{:,.0f}", "미혼 여": "{:,.0f}"}
    top = r.nlargest(12, "unmarried_ratio")[list(cols)].rename(columns=cols)
    bot = r.nsmallest(8, "unmarried_ratio")[list(cols)].rename(columns=cols)
    big = r[(r["미혼 남"] + r["미혼 여"]) > 60000].nlargest(8, "unmarried_ratio")[list(cols)].rename(columns=cols)
    border = ["강원 인제군", "강원 화천군", "강원 양구군", "강원 철원군", "경기 연천군", "강원 고성군"]
    bmask = r.label.isin(border)

    # 웹 페이지용 데이터
    web = {"year": YEAR, "nat": {k: float(nat[k]) for k in ["unmarried_ratio", "pop_ratio", "mrate_ratio", "f_unmarried_pct", "m_unmarried_pct", "excess_men"]},
           "regions": {x.code: [x.label, round(x.unmarried_ratio, 3), round(x.pop_ratio, 3), round(x.mrate_ratio, 3), int(x["미혼 남"]), int(x["미혼 여"]),
                                round(x.f_unmarried_pct, 1), round(x.m_unmarried_pct, 1), None if np.isnan(x.mig_F) else round(x.mig_F, 2),
                                None if np.isnan(x.mig_M) else round(x.mig_M, 2), None if np.isnan(x.change) else round(x.change, 3), x.kind]
                       for x in r.itertuples() for x in [r.loc[x.Index]]}}
    (D / "regions_web.json").write_text(json.dumps(web, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    L = [
        f"# 우리 동네 결혼 시장 성비 ({YEAR}, 시군구 {len(r)}곳)", "",
        "KOSIS 등록 기반 인구(내국인, 성·연령·혼인상태별, 시군구)로 계산했습니다. **미혼 성비 = 25~39세 미혼 남성 ÷ 미혼 여성**입니다. 1이면 같고, 2면 미혼 남성이 두 배입니다.", "",
        "## 1. 전국: 나이가 많을수록 벌어진다", "",
        md_table(ages, {"미혼 남 ÷ 미혼 여": "{:.2f}", "남는 미혼 남성 (명)": "{:,.0f}", "여성 미혼율": "{:.1f}%", "남성 미혼율": "{:.1f}%"}), "",
        f"- 25~39세 전체로 미혼 남성이 미혼 여성의 **{nat.unmarried_ratio:.2f}배**, 약 {nat.excess_men/1e4:.0f}만 명이 많습니다.",
        "- 20대 초반엔 1.1배인데 30대 후반엔 1.67배입니다. 인구 성비(1.05~1.12)보다 훨씬 큰 차이라, 대부분은 '남성이 결혼을 덜 해서'입니다. 여성은 같은 또래 남성보다 평균 2~3살 위 남성과 결혼하는 경우가 많아, 같은 나이끼리 비교하면 여성 쪽 미혼율이 낮게 나오는 효과도 섞여 있습니다.",
        f"- 2022→2025년 전국 미혼 성비는 {t22.loc['00','unmarried_ratio']:.2f} → {nat.unmarried_ratio:.2f}로 **줄었습니다**. 시군구 {(r.change < 0).mean():.0%}에서 줄었는데, 여성 미혼율이 남성보다 빨리 올랐기 때문입니다 (앞 분석: 30~34세 여성 미혼율 연 2.1%p 상승).", "",
        "## 2. 지도", "", "![지도](map.png)", "",
        f"- 가장 낮은 곳 서울 마포구 **{r.unmarried_ratio.min():.2f}** (미혼 여성이 더 많음), 가장 높은 곳 강원 인제군 **{r.unmarried_ratio.max():.2f}**. 같은 나라 안에서 3.6배 차이입니다.",
        f"- 미혼 여성이 더 많은(성비 1 미만) 곳은 {(r.unmarried_ratio < 1).sum()}곳이고, 모두 서울입니다.", "",
        "### 미혼 성비가 가장 높은 곳", "", md_table(top, fmt), "",
        "### 미혼 인구 6만 명 이상 큰 도시 중 가장 높은 곳", "", md_table(big, fmt), "",
        "### 가장 낮은 곳", "", md_table(bot, fmt), "",
        "## 3. 왜 다른가: 두 요인으로 나누기", "",
        "미혼 성비 = **인구 성비** (젊은 여성이 적은가) × **미혼율 비** (남성이 결혼을 덜 하는가).", "",
        "![분해](decomp.png)", "",
        f"- 지역 간 차이의 **{dec['인구 성비 몫']:.0%}는 인구 성비**, {dec['미혼율 비 몫']:.0%}는 미혼율 비에서 옵니다. 절반 넘게 '젊은 여성이 그 지역에 살지 않아서'입니다.",
        md_table(kinds, {"미혼성비": "{:.2f}", "인구성비": "{:.2f}", "미혼율비": "{:.2f}", "여성미혼율": "{:.0f}%", "남성미혼율": "{:.0f}%", "여성순이동": "{:+.2f}", "남성순이동": "{:+.2f}"}), "",
        "- **군 지역에선 여성 미혼율만 낮습니다.** 25~39세 여성 미혼율 중앙값이 군 55%, 광역시 65%로 10%p 차이인데, 남성은 77%와 76%로 거의 같습니다. 결혼한 여성은 남고 미혼 여성은 떠나는 그림과 맞습니다.",
        "- 순이동률(2015~2024 연평균, 20~34세, 100명당): 군 지역은 남녀 모두 해마다 5~6%씩 빠져나가지만 **여성이 더 많이** 빠져나갑니다.", "",
        md_table(pd.DataFrame([{"이동 지표": k, "인구 성비와 상관": a, "미혼 성비와 상관": b} for k, (a, b) in corr.items()]), {"인구 성비와 상관": "{:+.2f}", "미혼 성비와 상관": "{:+.2f}"}), "",
        "- 남녀 순이동률 차이(남성이 덜 떠나는 정도)가 인구 성비와 가장 강하게 연결됩니다 (+0.63). 지난 10년의 이동이 지금의 성비를 만들었다는 해석과 맞습니다. 상관이라 인과는 아닙니다.", "",
        "## 4. 극단값은 따로 봐야 한다", "",
        f"- 상위권의 접경 지역 6곳({', '.join(b.split(' ')[1] for b in border)})은 미혼 성비 평균 {r[bmask].unmarried_ratio.mean():.2f}입니다. 20대 인구 성비가 2배 안팎으로 치솟는데, **직업군인 주소 등록** 영향으로 보입니다 (이 표만으로는 확인 불가). 이곳의 높은 성비는 '결혼 시장'이라기보다 직업 구조입니다.",
        "- 당진·서산·진천 같은 산업도시는 제조업 일자리로 젊은 남성이 들어온 경우, 옹진·울릉은 섬입니다. 같은 '높은 성비'라도 원인이 다릅니다.",
        "- 반대로 서울 마포·강남·송파는 젊은 여성이 몰려 미혼 여성이 더 많습니다.", "",
        "## 한계", "",
        "- 등록 기반 인구라 실제 거주지와 다를 수 있습니다 (군인, 기숙사 대학생, 주소만 둔 사람).",
        "- 같은 연령대끼리 비교했습니다. 실제 결혼은 나이 차가 있어 '상대가 될 수 있는 사람' 수와는 다릅니다.",
        "- 연애 상대는 시군구 경계를 넘습니다. 통근권 단위로 보면 차이가 줄어듭니다.",
        "- 지도 경계는 2018년 통계청 시군구 경계를 2025년 행정구역으로 합친 것입니다 (군위군은 대구에 포함).",
    ]
    (OUT / "report.md").write_text("\n".join(L), encoding="utf-8")
    print(OUT / "report.md")
    print(kinds.round(2).to_string()); print(dec); print(corr)


if __name__ == "__main__":
    main()
