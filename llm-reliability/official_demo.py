"""한국·일본 여성의 연령대별 미혼율·혼인율·초혼연령을 공식 통계 API로 받아 비교.

    python official_demo.py            # KOSIS·e-Stat에서 새로 받음 → data_official/*.csv, output/official/report.md
    python official_demo.py --cached   # 저장된 CSV로 보고서만 다시 만듦 (키 없이 실행 가능)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from llmrel import official as O
from llmrel.report import md_table

ROOT = Path(__file__).parent
DATA = ROOT / "data_official" / "kr_jp_marriage_official.csv"
OUT = ROOT / "output" / "official"
COLOR = {"KR": "#c0392b", "JP": "#2c3e8f"}
NAME = {"KR": "한국", "JP": "일본"}
BANDS = ["25-29", "30-34", "35-39"]


def chart_unmarried(d: pd.DataFrame, path: Path) -> None:
    f = d[(d.indicator == "unmarried_pct") & (d.sex == "F") & (d.year >= 1980)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, band in zip(axes, BANDS):
        for c in ("JP", "KR"):
            s = f[(f.country == c) & (f.age_band == band)].sort_values("year")
            census = s[s.table_id != "DT_1MR2060"]
            ax.plot(census.year, census.value, "o-", color=COLOR[c], label=f"{NAME[c]} (인구조사)")
            reg = s[s.table_id == "DT_1MR2060"]
            if len(reg):
                ax.plot(reg.year, reg.value, "s--", color=COLOR[c], alpha=.6, label="한국 (등록 기반, 기준 다름)")
        ax.set_title(f"여성 {band}세 미혼율"); ax.set_ylabel("%"); ax.grid(alpha=.3)
    axes[0].legend(loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def chart_marriage_rate(d: pd.DataFrame, path: Path) -> None:
    f = d[(d.indicator == "marriage_rate") & (d.sex == "F") & (d.year >= 1990)]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=False)
    for ax, band in zip(axes, ["20-24"] + BANDS):
        for c in ("JP", "KR"):
            s = f[(f.country == c) & (f.age_band == band)].sort_values("year")
            ax.plot(s.year, s.value, "-o", ms=2.5, color=COLOR[c], label=NAME[c])
        ax.axvspan(2020, 2022, color="grey", alpha=.12)
        ax.set_title(f"여성 {band}세 혼인율 (천 명당)"); ax.grid(alpha=.3)
    axes[0].legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def chart_age(d: pd.DataFrame, path: Path) -> None:
    f = d[(d.indicator == "first_marriage_age") & (d.year >= 1990)]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for (c, sx), s in f.groupby(["country", "sex"]):
        ax.plot(s.year, s.value, "-" if sx == "F" else ":", color=COLOR[c], label=f"{NAME[c]} {'여성' if sx == 'F' else '남성'}")
    ax.set_title("평균 초혼연령"); ax.set_ylabel("세"); ax.grid(alpha=.3); ax.legend(ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def val(d, c, ind, band, yr, sex="F", table=None):
    s = d[(d.country == c) & (d.indicator == ind) & (d.age_band == band) & (d.year == yr) & (d.sex == sex)]
    if table:
        s = s[s.table_id == table]
    return float(s.value.iloc[0]) if len(s) else float("nan")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); DATA.parent.mkdir(exist_ok=True)
    if "--cached" in sys.argv:
        d, log = pd.read_csv(DATA), [dict(collector="저장된 CSV", ok=True, rows=0, years="")]
        d["age_band"] = d.age_band.astype(str)
    else:
        d, log = O.collect_all()
        d.to_csv(DATA, index=False)

    chart_unmarried(d, OUT / "unmarried.png")
    chart_marriage_rate(d, OUT / "marriage_rate.png")
    chart_age(d, OUT / "first_marriage_age.png")

    # 1) 미혼율: 같은 기준(인구조사)끼리만 비교
    rows = []
    for band in BANDS:
        kr15, kr20 = val(d, "KR", "unmarried_pct", band, 2015), val(d, "KR", "unmarried_pct", band, 2020)
        jp15, jp20 = val(d, "JP", "unmarried_pct", band, 2015), val(d, "JP", "unmarried_pct", band, 2020)
        kr05, jp05 = val(d, "KR", "unmarried_pct", band, 2005), val(d, "JP", "unmarried_pct", band, 2005)
        r22, r25 = val(d, "KR", "unmarried_pct", band, 2022), val(d, "KR", "unmarried_pct", band, 2025)
        rows.append({"연령": band, "한국 2005": kr05, "한국 2020": kr20, "한국 변화(%p/년)": (kr20 - kr05) / 15,
                     "일본 2005": jp05, "일본 2020": jp20, "일본 변화(%p/년)": (jp20 - jp05) / 15,
                     "2020 격차(한-일)": kr20 - jp20, "한국 등록 2022→2025(%p/년)": (r25 - r22) / 3})
    un = pd.DataFrame(rows)

    # 2) 혼인율: 코로나 이전(2019) 대비, 최저점 이후 반등
    rows = []
    for band in ["20-24"] + BANDS:
        for c, last in (("KR", 2025), ("JP", 2024)):
            s = d[(d.country == c) & (d.indicator == "marriage_rate") & (d.age_band == band) & (d.sex == "F") & (d.year >= 2019)].set_index("year").value
            low_yr = int(s.idxmin())
            rows.append({"연령": band, "나라": NAME[c], "2019": s.get(2019), "최저(연도)": f"{s.min():.1f} ({low_yr})",
                         f"최신": f"{s[last]:.1f} ({last})", "2019 대비": s[last] / s[2019] - 1,
                         "최저 대비 반등": s[last] / s.min() - 1})
    mr = pd.DataFrame(rows)

    age = d[d.indicator == "first_marriage_age"].pivot_table(index="year", columns=["country", "sex"], values="value")
    age_rows = [{"연도": str(y), "한국 여성": age.get(("KR", "F"), {}).get(y), "일본 여성": age.get(("JP", "F"), {}).get(y),
                 "한국 남성": age.get(("KR", "M"), {}).get(y), "일본 남성": age.get(("JP", "M"), {}).get(y)}
                for y in (1990, 2000, 2010, 2020, 2024, 2025) if y in age.index]

    kr_break = (val(d, "KR", "unmarried_pct", "30-34", 2022) - val(d, "KR", "unmarried_pct", "30-34", 2020))
    lines = [
        "# 한국·일본 여성 연령대별 결혼 지표 (공식 통계 API)", "",
        "KOSIS(통계청)와 e-Stat(일본 총무성 통계국) API에서 직접 받은 값입니다. 기사나 2차 자료를 거치지 않았습니다.", "",
        "## 수집 결과", "", md_table(pd.DataFrame(log)), "",
        f"전체 {len(d):,}행 → `data_official/kr_jp_marriage_official.csv` (열마다 출처 표 번호와 집계 기준 포함)", "",
        "## 1. 미혼율 — 같은 기준(인구조사)끼리 비교", "",
        "![미혼율](unmarried.png)", "",
        md_table(un, {c: "{:.1f}" for c in un.columns if c != "연령"}), "",
        "- 한국 여성 30대 초반 미혼율은 2005년 일본보다 13%p 낮았지만 2015년 무렵 역전했고, 2020년에는 10%p 높습니다.",
        "- 2005→2020 사이 한국은 연령대마다 해마다 1.0~1.7%p씩 올랐고, 일본은 0.2~0.3%p로 거의 멈췄습니다. **한국이 일본보다 3~8배 빠르게 변했습니다.**",
        "- 35~39세는 2020년에 두 나라가 거의 같아졌습니다(한국 22.9%, 일본 23.6%). 30대 후반까지 미혼 비율이 일본 수준을 따라잡았다는 뜻입니다.",
        f"- 한국 2022년 이후 값(점선)은 **혼인신고 기준 등록 자료**라 인구조사(본인 응답)와 기준이 다릅니다. 30~34세에서 2020→2022 사이 {kr_break:.1f}%p가 한 번에 뛴 것은 실제 변화보다 기준 차이가 큽니다. 그래서 등록 자료끼리의 추세(표의 마지막 열)만 따로 봅니다.",
        "- 일본은 2025년 국세조사 결과가 아직 API에 없어 2020년이 최신입니다.", "",
        "## 2. 연령별 혼인율 — 코로나 이후 반등이 어디서 일어나나", "",
        "![혼인율](marriage_rate.png)", "",
        md_table(mr, {"2019": "{:.1f}", "2019 대비": "{:+.0%}", "최저 대비 반등": "{:+.0%}"}), "",
        "- 한국의 반등은 **30~34세에 몰려 있습니다.** 2019년 수준을 이미 넘었고, 25~29세는 최저점 대비 크게 회복했지만 아직 2019년보다 낮습니다.",
        "- 일본은 모든 연령대에서 2019년보다 26~36% 낮고, 2023→2024는 거의 그대로입니다. 반등 신호가 없습니다.",
        "- 혼인의 중심 연령도 다릅니다. 일본은 여전히 25~29세가 가장 높고, 한국은 2021년부터 30~34세가 25~29세를 앞질렀습니다.",
        "- 기준 차이: 한국은 '그 해 신고된 혼인, 혼인 당시 연령', 일본은 '그 해 동거를 시작하고 신고한 혼인, 동거 시작 연령'입니다. 일본 쪽이 조금 낮게 나오는 정의라, 수준 비교보다 **변화율 비교**가 안전합니다.", "",
        "## 3. 평균 초혼연령", "", "![초혼연령](first_marriage_age.png)", "",
        md_table(pd.DataFrame(age_rows), {c: "{:.1f}" for c in ["한국 여성", "일본 여성", "한국 남성", "일본 남성"]}), "",
        "- 한국 여성의 초혼연령은 2009~2010년 무렵 일본을 넘어섰고, 2024년에는 약 1.8세 많습니다. 일본은 2015년 이후 29세대에 멈춰 있고 한국은 계속 오릅니다.", "",
        "## 요약: 가장 큰 차이", "",
        "1. **변화 속도** — 같은 기간 한국 여성의 연령대별 미혼율은 일본보다 3~8배 빠르게 올랐다. 일본은 이미 고원(plateau), 한국은 아직 상승 중.",
        "2. **결혼 시점** — 한국은 혼인이 30대 초반으로 옮겨 갔고, 최근 반등도 그 연령에서 일어난다. 일본은 20대 후반 중심이 유지되지만 전 연령에서 줄고 있다.",
        "3. **최근 방향** — 한국은 2023년 이후 30대 혼인율이 오르는데 같은 기간 미혼율(등록 기준)도 오른다. 늦게 결혼하는 사람이 늘어난 것이지 미혼 증가가 멈춘 것은 아니다. 일본은 반등 없이 낮은 수준이 굳어지고 있다.", "",
        "## 한계", "",
        "- 연애(교제) 여부는 두 나라 모두 공식 연간 통계가 없습니다. 일본 출생동향기본조사(5년 주기)·한국 가족과 출산 조사가 있지만 API로 받을 수 있는 표가 아니어서 이번 수집에서 빠졌습니다.",
        "- 한국 2020년 미혼율은 같은 구성의 표(DT_1PM2003)를 API가 거부해 출생지 유형 표(DT_1PB2004)의 합계를 썼습니다. 이 표는 15~24세가 한 구간이라 25세 이상만 사용했습니다.",
        "- 일본 미혼율은 외국인 포함·배우관계 불상 제외, 한국은 내국인 기준입니다.",
    ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(OUT / "report.md")


if __name__ == "__main__":
    main()
