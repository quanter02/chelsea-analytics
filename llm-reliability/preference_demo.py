"""결혼할 때 무엇을 보나: 말(설문) vs 실제(누가 결혼했나).

    python preference_demo.py            # KOSIS·e-Stat에서 새로 받음
    python preference_demo.py --cached   # data_preference/ CSV 사용
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from llmrel import preference as P
from llmrel.report import md_table

ROOT = Path(__file__).parent
D = ROOT / "data_preference"
OUT = ROOT / "output" / "preference"
F_COL, M_COL = "#5a46b5", "#1f7a8c"


def load():
    if "--cached" in sys.argv:
        return (pd.read_csv(D / "kr_stated_2021.csv"), pd.read_csv(D / "jp_employed_2022.csv"),
                pd.read_csv(D / "kr_edu_pairs.csv"), pd.read_csv(D / "kr_age_gap.csv", dtype={"gcode": str}))
    D.mkdir(exist_ok=True)
    s, j, e, a = P.stated(), P.jp_revealed(), P.kr_education_pairs(), P.kr_age_gap()
    s.to_csv(D / "kr_stated_2021.csv", index=False); j.to_csv(D / "jp_employed_2022.csv", index=False)
    e.to_csv(D / "kr_edu_pairs.csv", index=False); a.to_csv(D / "kr_age_gap.csv", index=False)
    a["gcode"] = a.gcode.astype(str)
    return s, j, e, a


def chart_stated(w: pd.DataFrame, path: Path) -> None:
    w = w.sort_values("F")
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    y = np.arange(len(w))
    ax.hlines(y, w.M, w.F, color="#c3cad1", lw=2)
    ax.scatter(w.M, y, color=M_COL, s=45, zorder=3, label="남성")
    ax.scatter(w.F, y, color=F_COL, s=45, zorder=3, label="여성")
    for i, r in enumerate(w.itertuples()):
        ax.text(max(r.F, r.M) + 1.2, i, f"{r.F - r.M:+.0f}%p", va="center", fontsize=8.5, color="#333")
    ax.set_yticks(y, w.index); ax.set_xlim(50, 102); ax.grid(axis="x", alpha=.3)
    ax.set_xlabel("'중요하다'(약간+매우) 응답 비율 (%)")
    ax.set_title("결혼을 결정할 때 중요하게 보는 것 (한국 미혼 19~49세, 2021)", fontsize=11)
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def chart_income(me: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    levels = P.LEVELS["income"]
    for sx, col, nm in (("M", M_COL, "남성"), ("F", F_COL, "여성")):
        r = me[sx][me[sx].factor == "income"].set_index("level").reindex(levels)
        ax.plot(range(len(levels)), r.married_pct, "o-", color=col, lw=2, label=nm)
        ax.text(len(levels) - 1 + .1, r.married_pct.iloc[-1], f"{nm} {r.married_pct.iloc[-1]:.0f}%", va="center", fontsize=9)
    ax.set_xticks(range(len(levels)), [f"{l}" for l in levels]); ax.set_xlabel("본인 연 소득 (만 엔)")
    ax.set_ylabel("기혼 비율 (%)"); ax.set_ylim(0, 100); ax.grid(alpha=.3); ax.set_xlim(-.3, len(levels) + .6)
    ax.set_title("나이·고용형태·학력을 같게 놓고 소득만 바꿨을 때 (일본 30~44세 취업자, 2022)", fontsize=10)
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def chart_kr(edu_share: pd.DataFrame, gap: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    ax = axes[0]
    ax.plot(gap.index, gap["여자연상(소계)"], "-", color=F_COL, lw=2); ax.plot(gap.index, gap["남자연상 3세 이상"], "-", color=M_COL, lw=2)
    ax.text(gap.index[-1] + .5, gap["여자연상(소계)"].iloc[-1], f"아내 연상 {gap['여자연상(소계)'].iloc[-1]:.0f}%", va="center", fontsize=9)
    ax.text(gap.index[-1] + .5, gap["남자연상 3세 이상"].iloc[-1], f"남편 3살+ 연상 {gap['남자연상 3세 이상'].iloc[-1]:.0f}%", va="center", fontsize=9)
    ax.set_xlim(1990, 2036); ax.set_xticks([1990, 2000, 2010, 2020, 2025]); ax.set_ylim(0, 65); ax.grid(alpha=.3); ax.set_ylabel("초혼 부부 중 비율 (%)")
    ax.set_title("한국 초혼 부부 나이 차", fontsize=11)
    ax = axes[1]
    ax.plot(edu_share.index, edu_share["아내 학력 높음"], "-o", color=F_COL, lw=2, ms=4)
    ax.plot(edu_share.index, edu_share["남편 학력 높음"], "-o", color=M_COL, lw=2, ms=4)
    ax.text(edu_share.index[-1] + .2, edu_share["아내 학력 높음"].iloc[-1], "아내 학력이 더 높음", va="center", fontsize=9)
    ax.text(edu_share.index[-1] + .2, edu_share["남편 학력 높음"].iloc[-1], "남편 학력이 더 높음", va="center", fontsize=9)
    ax.set_xlim(2014.5, 2028); ax.set_xticks([2015, 2018, 2021, 2024]); ax.set_ylim(10, 22); ax.grid(alpha=.3); ax.set_ylabel("초혼 신혼부부 중 비율 (%)")
    ax.set_title("한국 초혼 신혼부부 학력 조합 (혼인 1년차)", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    s, j, e, a = load()

    # 1. 말
    w = s.pivot_table(index="item", columns="group", values="important")
    very = s.pivot_table(index="item", columns="group", values="very")
    w["차이"] = w.F - w.M
    stated_tab = pd.DataFrame({"여성 중요": w.F, "남성 중요": w.M, "차이 (여−남)": w["차이"], "여성 '매우' 중요": very.F}).sort_values("차이 (여−남)", ascending=False)
    n = s.groupby("group").n.first()

    # 2. 실제 (일본)
    me, fitq = {}, {}
    for sx in ("M", "F"):
        c = P.cells(j, sx); b = P.fit_logit(c); me[sx] = P.marginal_effects(c, b)
        X, _ = P._design(c); p = 1 / (1 + np.exp(-X @ b.to_numpy()))
        fitq[sx] = (len(c), c.total.sum(), 100 * np.average(np.abs(p - c.married / c.total), weights=c.total))
        if sx == "M":
            coef_m = b
    span = {sx: me[sx].groupby("factor").married_pct.agg(lambda v: v.max() - v.min()) for sx in me}
    eff = pd.DataFrame({"남성: 가장 낮은~높은 칸 차이": span["M"], "여성: 가장 낮은~높은 칸 차이": span["F"]}).reindex(["income", "emp", "edu", "age"])
    eff.index = ["본인 소득", "고용형태", "학력", "나이"]
    m = me["M"].set_index(["factor", "level"]).married_pct; f = me["F"].set_index(["factor", "level"]).married_pct
    raw = P.married_share(j, "emp"); raw = raw[raw.age == "30～34歳"].set_index(["sex", "emp"]).married_pct

    # 3. 한국 실제 결혼 조합
    rank = {"중졸이하": 0, "고졸": 1, "대졸": 2, "대학원졸": 3}
    e = e.assign(diff=e.wife.map(rank) - e.husband.map(rank))
    e["rel"] = np.select([e["diff"] > 0, e["diff"] < 0], ["아내 학력 높음", "남편 학력 높음"], "같음")
    t = e.pivot_table(index="year", columns="rel", values="value", aggfunc="sum")
    edu_share = 100 * t.div(t.sum(axis=1), axis=0)
    m24 = e[e.year == e.year.max()].pivot_table(index="husband", columns="wife", values="value").reindex(index=list(rank), columns=list(rank))
    tot = a[a.gcode.isin(["10", "20", "30"])].pivot_table(index="year", columns="gap", values="value")
    gap = 100 * tot.div(tot.sum(axis=1), axis=0)
    sub = a.pivot_table(index="year", columns="gcode", values="value")
    gap["남자연상 3세 이상"] = 100 * sub[["12", "13", "14"]].sum(axis=1) / tot.sum(axis=1)

    chart_stated(w, OUT / "stated.png"); chart_income(me, OUT / "income.png"); chart_kr(edu_share, gap, OUT / "kr_pairs.png")

    # 웹용 데이터: 남성·여성 회귀 계수 + 기준 → 페이지에서 조건 바꿔 기혼 확률 계산
    web = {"levels": P.LEVELS, "ref": P.REF,
           "coef": {sx: P.fit_logit(P.cells(j, sx)).round(4).to_dict() for sx in ("M", "F")},
           "stated": stated_tab.round(1).reset_index().values.tolist(), "n": {k: int(v) for k, v in n.items()},
           "edu_share": edu_share.round(1).reset_index().values.tolist(), "gap": gap[["여자연상(소계)", "동갑", "남자연상 3세 이상"]].round(1).reset_index().values.tolist()}
    (D / "preference_web.json").write_text(json.dumps(web, ensure_ascii=False), encoding="utf-8")

    pct = lambda v: f"{v:.0f}%"
    L = [
        "# 결혼할 때 무엇을 보나: 말 vs 실제", "",
        "두 가지를 나란히 봅니다. **말하는 기준**은 설문에서 직접 물은 결과, **실제로 드러나는 기준**은 어떤 조건의 사람이 실제로 결혼해 있는지입니다.", "",
        "> 읽기 전에: 결혼 여부는 두 사람의 선택, 그리고 본인이 '결혼할 여건이 된다'고 느끼는지가 함께 만든 결과입니다. 아래 '실제' 수치를 여성의 선택만으로 해석할 수 없습니다. 모두 집단 통계이고 개인에 대한 판단이 아닙니다.", "",
        "## 1. 말: 결혼 결정 때 중요하게 보는 것 (한국, 2021)", "",
        f"보건사회연구원 가족과 출산 조사, 미혼 19~49세 (여성 {n['F']:,}명, 남성 {n['M']:,}명).", "",
        "![말](stated.png)", "",
        md_table(stated_tab.reset_index().rename(columns={"item": "항목"}), {c: "{:.1f}" for c in stated_tab.columns}), "",
        f"- 여성은 9개 항목 중 8개를 85% 넘게 '중요하다'고 답했습니다. 1위는 **사랑과 신뢰 ({w.F['부부간의 사랑과 신뢰']:.0f}%)**입니다.",
        f"- 남녀 차이가 가장 큰 것은 **배우자의 경제적 여건 (여성 {w.F['배우자의 경제적 여건']:.0f}% vs 남성 {w.M['배우자의 경제적 여건']:.0f}%)**과 **배우자의 일과 직장 ({w.F['배우자의 일과 직장']:.0f}% vs {w.M['배우자의 일과 직장']:.0f}%)**입니다. 그다음이 **평등한 관계(가사 분담)**입니다 ({w.F['평등한 관계 (공평한 가사분담 등)']:.0f}% vs {w.M['평등한 관계 (공평한 가사분담 등)']:.0f}%).",
        f"- '매우 중요' 기준으로는 여성의 배우자 경제적 여건이 {very.F['배우자의 경제적 여건']:.0f}%입니다.",
        f"- 남성도 **본인의** 경제적 여건은 {w.M['본인의 경제적 여건']:.0f}%가 중요하다고 답합니다. 남성 스스로 '돈이 준비돼야 결혼한다'고 보는 것입니다.", "",
        "## 2. 실제: 조건이 같을 때 소득만 다르면 (일본, 2022)", "",
        f"한국에는 성·나이·소득·고용형태·학력을 한꺼번에 나눈 공개 표가 없어 일본 취업구조기본조사를 썼습니다. 30~44세 **취업자** (남성 {fitq['M'][1]/1e4:,.0f}만 명, 여성 {fitq['F'][1]/1e4:,.0f}만 명 추정치, 겹치지 않는 칸 {fitq['M'][0]}·{fitq['F'][0]}개). 가중 로지스틱 회귀로 '다른 조건은 그대로 두고 하나만 바꿨을 때' 기혼 비율을 계산했습니다 (칸별 실제값과 평균 오차 {fitq['M'][2]:.1f}%p·{fitq['F'][2]:.1f}%p).", "",
        "![소득](income.png)", "",
        md_table(eff.reset_index().rename(columns={"index": "조건"}), {c: "{:.0f}%p" for c in eff.columns}), "",
        f"- **남성은 소득이 결혼과 가장 강하게 연결됩니다.** 연 200만 엔 미만 {pct(m['income','~199'])} → 300만엔대 {pct(m['income','300~399'])} → 500만엔대 {pct(m['income','500~599'])} → 800만 엔 이상 {pct(m['income','800+'])}.",
        f"- **고용형태도 남성에게만 차이가 납니다.** 같은 소득·학력이어도 비정규직은 정규직보다 {m['emp','정규직'] - m['emp','비정규직']:.0f}%p 낮습니다. 30~34세 원자료로는 남성 정규직 {raw['M','うち正規の職員・従業員']:.0f}%, 비정규직 {raw['M','うち非正規の職員・従業員']:.0f}%입니다.",
        f"- **학력은 소득을 고정하면 오히려 반대입니다.** 대학원졸 남성 {pct(m['edu','대학원졸'])}, 고졸 {pct(m['edu','고졸'])}. 학력 자체보다 학력이 가져오는 소득·안정성이 작동한다는 뜻입니다.",
        f"- **여성은 소득·고용형태와 거의 무관합니다** (200만 엔 이상 구간 60~64%). 소득 200만 엔 미만 여성의 기혼 비율이 높은 것({pct(f['income','~199'])})은 결혼·출산 뒤 시간제로 일하는 경우가 많아서 생긴 **역방향 관계**로 보입니다.",
        "- 해석 주의: 남성도 결혼 뒤 소득이 오르는 효과(나이·책임감·회사 처우)가 섞여 있어, 이 차이 전부를 '소득이 높아서 결혼했다'로 볼 수 없습니다. 또 1절에서 보듯 남성 스스로 경제적 준비를 결혼 조건으로 보기 때문에, 저소득 남성의 낮은 기혼율에는 본인의 미루기도 들어 있습니다.", "",
        "## 3. 실제: 한국 부부는 어떻게 맺어지나", "",
        "![한국](kr_pairs.png)", "",
        f"- **아내가 연상인 초혼 부부: {gap['여자연상(소계)'].iloc[0]:.0f}% (1990) → {gap['여자연상(소계)'].iloc[-1]:.0f}% ({gap.index[-1]})**. 남편이 3살 이상 많은 부부는 {gap['남자연상 3세 이상'].iloc[0]:.0f}% → {gap['남자연상 3세 이상'].iloc[-1]:.0f}%로 줄었습니다. 나이는 점점 덜 따지는 조건입니다.",
        f"- **아내 학력이 더 높은 부부({edu_share['아내 학력 높음'].iloc[-1]:.0f}%)가 남편 학력이 더 높은 부부({edu_share['남편 학력 높음'].iloc[-1]:.0f}%)보다 많고**, 그 차이가 해마다 벌어집니다. {e.year.max()}년 고졸 남편 + 대졸 아내 {m24.loc['고졸','대졸']:,.0f}쌍, 대졸 남편 + 고졸 아내 {m24.loc['대졸','고졸']:,.0f}쌍입니다. 여성의 대학 진학률이 남성을 넘어선 영향이 큽니다.", "",
        "## 4. 종합: 여성의 평가 기준을 숫자로 정리하면", "",
        "| 기준 | 말 (한국 여성 '중요' 비율) | 실제 결과에서 보이는 무게 |",
        "|---|---|---|",
        f"| 사랑과 신뢰 | {w.F['부부간의 사랑과 신뢰']:.0f}% (1위) | 통계로 잴 수 없음 |",
        f"| 배우자의 경제적 여건 | {w.F['배우자의 경제적 여건']:.0f}% (남성보다 +{w['차이']['배우자의 경제적 여건']:.0f}%p) | **매우 큼**: 남성 소득 칸별 기혼율 {span['M']['income']:.0f}%p 차이 (일본) |",
        f"| 배우자의 일과 직장 | {w.F['배우자의 일과 직장']:.0f}% (+{w['차이']['배우자의 일과 직장']:.0f}%p) | **큼**: 남성 비정규직 −{m['emp','정규직'] - m['emp','비정규직']:.0f}%p (소득 같아도) |",
        f"| 평등한 관계 (가사 분담) | {w.F['평등한 관계 (공평한 가사분담 등)']:.0f}% (+{w['차이']['평등한 관계 (공평한 가사분담 등)']:.0f}%p) | 이번 자료로는 측정 불가 (시간사용조사로 다음 단계 가능) |",
        "| 학력 | 설문 항목 없음 | **작음**: 아내 학력이 더 높은 결혼이 더 흔함. 소득을 고정하면 남성 학력이 높을수록 기혼율이 오히려 낮음 |",
        "| 나이 (남편 연상) | 설문 항목 없음 | **줄어드는 중**: 아내 연상 결혼 20% |", "",
        "말과 실제가 같은 방향을 가리키는 것은 **경제적 안정**입니다. 반대로 학력과 나이는 흔히 생각하는 것보다 덜 중요해졌습니다.", "",
        "## 한계", "",
        "- 설문(한국 2021)과 결과(일본 2022)는 나라가 다릅니다. 한국에서 같은 분석을 하려면 소득별 미혼율 표가 필요합니다 (통계청 신혼부부·인구주택총조사 마이크로데이터, API 비공개).",
        "- 일본 자료는 **취업자만** 포함합니다. 일하지 않는 기혼 여성이 빠져 여성 기혼 비율은 실제보다 낮게 나옵니다.",
        "- 설문 문항은 '결혼 결정 시 고려 정도'이지 '배우자 조건 순위'가 아닙니다. 외모·성격처럼 공식 통계에 없는 기준은 다루지 못합니다.",
        "- 표본조사 추정치라 칸이 작은 곳(고소득 여성 등)은 흔들립니다.",
    ]
    (OUT / "report.md").write_text("\n".join(L), encoding="utf-8")
    print(OUT / "report.md"); print(stated_tab.round(1)); print(eff.round(1))


if __name__ == "__main__":
    main()
