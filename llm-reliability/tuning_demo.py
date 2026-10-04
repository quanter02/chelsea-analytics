"""점진 개선 실험: 모델 3개 × 채택 규칙 3가지 + 축구 이번 시즌 확인 + 맨유 남은 경기 예측 저장.

    python tuning_demo.py    # → output/tuning/report.md, data_epl/predictions_<날짜>.csv
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from llmrel import epl, forecast_tune as FT, regional_tune as RT, tuning
from llmrel.report import md_table

ROOT = Path(__file__).parent
OUT = ROOT / "output" / "tuning"
MODE = {"best": "평균 1등만 확인 (1차 규칙)", "naive": "조금이라도 좋으면 채택", "certain": "가장 확실한 개선부터 (최종 규칙)"}
COL = {"best": "#9aa3ac", "naive": "#d9622b", "certain": "#2a63d4"}
TEAM = "Manchester United"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(ROOT / "data_official" / "kr_jp_marriage_official.csv"); d["age_band"] = d.age_band.astype(str)
    sgg, sido = RT.load()
    df = epl.load()
    specs = [epl.make_spec(df), FT.make_spec(d), RT.make_spec(sgg, sido)]
    rows, hist = [], {}
    for sp in specs:
        for mode in MODE:
            r = tuning.hill_climb(sp, mode=mode, max_steps=25)
            h = r["history"]
            hist[(sp.name, mode)] = h
            rows.append(dict(model=sp.name, mode=mode, accepted=int(h.accepted.sum()) - 1, tried=r["tried"],
                             val_gain=h.val.iloc[0] - h[h.accepted].val.iloc[-1], test_gain=r["test_gain"], lo=r["test_lo"], hi=r["test_hi"],
                             test0=r["test_start"], n=r["n_test"], final=r["final"]))
    res = pd.DataFrame(rows)

    # 축구: 튜닝에 안 쓴 이번 시즌 경기로 확인
    live = {}
    for mode in MODE:
        p = res[(res.model == specs[0].name) & (res["mode"] == mode)].final.iloc[0]
        r = epl.run(df, **p); live[mode] = epl.losses(r[r.season == "2026-27"])
    r0 = epl.run(df, **epl.START); live0 = epl.losses(r0[r0.season == "2026-27"])
    live_tab = pd.DataFrame([{"규칙": "시작 규칙", "RPS": live0.mean(), "시작 대비": np.nan, "95% 구간": ""}] +
                            [{"규칙": MODE[m], "RPS": v.mean(), "시작 대비": -tuning.paired_ci(live0, v)[0],
                              "95% 구간": "{:+.4f} ~ {:+.4f}".format(*[-x for x in tuning.paired_ci(live0, v)[2:0:-1]])} for m, v in live.items()])
    n_live = len(live0)

    # 맨유 남은 경기 예측 (최종 규칙) 저장
    final = res[(res.model == specs[0].name) & (res["mode"] == "certain")].final.iloc[0]
    rr = epl.run(df, **final)
    today = dt.date.today().isoformat()
    fut = rr[(rr.season == "2026-27") & ~rr.played].copy()
    fut.insert(0, "made_on", today)
    fut[["made_on", "date", "home", "away", "pH", "pD", "pA", "xh", "xa"]].round(3).to_csv(ROOT / "data_epl" / f"predictions_{today}.csv", index=False)
    mu = fut[(fut.home == TEAM) | (fut.away == TEAM)].head(10)
    mu_played = rr[(rr.season == "2026-27") & rr.played & ((rr.home == TEAM) | (rr.away == TEAM))]

    # 차트: 모델별 검증 개선 vs 시험 개선
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, sp in zip(axes, specs):
        sub = res[res.model == sp.name].set_index("mode").reindex(list(MODE))
        x = np.arange(len(sub)); w = 0.38
        ax.bar(x - w / 2, sub.val_gain, w, color="#c3cad1", label="검증 구간 개선 (튜닝에 사용)")
        ax.bar(x + w / 2, sub.test_gain, w, color=[COL[m] for m in sub.index], label="시험 구간 개선 (잠가 둔 데이터)")
        ax.errorbar(x + w / 2, sub.test_gain, yerr=[sub.test_gain - sub.lo, sub.hi - sub.test_gain], fmt="none", ecolor="#333", capsize=3, lw=1)
        ax.axhline(0, color="#999", lw=.8)
        ax.set_xticks(x, ["평균 1등만", "조금이라도", "가장 확실한"], fontsize=9); ax.set_title(sp.name, fontsize=11); ax.grid(axis="y", alpha=.3)
    axes[0].legend(fontsize=8, loc="upper left"); axes[0].set_ylabel("손실 감소 (클수록 좋음)")
    fig.tight_layout(); fig.savefig(OUT / "gains.png", dpi=140); plt.close(fig)

    tab = res.assign(채택규칙=res["mode"].map(MODE), 바뀐규칙=res.final.map(lambda p: ", ".join(f"{k}={round(v, 3)}" for k, v in p.items())))
    tab = tab[["model", "채택규칙", "accepted", "tried", "val_gain", "test_gain", "lo", "hi", "바뀐규칙"]].rename(columns={
        "model": "모델", "accepted": "채택 수", "tried": "시도 수", "val_gain": "검증 개선", "test_gain": "시험 개선", "lo": "시험 95% 하한", "hi": "시험 95% 상한"})
    steps = hist[(specs[2].name, "certain")]
    steps = steps[steps.accepted][["step", "change", "val", "gain", "lo", "hi"]].rename(columns={"step": "단계", "change": "바꾼 규칙", "val": "검증 손실", "gain": "개선", "lo": "95% 하한", "hi": "95% 상한"})
    mu_tab = mu.assign(경기=mu.date + " " + mu.home + " vs " + mu.away)[["경기", "pH", "pD", "pA", "xh", "xa"]].rename(
        columns={"pH": "홈 승", "pD": "무", "pA": "원정 승", "xh": "홈 기대골", "xa": "원정 기대골"})
    played_tab = mu_played.assign(경기=mu_played.date + " " + mu_played.home + " " + mu_played.hg.astype(int).astype(str) + "-" + mu_played.ag.astype(int).astype(str) + " " + mu_played.away)[["경기", "pH", "pD", "pA"]].rename(
        columns={"pH": "예측 홈 승", "pD": "예측 무", "pA": "예측 원정 승"})

    L = [
        "# 점진 개선 실험: 규칙을 하나씩 바꿔 모델 성능 올리기", "",
        "모델 3개를 같은 실험대에 올렸습니다. 데이터를 시간 순서로 **학습 → 검증 → 시험**으로 나누고, 규칙(파라미터)을 한 번에 하나씩 바꿔 **검증 구간**에서 좋아지면 채택합니다. 시험 구간은 끝에서 한 번만 엽니다. 채택 방식 세 가지를 비교했습니다.", "",
        "| 채택 방식 | 내용 |", "|---|---|",
        "| 평균 1등만 확인 (1차 규칙) | 평균 개선이 가장 큰 후보 하나가 '확실히'(부트스트랩 95% 구간 > 0) 좋아질 때만 채택, 아니면 멈춤 |",
        "| 조금이라도 좋으면 채택 | 평균이 조금이라도 좋아지면 채택 (흔한 방식) |",
        "| 가장 확실한 개선부터 (최종 규칙) | 개선 95% 구간 하한이 가장 큰 후보를 고르고, 하한 > 0 일 때만 채택. 확실한 후보가 하나도 없을 때 멈춤 |", "",
        "| 모델 | 데이터 | 학습 / 검증 / 시험 | 손실 |", "|---|---|---|---|",
        "| EPL 경기 예측 | openfootball 프리미어리그 2010/11~2026/27 결과 6,130경기 | 2010/11~2020/21 / 2021/22~2023/24 / 2024/25~2025/26 | RPS (순위 확률 점수) |",
        "| 연령별 혼인율 예측 | KOSIS·e-Stat 한·일 연령별 혼인율 | 원점 ~2007 / 2008~2014 / 2015~2021 (1~3년 뒤) | |로그 오차| |",
        "| 시군구 혼인 건수 예측 | KOSIS 시군구 연간 혼인 2005~2025 (243곳) | 원점 ~2012 / 2013~2016 / 2017~2023 (1~2년 뒤) | |로그 오차| |", "",
        "## 1. 결과", "", "![개선](gains.png)", "",
        md_table(tab, {"검증 개선": "{:.4f}", "시험 개선": "{:+.4f}", "시험 95% 하한": "{:+.4f}", "시험 95% 상한": "{:+.4f}"}), "",
        "- **'조금이라도 좋으면 채택'은 검증 점수를 가장 많이 올리지만, 축구·혼인율에서는 시험에서 효과가 대부분 사라졌습니다** (95% 구간이 0을 포함). 검증 구간에 맞춰진 것입니다.",
        "- **1차 규칙(평균 1등만 확인)은 시군구 모델에서 너무 일찍 멈췄습니다.** 2단계의 평균 1등 후보가 불확실해서 멈췄는데, 그 뒤에 확실하고 큰 개선(시도 추세에 더 기대기)이 있었습니다.",
        "- **최종 규칙(가장 확실한 개선부터)은 세 모델 모두에서 시험 성능이 가장 좋았거나 공동 1위**였습니다.",
        "- 주의: 최종 규칙은 1차 규칙의 시험 결과를 보고 고친 것이라, 위 시험 구간은 이 비교에서 더 이상 깨끗하지 않습니다. 깨끗한 확인은 아래 축구 이번 시즌 경기와, 앞으로 나올 데이터로 합니다.", "",
        "### 시군구 모델에서 채택된 순서 (최종 규칙)", "",
        md_table(steps, {"검증 손실": "{:.4f}", "개선": "{:.4f}", "95% 하한": "{:.4f}", "95% 상한": "{:.4f}"}), "",
        "- 작은 시군구는 해마다 숫자가 크게 출렁여서, **자기 추세보다 소속 시도의 추세를 따르는 편(alpha=1)이 더 정확**했습니다. 추세는 최근 2년만 보는 게 가장 좋았습니다.", "",
        f"## 2. 깨끗한 확인: 2026/27 시즌 {n_live}경기 (튜닝에 쓰지 않은 데이터)", "",
        md_table(live_tab, {"RPS": "{:.4f}", "시작 대비": "{:+.4f}"}), "",
        f"- 방향은 시험 구간과 같습니다: 최종 규칙이 가장 좋고, '조금이라도 좋으면 채택'은 시작 규칙보다도 나빴습니다. 다만 {n_live}경기라 차이가 통계적으로 확실하지는 않습니다. 경기가 쌓이면 다시 잽니다.",
        f"- 기준선(홈·무·원정 비율만 사용) RPS {epl.baseline(df, 'live').mean():.4f}보다는 모두 낫습니다.", "",
        "## 3. 맨유 2026/27", "",
        "### 치른 경기 (경기 전 예측)", "", md_table(played_tab, {c: "{:.0%}" for c in ["예측 홈 승", "예측 무", "예측 원정 승"]}), "",
        f"### 남은 경기 예측 ({today} 기준, 최종 규칙)", "", md_table(mu_tab, {"홈 승": "{:.0%}", "무": "{:.0%}", "원정 승": "{:.0%}", "홈 기대골": "{:.2f}", "원정 기대골": "{:.2f}"}), "",
        f"- 예측은 `data_epl/predictions_{today}.csv`에 저장했습니다 (이번 시즌 남은 전 경기). 결과가 나오면 채점해 보정 기록을 공개합니다.",
        "- 먼 경기일수록 그 사이 결과를 모르는 상태의 예측이라 덜 정확합니다. 매 라운드 다시 예측하는 게 원칙입니다.", "",
        "## 한계", "",
        "- 축구 자료는 골 결과뿐입니다 (부상, 이적, 감독 교체, 기대득점, 배당률 없음). 북메이커 배당과 비교하지 못했습니다. 실제로 이기려면 배당보다 나아야 하는데, 이 모델은 그 수준이 아닐 가능성이 큽니다.",
        "- 시험 구간의 항목들은 서로 독립이 아닙니다 (같은 팀, 같은 지역이 여러 번 나옴). 그래서 95% 구간은 실제보다 좁게 나왔을 수 있습니다.",
        "- 모델 세 개는 일부러 단순하게 만들었습니다. 규칙 튜닝으로 얻는 개선은 손실의 0.4~10% 수준이고, 큰 개선은 새 정보(배당, 선수 정보, 월별 통계)에서 옵니다.",
    ]
    (OUT / "report.md").write_text("\n".join(L), encoding="utf-8")
    print(OUT / "report.md")
    print(tab.round(4).to_string()); print(live_tab.round(4).to_string()); print(mu_tab.round(2).to_string())


if __name__ == "__main__":
    main()
