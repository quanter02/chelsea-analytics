"""베테랑 노화 곡선: 나이가 들면 출전 시간과 90분당 공격 기여가 얼마나 떨어지나 (원칙 3 '싼 베테랑' 시험).

    python football_aging.py     # data_cache/fbref/big5_player_standard.rds 필요 → output/football/aging.md, aging.png

자료: fbref 5대 리그 선수 시즌 기록 (worldfootballR_data 공개 사본). 원 권리는 Opta/Stats Perform → 연구용, 원자료는 레포에 올리지 않음.
가설은 결과 보기 전에 장부에 등록함 (kind=football_hypothesis, aging_H1~H3).
방법: 같은 선수를 연속 두 시즌 비교 (델타 방식). 이적한 시즌은 합쳐서 한 시즌으로.
  H1·H3 90분당 기여 = (npxG + xAG) / 90분, 공격수·미드필더, 이번 시즌 900분+, 다음 시즌 450분+ (xG는 2017/18~)
  H2 출전 유지율 = 다음 시즌 분 / 이번 시즌 분, 5대 리그에서 사라지면 0, 골키퍼 제외, 이번 시즌 900분+
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadr

from llmrel import ledger

ROOT = Path(__file__).parent
SRC = ROOT / "data_cache" / "fbref" / "big5_player_standard.rds"
OUT = ROOT / "output" / "football"
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False
LAST_FULL = 2025          # 2025/26 시즌까지 완료 (2026은 진행 중일 수 있어 다음 시즌 자료로 쓰지 않음)


def seasons() -> pd.DataFrame:
    d = pyreadr.read_r(SRC)[None]
    d = d[d.Url.notna()]
    d["pos"] = d.Pos.fillna("").str.split(",").str[0]
    d["contrib"] = d.npxG_Expected.fillna(0) + d.xAG_Expected.fillna(0)
    d["has_xg"] = d.npxG_Expected.notna()
    g = d.groupby(["Url", "Season_End_Year"]).agg(player=("Player", "first"), age=("Age", "min"), pos=("pos", "first"),
                                                 minutes=("Min_Playing", "sum"), contrib=("contrib", "sum"), has_xg=("has_xg", "all")).reset_index()
    g["age"] = pd.to_numeric(g.age.astype(str).str.split("-").str[0], errors="coerce")   # 2023/24부터 "29-123"(세-일) 표기
    return g


def pairs(g: pd.DataFrame) -> pd.DataFrame:
    nxt = g[["Url", "Season_End_Year", "minutes", "contrib", "has_xg"]].copy()
    nxt["Season_End_Year"] -= 1
    p = g.merge(nxt, on=["Url", "Season_End_Year"], how="left", suffixes=("", "_n"))
    p = p[(p.Season_End_Year + 1 <= LAST_FULL) & (p.minutes >= 900) & p.age.between(18, 38) & (p.pos != "GK")]
    p["minutes_n"] = p.minutes_n.fillna(0)
    p["keep"] = p.minutes_n / p.minutes
    return p


def boot_mean(x: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def per90_change(p: pd.DataFrame) -> pd.DataFrame:
    q = p[p.pos.isin(["FW", "MF"]) & p.has_xg & (p.has_xg_n == True) & (p.minutes_n >= 450)].copy()  # noqa: E712
    q["p0"] = q.contrib / (q.minutes / 90)
    q["p1"] = q.contrib_n / (q.minutes_n / 90)
    q = q[q.p0 > 0.05]                                            # 기여가 거의 0인 선수는 비율이 튀므로 제외
    q["chg"] = np.log((q.p1 + 0.02) / (q.p0 + 0.02))              # 로그 변화 (작은 값 보정)
    return q


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    g = seasons()
    p = pairs(g)
    q = per90_change(p)
    ages = range(20, 36)
    curve = pd.DataFrame([{"age": a, "n_keep": int((p.age == a).sum()), "keep": p[p.age == a].keep.clip(upper=2).mean(),
                           "n_per90": int((q.age == a).sum()), "per90": np.expm1(q[q.age == a].chg.mean()) if (q.age == a).any() else np.nan}
                          for a in ages])
    grp = lambda df, lo, hi: df[df.age.between(lo, hi)]
    # H1
    h1 = boot_mean(grp(q, 31, 33).chg.to_numpy())
    h1_pct = [np.expm1(v) for v in h1]
    # H2
    k31, k25 = grp(p, 31, 33).keep.clip(upper=2).to_numpy(), grp(p, 25, 29).keep.clip(upper=2).to_numpy()
    rng = np.random.default_rng(1)
    diff = [k25[rng.integers(0, len(k25), len(k25))].mean() - k31[rng.integers(0, len(k31), len(k31))].mean() for _ in range(2000)]
    h2 = (k25.mean() - k31.mean(), np.percentile(diff, 2.5), np.percentile(diff, 97.5))
    gone31, gone25 = (grp(p, 31, 33).minutes_n == 0).mean(), (grp(p, 25, 29).minutes_n == 0).mean()
    # H3
    c31, c25 = grp(q, 31, 32).chg.to_numpy(), grp(q, 25, 29).chg.to_numpy()
    d3 = [np.expm1(c25[rng.integers(0, len(c25), len(c25))].mean()) - np.expm1(c31[rng.integers(0, len(c31), len(c31))].mean()) for _ in range(2000)]
    h3 = (np.expm1(c25.mean()) - np.expm1(c31.mean()), np.percentile(d3, 2.5), np.percentile(d3, 97.5))
    def judge(lo, hi, ok):            # 3단계: 95% 범위 전체가 가설 쪽이면 맞음, 전체가 반대쪽이면 틀림, 걸치면 불확실(점추정 방향 표시)
        a, b_ = ok(lo), ok(hi)
        return "맞음" if a and b_ else ("틀림" if not a and not b_ else "불확실")
    verdict = {
        "aging_H1": dict(result=f"{h1_pct[0]:+.1%} (95% {h1_pct[1]:+.1%} ~ {h1_pct[2]:+.1%}), n={len(grp(q, 31, 33))}",
                         verdict=judge(h1_pct[1], h1_pct[2], lambda v: v > -0.10), point_supports=bool(h1_pct[0] > -0.10)),
        "aging_H2": dict(result=f"유지율 차이 {h2[0]:+.1%}p (95% {h2[1]:+.1%} ~ {h2[2]:+.1%}), 31~33세 {k31.mean():.0%} vs 25~29세 {k25.mean():.0%}",
                         verdict=judge(h2[1], h2[2], lambda v: v >= 0.20), point_supports=bool(h2[0] >= 0.20)),
        "aging_H3": dict(result=f"하락 차이 {h3[0]:+.1%}p (95% {h3[1]:+.1%} ~ {h3[2]:+.1%})",
                         verdict=judge(h3[1], h3[2], lambda v: abs(v) <= 0.05), point_supports=bool(abs(h3[0]) <= 0.05)),
    }
    for v in verdict.values():
        v["note"] = "판정 규칙(3단계)은 1차 실행 뒤 수정: 1차는 95% 하한만 봤음. 이전 줄은 지우지 않음"
    ledger.append([dict(kind="football_result", key=k, payload=v, model="aging_curve", evidence="fbref 5대 리그 2009/10~2025/26") for k, v in verdict.items()])

    # 그림: 나이별 출전 유지율, 90분당 기여 변화
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(curve.age, curve.keep * 100, "o-", color="#2a63d4", lw=2)
    ax[0].axvspan(30.5, 33.5, color="#df6330", alpha=.12)
    ax[0].set_title("다음 시즌 출전 시간 유지율 (%)", fontsize=11); ax[0].set_xlabel("이번 시즌 나이"); ax[0].grid(alpha=.25)
    ax[1].axhline(0, color="#9aa3ac", lw=1)
    ax[1].bar(curve.age, curve.per90 * 100, color=np.where(curve.age.between(31, 33), "#df6330", "#2a63d4"))
    ax[1].set_title("다음 시즌 90분당 공격 기여 변화 (%)", fontsize=11); ax[1].set_xlabel("이번 시즌 나이"); ax[1].grid(alpha=.25, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "aging.png", dpi=150); plt.close(fig)

    tbl = "\n".join(f"| {r.age} | {r.n_keep} | {r.keep:.0%} | {r.n_per90} | {r.per90:+.1%} |" for r in curve.itertuples() if r.n_keep)
    (OUT / "aging.md").write_text(f"""# 베테랑 노화 곡선 (원칙 3 시험)

자료: fbref 5대 리그 선수 시즌 기록 2009/10~2025/26, 같은 선수 연속 두 시즌 비교. 이번 시즌 900분 이상, 골키퍼 제외.
90분당 공격 기여 = npxG + xAG (2017/18~, 공격수·미드필더, 다음 시즌 450분 이상). 가설은 결과 보기 전 장부 등록. 판정: 95% 범위 전체가 가설 쪽이면 맞음, 전체가 반대면 틀림, 걸치면 불확실.

## 가설 판정
| 가설 | 결과 | 판정 |
|---|---|---|
| H1 31~33세 90분당 기여 하락 10% 미만 (자신감 60%) | {verdict['aging_H1']['result']} | {verdict['aging_H1']['verdict']}{' (점추정은 가설 쪽)' if verdict['aging_H1']['verdict'] == '불확실' and verdict['aging_H1']['point_supports'] else (' (점추정은 반대쪽)' if verdict['aging_H1']['verdict'] == '불확실' else '')} |
| H2 출전 유지율이 25~29세보다 20%p 이상 낮음 (자신감 70%) | {verdict['aging_H2']['result']} | {verdict['aging_H2']['verdict']}{' (점추정은 가설 쪽)' if verdict['aging_H2']['verdict'] == '불확실' and verdict['aging_H2']['point_supports'] else (' (점추정은 반대쪽)' if verdict['aging_H2']['verdict'] == '불확실' else '')} |
| H3 31~32세 하락이 25~29세와 5%p 이내 (자신감 50%) | {verdict['aging_H3']['result']} | {verdict['aging_H3']['verdict']}{' (점추정은 가설 쪽)' if verdict['aging_H3']['verdict'] == '불확실' and verdict['aging_H3']['point_supports'] else (' (점추정은 반대쪽)' if verdict['aging_H3']['verdict'] == '불확실' else '')} |

- 다음 시즌 5대 리그에서 사라진 비율: 31~33세 {gone31:.0%}, 25~29세 {gone25:.0%} (은퇴·하부 리그·해외 이적 포함, 실력 하락만은 아님)

![노화 곡선](aging.png)

## 나이별
| 나이 | 선수-시즌 | 다음 시즌 출전 유지율 | 90분당 표본 | 90분당 기여 변화 |
|---|---|---|---|---|
{tbl}

## 한계
- 다음 시즌에도 뛴 선수만 90분당 변화를 잴 수 있음 → 크게 떨어진 선수는 빠졌을 수 있어 하락이 덜 보일 수 있음 (생존자 편향)
- 출전 유지율 0에는 하부 리그·사우디·MLS 이적이 섞여 있음 (실력 하락만은 아님)
- 수비수의 기여는 이 지표로 못 잼. 가격(이적료·연봉) 자료 없음 → '가격 대비' 결론은 아직 못 냄
""", encoding="utf-8")
    print((OUT / "aging.md").read_text()[:2400])


if __name__ == "__main__":
    main()
