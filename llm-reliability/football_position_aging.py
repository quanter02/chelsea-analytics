"""포지션별 노화: 조율형(순수 MF)은 나이에 강하고, 활동량형(FW·측면)은 약한가.

    python football_position_aging.py   # data_cache/fbref/big5_player_standard.rds 필요 → output/football/position_aging.md

가설은 결과 보기 전에 장부 등록 (kind=football_hypothesis, pos_aging_P1~P3).
한계: fbref 포지션은 DF/MF/FW와 조합뿐이라 수비형·중앙 MF를 나누지 못함. 활동량(뛴 거리) 자료 없음 → 대리 지표 사용.
판정: 95% 범위 전체가 가설 쪽이면 맞음, 전체가 반대면 틀림, 걸치면 불확실.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr

from llmrel import ledger

ROOT = Path(__file__).parent
SRC = ROOT / "data_cache" / "fbref" / "big5_player_standard.rds"
OUT = ROOT / "output" / "football"
LAST_FULL = 2025


def load() -> pd.DataFrame:
    d = pyreadr.read_r(SRC)[None]
    d = d[d.Url.notna()].copy()
    d["contrib"] = d.npxG_Expected.fillna(0) + d.xAG_Expected.fillna(0)
    g = d.groupby(["Url", "Season_End_Year"]).agg(age=("Age", "min"), pos=("Pos", "first"), minutes=("Min_Playing", "sum"),
                                                 contrib=("contrib", "sum"), prgp=("PrgP_Progression", "sum"),
                                                 has_xg=("npxG_Expected", lambda s: s.notna().all()),
                                                 has_prg=("PrgP_Progression", lambda s: s.notna().all())).reset_index()
    g["age"] = pd.to_numeric(g.age.astype(str).str.split("-").str[0], errors="coerce")   # 2023/24부터 "29-123"(세-일) 표기
    g["grp"] = np.select([g.pos == "MF", g.pos.fillna("").str.contains("FW")], ["MF", "FW·측면"], "기타")
    nxt = g[["Url", "Season_End_Year", "minutes", "contrib", "prgp", "has_xg", "has_prg"]].copy()
    nxt["Season_End_Year"] -= 1
    p = g.merge(nxt, on=["Url", "Season_End_Year"], how="left", suffixes=("", "_n"))
    p = p[(p.Season_End_Year + 1 <= LAST_FULL) & (p.minutes >= 900) & p.age.between(18, 38)]
    p["minutes_n"] = p.minutes_n.fillna(0)
    p["keep"] = (p.minutes_n / p.minutes).clip(upper=2)
    return p


def logchg(x0, m0, x1, m1, eps):
    return np.log((x1 / (m1 / 90) + eps) / (x0 / (m0 / 90) + eps))


def boot_diff(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    d = [a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean() for _ in range(n)]
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def judge(lo, hi, ok):
    return "맞음" if ok(lo) and ok(hi) else ("틀림" if not ok(lo) and not ok(hi) else "불확실")


def main():
    p = load()
    old, prime = p.age.between(31, 33), p.age.between(25, 29)
    # P1 출전 유지율
    mf_k, fw_k = p[old & (p.grp == "MF")].keep.to_numpy(), p[old & (p.grp == "FW·측면")].keep.to_numpy()
    p1 = boot_diff(mf_k, fw_k)
    # P2 MF 전진 패스 vs FW 공격 기여
    q = p[(p.minutes_n >= 450)]
    mf = q[(q.grp == "MF") & q.has_prg & (q.has_prg_n == True) & (q.prgp > 0)].copy()  # noqa: E712
    mf["chg"] = logchg(mf.prgp, mf.minutes, mf.prgp_n, mf.minutes_n, 0.5)
    fw = q[(q.grp == "FW·측면") & q.has_xg & (q.has_xg_n == True) & (q.contrib / (q.minutes / 90) > 0.05)].copy()  # noqa: E712
    fw["chg"] = logchg(fw.contrib, fw.minutes, fw.contrib_n, fw.minutes_n, 0.02)
    p2 = boot_diff(mf[mf.age.between(31, 33)].chg.to_numpy(), fw[fw.age.between(31, 33)].chg.to_numpy())
    # P3 MF 전진 패스: 31~33 vs 25~29
    p3 = boot_diff(mf[mf.age.between(25, 29)].chg.to_numpy(), mf[mf.age.between(31, 33)].chg.to_numpy())
    pct = lambda t: tuple(np.expm1(v) for v in t)
    res = {
        "pos_aging_P1": dict(result=f"유지율 MF {mf_k.mean():.0%} vs FW·측면 {fw_k.mean():.0%}, 차이 {p1[0]:+.1%}p (95% {p1[1]:+.1%} ~ {p1[2]:+.1%}), n={len(mf_k)}/{len(fw_k)}",
                             verdict=judge(p1[1], p1[2], lambda v: v > 0)),
        "pos_aging_P2": dict(result=f"MF 전진 패스 {np.expm1(mf[mf.age.between(31, 33)].chg.mean()):+.1%} vs FW·측면 공격 기여 {np.expm1(fw[fw.age.between(31, 33)].chg.mean()):+.1%}, 로그 차이 {p2[0]:+.3f} (95% {p2[1]:+.3f} ~ {p2[2]:+.3f})",
                             verdict=judge(p2[1], p2[2], lambda v: v > 0)),
        "pos_aging_P3": dict(result=f"MF 전진 패스 하락 차이(25~29세 − 31~33세) {p3[0]:+.1%}p 근사 (95% {p3[1]:+.3f} ~ {p3[2]:+.3f} 로그)",
                             verdict=judge(p3[1], p3[2], lambda v: abs(v) <= 0.05)),
    }
    ledger.append([dict(kind="football_result", key=k, payload=v, model="position_aging", evidence="fbref 5대 리그, 3단계 판정") for k, v in res.items()])
    rows = []
    for a in range(24, 36):
        r = {"age": a}
        for g_ in ["MF", "FW·측면", "기타"]:
            r[f"keep_{g_}"] = p[(p.age == a) & (p.grp == g_)].keep.mean()
        r["mf_prg"] = np.expm1(mf[mf.age == a].chg.mean()) if (mf.age == a).any() else np.nan
        r["fw_con"] = np.expm1(fw[fw.age == a].chg.mean()) if (fw.age == a).any() else np.nan
        rows.append(r)
    t = "\n".join(f"| {r['age']} | {r['keep_MF']:.0%} | {r['keep_FW·측면']:.0%} | {r['keep_기타']:.0%} | {r['mf_prg']:+.1%} | {r['fw_con']:+.1%} |" for r in rows)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "position_aging.md").write_text(f"""# 포지션별 노화: 조율형 vs 활동량형

자료·조건은 aging.md와 같음 (fbref 5대 리그, 연속 두 시즌, 이번 시즌 900분+). 순수 MF = 그 시즌 포지션 표기가 'MF'만인 선수 (조율형 대리),
FW·측면 = 표기에 FW가 들어간 선수 (공격수·윙어, 활동량형 대리), 기타 = 수비수 등. 전진 패스는 2017/18~ 자료.

## 가설 판정 (결과 보기 전 등록)
| 가설 | 결과 | 판정 |
|---|---|---|
| P1 31~33세 출전 유지율: 순수 MF > FW·측면 (자신감 65%) | {res['pos_aging_P1']['result']} | {res['pos_aging_P1']['verdict']} |
| P2 31~33세 MF 전진 패스가 FW·측면 공격 기여보다 덜 떨어짐 (자신감 60%) | {res['pos_aging_P2']['result']} | {res['pos_aging_P2']['verdict']} |
| P3 MF 전진 패스는 31~33세도 25~29세와 하락 차이 5%p 이내 (자신감 55%) | {res['pos_aging_P3']['result']} | {res['pos_aging_P3']['verdict']} |

## 나이별
| 나이 | 유지율 MF | 유지율 FW·측면 | 유지율 기타 | MF 전진 패스 변화 | FW·측면 공격 기여 변화 |
|---|---|---|---|---|---|
{t}

## 한계
- 수비형·중앙 MF를 나누지 못하고, 뛴 거리(활동량) 자료가 없음 → 포지션 표기와 패스 지표로 대신함
- 전진 패스는 팀 전술(점유율)에 크게 좌우됨. 이적하면 바뀔 수 있음
- 생존자 편향: 다음 시즌에도 뛴 선수만 90분당 변화를 잼
""", encoding="utf-8")
    print((OUT / "position_aging.md").read_text())


if __name__ == "__main__":
    main()
