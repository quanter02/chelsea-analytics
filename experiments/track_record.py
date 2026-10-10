"""적중 기록 미리 재기: 채점 기간에 매달 뉴스레터를 냈다면, 그 호의 경보가 연말 확정치로 몇 % 맞았나.

규칙·기준값은 이미 정해진 것(보정 기간에서 고른 α, m)을 그대로 쓰고, 여기서 고르는 값은 없습니다.
- 발행 월 k(3·6·9월): k월까지 자료로 '유지' 중인 경보만 셈 (뉴스레터 표에 실리는 것과 같음).
- 같은 방향: 연간 이탈률의 부호가 경보 방향과 같음.
- 진짜 이탈: 같은 방향이면서 |연간 이탈률| ≥ 이탈 해 기준 (채점과 같은 정의).
- 평범한 해: |연간 이탈률| < 정상 해 기준 (= 잘못 울린 경보).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import kosis_monitor as K


def issue_alerts(out, k: int) -> pd.DataFrame:
    tp, cb, U = out["topic"], out["calibration"], out["units"]
    rows = []
    for (unit, y), u in U.items():
        if y not in tp.test or u["n"] < 12 or u["size"] < tp.min_size: continue
        mo, s = K.alarm(u, K._phi(cb, unit, y), cb.rule, cb.alpha, cb.m)
        if s == 0 or mo > k: continue
        S, Sc = u["e"][:k].sum(), u["scale_inc"][:k].sum()
        if not (np.sign(S) == s and abs(S / Sc) >= cb.m): continue      # 그달 '해소'면 표에 안 실림
        rows.append(dict(unit=unit, name=u["name"], year=y, dir=s, dev=u["dev"]))
    return pd.DataFrame(rows, columns=["unit", "name", "year", "dir", "dev"])


def precision(out, months=(3, 6, 9)) -> pd.DataFrame:
    cb = out["calibration"]
    res = []
    for k in months:
        A = issue_alerts(out, k)
        n_years = len(out["topic"].test)
        if len(A) == 0:
            res.append(dict(발행월=f"{k}월", 연평균_경보=0, 같은방향=np.nan, 진짜이탈=np.nan, 평범한해=np.nan)); continue
        same = np.sign(A.dev) == A.dir
        res.append(dict(발행월=f"{k}월", 연평균_경보=round(len(A) / n_years, 1), 같은방향=round(same.mean(), 3),
                        진짜이탈=round((same & (A.dev.abs() >= cb.dev)).mean(), 3), 평범한해=round((A.dev.abs() < cb.clear).mean(), 3)))
    return pd.DataFrame(res)
