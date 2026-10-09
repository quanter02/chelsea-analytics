"""미분양 월간 경보: 통일 점진 규칙 2차 vs 5차 (사전 등록 문서 experiments/preregistration_unsold_v2_v5.md 그대로)."""
import os
from math import log, sqrt
import numpy as np, pandas as pd

DATA = "data_kosis/unsold_housing_monthly_sigungu_2000_2026.csv"
DATA_URL = ("https://raw.githubusercontent.com/quanter02/chelsea-analytics/claude/great-dirac-3wwgzy/"
            "experiments/data_kosis/unsold_housing_monthly_sigungu_2000_2026.csv")
CAL, TEST, LIVE = range(2010, 2017), range(2017, 2026), 2026
FLOOR, MIN_L0 = 50, 100
DEV_Q, CLEAR_Q, FA_TARGET = 0.80, 0.50, 0.10
SIZE_CUTS, SIZE_LABELS = [100, 500, 2000, np.inf], ["100~500", "500~2천", "2천 이상"]
M_MULT = [0, 0.2, 0.4, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0]
A_V2 = [0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.01]
A_V5 = [0.99, 0.95, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.01]

def load(path=DATA):
    d = pd.read_csv(path if os.path.exists(path) else DATA_URL, dtype={"ym": str})
    d["year"] = d.ym.str[:4].astype(int); d["month"] = d.ym.str[4:].astype(int)
    return d

def build_units(d):
    """(시도, 시군구, 연도) → L0(작년 12월), 월별 수준 L1..L12, 공표된 달 수."""
    W = d.pivot_table(index=["sido", "sigungu"], columns=["year", "month"], values="unsold")
    years = sorted({y for y, _ in W.columns})
    U = {}
    for key, row in W.iterrows():
        for Y in years:
            L0 = row.get((Y - 1, 12), np.nan)
            if np.isnan(L0): continue
            L = np.array([row.get((Y, m), np.nan) for m in range(1, 13)])
            n = 0
            while n < 12 and not np.isnan(L[n]): n += 1
            if n == 0: continue
            U[(key, Y)] = dict(L0=L0, L=L, n=n, s=L0 + FLOOR)
    return U

def phi_from(U):
    r = []
    for (k, y), u in U.items():
        if y in CAL and u["L0"] >= MIN_L0 and u["n"] == 12:
            e = np.diff(np.r_[u["L0"], u["L"]])
            r.append((e ** 2 / u["s"]).mean())
    return float(np.mean(r))

def thresholds(U):
    x = np.abs([(u["L"][11] - u["L0"]) / u["s"] for (k, y), u in U.items() if y in CAL and u["L0"] >= MIN_L0 and u["n"] == 12])
    return float(np.quantile(x, DEV_Q)), float(np.quantile(x, CLEAR_Q))

def hw(V, V0, a):
    return sqrt((V + V0) * (log(1 + V / V0) + 2 * log(1 / a)))

def alarm(u, phi, rule, a, m):
    """rule: 'v2' = 하한/s > m, 'v5' = 하한 > 0 그리고 추정치/s > m, 'simple' = |추정치/s| ≥ m. 첫 경보 월과 방향."""
    s = u["s"]; V0 = 6 * phi * s
    for n in range(1, u["n"] + 1):
        S = u["L"][n - 1] - u["L0"]; V = n * phi * s
        h = hw(V, V0, a / 2) if rule != "simple" else 0.0
        if rule == "v2":
            if (S - h) / s > m: return n, 1
            if (S + h) / s < -m: return n, -1
        elif rule == "v5":
            if S - h > 0 and S / s > m: return n, 1
            if S + h < 0 and S / s < -m: return n, -1
        else:
            if S / s >= m and m > 0: return n, 1
            if S / s <= -m and m > 0: return n, -1
    return None, 0

def evaluate(U, years, phi, rule, a, m, dev, clear):
    rows = []
    for ((sd, sg), y), u in U.items():
        if y not in years or u["L0"] < MIN_L0 or u["n"] < 12: continue
        mo, sgn = alarm(u, phi, rule, a, m)
        rows.append(dict(sido=sd, sigungu=sg, year=y, L0=u["L0"], dev=(u["L"][11] - u["L0"]) / u["s"], alarm_m=mo, alarm_dir=sgn))
    R = pd.DataFrame(rows)
    R["규모"] = pd.cut(R.L0, SIZE_CUTS, right=False, labels=SIZE_LABELS)
    R["이탈"] = R.dev.abs() >= dev; R["정상"] = R.dev.abs() < clear
    R["감지"] = R.이탈 & (R.alarm_dir == np.sign(R.dev))
    return R

def summary(R):
    hit = R[R.감지]
    tot = dict(이탈해=int(R.이탈.sum()), 정상해=int(R.정상.sum()), 감지율=round(R.감지.sum() / max(R.이탈.sum(), 1), 3),
               상반기내_감지율=round(float((hit.alarm_m <= 6).sum()) / max(R.이탈.sum(), 1), 3),
               감지월_중앙값=float(hit.alarm_m.median()) if len(hit) else np.nan,
               잘못된경보율=round(float((R[R.정상].alarm_dir != 0).mean()), 3))
    grp = R.groupby("규모", observed=True).apply(lambda x: pd.Series(dict(
        단위연도=len(x), 이탈해=int(x.이탈.sum()), 감지율=round(x.감지.sum() / max(x.이탈.sum(), 1), 3),
        정상해=int(x.정상.sum()), 잘못된경보율=round(float((x[x.정상].alarm_dir != 0).mean()), 3) if x.정상.sum() else np.nan)))
    return tot, grp

def pick(U, phi, rule, dev, clear):
    """사전 등록된 순서·기준으로 (α, m) 선택."""
    ms = [x * clear for x in M_MULT]
    if rule == "v2":
        order = [(a, m) for m in ms for a in A_V2]
        ok = lambda tot, grp: tot["잘못된경보율"] <= FA_TARGET
    elif rule == "v5":
        order = [(a, m) for a in A_V5 for m in ms]
        ok = lambda tot, grp: bool((grp.잘못된경보율.fillna(0) <= FA_TARGET).all())
    else:
        order = [(None, m) for m in ms if m > 0]
        ok = lambda tot, grp: tot["잘못된경보율"] <= FA_TARGET
    for a, m in order:
        tot, grp = summary(evaluate(U, CAL, phi, rule, a, m, dev, clear))
        if ok(tot, grp): return a, m, tot, grp
    return a, m, tot, grp

def passes(tot, grp):
    return tot["잘못된경보율"] <= FA_TARGET and bool((grp.잘못된경보율.fillna(0) <= FA_TARGET).all())

def decide(t2, g2, t5, g5):
    p2, p5 = passes(t2, g2), passes(t5, g5)
    if p5 and not p2: return "5차 채택 (5차만 통과)"
    if p5 and p2: return "5차 채택 (둘 다 통과, 감지율 5%p 이내)" if t5["감지율"] >= t2["감지율"] - 0.05 else "2차 유지 (둘 다 통과, 5차 감지율이 5%p 넘게 낮음)"
    if p2: return "2차 유지 (2차만 통과)"
    return "둘 다 실패 → 미분양에서는 미검증"

def live(U, phi, rule, a, m):
    rows = []
    for ((sd, sg), y), u in U.items():
        if y != LIVE or u["L0"] < MIN_L0: continue
        mo, sgn = alarm(u, phi, rule, a, m)
        if sgn == 0: continue
        n = u["n"]
        rows.append(dict(지역=f"{sd} {sg}", 반영=f"1~{n}월", 경보월=f"{mo}월", 방향="증가" if sgn > 0 else "감소",
                         작년12월=int(u["L0"]), 최근=int(u["L"][n - 1]), 변화=int(u["L"][n - 1] - u["L0"])))
    return pd.DataFrame(rows).sort_values("변화", key=lambda s: -s.abs()).reset_index(drop=True) if rows else pd.DataFrame()
