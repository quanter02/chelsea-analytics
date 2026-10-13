"""출생아 예측: 반등은 언제 꺾이나 + 우리 동네 출생 수요.

세 층으로 나눈다.
  1. 세대 효과 (cohort): 연도별 출생아 수(1970~)로 각 나이의 여성 수를 재구성하고, 출산율을 2025년에 고정했을 때
     출생아가 어떻게 움직이는지 본다. 사람 수만으로 생기는 변화.
  2. 행동 효과 (rate): 실제 출생아 ÷ 세대 효과 = 출산율 변화. 2026년 반등이 어느 쪽에서 왔는지 가른다.
  3. 선행 지표: 혼인 증가율(1년 전)이 출생 증가율을 이끈다 (1995~2025 회귀).

시군구: 2025년 출생아 × 시도의 2026년 1~7월 증가율. 80% 범위는 과거 '시군구 증가율 − 시도 증가율' 흩어짐을
작은 동네일수록 넓게(c² + φ/n) 맞춘다 — 혼인 예측의 disp 규칙과 같은 꼴.

자료: KOSIS DT_1B8000F(연간 인구동태), DT_1B81A21(모의 연령별 출산율), DT_1B8000G(월별, 시도), DT_1B81A01(시군구 연간 출생).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data_births"
BANDS = {"모의 연령별출산율:15-19세": (15, 19), "20-24세": (20, 24), "25-29세": (25, 29), "30-34세": (30, 34),
         "35-39세": (35, 39), "40-44세": (40, 44), "45-49세": (45, 49)}
Z80 = 1.2816


def annual() -> pd.DataFrame:
    b = pd.read_csv(DATA / "_raw_1B8000F.csv")
    return b.pivot_table(index="year", columns="C1", values="value").rename(
        columns={11: "births", 30: "tfr", 31: "srb", 41: "marriages"})[["births", "tfr", "srb", "marriages"]]


def asfr_by_year() -> dict[int, np.ndarray]:
    a = pd.read_csv(DATA / "_raw_asfr.csv")
    a = a[a.C1_NM == "전국"]
    out = {}
    for y, g in a.groupby("year"):
        f = np.zeros(50)
        for _, r in g.iterrows():
            if r.ITM_NM in BANDS:
                lo, hi = BANDS[r.ITM_NM]
                f[lo:hi + 1] = r.value / 1000
        out[int(y)] = f
    return out


def women(t: int, ann: pd.DataFrame) -> np.ndarray:
    """t년의 15~49세 여성 수 ≈ 그 세대가 태어난 해의 여아 출생아 수 (사망·이동은 보정계수 k로)."""
    w = np.zeros(50)
    for a in range(15, 50):
        y0 = t - a
        if y0 in ann.index and not np.isnan(ann.births.get(y0, np.nan)):
            srb = ann.srb.get(y0, np.nan)
            srb = 107.0 if np.isnan(srb) else srb
            w[a] = ann.births[y0] * 100 / (100 + srb)
    return w


def cohort_index(years=range(2015, 2046), base_rate_year: int = 2025) -> pd.DataFrame:
    ann, f = annual(), asfr_by_year()
    k = np.mean([ann.births[y] / np.sum(women(y, ann) * f[y]) for y in range(2015, 2026)])
    rows = []
    for t in years:
        w = women(t, ann) * k
        rows.append({"year": t, "cohort": float(np.sum(w * f[base_rate_year])), "women_30_34": float(w[30:35].sum()),
                     "actual": ann.births.get(t, np.nan)})
    d = pd.DataFrame(rows).set_index("year")
    d["cohort_rel"] = d.cohort / d.cohort[base_rate_year]
    d.attrs["k"] = k
    return d


def marriage_lead() -> dict:
    ann = annual()
    g = pd.DataFrame({"gB": np.log(ann.births).diff(), "gM1": np.log(ann.marriages).diff().shift(1)}).loc[1995:2025].dropna()
    slope, icpt = np.polyfit(g.gM1, g.gB, 1)
    resid = g.gB - (slope * g.gM1 + icpt)
    return {"slope": slope, "intercept": icpt, "corr": float(np.corrcoef(g.gM1, g.gB)[0, 1]), "resid_sd": float(resid.std(ddof=2)), "n": len(g)}


def national(ytd_months: int = 7) -> dict:
    p = pd.read_csv(DATA / "national_monthly_2024_2026.csv", index_col=0)
    B25 = p.loc[202501:202512, "출생아수(명)"].sum()
    rb = p.loc[202601:202600 + ytd_months, "출생아수(명)"].sum() / p.loc[202501:202500 + ytd_months, "출생아수(명)"].sum()
    rm = p.loc[202601:202600 + ytd_months, "혼인건수(건)"].sum() / p.loc[202501:202500 + ytd_months, "혼인건수(건)"].sum()
    c = cohort_index()
    ml = marriage_lead()
    est26 = B25 * rb
    sd26 = 0.015                                                   # 남은 5개월 + 연말 몰림 변동, 판단값 (과거 월별 비율 흩어짐보다 넓게)
    g27 = ml["slope"] * np.log(rm) + ml["intercept"] + np.log(c.cohort_rel[2027] / c.cohort_rel[2026])
    est27 = est26 * np.exp(g27)
    sd27 = np.sqrt(sd26 ** 2 + ml["resid_sd"] ** 2)
    tfr25 = annual().tfr[2025]
    tfr26 = tfr25 * est26 / (B25 * c.cohort_rel[2026])
    return {"B25": B25, "ytd_ratio": rb, "marriage_ytd_ratio": rm, "est26": est26, "lo26": est26 * np.exp(-Z80 * sd26), "hi26": est26 * np.exp(Z80 * sd26),
            "est27": est27, "lo27": est27 * np.exp(-Z80 * sd27), "hi27": est27 * np.exp(Z80 * sd27), "g27": float(np.expm1(g27)),
            "tfr25": tfr25, "tfr26": tfr26, "lead": ml, "cohort": c}


def scenarios(n: dict, years=(2026, 2027, 2028, 2030, 2032, 2035, 2040)) -> pd.DataFrame:
    c, B25, t25, t26 = n["cohort"].cohort_rel, n["B25"], n["tfr25"], n["tfr26"]
    out = []
    for y in years:
        out.append({"year": y, "세대 효과": c[y],
                    "출산율 유지": B25 * c[y] * t26 / t25,
                    "3년에 걸쳐 2025 수준으로": B25 * c[y] * (t25 + (t26 - t25) * max(0.0, 1 - (y - 2026) / 3)) / t25,
                    "매년 +0.03 상승": B25 * c[y] * min(t26 + 0.03 * (y - 2026), 1.2) / t25,
                    "2026 출생 유지에 필요한 출산율": t26 * c[2026] / c[y]})
    return pd.DataFrame(out).set_index("year")


def sigungu(n: dict) -> pd.DataFrame:
    """시군구 2026 출생아 예측 + 80% 범위."""
    d = pd.read_csv(DATA / "_raw_sigungu_annual.csv", dtype={"C1": str})
    d = d[d.ITM_NM == "계"].pivot_table(index=["C1", "C1_NM"], columns="year", values="value").reset_index()
    d["sido"] = d.C1.str[:2]
    sido = d[d.C1.str.len() == 2].set_index("C1")
    sgg = d[d.C1.str.len() == 5].copy()
    # 과거 흩어짐: log(시군구 증가율) − log(시도 증가율), 작은 곳일수록 크다 → r² ≈ c² + φ/n
    res = []
    for y in range(2016, 2026):
        if y not in d.columns or y - 1 not in d.columns:
            continue
        g_sgg = np.log(sgg[y] / sgg[y - 1])
        g_sido = np.log(sgg.sido.map(sido[y]) / sgg.sido.map(sido[y - 1]))
        r = (g_sgg - g_sido).replace([np.inf, -np.inf], np.nan)
        res.append(pd.DataFrame({"r": r, "n": sgg[y - 1]}))
    R = pd.concat(res).dropna()
    R = R[(R.n > 0) & (R.r.abs() < 1.5)]
    X = np.c_[np.ones(len(R)), 1 / R.n]
    c2, phi = np.linalg.lstsq(X, R.r ** 2, rcond=None)[0]
    c2, phi = max(c2, 1e-4), max(phi, 0.0)
    # 2026 시도 증가율 (1~7월)
    s = pd.read_csv(DATA / "_raw_sido_monthly.csv", dtype={"C1": str})
    s["ym"] = s.PRD_DE.astype(int)
    ratio = {}
    for code, g in s.groupby("C1"):
        g = g.set_index("ym").value
        a, b = g.loc[202601:202607].sum(), g.loc[202501:202507].sum()
        ratio[code] = a / b if b > 0 else np.nan
    sgg["r_sido"] = sgg.sido.map(ratio).fillna(n["ytd_ratio"])
    sgg["pred"] = sgg[2025] * sgg.r_sido
    sd = np.sqrt(c2 + phi / sgg[2025].clip(lower=1))
    sgg["lo"] = sgg.pred * np.exp(-Z80 * sd)
    sgg["hi"] = sgg.pred * np.exp(Z80 * sd)
    # 같은 규칙의 과거 적중률 (그해 시도 연간 증가율을 안다고 놓고)
    hit = []
    for y in range(2017, 2026):
        p = sgg[y - 1] * (sgg.sido.map(sido[y]) / sgg.sido.map(sido[y - 1]))
        sdy = np.sqrt(c2 + phi / sgg[y - 1].clip(lower=1))
        ok = (sgg[y] >= p * np.exp(-Z80 * sdy)) & (sgg[y] <= p * np.exp(Z80 * sdy))
        hit.append(ok[sgg[y - 1] > 0].mean())
    codes = set(sgg.C1)
    sgg["general_gu"] = (sgg.C1.str[-1] != "0") & (sgg.C1.str[:4] + "0").isin(codes)   # 시 아래 일반구 (합계에서 빼야 이중 집계 안 됨)
    out = sgg.rename(columns={"C1": "code", "C1_NM": "name", 2024: "b2024", 2025: "b2025"})[["code", "name", "sido", "general_gu", "b2024", "b2025", "r_sido", "pred", "lo", "hi"]]
    out.attrs.update(c2=c2, phi=phi, coverage=float(np.mean(hit)))
    return out
