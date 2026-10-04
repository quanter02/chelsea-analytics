"""결혼할 때 무엇을 보나: 말하는 기준(설문) vs 실제로 드러나는 기준(누가 결혼했나).

말하는 기준   한국 보건사회연구원 가족과 출산 조사 2021, 미혼 남녀(19~49세)
              '결혼 결정 시 중요하게 고려해야 하는 정도' 9개 항목, 성별 (KOSIS orgId 331)
실제 결과     일본 취업구조기본조사 2022: 성·나이·소득·고용형태·학력별 미혼 인구 (취업자만, e-Stat)
              한국 신혼부부통계: 남편 × 아내 학력 조합 (2015~2024)
              한국 혼인통계: 초혼 부부 나이 차 (1990~2025)

결혼 여부는 두 사람의 선택과 경제 여건이 함께 만든 결과다. 그래서 '실제 결과'의 차이를
어느 한쪽의 선호로만 해석하지 않는다 (보고서에서 명시).
"""
from __future__ import annotations

import pandas as pd

from .official import estat, kosis

ITEMS = {  # 표 번호 끝자리 → 항목 이름
    "CG020": "본인의 경제적 여건", "CG021": "본인의 일과 직장", "CG022": "배우자의 경제적 여건", "CG023": "배우자의 일과 직장",
    "CG024": "평등한 관계 (공평한 가사분담 등)", "CG025": "양가와의 원만한 관계", "CG026": "안정된 주거 마련",
    "CG027": "자녀계획 일치", "CG028": "부부간의 사랑과 신뢰",
}
SCORE = {"전혀 중요하지 않다": 1, "별로 중요하지 않다": 2, "보통이다": 3, "약간 중요하다": 4, "매우 중요하다": 5}


def stated() -> pd.DataFrame:
    """항목 × 성별: '중요(약간+매우)' 비율, '매우 중요' 비율, 평균 점수(1~5)."""
    rows = []
    for code, item in ITEMS.items():
        d = kosis(f"DT_331001_2021{code}", "2021", "2021", prd="F", org="331", objL1="A01 A0301 A0302", objL2="ALL")
        for grp, g in d.groupby("C1_NM"):
            p = g[g.C2_NM.isin(SCORE)].set_index("C2_NM").value
            rows.append(dict(item=item, group={"전체": "all", "남자": "M", "여자": "F"}[grp],
                             important=p["약간 중요하다"] + p["매우 중요하다"], very=p["매우 중요하다"],
                             mean=sum(SCORE[k] * v for k, v in p.items()) / p.sum(),
                             n=int(g[g.C2_NM == "응답자수"].value.iloc[0])))
    return pd.DataFrame(rows)


def jp_revealed() -> pd.DataFrame:
    """취업자의 성·나이·고용형태·소득·학력별 인구와 미혼 인구 (2022)."""
    d = estat("0004008157", cdCat03="04,05,06")                 # 30~34, 35~39, 40~44세
    d = d.rename(columns={"cat01": "sex", "cat02": "marital", "cat03": "age", "cat04": "emp", "cat05": "income", "cat06": "edu"})
    d["sex"] = d.sex.map({"男": "M", "女": "F", "総数": "all"})
    d["marital"] = d.marital.map({"総数": "total", "うち未婚": "unmarried"})
    return d[["sex", "marital", "age", "emp", "income", "edu", "value"]]


def married_share(jp: pd.DataFrame, by: str, **fixed) -> pd.DataFrame:
    """한 요인(by)별 기혼 비율 = 1 − 미혼/전체. 나머지 차원은 fixed로 고정 (기본 '総数')."""
    base = {"emp": "総数", "income": "総数", "edu": "総数"}
    base.update(fixed); base.pop(by, None)
    x = jp
    for k, v in base.items():
        x = x[x[k] == v]
    w = x.pivot_table(index=["sex", "age", by], columns="marital", values="value", aggfunc="sum").reset_index()
    w["married_pct"] = 100 * (1 - w.unmarried / w.total)
    return w


def kr_education_pairs(years=range(2015, 2025)) -> pd.DataFrame:
    """초혼 신혼부부(혼인 1년차)의 남편 × 아내 학력 조합."""
    d = kosis("DT_1NW2006", str(min(years)), str(max(years)), objL1="1", objL2="10 20 30 40", objL3="10 20 30 40")
    d = d.rename(columns={"C2_NM": "husband", "C3_NM": "wife", "PRD_DE": "year_s"})
    d["year"] = d.year_s.astype(int)
    return d[["year", "husband", "wife", "value"]]


def kr_age_gap(start=1990, end=2025) -> pd.DataFrame:
    """초혼 부부 나이 차: 남자 연상 / 동갑 / 여자 연상 (전국)."""
    d = kosis("DT_1B83A08", str(start), str(end), objL1="00", objL2="10 11 12 13 14 20 30 31 32 33 34")
    d = d.rename(columns={"C2_NM": "gap", "C2": "gcode"})
    return d[["year", "gcode", "gap", "value"]]


# ── 평가 함수 추정: 조건별 기혼 확률 (가중 로지스틱 회귀) ─────────────────────
EMP = {"うち正規の職員・従業員": "정규직", "うち非正規の職員・従業員": "비정규직", "うち自営業主": "자영업"}
EDU = {"小学・中学（卒業者）": "중졸", "高校・旧制中（卒業者）": "고졸", "専門学校（２年未満）（卒業者）": "전문·단대",
       "専門学校（２～４年未満）（卒業者）": "전문·단대", "専門学校（４年以上）（卒業者）": "전문·단대", "短大（卒業者）": "전문·단대",
       "高専（卒業者）": "전문·단대", "大学（卒業者）": "대졸", "大学院（卒業者）": "대학원졸"}
INCOME = {"50万円未満": "~199", "50～99万円": "~199", "100～149万円": "~199", "150～199万円": "~199",
          "200～249万円": "200~299", "250～299万円": "200~299", "300～399万円": "300~399", "400～499万円": "400~499",
          "500～599万円": "500~599", "600～699万円": "600~799", "700～799万円": "600~799", "800～899万円": "800+",
          "900～999万円": "800+", "1000～1249万円": "800+", "1250～1499万円": "800+", "1500万円以上": "800+"}
AGE = {"30～34歳": "30~34", "35～39歳": "35~39", "40～44歳": "40~44"}
REF = {"age": "30~34", "emp": "정규직", "edu": "고졸", "income": "300~399"}
LEVELS = {"age": list(AGE.values()), "emp": list(dict.fromkeys(EMP.values())), "edu": list(dict.fromkeys(EDU.values())),
          "income": list(dict.fromkeys(INCOME.values()))}


def cells(jp: pd.DataFrame, sex: str) -> pd.DataFrame:
    """서로 겹치지 않는 칸(나이 × 고용형태 × 학력 × 소득)으로 묶은 기혼·전체 인원."""
    x = jp[(jp.sex == sex) & jp.emp.isin(EMP) & jp.edu.isin(EDU) & jp.income.isin(INCOME) & jp.age.isin(AGE)].copy()
    for k, mp in (("emp", EMP), ("edu", EDU), ("income", INCOME), ("age", AGE)):
        x[k] = x[k].map(mp)
    w = x.pivot_table(index=["age", "emp", "edu", "income"], columns="marital", values="value", aggfunc="sum").reset_index()
    w = w[w.total > 0]
    w["married"] = (w.total - w.unmarried).clip(lower=0)
    return w


def _design(c: pd.DataFrame) -> tuple:
    import numpy as np
    cols, X = ["const"], [np.ones(len(c))]
    for k, levels in LEVELS.items():
        for lv in levels:
            if lv != REF[k]:
                cols.append(f"{k}={lv}"); X.append((c[k] == lv).to_numpy(float))
    return np.column_stack(X), cols


def fit_logit(c: pd.DataFrame, iters: int = 50) -> pd.Series:
    """가중 로지스틱 회귀 (IRLS). 인원은 표본조사 추정치라 표준오차는 내지 않는다 (계수 크기만 해석)."""
    import numpy as np
    X, cols = _design(c)
    y, n = c.married.to_numpy(), c.total.to_numpy()
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        W = n * p * (1 - p) + 1e-9
        z = X @ b + (y - n * p) / W
        b_new = np.linalg.solve(X.T @ (W[:, None] * X) + 1e-6 * np.eye(len(b)), X.T @ (W * z))
        if np.max(np.abs(b_new - b)) < 1e-9:
            b = b_new; break
        b = b_new
    return pd.Series(b, index=cols)


def marginal_effects(c: pd.DataFrame, coef: pd.Series) -> pd.DataFrame:
    """조건 하나만 바꿨을 때 평균 기혼 확률 변화 (%p). 실제 인원 분포를 그대로 두고 그 요인만 바꿔 평균."""
    import numpy as np
    rows = []
    base_X, _ = _design(c)
    w = c.total.to_numpy()
    for k, levels in LEVELS.items():
        preds = {}
        for lv in levels:
            cc = c.copy(); cc[k] = lv
            X, _ = _design(cc)
            preds[lv] = float(np.average(1 / (1 + np.exp(-X @ coef.to_numpy())), weights=w))
        for lv in levels:
            rows.append(dict(factor=k, level=lv, married_pct=100 * preds[lv], vs_ref_pp=100 * (preds[lv] - preds[REF[k]])))
    return pd.DataFrame(rows)
