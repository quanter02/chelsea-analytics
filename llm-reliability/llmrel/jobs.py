"""일자리 구성 → 결혼: 일본 취업구조기본조사 2012·2017·2022 (도도부현) 로 만든 예측 모델.

핵심 식:  기혼 비율(지역, 해) = Σ_칸  인원 비중(칸) × 기혼 확률(칸)
  칸 = 나이 × 고용형태(정규·비정규·자영) × 본인 소득 구간
  인원 비중 = '일자리 구성' (산업·일자리 변화가 바꾸는 부분)
  칸별 기혼 확률 = '행동' (같은 조건의 사람이 결혼하는 정도)

그래서 두 시점 사이 변화를 '일자리 구성 효과'와 '행동 효과'로 나눌 수 있고,
앞 시점에서만 배운 확률로 다음 시점을 예측해 맞는지 확인할 수 있다.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .collect import _curl_json, clean_key

INCOME = {"50万円未満": "~199", "50～99万円": "~199", "100～149万円": "~199", "150～199万円": "~199",
          "200～249万円": "200~299", "250～299万円": "200~299", "300～399万円": "300~399", "400～499万円": "400~499",
          "500～599万円": "500~599", "600～699万円": "600~799", "700～799万円": "600~799", "800～899万円": "800+",
          "900～999万円": "800+", "1000～1249万円": "800+", "1250～1499万円": "800+", "1500万円以上": "800+"}
AGES = {"25～29歳": "25~29", "30～34歳": "30~34", "35～39歳": "35~39", "40～44歳": "40~44"}
EMP = {"うち正規の職員・従業員": "정규직", "正規の職員・従業員": "정규직", "うち非正規の職員・従業員": "비정규직",
       "非正規の職員・従業員": "비정규직", "うち自営業主": "자영업", "自営業主": "자영업"}
LEVELS = {"age": list(AGES.values()), "emp": ["정규직", "비정규직", "자영업"], "income": list(dict.fromkeys(INCOME.values()))}
REF = {"age": "30~34", "emp": "정규직", "income": "300~399"}
WAVES = {2012: "0003086551", 2017: "0003222823", 2022: "0004008500"}
PREFS = ",".join(["00000"] + [f"{i:02d}000" for i in range(1, 48)])   # 전국 + 47개 도도부현


def _get(sid: str, **filters) -> tuple[list, dict]:
    """e-Stat 통계 데이터 (10만 셀 넘으면 이어받기)."""
    app = clean_key("ESTAT_APP_ID")
    vals, names, start = [], {}, 1
    while True:
        for _ in range(3):
            code, body = _curl_json("https://api.e-stat.go.jp/rest/3.0/app/json/getStatsData",
                                    dict(appId=app, statsDataId=sid, metaGetFlg="Y" if start == 1 else "N", cntGetFlg="N",
                                         startPosition=start, **filters), timeout=120)
            if code and body.strip():
                break
        d = json.loads(body)["GET_STATS_DATA"]
        if d["RESULT"]["STATUS"] not in (0, 1):
            raise RuntimeError(f"e-Stat {sid}: {d['RESULT']['ERROR_MSG']}")
        sd = d["STATISTICAL_DATA"]
        if start == 1:
            for o in sd["CLASS_INF"]["CLASS_OBJ"]:
                cl = o["CLASS"] if isinstance(o["CLASS"], list) else [o["CLASS"]]
                names[o["@id"]] = [(c["@code"], c["@name"].replace("　", ""), c.get("@level")) for c in cl]
        v = sd.get("DATA_INF", {}).get("VALUE", [])
        vals += v if isinstance(v, list) else [v]
        nxt = sd.get("RESULT_INF", {}).get("NEXT_KEY")
        if not nxt or "limit" in filters:                                 # limit 지정 = 구조만 보려는 호출
            return vals, names
        start = int(nxt)


def fetch_wave(year: int) -> pd.DataFrame:
    """한 시점, 남녀 × 25~44세 × 고용형태 × 소득 × 지역(전국+도도부현)의 전체·미혼 인원."""
    sid = WAVES[year]
    rows = []
    if year == 2012:
        # 고용형태와 소득이 한 분류(cat03)에 묶여 있다: 고용형태 머리 코드 뒤에 소득 16구간이 이어짐
        _, names = _get(sid, cdCat01="202", cdCat04="001", cdArea="00000", limit=1)
        emp_of, cur = {}, None
        for code, nm, lv in names["cat03"]:
            if lv != "5":
                cur = EMP.get(nm)
            elif cur:
                emp_of[code] = (cur, INCOME.get(nm))
        codes = ",".join(c for c, v in emp_of.items() if v[1])
        lab = {k: {c: n for c, n, _ in v} for k, v in names.items()}
        for sex_code, sex in (("001", "M"), ("002", "F")):
            vals, _ = _get(sid, cdCat01="202,203,204,205", cdCat04=sex_code, cdCat03=codes, cdArea=PREFS)
            for v in vals:
                e, inc = emp_of[v["@cat03"]]
                rows.append(dict(area=v["@area"], sex=sex, age=AGES.get(lab["cat01"][v["@cat01"]]), emp=e, income=inc,
                                 kind="total" if v["@cat02"] == "000" else "unmarried", value=pd.to_numeric(v["$"], errors="coerce")))
    else:
        _, names = _get(sid, cdCat01="1", cdArea="00000", limit=1)
        lab = {k: {c: n for c, n, _ in v} for k, v in names.items()}
        inc_codes = ",".join(c for c, n in lab["cat04"].items() if n in INCOME)
        emp_codes = ",".join(c for c, n in lab["cat03"].items() if n in EMP)
        age_codes = ",".join(c for c, n in lab["cat05"].items() if n in AGES)
        for sex_code, sex in (("1", "M"), ("2", "F")):
            vals, _ = _get(sid, cdCat01=sex_code, cdCat03=emp_codes, cdCat04=inc_codes, cdCat05=age_codes, cdArea=PREFS)
            for v in vals:
                rows.append(dict(area=v["@area"], sex=sex, age=AGES[lab["cat05"][v["@cat05"]]], emp=EMP[lab["cat03"][v["@cat03"]]],
                                 income=INCOME[lab["cat04"][v["@cat04"]]], kind="total" if v["@cat02"] == "0" else "unmarried",
                                 value=pd.to_numeric(v["$"], errors="coerce")))
    df = pd.DataFrame(rows).dropna(subset=["age"])
    df = df[df.area.str.endswith("000")]                                  # 전국(00000) + 47개 도도부현
    w = df.pivot_table(index=["area", "sex", "age", "emp", "income"], columns="kind", values="value", aggfunc="sum").reset_index()
    w["year"] = year
    w = w.fillna({"unmarried": 0})
    w["married"] = (w.total - w.unmarried).clip(lower=0)
    return w[w.total > 0]


# ── 모델 ──────────────────────────────────────────────────────────────────
def _X(c: pd.DataFrame, areas: list | None = None) -> np.ndarray:
    cols = [np.ones(len(c))]
    for k, lv in LEVELS.items():
        cols += [(c[k] == x).to_numpy(float) for x in lv if x != REF[k]]
    if areas:                                                             # 지역 고정효과 (전국 기준 대비)
        cols += [(c.area == a).to_numpy(float) for a in areas[1:]]
    return np.column_stack(cols)


def fit(c: pd.DataFrame, areas: list | None = None, ridge: float = 1e-4) -> np.ndarray:
    X, y, n = _X(c, areas), c.married.to_numpy(), c.total.to_numpy()
    b = np.zeros(X.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-X @ b)); W = n * p * (1 - p) + 1e-9
        z = X @ b + (y - n * p) / W
        nb = np.linalg.solve(X.T @ (W[:, None] * X) + ridge * np.eye(len(b)), X.T @ (W * z))
        if np.max(np.abs(nb - b)) < 1e-10:
            return nb
        b = nb
    return b


def predict(c: pd.DataFrame, b: np.ndarray, areas: list | None = None, shift: float = 0.0) -> np.ndarray:
    """칸별 기혼 확률. shift = 모든 칸에 같은 크기로 더하는 로짓 변화 (행동 변화)."""
    return 1 / (1 + np.exp(-(_X(c, areas) @ b + shift)))


def share(c: pd.DataFrame, p: np.ndarray | None = None) -> float:
    """구성(인원)으로 가중한 기혼 비율. p 없으면 실제 값."""
    m = c.married if p is None else p * c.total
    return float(m.sum() / c.total.sum())


def behavior_shift(c_new: pd.DataFrame, b_old: np.ndarray, areas=None) -> float:
    """새 시점 구성에 옛 확률을 적용했을 때 실제와 같아지려면 로짓을 얼마나 옮겨야 하나 (전국 단일 값)."""
    target = c_new.married.sum()
    lo, hi = -3.0, 3.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if (predict(c_new, b_old, areas, mid) * c_new.total).sum() > target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def composition(c: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = c.groupby(by).total.sum()
    return (g / g.groupby(level=[i for i, b in enumerate(by) if b in ("sex", "age", "area", "year")] or None).transform("sum")).rename("share").reset_index()
