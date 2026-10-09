"""KOSIS 월별 자료 → 백테스트 → 실시간 경보: 범용 파이프라인.

설정(Topic) 하나만 바꾸면 새 주제에 같은 절차를 적용한다.
  1. fetch()      KOSIS에서 받기 (40,000셀 한도에 걸리면 기간 → 분류 순으로 자동 분할, 빈 구간은 건너뜀, 파일로 저장)
  2. units()      지역 × 연도 단위로 '예측 · 증거 · 분산 · 규모' 만들기 (흐름 / 순흐름 / 재고)
  3. calibrate()  보정 기간에서 판정 기준(α, m)을 사전 등록된 절차로 고르기
  4. backtest()   채점 기간 성적 (전체 · 규모별)
  5. live()       올해 공표분 기준 실시간 경보 목록
  6. report()     위 결과를 마크다운 한 장으로

판정 규칙: 통일 점진 규칙 (experiments/unified_incremental_rule.ipynb)
  v2 (현재 채택)  하한/규모 > m 이면 판정. 보정: 가장 작은 m → 가장 민감한 α, 보정 기간 전체 잘못된 경보 ≤ 10%
  v5 (후보)       하한 > 0 그리고 추정치/규모 > m. 보정: 가장 민감한 α → 가장 작은 m, 모든 규모 구간 각각 ≤ 10%
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from math import log, sqrt
from typing import Callable

import numpy as np
import pandas as pd

KOSIS_DATA = "https://kosis.kr/openapi/Param/statisticsParameterData.do"
KOSIS_META = "https://kosis.kr/openapi/statisticsData.do"
CELL_LIMIT_MSG = "40,000"


# ───────────────────────── 설정 ─────────────────────────
@dataclass
class Series:
    """KOSIS에서 받을 계열 하나. objs: {"objL1": "ALL", "objL2": "0", ...}. 단위 열(unit_cols) 외의 분류는 합산."""
    itm: str
    objs: dict


@dataclass
class Topic:
    name: str
    table: str
    org: str = "101"
    kind: str = "flow"                     # flow: 건수 1개 / net: 전입(in)−전출(out) / stock: 재고 수준
    series: dict = field(default_factory=dict)   # {"y": Series} (flow·stock) 또는 {"in": Series, "out": Series} (net)
    unit_cols: tuple = ("C1",)             # 지역 단위를 정하는 분류 열 (예: ("C1",) 또는 ("C1", "C2"))
    unit_filter: Callable | None = None    # (unit, name) → bool
    start: str = "200001"
    end: str = "202612"
    cal: range = range(2003, 2013)
    test: range = range(2013, 2026)
    live_year: int = 2026
    min_size: float = 300                  # 평가 대상 최소 규모 (flow: 작년 건수, net: 작년 전입, stock: 작년 12월 수준)
    thresholds: tuple = ("fixed", 0.10, 0.05)    # ("fixed", 이탈, 정상) 또는 ("quantile", 0.8, 0.5)
    size_cuts: tuple = (0, 1000, 3000, np.inf)
    size_labels: tuple = ("소", "중", "대")
    stock_floor: float = 50                # 재고 규모 = 작년 12월 + floor
    cache: str | None = None


# ───────────────────────── 1. 받기 ─────────────────────────
def _key(key=None):
    if key: return key
    k = (os.environ.get("KOSIS_API_KEY") or "").strip().strip("\"'")
    if not k:
        try:
            from google.colab import userdata   # 코랩 '보안 비밀'
            k = userdata.get("KOSIS_API_KEY")
        except Exception:
            pass
    if not k: raise RuntimeError("KOSIS_API_KEY가 필요합니다 (환경 변수 또는 코랩 보안 비밀)")
    return k


def _get(url, params, tries=6):
    """KOSIS 연결이 가끔 끊기므로 간격을 늘려 가며 다시 시도."""
    q = url + "?" + urllib.parse.urlencode(params)
    for i in range(tries):
        try:
            return json.loads(urllib.request.urlopen(q, timeout=120).read())
        except Exception:
            if i == tries - 1: raise
            time.sleep(min(60, 2 ** (i + 1)))


def obj_values(topic, obj_id, key=None):
    """분류 하나(예: objL1)에 들어갈 수 있는 값 목록 (자동 분할용)."""
    meta = _get(KOSIS_META, dict(method="getMeta", type="ITM", orgId=topic.org, tblId=topic.table, apiKey=_key(key), format="json", jsonVD="Y"))
    order = []
    for m in meta:
        if m.get("OBJ_ID") not in order and m.get("OBJ_ID") != "ITEM": order.append(m["OBJ_ID"])
    idx = int(obj_id[-1]) - 1
    oid = order[idx]
    return [m["ITM_ID"] for m in meta if m.get("OBJ_ID") == oid]


def _part_path(topic, s, itm, objs, start, end):
    if not topic.cache: return None
    import hashlib
    h = hashlib.md5(json.dumps([topic.table, s, itm, objs, start, end], sort_keys=True).encode()).hexdigest()[:16]
    return os.path.join(topic.cache + ".parts", f"{h}.json")


def _fetch_range(topic, s, itm, objs, start, end, key, log_):
    part = _part_path(topic, s, itm, objs, start, end)
    if part and os.path.exists(part):                     # 이미 받은 묶음 → 이어받기
        d = json.load(open(part, encoding="utf-8"))
        return [pd.DataFrame(d)] if d else []
    p = dict(method="getList", apiKey=key, orgId=topic.org, tblId=topic.table, itmId=itm, format="json", jsonVD="Y",
             prdSe="M", startPrdDe=start, endPrdDe=end, **objs)
    d = _get(KOSIS_DATA, p)
    if part and (isinstance(d, list) or CELL_LIMIT_MSG not in str(d.get("errMsg", d))):
        os.makedirs(os.path.dirname(part), exist_ok=True)
        json.dump(d if isinstance(d, list) else [], open(part, "w", encoding="utf-8"), ensure_ascii=False)
    if isinstance(d, list):
        return [pd.DataFrame(d)]
    msg = str(d.get("errMsg", d))
    if CELL_LIMIT_MSG not in msg:          # 자료 없음 등 → 건너뜀
        log_(f"  {s} {start}~{end}: {msg[:40]} → 건너뜀")
        return []
    y0, y1 = int(start[:4]), int(end[:4])
    if y1 > y0:                            # 기간을 반으로
        mid = (y0 + y1) // 2
        return (_fetch_range(topic, s, itm, objs, start, f"{mid}12", key, log_) +
                _fetch_range(topic, s, itm, objs, f"{mid + 1}01", end, key, log_))
    for ok, ov in objs.items():            # 한 해도 크면 'ALL'인 분류를 값별로
        if ov == "ALL":
            out = []
            for v in obj_values(topic, ok, key):
                out += _fetch_range(topic, s, itm, {**objs, ok: v}, start, end, key, log_)
            return out
    raise RuntimeError(f"{s} {start}~{end}: 더 나눌 수 없음")


def _plan(topic, objs, key):
    """표 구조로 한 번에 받을 수 있는 기간(년)을 계산. 한 해도 넘으면 가장 큰 'ALL' 분류를 값별로 나눔.
    반환: [(objs, 한 번에 받을 햇수)]"""
    meta = _get(KOSIS_META, dict(method="getMeta", type="ITM", orgId=topic.org, tblId=topic.table, apiKey=key, format="json", jsonVD="Y"))
    order = []
    for m in meta:
        if m.get("OBJ_ID") not in order and m.get("OBJ_ID") != "ITEM": order.append(m["OBJ_ID"])
    vals = {f"objL{i + 1}": [m["ITM_ID"] for m in meta if m.get("OBJ_ID") == oid] for i, oid in enumerate(order)}
    def per_month(o):
        c = 1
        for k, v in o.items():
            c *= len(vals.get(k, [])) if v == "ALL" else len(str(v).split())
        return c
    cm = per_month(objs)
    years = (40000 // max(cm, 1)) // 12
    if years >= 1:
        return [(objs, years)]
    span = int(topic.end[:4]) - int(topic.start[:4]) + 1
    def calls(k):                      # 이 분류를 값별로 나누면 필요한 요청 수
        pm = cm // max(len(vals.get(k, [])), 1)
        y = max(1, (40000 // max(pm, 1)) // 12)
        return len(vals.get(k, [])) * -(-span // y)
    big = min((k for k, v in objs.items() if v == "ALL"), key=calls)   # 요청 수가 가장 적어지는 분류로 나눔
    out = []
    for v in vals[big]:
        out += _plan(topic, {**objs, big: v}, key) if per_month({**objs, big: v}) * 12 > 40000 else [({**objs, big: v}, (40000 // per_month({**objs, big: v})) // 12)]
    return out


REPO_RAW = "https://raw.githubusercontent.com/quanter02/chelsea-analytics/claude/great-dirac-3wwgzy/experiments/"


def search(keyword: str, org: str | None = None, key=None, n=30) -> pd.DataFrame:
    """KOSIS 통계표 검색. 새 주제를 찾을 때: search("시군구 월별 출생")"""
    p = dict(method="getList", apiKey=_key(key), searchNm=keyword, sort="RANK", startCount=1, resultCount=n, format="json", jsonVD="Y")
    if org: p["orgId"] = org
    d = _get("https://kosis.kr/openapi/statisticsSearch.do", p)
    if not isinstance(d, list): return pd.DataFrame()
    return pd.DataFrame([dict(org=x.get("ORG_ID"), table=x.get("TBL_ID"), 이름=x.get("TBL_NM"),
                              시작=x.get("STRT_PRD_DE"), 끝=x.get("END_PRD_DE")) for x in d])


def describe(table: str, org: str = "101", key=None):
    """표 구조: 주기(월별이 있는지, 최신 시점)와 분류·항목. Topic.series의 objs를 정할 때 씀."""
    k = _key(key)
    prd = _get(KOSIS_META, dict(method="getMeta", type="PRD", orgId=org, tblId=table, apiKey=k, format="json", jsonVD="Y"))
    itm = _get(KOSIS_META, dict(method="getMeta", type="ITM", orgId=org, tblId=table, apiKey=k, format="json", jsonVD="Y"))
    order = []
    for m in itm:
        if m.get("OBJ_ID") not in order: order.append(m.get("OBJ_ID"))
    rows = []
    for i, oid in enumerate(order):
        ms = [m for m in itm if m.get("OBJ_ID") == oid]
        rows.append(dict(인자="itmId" if oid == "ITEM" else f"objL{i if order[0] == 'ITEM' else i + 1}", 분류=ms[0].get("OBJ_NM"),
                         값개수=len(ms), 예시=", ".join(f"{m['ITM_ID']}={m['ITM_NM']}" for m in ms[:6])))
    return pd.DataFrame(prd), pd.DataFrame(rows)


def fetch(topic: Topic, key=None, verbose=True) -> pd.DataFrame:
    """정돈된 표: unit, name, ym, series, value.
    순서: ① topic.cache 파일 ② 저장소에 올려 둔 같은 파일(코랩용) ③ KOSIS에서 받기(키 필요)."""
    if topic.cache and os.path.exists(topic.cache):
        return pd.read_csv(topic.cache, dtype={"unit": str, "ym": str})
    if topic.cache:
        try:
            d = pd.read_csv(REPO_RAW + topic.cache, dtype={"unit": str, "ym": str})
            os.makedirs(os.path.dirname(topic.cache) or ".", exist_ok=True); d.to_csv(topic.cache, index=False)
            return d
        except Exception:
            pass
    key = _key(key)
    log_ = print if verbose else (lambda *a: None)
    rows = []
    y0, y1 = int(topic.start[:4]), int(topic.end[:4])
    for s, ser in topic.series.items():
        plan = _plan(topic, ser.objs, key)
        log_(f"{topic.name} · {s}: 요청 묶음 {len(plan)}개 × 기간 분할")
        for objs, step in plan:
            step = max(1, min(step, y1 - y0 + 1))
            for a in range(y0, y1 + 1, step):
                st = topic.start if a == y0 else f"{a}01"
                en = topic.end if a + step - 1 >= y1 else f"{a + step - 1}12"
                for df in _fetch_range(topic, s, ser.itm, objs, st, en, key, log_):   # 혹시 넘으면 기존 방식으로 다시 분할
                    df["series"] = s; rows.append(df)
        log_(f"{topic.name} · {s} 완료")
    d = pd.concat(rows, ignore_index=True)
    d["value"] = pd.to_numeric(d.DT, errors="coerce")
    d["unit"] = d[list(topic.unit_cols)].astype(str).agg("|".join, axis=1)
    name_cols = [c + "_NM" for c in topic.unit_cols]
    d["name"] = d[name_cols].astype(str).agg(" ".join, axis=1)
    out = d.groupby(["unit", "name", "PRD_DE", "series"], as_index=False).value.sum(min_count=1).rename(columns={"PRD_DE": "ym"})
    if topic.cache:
        os.makedirs(os.path.dirname(topic.cache) or ".", exist_ok=True); out.to_csv(topic.cache, index=False)
    return out


# ───────────────────────── 2. 단위 만들기 ─────────────────────────
SIDO_STD = {"11": "서울", "12": "전남광주", "26": "부산", "27": "대구", "28": "인천", "29": "광주", "30": "대전", "31": "울산",
            "36": "세종", "41": "경기", "43": "충북", "44": "충남", "46": "전남", "47": "경북", "48": "경남", "50": "제주",
            "51": "강원", "52": "전북"}   # 행정표준코드 (인구이동 표). 혼인 표처럼 자료 안에 시도 행이 있으면 그 이름을 씀
SIDO_SHORT = {"서울특별시": "서울", "부산광역시": "부산", "대구광역시": "대구", "인천광역시": "인천", "광주광역시": "광주",
              "대전광역시": "대전", "울산광역시": "울산", "세종특별자치시": "세종", "경기도": "경기", "강원도": "강원",
              "강원특별자치도": "강원", "충청북도": "충북", "충청남도": "충남", "전라북도": "전북", "전북특별자치도": "전북",
              "전라남도": "전남", "경상북도": "경북", "경상남도": "경남", "제주특별자치도": "제주"}
def _plan_a_flow(row, Y, years):
    """흐름: 작년 연간 × 최근 5년 평균 월별 비중."""
    if any(Y - i not in years for i in range(1, 6)): return None, None
    M = np.array([[row.get((y, m), np.nan) for m in range(1, 13)] for y in [Y - i for i in range(1, 6)]])
    if np.isnan(M).any() or (M.sum(1) <= 0).any(): return None, None
    shares = (M / M.sum(1, keepdims=True)).mean(0)
    return M[0].sum() * shares, M[0].sum()


def units(topic: Topic, tidy: pd.DataFrame) -> dict:
    """{(unit, Y): dict(e, var_raw, scale_inc, n, size, dev, name)}. var_raw는 φ를 곱하기 전 분산."""
    t = tidy.copy()
    t["year"] = t.ym.str[:4].astype(int); t["month"] = t.ym.str[4:].astype(int)
    names = t.drop_duplicates("unit").set_index("unit").name
    sido = {u: str(n).strip() for u, n in names.items() if len(u) == 2}   # 필터 전에 시도 이름 확보 (자료마다 코드 체계가 다름)
    if topic.unit_filter is not None:  # 필터는 정리 전 원래 이름에 적용 (예: "서울 계" 제외)
        keep = [u for u in names.index if topic.unit_filter(u, names[u])]
        t = t[t.unit.isin(keep)]; names = names[keep]
    names = names.map(lambda x: str(x).replace(" ", "") if len(str(x).replace(" ", "")) <= 3 else str(x))   # "남  구" → "남구"
    def full(u):                       # 시군구 코드(5자리) 앞에 시도 이름
        n = names[u]
        if len(u) == 5 and u.isdigit():
            sd = sido.get(u[:2]) or SIDO_STD.get(u[:2])
            if sd: return f"{SIDO_SHORT.get(sd, sd)} {n}"
        return n
    names = pd.Series({u: full(u) for u in names.index})
    wide = {s: t[t.series == s].pivot_table(index="unit", columns=["year", "month"], values="value") for s in t.series.unique()}
    years = sorted({y for w in wide.values() for y, _ in w.columns})
    U = {}
    first = next(iter(wide.values()))
    for unit in first.index:
        rows = {s: (w.loc[unit] if unit in w.index else None) for s, w in wide.items()}
        if any(r is None for r in rows.values()): continue
        for Y in years:
            act = {s: np.array([r.get((Y, m), np.nan) for m in range(1, 13)]) for s, r in rows.items()}
            n = 0
            while n < 12 and all(not np.isnan(a[n]) for a in act.values()): n += 1
            if n == 0: continue
            if topic.kind == "flow":
                mu, prev = _plan_a_flow(rows["y"], Y, years)
                if mu is None: continue
                e = np.nan_to_num(act["y"] - mu); var = mu; sc = mu; size = prev
                dev = act["y"].sum() / mu.sum() - 1 if n == 12 else np.nan
            elif topic.kind == "net":
                mi, prev = _plan_a_flow(rows["in"], Y, years); mo, _ = _plan_a_flow(rows["out"], Y, years)
                if mi is None or mo is None: continue
                e = np.nan_to_num((act["in"] - act["out"]) - (mi - mo)); var = mi + mo; sc = mi; size = prev
                dev = e.sum() / mi.sum() if n == 12 else np.nan
            elif topic.kind == "stock":
                L0 = rows["y"].get((Y - 1, 12), np.nan)
                if np.isnan(L0): continue
                s_ = L0 + topic.stock_floor
                L = act["y"]
                e = np.nan_to_num(np.diff(np.r_[L0, L])); var = np.full(12, s_); sc = np.r_[s_, np.zeros(11)]; size = L0
                dev = (L[11] - L0) / s_ if n == 12 else np.nan
            else:
                raise ValueError(topic.kind)
            U[(unit, Y)] = dict(e=e, var_raw=np.asarray(var, float), scale_inc=np.asarray(sc, float), n=n, size=size, dev=dev, name=names[unit])
    return U


# ───────────────────────── 3~5. 판정 · 보정 · 채점 · 실시간 ─────────────────────────
def _hw(V, V0, a):
    return sqrt((V + V0) * (log(1 + V / V0) + 2 * log(1 / a)))


def alarm(u, phi, rule, a, m):
    """통일 규칙. 첫 판정 월과 방향(+1 증가 / −1 감소)."""
    S = V = Sc = 0.0
    var = phi * u["var_raw"]; V0 = float(var[:6].sum())
    for i in range(u["n"]):
        S += u["e"][i]; V += var[i]; Sc += u["scale_inc"][i]
        h = _hw(V, V0, a / 2)
        if rule == "v2":
            if (S - h) / Sc > m: return i + 1, 1
            if (S + h) / Sc < -m: return i + 1, -1
        elif rule in ("v5", "v6"):        # v6: 판정은 v5와 같고 φ만 지역별 (unit_phis)
            if S - h > 0 and S / Sc > m: return i + 1, 1
            if S + h < 0 and S / Sc < -m: return i + 1, -1
        else:
            raise ValueError(rule)
    return None, 0


@dataclass
class Calibration:
    phi: float
    dev: float
    clear: float
    rule: str
    alpha: float
    m: float
    cal_summary: dict
    phi_u: dict | None = None             # v6: (unit, 연도) → 지역별 φ


K_SHRINK = 24   # v6: 지역별 φ를 전체 φ 쪽으로 당기는 강도 (월 수)


def unit_phis(U, phi_g, k=K_SHRINK):
    """v6: 지역별 φ = (k·전체 φ + 그 지역의 직전 5년 월별 (e²/분산) 합) / (k + 월 수). 해당 연도 이전 자료만 씀."""
    out = {}
    for (unit, Y) in U:
        r = []
        for y in range(Y - 5, Y):
            u = U.get((unit, y))
            if u and u["n"] == 12:
                r.extend(((u["e"] ** 2) / np.maximum(u["var_raw"], 1e-9)).tolist())
        out[(unit, Y)] = (k * phi_g + float(np.sum(r))) / (k + len(r))
    return out


def _phi(cb, unit, y):
    return cb.phi_u[(unit, y)] if cb.rule == "v6" and cb.phi_u is not None else cb.phi


A_V2 = [0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.01]
A_V5 = [0.99, 0.95, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.01]
M_MULT = [0, 0.2, 0.4, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0]
FA_TARGET = 0.10


def _evaluate(topic, U, years, cb, a, m):
    rows = []
    for (unit, y), u in U.items():
        if y not in years or u["size"] < topic.min_size or u["n"] < 12: continue
        mo, s = alarm(u, _phi(cb, unit, y), cb.rule, a, m)
        rows.append(dict(unit=unit, name=u["name"], year=y, size=u["size"], dev=u["dev"], alarm_m=mo, alarm_dir=s))
    R = pd.DataFrame(rows)
    R["규모"] = pd.cut(R["size"], list(topic.size_cuts), right=False, labels=list(topic.size_labels))
    R["이탈"] = R.dev.abs() >= cb.dev; R["정상"] = R.dev.abs() < cb.clear
    R["감지"] = R.이탈 & (R.alarm_dir == np.sign(R.dev))
    return R


def _summary(R):
    hit = R[R.감지]
    tot = dict(단위연도=len(R), 이탈해=int(R.이탈.sum()), 정상해=int(R.정상.sum()),
               감지율=round(R.감지.sum() / max(R.이탈.sum(), 1), 3),
               상반기내_감지율=round(float((hit.alarm_m <= 6).sum()) / max(R.이탈.sum(), 1), 3),
               감지월_중앙값=float(hit.alarm_m.median()) if len(hit) else np.nan,
               잘못된경보율=round(float((R[R.정상].alarm_dir != 0).mean()), 3) if R.정상.sum() else np.nan)
    grp = R.groupby("규모", observed=True).apply(lambda x: pd.Series(dict(
        단위연도=len(x), 이탈해=int(x.이탈.sum()), 감지율=round(x.감지.sum() / max(x.이탈.sum(), 1), 3),
        정상해=int(x.정상.sum()), 잘못된경보율=round(float((x[x.정상].alarm_dir != 0).mean()), 3) if x.정상.sum() else np.nan)))
    return tot, grp


def calibrate(topic: Topic, U: dict, rule="v2") -> Calibration:
    sel = [(k, u) for k, u in U.items() if k[1] in topic.cal and u["size"] >= topic.min_size and u["n"] == 12]
    phi = float(np.mean([((u["e"] ** 2) / u["var_raw"]).mean() for _, u in sel]))
    x = np.abs([u["dev"] for _, u in sel])
    if topic.thresholds[0] == "fixed":
        dev, clear = topic.thresholds[1], topic.thresholds[2]
    else:
        dev, clear = float(np.quantile(x, topic.thresholds[1])), float(np.quantile(x, topic.thresholds[2]))
    ms = [k * clear for k in M_MULT]
    cb = Calibration(phi, dev, clear, rule, None, None, {}, unit_phis(U, phi) if rule == "v6" else None)
    if rule == "v2":
        order = [(a, m) for m in ms for a in A_V2]
        ok = lambda t, g: t["잘못된경보율"] <= FA_TARGET
    else:                              # v5·v6: 가장 민감한 α 먼저 → 가장 작은 m, 모든 규모 구간 각각 ≤ 10%
        order = [(a, m) for a in A_V5 for m in ms]
        ok = lambda t, g: bool((g.잘못된경보율.fillna(0) <= FA_TARGET).all())
    for a, m in order:
        t, g = _summary(_evaluate(topic, U, topic.cal, cb, a, m))
        if ok(t, g):
            cb.alpha, cb.m, cb.cal_summary = a, m, t
            return cb
    cb.alpha, cb.m, cb.cal_summary = a, m, t
    return cb


def backtest(topic, U, cb):
    R = _evaluate(topic, U, topic.test, cb, cb.alpha, cb.m)
    t, g = _summary(R)
    return R, t, g


def live(topic, U, cb, year=None):
    year = year or topic.live_year
    rows = []
    for (unit, y), u in U.items():
        if y != year or u["size"] < topic.min_size: continue
        mo, s = alarm(u, _phi(cb, unit, y), cb.rule, cb.alpha, cb.m)
        if s == 0: continue
        n = u["n"]; Sc = u["scale_inc"][:n].sum(); S = u["e"][:n].sum()
        rows.append(dict(지역=u["name"], 반영=f"1~{n}월", 경보월=f"{mo}월", 방향="증가" if s > 0 else "감소",
                         예측대비_누적차이=round(float(S), 1), 이탈률=f"{S / Sc:+.1%}",
                         상태="유지" if np.sign(S) == s and abs(S / Sc) >= cb.m else "해소"))
    if not rows: return pd.DataFrame(columns=["지역", "반영", "경보월", "방향", "예측대비_누적차이", "이탈률", "상태"])
    return pd.DataFrame(rows).sort_values("예측대비_누적차이", key=lambda s: -s.abs()).reset_index(drop=True)


def run(topic: Topic, tidy: pd.DataFrame | None = None, rule="v2", key=None, verbose=True):
    """한 번에: 받기 → 단위 → 보정 → 채점 → 실시간."""
    tidy = fetch(topic, key=key, verbose=verbose) if tidy is None else tidy
    U = units(topic, tidy)
    cb = calibrate(topic, U, rule)
    R, t, g = backtest(topic, U, cb)
    L = live(topic, U, cb)
    return dict(topic=topic, units=U, calibration=cb, results=R, test=t, groups=g, live=L)


def _md(df, index=True):
    try:
        return df.to_markdown(index=index)
    except Exception:          # tabulate가 없으면 일반 표
        return "```\n" + df.to_string(index=index) + "\n```"


def report(out) -> str:
    tp, cb, t, g, L = out["topic"], out["calibration"], out["test"], out["groups"], out["live"]
    lines = [f"# {tp.name}: 월간 경보 리포트", "",
             f"- 자료: KOSIS `{tp.table}` (orgId {tp.org}), 종류 `{tp.kind}`",
             f"- 판정 규칙: 통일 점진 규칙 {cb.rule} · 보정 기간 {tp.cal.start}~{tp.cal.stop - 1}에서 고른 값 α = {cb.alpha}, m = {cb.m:.3f}",
             f"- 이탈 해: |이탈률| ≥ {cb.dev:.1%}, 정상 해: < {cb.clear:.1%}, 과분산 φ = {cb.phi:.2f}", "",
             f"## 채점 기간 {tp.test.start}~{tp.test.stop - 1}", "",
             f"감지율 **{t['감지율']:.0%}** (이탈 해 {t['이탈해']}건), 상반기 내 {t['상반기내_감지율']:.0%}, 감지 월 중앙값 {t['감지월_중앙값']:.0f}월, "
             f"잘못된 경보 **{t['잘못된경보율']:.1%}** (정상 해 {t['정상해']}건)", "",
             _md(g), "",
             f"## 실시간 경보 ({tp.live_year}년 공표분)", "",
             f"경보 {len(L)}곳 (유지 {int((L.상태 == '유지').sum()) if len(L) else 0}곳)", "",
             _md(L.head(30), index=False) if len(L) else "(경보 없음)"]
    return "\n".join(lines)


# ───────────────────────── 이미 검증한 주제 설정 ─────────────────────────
def _sgg_marriage(unit, name):          # 시·군·자치구 (일반구·개편 전 계열 제외)
    return len(unit) == 5 and unit.endswith("0") and "변동전" not in name


TOPICS = {
    "혼인": Topic(
        name="시군구 혼인", table="DT_1B83A35", kind="flow",
        series={"y": Series("T3", {"objL1": "ALL"})}, unit_cols=("C1",), unit_filter=_sgg_marriage,
        start="199701", end="202512", cal=range(2003, 2013), test=range(2013, 2026), live_year=2025,
        min_size=300, thresholds=("fixed", 0.10, 0.05), size_cuts=(0, 1000, 3000, np.inf), size_labels=("300~1천", "1천~3천", "3천 이상"),
        cache="data_kosis/pipeline_marriage.csv"),
    "청년 순이동": Topic(
        name="시군구 청년(20~39세) 순이동", table="DT_1B26001", kind="net",
        series={"in": Series("T10", {"objL1": "ALL", "objL2": "0", "objL3": "120 130 150 160"}),
                "out": Series("T20", {"objL1": "ALL", "objL2": "0", "objL3": "120 130 150 160"})},
        unit_cols=("C1",), unit_filter=lambda u, n: len(u) == 5,
        start="200501", end="202608", cal=range(2010, 2018), test=range(2018, 2026), live_year=2026,
        min_size=2000, thresholds=("quantile", 0.8, 0.5), size_cuts=(0, 5000, 15000, np.inf), size_labels=("2천~5천", "5천~1.5만", "1.5만 이상"),
        cache="data_kosis/pipeline_youth_migration.csv"),
    "출생": Topic(
        name="시군구 출생", table="DT_1B81A01", kind="flow",
        series={"y": Series("T1", {"objL1": "ALL"})}, unit_cols=("C1",), unit_filter=_sgg_marriage,
        start="199701", end="202512", cal=range(2003, 2013), test=range(2013, 2026), live_year=2025,
        min_size=300, thresholds=("quantile", 0.8, 0.5), size_cuts=(0, 1000, 3000, np.inf), size_labels=("300~1천", "1천~3천", "3천 이상"),
        cache="data_kosis/pipeline_births.csv"),
    "미분양": Topic(
        name="시군구 미분양 주택", table="DT_MLTM_2082", org="116", kind="stock",
        series={"y": Series("13103871087T1", {"objL1": "ALL", "objL2": "ALL"})}, unit_cols=("C1", "C2"),
        unit_filter=lambda u, n: not n.endswith(" 계") and not n.startswith("전국"),
        start="200012", end="202608", cal=range(2010, 2017), test=range(2017, 2026), live_year=2026,
        min_size=100, thresholds=("quantile", 0.8, 0.5), size_cuts=(100, 500, 2000, np.inf), size_labels=("100~500", "500~2천", "2천 이상"),
        stock_floor=50, cache="data_kosis/pipeline_unsold.csv"),
}


def tidy_from_legacy(kind: str, path: str) -> pd.DataFrame:
    """이전 실험에서 저장한 CSV를 파이프라인 형식으로 (회귀 검사용)."""
    d = pd.read_csv(path, dtype={"code": str, "ym": str})
    if kind == "혼인":
        return d.rename(columns={"code": "unit", "marriages": "value"}).assign(series="y")[["unit", "name", "ym", "series", "value"]]
    if kind == "청년 순이동":
        a = d.rename(columns={"code": "unit", "in_": "value"}).assign(series="in")[["unit", "name", "ym", "series", "value"]]
        b = d.rename(columns={"code": "unit", "out": "value"}).assign(series="out")[["unit", "name", "ym", "series", "value"]]
        return pd.concat([a, b], ignore_index=True)
    raise ValueError(kind)
