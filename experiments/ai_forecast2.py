"""AI 예측 검증 2차 (사전 등록: preregistration_ai_forecast_v2.md).

1차와 다른 점
- 문항을 3개 주제로 넓힘: 시군구 순이동(플랜 A 예측보다 높은가), 미분양(전달보다 늘었나), 주택 매매 거래량(1년 전 같은 달보다 많은가).
- AI 예측을 손으로 고르지 않고, 모든 지역에 같은 정보 묶음과 같은 지시문으로 Claude API를 호출 (지시문 SHA-256 고정).
- 기권(0.5) 허용, 근거 한 줄. llm-reliability 프로파일(정답·확신 오답·기권)과 보정(ECE)까지 채점.

흐름: build() → 기준선 B0·B1 → ai_forecast() (API 키 필요) → freeze() 지문 → (공표 뒤) resolve() → score().
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd

import housing_lead as H
import kosis_monitor as K
import niche_candidates as N

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "ai_forecast", "v2")
TARGET = [f"2026{m:02d}" for m in (10, 11, 12)] + [f"2027{m:02d}" for m in (1, 2, 3)]   # 예측 대상 6개월
MODELS = ["claude-opus-5-5"]          # 비교할 AI. 다른 모델을 더하려면 여기 추가 (사전 등록 전에만)
HISTORY_MONTHS = 24                   # AI에게 주는 과거 월 수

SYSTEM = """당신은 한국 시군구 통계를 예측하는 분석가입니다. 주어진 과거 자료와 당신이 아는 지역 사정(정책 발표, 대규모 입주, 산업 변화 등)을 근거로, 각 달에 대해 질문이 '예'일 확률을 0~1로 답하세요.
- 근거가 약하면 0.5에 가깝게(기권) 답하세요. 확신이 있을 때만 0.2 이하나 0.8 이상을 쓰세요.
- 확률은 잘 보정되어야 합니다: 0.7이라고 답한 문항들은 실제로 약 70%가 '예'여야 합니다.
- 근거는 한두 문장으로, 지역 사정을 썼다면 무엇인지 적으세요."""

QUESTION = {
    "순이동": "그달 순이동(전입 − 전출)이 '플랜 A 예측값'보다 높은가?",
    "미분양": "그달 말 미분양 주택 수가 바로 전달보다 많은가?",
    "주택거래": "그달 주택 매매 거래량이 1년 전 같은 달보다 많은가?",
}


# ───────────────────────── 1. 문항 만들기 ─────────────────────────
def _wide(tidy, series=None):
    t = tidy if series is None else tidy[tidy.series == series]
    return t.pivot_table(index="unit", columns="ym", values="value", aggfunc="sum")


def _months_before(ym: str, n: int) -> list[str]:
    y, m = int(ym[:4]), int(ym[4:])
    out = []
    for _ in range(n):
        m -= 1
        if m == 0: y, m = y - 1, 12
        out.append(f"{y}{m:02d}")
    return out[::-1]


def build(cutoff: str) -> tuple[pd.DataFrame, dict]:
    """cutoff(YYYYMM)까지 공표된 자료로 문항·기준선·AI 정보 묶음을 만듦. cutoff 뒤 자료는 쓰지 않음."""
    rows, packets = [], {}
    # (1) 순이동: 1차와 같은 정의, 예측값 고정
    tp = N.CANDIDATES["E 전체 순이동"]
    tidy = K.fetch(tp, verbose=False); tidy = tidy[tidy.ym <= cutoff]
    U = K.units(tp, tidy)
    t = tidy.copy(); t["y"] = t.ym.str[:4].astype(int); t["m"] = t.ym.str[4:].astype(int)
    years = sorted(t.y.unique())
    win, wout = _wide(tidy, "in"), _wide(tidy, "out")
    for (unit, y), u in U.items():
        if y != int(cutoff[:4]) or u["size"] < tp.min_size or u["n"] != int(cutoff[4:]): continue   # 기준 달까지 자료가 끊긴 지역(행정구역 개편) 제외
        g = t[t.unit == unit]
        r = {s: {(a.y, a.m): a.value for a in g[g.series == s].itertuples()} for s in ("in", "out")}
        fc = {}
        for yy in sorted({int(x[:4]) for x in TARGET}):
            mi, _ = K._plan_a_flow(r["in"], yy, years) if yy <= int(cutoff[:4]) else (None, None)
            mo, _ = K._plan_a_flow(r["out"], yy, years) if yy <= int(cutoff[:4]) else (None, None)
            if mi is None:   # 다음 해 예측: 기준 해(작년)가 아직 끝나지 않았으면 올해 예측값을 그대로 씀(미리 정한 규칙)
                mi, _ = K._plan_a_flow(r["in"], int(cutoff[:4]), years); mo, _ = K._plan_a_flow(r["out"], int(cutoff[:4]), years)
            if mi is None: break
            for ym in TARGET:
                if int(ym[:4]) == yy: fc[ym] = float(mi[int(ym[4:]) - 1] - mo[int(ym[4:]) - 1])
        if len(fc) != len(TARGET): continue
        e = u["e"][:u["n"]]; b1 = ((e > 0).sum() + 1) / (len(e) + 2)
        hist = _months_before(_next(cutoff), HISTORY_MONTHS)
        net = (win.loc[unit, [m for m in hist if m in win.columns]] - wout.loc[unit, [m for m in hist if m in wout.columns]])
        key = f"순이동|{u['name']}"
        packets[key] = dict(주제="순이동", 지역=u["name"], 질문=QUESTION["순이동"], 과거_순이동=net.round(0).astype(int).to_dict(),
                            플랜A_예측값={m: round(v, 1) for m, v in fc.items()}, 올해_예측보다_높았던_달=f"{int((e > 0).sum())}/{len(e)}")
        rows += [dict(key=key, 주제="순이동", unit=unit, 지역=u["name"], ym=m, 비교값=round(fc[m], 1), B0=0.5, B1=round(b1, 4)) for m in TARGET]
    # (2) 미분양: 전달보다 많은가
    tp = K.TOPICS["미분양"]
    tidy = K.fetch(tp, verbose=False); tidy = tidy[tidy.ym <= cutoff]
    names = {unit: u["name"] for (unit, y), u in K.units(tp, tidy).items()}
    w = _wide(tidy)
    hist = _months_before(_next(cutoff), HISTORY_MONTHS)
    for unit, name in names.items():
        if unit not in w.index: continue
        s = w.loc[unit, [m for m in hist if m in w.columns]].dropna()
        if len(s) < 12 or s.iloc[-1] < 20: continue                      # 미분양 20호 미만 지역 제외
        d = np.diff(s.values[-13:]); b1 = ((d > 0).sum() + 1) / (len(d) + 2)
        key = f"미분양|{name}"
        packets[key] = dict(주제="미분양", 지역=name, 질문=QUESTION["미분양"], 과거_미분양=s.astype(int).to_dict())
        rows += [dict(key=key, 주제="미분양", unit=unit, 지역=name, ym=m, 비교값=np.nan, B0=0.5, B1=round(b1, 4)) for m in TARGET]
    # (3) 주택 거래량: 1년 전 같은 달보다 많은가
    tidy = K.fetch(H.TRADES, verbose=False); tidy = tidy[tidy.ym <= cutoff]
    names = H.trade_names(tidy)
    w = _wide(tidy)
    for unit, name in names.items():
        if unit not in w.index: continue
        s = w.loc[unit, [m for m in hist if m in w.columns]].dropna()
        if len(s) < 18 or s.iloc[-12:].sum() < 300: continue             # 최근 1년 거래 300건 미만 제외
        yoy = (s.values[-6:] > s.values[-18:-12]); b1 = (yoy.sum() + 1) / (len(yoy) + 2)
        key = f"주택거래|{name}"
        packets[key] = dict(주제="주택거래", 지역=name, 질문=QUESTION["주택거래"], 과거_거래량=s.astype(int).to_dict())
        rows += [dict(key=key, 주제="주택거래", unit=unit, 지역=name, ym=m, 비교값=np.nan, B0=0.5, B1=round(b1, 4)) for m in TARGET]
    return pd.DataFrame(rows), packets


def _next(ym: str) -> str:
    y, m = int(ym[:4]), int(ym[4:]) + 1
    return f"{y + (m > 12)}{(m - 1) % 12 + 1:02d}"


# ───────────────────────── 2. AI 예측 (Claude API) ─────────────────────────
SCHEMA = {
    "type": "object",
    "properties": {
        "probabilities": {"type": "object", "properties": {m: {"type": "number"} for m in TARGET},
                          "required": TARGET, "additionalProperties": False},
        "reason": {"type": "string"},
    },
    "required": ["probabilities", "reason"],
    "additionalProperties": False,
}


def prompt_hash() -> str:
    return hashlib.sha256(json.dumps([SYSTEM, QUESTION, SCHEMA, TARGET], ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def ai_forecast(packets: dict, model: str, limit: int | None = None) -> pd.DataFrame:
    """지역·주제마다 한 번 호출해 6개월 확률을 받음. ANTHROPIC_API_KEY(또는 ant 로그인)가 필요."""
    import anthropic
    client = anthropic.Anthropic()
    out = []
    for i, (key, p) in enumerate(packets.items()):
        if limit and i >= limit: break
        user = "아래 자료로 대상 달마다 질문이 '예'일 확률을 답하세요.\n대상 달: " + ", ".join(TARGET) + "\n" + json.dumps(p, ensure_ascii=False)
        try:
            r = client.beta.messages.create(
                model=model, max_tokens=4000, system=SYSTEM,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            )
        except anthropic.RateLimitError:
            raise
        except anthropic.APIStatusError as e:
            out.append(dict(key=key, model=model, error=f"{e.status_code}")); continue
        if r.stop_reason == "refusal":
            out.append(dict(key=key, model=model, error="refusal")); continue
        text = next((b.text for b in r.content if b.type == "text"), "{}")
        try:
            j = json.loads(text)
        except json.JSONDecodeError:
            out.append(dict(key=key, model=model, error="json")); continue
        for m in TARGET:
            pr = float(np.clip(j["probabilities"].get(m, 0.5), 0.01, 0.99))
            out.append(dict(key=key, model=model, served_by=r.model, ym=m, p=pr, reason=j.get("reason", "")))
    return pd.DataFrame(out)


# ───────────────────────── 3. 고정 · 채점 ─────────────────────────
def freeze(Q: pd.DataFrame, packets: dict, ai: pd.DataFrame | None, tag: str) -> str:
    os.makedirs(OUT, exist_ok=True)
    P = Q.copy()
    if ai is not None and len(ai):
        for model, g in ai.dropna(subset=["p"]).groupby("model"):
            P = P.merge(g[["key", "ym", "p"]].rename(columns={"p": f"AI:{model}"}), on=["key", "ym"], how="left")
    path = os.path.join(OUT, f"predictions_{tag}.csv")
    P.to_csv(path, index=False)
    json.dump(packets, open(os.path.join(OUT, f"packets_{tag}.json"), "w", encoding="utf-8"), ensure_ascii=False)
    if ai is not None: ai.to_csv(os.path.join(OUT, f"ai_raw_{tag}.csv"), index=False)
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    open(path + ".sha256", "w").write(f"{h}\nprompt {prompt_hash()}\n")
    return h


def resolve(P: pd.DataFrame) -> pd.DataFrame:
    """공표된 달만 답을 채움. 순이동은 고정된 예측값과, 미분양은 전달과, 거래량은 1년 전 같은 달과 비교."""
    P = P.copy(); P["답"] = np.nan
    mig = _wide(K.fetch(N.CANDIDATES["E 전체 순이동"], verbose=False), "in") - _wide(K.fetch(N.CANDIDATES["E 전체 순이동"], verbose=False), "out")
    uns = _wide(K.fetch(K.TOPICS["미분양"], verbose=False))
    trd = _wide(K.fetch(H.TRADES, verbose=False))
    prev_month = lambda ym: _months_before(ym, 1)[0]
    last_year = lambda ym: f"{int(ym[:4]) - 1}{ym[4:]}"
    for i, r in P.iterrows():
        if r.주제 == "순이동" and r.unit in mig.index and r.ym in mig.columns and not np.isnan(mig.loc[r.unit, r.ym]):
            P.at[i, "답"] = float(mig.loc[r.unit, r.ym] > r.비교값)
        elif r.주제 == "미분양" and r.unit in uns.index and r.ym in uns.columns and prev_month(r.ym) in uns.columns:
            a, b = uns.loc[r.unit, r.ym], uns.loc[r.unit, prev_month(r.ym)]
            if not (np.isnan(a) or np.isnan(b)): P.at[i, "답"] = float(a > b)
        elif r.주제 == "주택거래" and r.unit in trd.index and r.ym in trd.columns and last_year(r.ym) in trd.columns:
            a, b = trd.loc[r.unit, r.ym], trd.loc[r.unit, last_year(r.ym)]
            if not (np.isnan(a) or np.isnan(b)): P.at[i, "답"] = float(a > b)
    return P


def _ece(p, y, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def score(R: pd.DataFrame, n_boot=4000, seed=0) -> dict:
    R = R.dropna(subset=["답"])
    ai_cols = [c for c in R.columns if c.startswith("AI:")]
    out = dict(문항=len(R), 주제별=R.groupby("주제").size().to_dict())
    for c in ["B0", "B1", *ai_cols]:
        x = R.dropna(subset=[c])
        out[c] = dict(브라이어=float(((x[c] - x.답) ** 2).mean()), ECE=_ece(x[c].values, x.답.values),
                      주제별_브라이어={k: float(((g[c] - g.답) ** 2).mean()) for k, g in x.groupby("주제")})
    rng = np.random.default_rng(seed)
    for c in ai_cols:
        x = R.dropna(subset=[c]).assign(d=lambda t: (t.B1 - t.답) ** 2 - (t[c] - t.답) ** 2)
        g = x.groupby("key").d.agg(["sum", "count"])
        boots = [(lambda b: b["sum"].sum() / b["count"].sum())(g.iloc[rng.integers(0, len(g), len(g))]) for _ in range(n_boot)]
        lo, hi = np.quantile(boots, [0.05, 0.95])
        p = x[c]; hit = (p > 0.5) == (x.답 == 1)
        out[c].update(개선=float(x.d.mean()), 개선_90구간=(float(lo), float(hi)),
                      판정="AI 우위 확인" if lo > 0 else "방향만 맞음 (불확실)" if x.d.mean() > 0 else "AI 우위 확인 안 됨",
                      보정_판정="보정 양호" if out[c]["ECE"] <= 0.05 else "보정 미흡",
                      프로파일=dict(기권=float(((p >= 0.4) & (p <= 0.6)).mean()), 정답=float((hit & ((p < 0.4) | (p > 0.6))).mean()),
                                  확신_오답=float((~hit & ((p <= 0.2) | (p >= 0.8))).mean())))
    return out
