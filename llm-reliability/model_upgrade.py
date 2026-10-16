"""시군구 혼인 예측 개선 후보: 사전등록 → 시험 구간(2019~2025) 한 번 판정.

    python model_upgrade.py      # 후보를 장부에 등록(같은 내용이면 건너뜀)하고 시험 구간 채점 → output/model_upgrade/report.md

진단(2010~2018만 사용, 2026-10-16): 오차 분산의 94%가 시군구 고유 흔들림, 고유 오차의 1년 자기상관 −0.25(되돌림).
후보 (모든 상수는 2010~2018에서 정함, 시험 구간에서 고치지 않음)
  A  되돌림 보정: 예측 = 작년 + 시도 1~6월 증가율 − ρ·(작년 고유 변화), ρ = 0.25
     작년 고유 변화 = (작년 − 재작년 로그 건수) − 작년 시도 연간 증가율
  B  A + 신호 달 수 7개월 (10월에 7월 자료까지 쓸 수 있음)
  C  1월 시점 예측(그해 신호 없음): 작년 − ρ·(작년 고유 변화)  vs  작년 그대로
판정: 현재 모델 대비 평균 절대 로그 오차 이득의 95% 범위(시군구 단위 부트스트랩)가 0보다 크면 채택.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from llmrel import ledger, nowcast as NC, regional_tune as RT

ROOT = Path(__file__).parent
OUT = ROOT / "output" / "model_upgrade"
RHO = 0.25
TEST = range(2019, 2026)
SPEC = dict(version="v1", diagnosis="2010~2018: 고유 오차 94%, 자기상관 -0.25, 범위 적중 78.9%",
            A="작년 + 시도 1~6월 증가율 − 0.25 × 작년 고유 변화", B="A + 신호 7개월", C="1월 시점: 작년 − 0.25 × 작년 고유 변화 vs 작년 그대로",
            test="2019~2025 한 번", rule="평균 절대 로그 오차 이득 95% 범위 > 0 이면 채택", rho=RHO)


def frame(sgg: pd.DataFrame) -> pd.DataFrame:
    g6, g7, g12 = NC.ytd_growth(6), NC.ytd_growth(7), NC.ytd_growth(12)
    rows = []
    for y in TEST:
        for c in sgg.columns:
            s = c[:2]
            last, prev, truth = sgg.loc[y - 1, c], sgg.loc[y - 2, c], sgg.loc[y, c]
            idio = (last - prev) - g12.loc[y - 1].get(s, np.nan)
            rows.append(dict(year=y, code=c, truth=truth, last=last, idio=idio,
                             cur=last + g6.loc[y].get(s, np.nan), A=last + g6.loc[y].get(s, np.nan) - RHO * idio,
                             B=last + g7.loc[y].get(s, np.nan) - RHO * idio, C0=last, C=last - RHO * idio))
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna()


def gain(f: pd.DataFrame, base: str, new: str, n: int = 3000, seed: int = 0):
    d = (f.truth - f[base]).abs() - (f.truth - f[new]).abs()           # 양수면 new가 나음
    by = d.groupby(f.code).mean()
    rng = np.random.default_rng(seed)
    bs = [by.sample(len(by), replace=True, random_state=int(rng.integers(1e9))).mean() for _ in range(n)]
    mae_b, mae_n = np.expm1((f.truth - f[base]).abs()).mean(), np.expm1((f.truth - f[new]).abs()).mean()
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), mae_b, mae_n


def main():
    ledger.append([dict(kind="model_candidate", key="marriage_upgrade_v1", payload=SPEC, model="model_upgrade",
                        evidence="시험 구간을 열기 전 등록")])
    sgg, _ = RT.load()
    f = frame(sgg)
    res = {}
    lines = []
    for name, base, new, label in [("A", "cur", "A", "되돌림 보정"), ("B", "cur", "B", "되돌림 보정 + 7개월 신호"), ("C", "C0", "C", "1월 시점: 되돌림 보정 vs 작년 그대로")]:
        g, lo, hi, mb, mn = gain(f, base, new)
        v = "채택" if lo > 0 else ("기각 (나빠짐)" if hi < 0 else "보류 (불확실)")
        res[name] = dict(gain_log=round(g, 5), ci=[round(lo, 5), round(hi, 5)], mae_before=round(mb, 4), mae_after=round(mn, 4), verdict=v)
        lines.append(f"| {name} | {label} | {mb:.2%} → {mn:.2%} | {g:+.4f} ({lo:+.4f} ~ {hi:+.4f}) | {v} |")
    ledger.append([dict(kind="model_result", key="marriage_upgrade_v1", payload=res, model="model_upgrade", evidence="시험 구간 2019~2025, 1회")])
    # 크기별 (A)
    f["size"] = pd.cut(np.exp(f["last"]), [0, 300, 1000, 3000, 1e9], labels=["300 미만", "300~1천", "1천~3천", "3천 이상"])
    sz = f.groupby("size", observed=True).apply(lambda d: pd.Series({"n": len(d), "현재": np.expm1((d.truth - d.cur).abs()).mean(), "A": np.expm1((d.truth - d.A).abs()).mean()}))
    szl = "\n".join(f"| {i} | {int(r.n)} | {r['현재']:.2%} | {r['A']:.2%} |" for i, r in sz.iterrows())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.md").write_text(f"""# 시군구 혼인 예측 개선 v1

후보는 시험 구간을 열기 전에 장부에 등록했습니다 (kind=model_candidate). 상수 ρ=0.25는 2010~2018 진단에서 정했고 시험 구간에서 바꾸지 않았습니다.
시험 구간 2019~2025, 시군구 {f.code.nunique()}곳 × {len(TEST)}년. 오차 = 평균 절대 오차(%), 이득 = 로그 오차 감소(양수면 개선), 95% 범위는 시군구 단위 부트스트랩.

| 후보 | 내용 | 평균 절대 오차 | 이득 (95%) | 판정 |
|---|---|---|---|---|
{chr(10).join(lines)}

## 크기별 (후보 A)
| 작년 혼인 건수 | 시군구-연도 | 현재 | A |
|---|---|---|---|
{szl}

## 읽는 법
- 진단: 오차의 94%가 시군구 고유 흔들림이고, 작년에 튄 지역은 다음 해 되돌아오는 경향(자기상관 −0.25)이 있었습니다. A는 그 되돌림을 출발점에 반영합니다.
- 범위(80%)는 아직 바꾸지 않았습니다. 채택된 후보로 범위를 다시 맞추는 것은 다음 단계입니다.
""", encoding="utf-8")
    print((OUT / "report.md").read_text())


if __name__ == "__main__":
    main()
