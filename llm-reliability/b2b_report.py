"""지자체용 샘플 리포트: 시군구 하나를 넣으면 같은 틀로 리포트를 만든다 (같은 틀을 여러 지역에 반복 판매).

    python b2b_report.py 31070          # → content/b2b/report_31070.html (+ .pdf)

담는 것: 결혼 시장 성비(2025)와 원인 분해, 20~34세 순이동, 혼인 건수 추이(2005~2025),
2026년 혼인 건수 예측(1~6월 신호, 예측 장부 기록), 과거 예측 성적, 한계.
숫자는 모두 KOSIS 원자료에서 이 레포 코드로 계산한 값만 쓴다.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from llmrel import district_select as D, ledger, nowcast as NC, regional_tune as RT

ROOT = Path(__file__).parent
OUT = ROOT / "content" / "b2b"


def svg_line(years, vals, other, w=640, h=220, pad=36):
    lo, hi = min(min(vals), min(other)) * 0.9, max(max(vals), max(other)) * 1.05
    x = lambda i: pad + i * (w - 2 * pad) / (len(years) - 1)
    y = lambda v: h - pad - (v - lo) / (hi - lo) * (h - 2 * pad)
    pts = lambda vs: " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vs))
    ticks = "".join(f'<text x="{x(i):.1f}" y="{h-12}" text-anchor="middle">{yr}</text>' for i, yr in enumerate(years) if yr % 5 == 0 or yr == years[-1])
    return (f'<svg viewBox="0 0 {w} {h}" class="chart"><g font-size="11" fill="#5a6672">{ticks}</g>'
            f'<polyline points="{pts(other)}" fill="none" stroke="#9aa3ac" stroke-width="2" stroke-dasharray="5 4"/>'
            f'<polyline points="{pts(vals)}" fill="none" stroke="#2a63d4" stroke-width="3"/>'
            f'<text x="{x(len(vals)-1)+4:.1f}" y="{y(vals[-1]):.1f}" font-size="12" fill="#2a63d4">{vals[-1]:.0f}</text></svg>')


_P = None


def cand_table(code: str):
    """후보 4개의 과거 오차(2008~2025)와 앞·뒤 구간 성적."""
    global _P
    if _P is None:
        _P = D.predictions(range(2008, 2026))
    e = _P[_P.code == code].pivot_table(index="year", columns="cand", values="err")
    pct = lambda s: float(np.expm1(s).mean() * 100)
    rows = {k: (pct(e[k][e.index.isin(D.PICK)]), pct(e[k][e.index.isin(D.TEST)]), pct(e[k])) for k in D.CANDS}
    return rows


def build(code: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    web = json.load(open(ROOT / "data_regional" / "regions_web.json", encoding="utf-8"))
    m = pd.read_csv(ROOT / "data_regional" / "kr_marriages_by_region.csv", dtype={"code": str})
    parent = None
    if code not in web["regions"]:                      # 일반구: 성비·이동은 소속 시 전체 값
        parent = code[:4] + "0"
    nat, r = web["nat"], web["regions"][parent or code]
    name, ratio, popr, mratio, men, women, fpct, mpct, mig_f, mig_m, chg, kind = r
    ratio = men / women
    scope = ""
    if parent:
        gu = m[m.code == code].name.iloc[0]
        scope = f"<p class=\"note\">성비·이동 지표는 {name} 전체 값입니다(구 단위 자료 없음). 혼인 건수와 예측은 {gu} 단위입니다.</p>"
        name = f"{name} {gu}"
    ts = m[m.code == code].set_index("year").value
    sido = m[m.code == code[:2]].set_index("year").value
    years = list(ts.index)
    idx = (ts / ts.iloc[0] * 100).round(1).tolist(); sidx = (sido.loc[years] / sido.loc[years[0]] * 100).round(1).tolist()
    sgg, _ = RT.load()
    nc = float(np.exp(NC.predict(sgg, 2026, 6)[code]))
    last = float(ts.loc[2025])
    bt = NC.backtest(sgg, range(2008, 2026), 6); bt = bt[bt.code == code]
    err_model, err_base = float(np.expm1(bt.model).mean() * 100), float(np.expm1(bt.base).mean() * 100)
    wins = int((bt.model < bt.base).sum())
    led = [x for x in ledger.read() if x["kind"] == "nowcast_2026" and x["key"] == code]
    h = led[-1]["hash"][:16] if led else "(미기록)"
    ct = cand_table(code)
    cand_rows = "".join(f'<tr><td>{D.LABEL[k]}{" (사용)" if k == "monthly" else ""}</td><td class="r">{a:.1f}%</td><td class="r">{b:.1f}%</td><td class="r">{c:.1f}%</td></tr>' for k, (a, b, c) in ct.items())
    m_ok = ct["monthly"][1] < ct["naive"][1] and ct["monthly"][0] < ct["naive"][0]
    verdict = ("이 지역에서는 앞·뒤 구간 모두 1~6월 신호가 '작년 값 그대로'보다 정확했습니다." if m_ok else
               "이 지역에서는 1~6월 신호가 '작년 값 그대로'보다 항상 낫지는 않았습니다. 예측 범위를 넓게 보시고, 월별 갱신으로 보완하는 것을 권합니다.")
    decomp_pop = np.log(popr) / np.log(ratio) * 100 if ratio > 1 else float("nan")
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>{name} 결혼·인구 리포트 (샘플)</title>
<style>
@page {{ size: A4; margin: 16mm }}
body {{ font-family: "NanumSquare","NanumGothic",sans-serif; color:#15202b; margin:0; font-size:13.5px; line-height:1.6 }}
.wrap {{ max-width: 760px; margin: 0 auto; padding: 24px }}
.top {{ display:flex; justify-content:space-between; align-items:flex-end; border-bottom:3px solid #15202b; padding-bottom:10px }}
.brand {{ font-weight:800; font-size:22px }} .brand span {{ color:#2a63d4 }} .tag {{ font-size:11px; color:#5a6672 }}
h1 {{ font-size:24px; margin:18px 0 4px }} h2 {{ font-size:16px; margin:22px 0 8px; border-left:4px solid #2a63d4; padding-left:8px }}
.kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:14px 0 }}
.kpi {{ border:1px solid #dce2e7; border-radius:8px; padding:10px }} .kpi b {{ display:block; font-size:22px; font-family:monospace }} .kpi small {{ color:#5a6672 }}
table {{ border-collapse:collapse; width:100% }} td,th {{ border-bottom:1px solid #dce2e7; padding:6px 8px; text-align:left }} th {{ color:#5a6672; font-weight:600; font-size:12px }}
td.r {{ text-align:right; font-family:monospace }}
.chart {{ width:100%; height:auto }} .note {{ font-size:11.5px; color:#5a6672; border-left:3px solid #dce2e7; padding-left:8px }}
.sample {{ background:#fff4d6; border:1px solid #b9850f; border-radius:6px; padding:6px 10px; font-size:12px }}
</style></head><body><div class="wrap">
<div class="top"><div class="brand">셈법<span>✓</span></div><div class="tag">미리 적고, 채점까지 공개합니다 · 2026-10-06 작성</div></div>
<p class="sample">샘플 리포트 — 공개 통계만으로 만든 예시입니다. 실제 납품본은 지역과 협의한 지표·월별 갱신을 포함합니다.</p>
<h1>{name} 결혼·인구 리포트</h1>
<p>25~39세 미혼 인구 구성, 혼인 추이, 2026년 혼인 건수 예측을 한 장에 정리했습니다. 모든 숫자는 통계청 KOSIS 원자료로 직접 계산했습니다.</p>
<div class="kpis">
<div class="kpi"><small>미혼 남 ÷ 미혼 여 (25~39세)</small><b>{ratio:.2f}</b><small>전국 {nat['unmarried_ratio']:.2f}</small></div>
<div class="kpi"><small>2025년 혼인 건수</small><b>{last:,.0f}</b><small>2005년 대비 {idx[-1]-100:+.0f}%</small></div>
<div class="kpi"><small>2026년 예측 (1~6월 신호)</small><b>{nc:,.0f}</b><small>{(nc/last-1)*100:+.1f}% · 장부 {h}</small></div>
<div class="kpi"><small>20~34세 순이동 (연, 2015~24)</small><b>{mig_f:+.1f}% / {mig_m:+.1f}%</b><small>여성 / 남성</small></div>
</div>
<h2>1. 결혼 시장 성비</h2>{scope}
<table><tr><th>지표</th><th class="r">{name}</th><th class="r">전국</th></tr>
<tr><td>미혼 남성 / 미혼 여성 (25~39세)</td><td class="r">{men:,} / {women:,}</td><td class="r">—</td></tr>
<tr><td>미혼 성비 (남÷여)</td><td class="r">{ratio:.2f}</td><td class="r">{nat['unmarried_ratio']:.2f}</td></tr>
<tr><td>인구 성비 (젊은 여성이 적은가)</td><td class="r">{popr:.2f}</td><td class="r">{nat['pop_ratio']:.2f}</td></tr>
<tr><td>미혼율 비 (남성이 결혼을 덜 하는가)</td><td class="r">{mratio:.2f}</td><td class="r">{nat['mrate_ratio']:.2f}</td></tr>
<tr><td>미혼율 여 / 남</td><td class="r">{fpct:.0f}% / {mpct:.0f}%</td><td class="r">{nat['f_unmarried_pct']:.0f}% / {nat['m_unmarried_pct']:.0f}%</td></tr>
<tr><td>2022→2025 미혼 성비 변화</td><td class="r">{chg:+.3f}</td><td class="r">1.38 → 1.33</td></tr></table>
<p>{name}의 미혼 성비 {ratio:.2f} 중 인구 성비 몫은 약 {decomp_pop:.0f}%, 남성 미혼율이 더 높은 몫이 {100-decomp_pop:.0f}%입니다. 20~34세 순이동은 여성 {mig_f:+.1f}%, 남성 {mig_m:+.1f}%로 {'남성 유입이 더 많아 성비를 키우는' if mig_m > mig_f else '여성 유입이 더 많아 성비를 줄이는'} 방향입니다.</p>
<h2>2. 혼인 건수 추이 (2005 = 100)</h2>
{svg_line(years, idx, sidx)}
<p class="note">파란 선 {name}, 회색 점선 소속 시도. 2005년 {ts.iloc[0]:,.0f}건 → 2025년 {last:,.0f}건.</p>
<h2>3. 2026년 혼인 건수 예측</h2>
<p>2026년 1~6월 소속 시도의 혼인 건수가 전년 같은 기간보다 늘어난 비율을 {name}의 2025년 건수에 곱했습니다. 예측값 <b>{nc:,.0f}건</b>(전년 대비 {(nc/last-1)*100:+.1f}%)은 공개 예측 장부에 hash <code>{h}</code>로 기록돼 있고, 2027년 통계 공표 후 채점합니다.</p>
<table><tr><th>같은 방식의 과거 성적 (2008~2025, 8월 말 시점 예측)</th><th class="r">값</th></tr>
<tr><td>평균 오차 — 이 방식</td><td class="r">{err_model:.1f}%</td></tr>
<tr><td>평균 오차 — '작년 값 그대로'</td><td class="r">{err_base:.1f}%</td></tr>
<tr><td>18년 중 '작년 값 그대로'보다 정확했던 해</td><td class="r">{wins}년</td></tr></table>
<h2>4. 예측 방식 비교 (이 지역의 과거 성적)</h2>
<table><tr><th>방식 (모두 8월 말 시점 정보만)</th><th class="r">2008~2018</th><th class="r">2019~2025</th><th class="r">전체</th></tr>{cand_rows}</table>
<p>{verdict} 전국 243개 시군구를 같은 방식으로 시험했을 때, 지역마다 과거에 가장 좋았던 방식을 골라 쓰는 것보다 <b>모든 지역에 1~6월 신호 하나를 쓰는 것</b>이 이후 성적이 더 좋았습니다(평균 오차 9.0% → 8.6%). 그래서 모든 리포트는 같은 방식을 씁니다.</p>
<h2>5. 한계와 다음 단계</h2>
<ul><li>등록 주소 기준입니다. 실제 거주·통근권과 다를 수 있습니다.</li>
<li>예측은 시도 단위 월별 신호를 씁니다. 시군구 고유 사정(대형 개발, 공공기관 이전 등)은 반영하지 못합니다.</li>
<li>납품본 제안: 월별 갱신(매달 인구동향 발표 후), 인접 지역 비교, 연령별 혼인율, 정책 시행 전후 효과 추적.</li></ul>
<p class="note">자료: 통계청 KOSIS(등록 기반 인구, 인구동향, 시군구 혼인). 계산 코드와 예측 장부는 공개 레포에서 검증할 수 있습니다. 투자·정책 결정의 유일한 근거로 쓰지 마십시오.</p>
</div></body></html>"""
    p = OUT / f"report_{code}.html"
    p.write_text(html, encoding="utf-8")
    return p


def build_all() -> pd.DataFrame:
    """243곳 전체 리포트 + 영업 우선순위 목록."""
    sgg, _ = RT.load()
    p = D.predictions(range(2008, 2026))
    e = p.pivot_table(index=["code", "year"], columns="cand", values="err").reset_index()
    gain = lambda d: (1 - np.expm1(d.monthly).mean() / np.expm1(d.naive).mean()) * 100
    g = e.groupby("code").apply(lambda d: pd.Series({"앞 구간 개선(%)": gain(d[d.year.isin(D.PICK)]), "뒤 구간 개선(%)": gain(d[d.year.isin(D.TEST)])}))
    web = json.load(open(ROOT / "data_regional" / "regions_web.json", encoding="utf-8"))["regions"]
    m = pd.read_csv(ROOT / "data_regional" / "kr_marriages_by_region.csv", dtype={"code": str})
    last = m[m.year == 2025].set_index("code").value
    nc = np.exp(NC.predict(sgg, 2026, 6))
    rows = []
    for c in sgg.columns:
        build(c)
        w = web.get(c) or web.get(c[:4] + "0")
        rows.append({"code": c, "지역": (w[0] if c in web else f"{w[0]} {m[m.code == c].name.iloc[0]}") if w else c, "미혼 성비": round(w[4] / w[5], 2) if w else None,
                     "2025 혼인": int(last[c]), "2026 예측": int(round(nc[c])), "앞 구간 개선(%)": round(g.loc[c].iloc[0], 1),
                     "뒤 구간 개선(%)": round(g.loc[c].iloc[1], 1)})
    t = pd.DataFrame(rows)
    t["강함"] = (t["앞 구간 개선(%)"] >= 10) & (t["뒤 구간 개선(%)"] >= 10)
    t = t.sort_values(["강함", "뒤 구간 개선(%)"], ascending=[False, False])
    t.to_csv(OUT / "districts_all.csv", index=False)
    return t


if __name__ == "__main__":
    if sys.argv[1:] == ["all"]:
        t = build_all()
        print(len(t), "reports;", int(t["강함"].sum()), "strong")
    else:
        for c in sys.argv[1:] or ["31070"]:
            print(build(c))
