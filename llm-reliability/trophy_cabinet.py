"""트로피 진열장: 목표는 '매 시즌 메이저 우승 최소 1개'.

    python trophy_cabinet.py    # → output/season_sim/trophies.md

- 시즌 = 달력 연도 (EPL은 시즌이 끝나는 해, 연간 모델은 예측 목표 연도)
- 메이저 = 강한 상대를 확실히 이김 (95% 하한 > 0)
    연간 모델: '마지막 값 그대로' (공식 통계에서 이기기 어려운 기준선)
    EPL: Opta·배당 (과거 기록 없음 → 공개 장부로 쌓는 중). 리그 평균 비율을 이긴 건 '컵 대회'로만 셈
- 스쿼드 운영: 핵심(1군) / 유망주(키움) / 판매. 후보 선수는 '빈 시즌을 메워 주는가'로 평가
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import season_sim as S
from llmrel import forecast_tune as FT, ledger, nowcast as NC, regional_tune as RT
from llmrel.report import md_table

ROOT = Path(__file__).parent
OUT = ROOT / "output" / "season_sim"


def yearly(x: pd.DataFrame, year_col: str) -> dict[int, int]:
    t = S.seg_table(x, year_col)
    return {int(k): {"🏆": 1, "➖": 0, "❌": -1}[v] for k, v in zip(t["구간"], t["판정"])}


def annual(sp, params, origins, keys, name):
    m = sp.detail(params, origins).rename(columns={"err": "model"})
    b = sp.detail({**params, "s": 0.0}, origins).rename(columns={"err": "base"})
    x = m.merge(b, on=keys); x["year"] = x.origin + x.h
    return yearly(x, "year")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(ROOT / "data_official" / "kr_jp_marriage_official.csv"); d["age_band"] = d.age_band.astype(str)
    sgg, sido = RT.load()
    fs, rs = FT.make_spec(d), RT.make_spec(sgg, sido)
    mo, ro, mk, rk = range(2006, 2025), range(2007, 2025), ["origin", "h", "key"], ["origin", "h", "code"]
    r = S.epl_detail(); r["year"] = r.season.str[:4].astype(int) + 1
    squad = {   # 이름: (역할, 결과)
        "EPL (리그 평균 상대, 컵)": ("핵심", yearly(r, "year")),
        "혼인율 · 현재 규칙": ("핵심", annual(fs, FT.FINAL, mo, mk, "")),
        "시군구 · 현재 규칙": ("핵심", annual(rs, RT.FINAL, ro, rk, "")),
        "혼인율 · 추세 상한 3%": ("유망주", annual(fs, {**FT.FINAL, "cap": 0.03}, mo, mk, "")),
        "시군구 · 추세 상한 3%": ("유망주", annual(rs, {**RT.FINAL, "cap": 0.03}, ro, rk, "")),
        "시군구 · 추세 절반": ("유망주", annual(rs, {**RT.FINAL, "s": 0.5}, ro, rk, "")),
        "시군구 · 1~6월 신호 (8월 말 예측)": ("유망주", yearly(NC.backtest(sgg, range(2008, 2026), 6), "year")),
    }
    years = list(range(2008, 2026))
    rows = []
    for nm, (role, res) in squad.items():
        rows.append({"선수": nm, "역할": role, **{str(y): {1: "🏆", 0: "·", -1: "❌"}.get(res.get(y), "") for y in years}})
    grid = pd.DataFrame(rows)
    major = [n for n in squad if "컵" not in n]
    core = [n for n in major if squad[n][0] == "핵심"]

    def covered(names):
        return {y for y in years if any(squad[n][1].get(y) == 1 for n in names)}

    c_core, c_all = covered(core), covered(major)
    cup = {y for y in years if squad["EPL (리그 평균 상대, 컵)"][1].get(y) == 1}
    # 유망주마다: 지금 핵심 위에 더하면 빈 시즌을 몇 개 메우나 / 핵심과 같은 시즌에 지나
    pros = []
    for n in major:
        if squad[n][0] != "유망주":
            continue
        res = squad[n][1]
        pros.append({"유망주": n, "메이저 수": sum(v == 1 for v in res.values()), "메워 주는 빈 시즌": ", ".join(str(y) for y in sorted(covered([n]) - c_core)) or "없음",
                     "확실한 패배": sum(v == -1 for v in res.values())})
    pros = pd.DataFrame(pros)
    empty_core = [y for y in years if y not in c_core]
    empty_all = [y for y in years if y not in c_all]
    L = ["# 트로피 진열장: 매 시즌 메이저 1개 이상", "",
         "목표는 매 시즌 1등이 아니라 **매 시즌 메이저 우승 최소 1개**. 시즌 = 달력 연도. 메이저 = 강한 상대(공식 통계는 '마지막 값 그대로')를 확실히 이김.",
         "EPL은 리그 평균 비율만 이긴 기록이라 **컵 대회**로만 셉니다. EPL 메이저(Opta·배당 상대)는 과거 배당 자료(football-data.co.uk)가 이 환경에서 차단돼 있어, 공개 장부로 이번 시즌부터 쌓습니다.", "",
         "## 요약", "",
         f"- 핵심 선수만: 메이저 있는 시즌 **{len(c_core)}/{len(years)}** · 빈 시즌 {', '.join(map(str, empty_core)) or '없음'}",
         f"- 유망주까지: **{len(c_all)}/{len(years)}** · 빈 시즌 {', '.join(map(str, empty_all)) or '없음'}",
         f"- 컵(EPL 리그 평균 상대)까지 넣으면 {len(c_all | cup)}/{len(years)} — 컵은 매년 들지만 메이저로 세지 않음", "",
         "주의: 유망주 성적은 전 기간을 보고 만든 후보라 실제보다 좋게 보입니다. 1군 승격은 앞으로의 시즌(2026년 통계부터)에서 빈 시즌을 실제로 메울 때만.", "",
         "## 시즌별 진열장", "", "🏆 메이저(컵) 우승 · · 우연 범위 · ❌ 확실한 패배", "",
         md_table(grid, {}), "",
         "## 유망주 평가: 핵심이 비운 시즌을 메워 주는가", "", md_table(pros, {}), "",
         "## 새 유망주: 1~6월 신호 (시즌 중반 예측)", "",
         "KOSIS 월별 인구동향(시도별 혼인 건수, 공짜)을 FA로 영입. 올해 1~6월이 작년 같은 기간보다 늘어난 만큼 각 시군구의 작년 건수에 곱합니다. 파라미터가 없어 과거에 맞춰 고를 것이 없습니다(1~3월 신호도 시험했는데 더 약해서 1~6월만 씀).",
         "- 핵심 선수들과 **다른 시즌에 이깁니다**: 2008~2011, 2024 — 핵심이 비운 시즌 대부분을 메움. 추세 모델이 못 보는 방향 전환을 올해 실제 숫자로 봄.",
         "- 단, 예측 시점이 8월 말이라 연초 예측과는 다른 대회입니다. 연초 예측(핵심)은 그대로 두고, 8월 말 갱신을 따로 둡니다.",
         f"- 2026년 예측은 1~6월 자료로 지금 기록했습니다 (`data_monthly/nowcast_2026.csv`). 2027년 공표되는 2026년 통계로 채점합니다.", "",
         "## 스쿼드 운영 규칙", "",
         "1. **영입**: FA(공짜·저비용 데이터, 규칙 후보)는 값어치보다 확실히 싸면 일단 영입. 전술 적합은 출전 기준이지 영입 기준이 아님.",
         "2. **핵심**: 메이저를 실제로 들어 올리는 선수. 지금은 혼인율·시군구 현재 규칙.",
         "3. **유망주**: 핵심이 비우는 시즌(국면 전환기)을 메워 줄 후보를 키움. 주급(관리 비용)이 거의 0이고 자동으로 채점될 것. 앞으로의 시즌에서 빈 시즌을 메우면 1군 승격.",
         "4. **판매**: 1군 역할이 없는 영입은 콘텐츠·리포트로 판매 (정보는 팔아도 남음).",
         "5. **스쿼드 평가 기준**: 평균 성적이 아니라 '메이저 없는 시즌 수'(최소치). 새 선수는 핵심과 다른 시즌에 이길수록 가치가 큼.", ""]
    nc = NC.predict(sgg, 2026, 6)
    pd.DataFrame({"made_on": "2026-10-05", "code": sgg.columns, "last_2025": np.exp(sgg.loc[2025]).round().astype(int).to_numpy(),
                  "pred_2026": np.exp(nc).round().astype(int).to_numpy()}).to_csv(ROOT / "data_monthly" / "nowcast_2026.csv", index=False)
    names = pd.read_csv(ROOT / "data_regional" / "kr_marriages_by_region.csv", dtype={"code": str}).drop_duplicates("code").set_index("code").name
    ledger.append([dict(kind="nowcast_2026", key=c, payload={"region": names.get(c, ""), "pred_2026": int(round(float(np.exp(v))))},
                        model="1~6월 신호 (KOSIS DT_1B8000G, 2026년 1~6월)", evidence="2025 연간 건수 × 소속 시도 1~6월 증가율")
                   for c, v in nc.items()])
    (OUT / "trophies.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
