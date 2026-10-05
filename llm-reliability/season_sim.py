"""수석코치 빠른 진행: 모델을 과거 시즌마다 처음부터 다시 돌려 성적표를 만든다.

    python season_sim.py    # → output/season_sim/report.md

- 각 시즌(연도)마다 '그 시점까지 알던 것'만으로 예측하고, 단순 기준선과 겨룬다.
- 우승 🏆 = 그 시즌 기준선보다 확실히 나음 (개선의 95% 구간 하한 > 0)
  무승부 ➖ = 우연과 구분 안 됨,  패배 ❌ = 기준선보다 확실히 나쁨
- 구간별 진단: 어디서 모델이 약한지 (시즌 초반, 승격팀, 코로나, 짧은/긴 예측 거리 등)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from llmrel import epl, forecast_tune as FT, regional_tune as RT, tuning
from llmrel.report import md_table

ROOT = Path(__file__).parent
OUT = ROOT / "output" / "season_sim"
ICON = {1: "🏆", 0: "➖", -1: "❌"}


def verdict(base: np.ndarray, model: np.ndarray) -> tuple[float, float, float, int]:
    """기준선 대비 개선률(%)과 95% 구간, 판정."""
    m, lo, hi = tuning.paired_ci(base, model)
    b = base.mean()
    g, glo, ghi = -m / b * 100, -hi / b * 100, -lo / b * 100
    return g, glo, ghi, (1 if glo > 0 else -1 if ghi < 0 else 0)


def seg_table(d: pd.DataFrame, by: str, order=None) -> pd.DataFrame:
    rows = []
    for k, g in d.groupby(by, sort=False):
        gg, lo, hi, v = verdict(g.base.to_numpy(), g.model.to_numpy())
        rows.append({"구간": k, "n": len(g), "모델": g.model.mean(), "기준선": g.base.mean(), "개선": gg, "95% 구간": f"{lo:+.1f} ~ {hi:+.1f}%", "판정": ICON[v]})
    t = pd.DataFrame(rows)
    if order:
        t["_o"] = t["구간"].map({k: i for i, k in enumerate(order)}); t = t.sort_values("_o").drop(columns="_o")
    return t


def season_table(d: pd.DataFrame, col: str, label: str) -> tuple[pd.DataFrame, dict]:
    t = seg_table(d, col).rename(columns={"구간": label})
    t = t.sort_values(label)
    cnt = {k: int((t["판정"] == ICON[k]).sum()) for k in (1, 0, -1)}
    return t, cnt


# ── EPL ─────────────────────────────────────────────────────────
def epl_detail() -> pd.DataFrame:
    df = epl.load()
    r = epl.run(df, **epl.GOALS_MODE)
    r = r[r.played & (r.season != r.season.min())].copy()            # 첫 시즌은 워밍업
    r["model"] = epl.losses(r)
    # 기준선: 그 시즌 전까지 누적된 홈승·무·원정승 비율 (미래 정보 없음)
    seasons = sorted(df.season.unique())
    base_rows = []
    for s in seasons[1:]:
        prev = df[df.season.isin(seasons[:seasons.index(s)]) & df.played]
        f = prev.result.value_counts(normalize=True).reindex(["H", "D", "A"]).to_numpy()
        cur = r[r.season == s].copy(); cur[["pH", "pD", "pA"]] = f
        base_rows.append(pd.Series(epl.losses(cur), index=cur.index))
    r["base"] = pd.concat(base_rows)
    # 구간 표시
    r = r.sort_values(["season", "date"])
    n_home = r.groupby(["season", "home"]).cumcount(); n_away = r.groupby(["season", "away"]).cumcount()
    gw = r.apply(lambda x: 0, axis=1)
    cnt: dict = {}
    mw = []
    for s, h, a in zip(r.season, r.home, r.away):
        k1, k2 = (s, h), (s, a)
        cnt[k1] = cnt.get(k1, 0) + 1; cnt[k2] = cnt.get(k2, 0) + 1
        mw.append(max(cnt[k1], cnt[k2]))
    r["mw"] = mw
    r["시기"] = pd.cut(r.mw, [0, 6, 19, 38], labels=["1~6라운드", "7~19라운드", "20~38라운드"]).astype(str)
    teams = {s: set(df[df.season == s].home) for s in seasons}
    prev_of = {s: seasons[i - 1] for i, s in enumerate(seasons) if i}
    r["승격팀"] = [("승격팀 포함" if (h not in teams[prev_of[s]] or a not in teams[prev_of[s]]) else "기존 팀끼리") for s, h, a in zip(r.season, r.home, r.away)]
    fav = r[["pH", "pA"]].max(axis=1)
    r["강약"] = pd.cut(fav, [0, 0.45, 0.6, 1], labels=["팽팽 (최고 확률 45% 미만)", "보통 (45~60%)", "확실한 우세 (60% 이상)"]).astype(str)
    r["결과"] = r.result.map({"H": "홈 승", "D": "무승부", "A": "원정 승"})
    fav_side = np.where(r.pH >= r.pA, "H", "A")
    r["이변"] = np.where(r.result == "D", "무승부", np.where(r.result == fav_side, "우세 팀 승리", "약한 팀 승리 (이변)"))
    r["관중"] = np.where(r.season == "2020-21", "무관중 시즌 2020/21", "관중 있음")
    r["fav_hit"] = (r.result == fav_side)
    return r


def epl_section() -> list[str]:
    r = epl_detail()
    st, cnt = season_table(r, "season", "시즌")
    hit = r.groupby("season").fav_hit.mean()
    st["우세 팀 적중"] = st["시즌"].map(hit).map(lambda x: f"{x:.0%}")
    L = ["## 1. EPL 경기 예측 (골 모드, 시즌 16개)", "",
         f"매 경기 전, 그때까지의 결과만 보고 홈승·무·원정승 확률을 냅니다. 기준선은 '지난 시즌들의 홈승·무·원정승 비율을 모든 경기에 똑같이 쓰기'. 손실은 RPS(낮을수록 좋음).", "",
         f"**성적: 🏆 {cnt[1]}회 · ➖ {cnt[0]}회 · ❌ {cnt[-1]}회** (2026/27은 진행 중)", "",
         md_table(st, {"모델": "{:.4f}", "기준선": "{:.4f}", "개선": "{:+.1f}%"}), ""]
    for by, title, order in [("시기", "시즌 안 시기", ["1~6라운드", "7~19라운드", "20~38라운드"]), ("승격팀", "승격팀", None),
                             ("강약", "경기 강약 (모델이 본 우세 정도)", ["팽팽 (최고 확률 45% 미만)", "보통 (45~60%)", "확실한 우세 (60% 이상)"]),
                             ("이변", "결과 유형", ["우세 팀 승리", "무승부", "약한 팀 승리 (이변)"]), ("관중", "관중", None)]:
        L += [f"### {title}", "", md_table(seg_table(r, by, order), {"모델": "{:.4f}", "기준선": "{:.4f}", "개선": "{:+.1f}%"}), ""]
    return L, r, st, cnt


# ── 연령별 혼인율 ────────────────────────────────────────────────
def marriage_section() -> list[str]:
    d = pd.read_csv(ROOT / "data_official" / "kr_jp_marriage_official.csv"); d["age_band"] = d.age_band.astype(str)
    sp = FT.make_spec(d)
    origins = range(2006, 2025)
    m = sp.detail(FT.FINAL, origins).rename(columns={"err": "model"})
    b = sp.detail({**FT.FINAL, "s": 0.0}, origins).rename(columns={"err": "base"})       # 기준선: 마지막 값 유지
    x = m.merge(b, on=["origin", "h", "key"])
    x["target"] = x.origin + x.h
    k = x.key.str.split("|", expand=True)
    x["나라"] = k[0].map({"KR": "한국", "JP": "일본"}); x["성별"] = k[2].map({"F": "여성", "M": "남성"}); x["연령"] = k[3]
    x["거리"] = x.h.map(lambda h: f"{h}년 뒤")
    x["시기"] = pd.cut(x.target, [0, 2019, 2022, 2030], labels=["평상시 (목표 ~2019)", "코로나 급감 (2020~2022)", "반등기 (2023~2025)"]).astype(str)
    x["시즌"] = x.target
    st, cnt = season_table(x, "시즌", "목표 연도")
    L = ["## 2. 연령별 혼인율 예측 (한·일 20계열)", "",
         "각 원점 연도까지의 통계만 보고 1~3년 뒤를 예측. 기준선은 '마지막 값 그대로'. 손실은 |로그 오차| (0.05 ≈ 5%). 시즌 = 목표 연도.", "",
         f"**성적: 🏆 {cnt[1]}회 · ➖ {cnt[0]}회 · ❌ {cnt[-1]}회**", "",
         md_table(st, {"모델": "{:.4f}", "기준선": "{:.4f}", "개선": "{:+.1f}%"}), ""]
    for by, title, order in [("시기", "시기", ["평상시 (목표 ~2019)", "코로나 급감 (2020~2022)", "반등기 (2023~2025)"]), ("거리", "예측 거리", None),
                             ("나라", "나라", None), ("연령", "연령대", None), ("성별", "성별", None)]:
        L += [f"### {title}", "", md_table(seg_table(x, by, order), {"모델": "{:.4f}", "기준선": "{:.4f}", "개선": "{:+.1f}%"}), ""]
    return L, x, st, cnt


# ── 시군구 혼인 건수 ─────────────────────────────────────────────
def regional_section() -> list[str]:
    sgg, sido = RT.load()
    sp = RT.make_spec(sgg, sido)
    origins = range(2007, 2025)
    m = sp.detail(RT.FINAL, origins).rename(columns={"err": "model"})
    b = sp.detail({**RT.FINAL, "s": 0.0}, origins).rename(columns={"err": "base"})
    x = m.merge(b, on=["origin", "h", "code"])
    x["target"] = x.origin + x.h
    size = np.exp(sgg.mean())
    q = pd.qcut(size, 4, labels=["작음 (하위 25%)", "중간 아래", "중간 위", "큼 (상위 25%)"])
    x["규모"] = x.code.map(q).astype(str)
    x["거리"] = x.h.map(lambda h: f"{h}년 뒤")
    x["시기"] = pd.cut(x.target, [0, 2019, 2022, 2030], labels=["평상시 (목표 ~2019)", "코로나 급감 (2020~2022)", "반등기 (2023~2025)"]).astype(str)
    names = pd.read_csv(ROOT / "data_regional" / "kr_marriages_by_region.csv", dtype={"code": str}).drop_duplicates("code").set_index("code")
    sido_name = {c: names.at[c, "name"] if "name" in names.columns and c in names.index else c for c in sido.columns}
    x["시도"] = x.code.str[:2].map(sido_name)
    st, cnt = season_table(x, "target", "목표 연도")
    sd = seg_table(x, "시도").sort_values("개선")
    L = ["## 3. 시군구 혼인 건수 예측 (243곳)", "",
         "각 원점 연도까지의 건수만 보고 1~2년 뒤를 예측(소속 시도의 최근 2년 추세 사용). 기준선은 '마지막 값 그대로'. 손실은 |로그 오차|.", "",
         f"**성적: 🏆 {cnt[1]}회 · ➖ {cnt[0]}회 · ❌ {cnt[-1]}회**", "",
         md_table(st, {"모델": "{:.4f}", "기준선": "{:.4f}", "개선": "{:+.1f}%"}), ""]
    for by, title, order in [("시기", "시기", ["평상시 (목표 ~2019)", "코로나 급감 (2020~2022)", "반등기 (2023~2025)"]), ("거리", "예측 거리", None),
                             ("규모", "시군구 규모 (연 혼인 건수)", ["작음 (하위 25%)", "중간 아래", "중간 위", "큼 (상위 25%)"])]:
        L += [f"### {title}", "", md_table(seg_table(x, by, order), {"모델": "{:.4f}", "기준선": "{:.4f}", "개선": "{:+.1f}%"}), ""]
    L += ["### 시도별 (개선이 작은 순)", "", md_table(sd, {"모델": "{:.4f}", "기준선": "{:.4f}", "개선": "{:+.1f}%"}), ""]
    return L, x, st, cnt


def fixes() -> list[str]:
    """약한 구간(반등기)에 대한 후보 규칙을 전 기간으로 진단. 채택 아님 — 가설 기록용."""
    def per(sp, p, base, origins, keys):
        x = sp.detail(p, origins).merge(sp.detail(base, origins), on=keys, suffixes=("", "_b"))
        x["t"] = x.origin + x.h
        x["per"] = pd.cut(x.t, [0, 2019, 2022, 2030], labels=["평상시", "코로나 급감", "반등기"])
        row = {"전체": (1 - x.err.mean() / x.err_b.mean()) * 100}
        for k, g in x.groupby("per", observed=True):
            row[str(k)] = (1 - g.err.mean() / g.err_b.mean()) * 100
        return row
    d = pd.read_csv(ROOT / "data_official" / "kr_jp_marriage_official.csv"); d["age_band"] = d.age_band.astype(str)
    sgg, sido = RT.load()
    rows = []
    for mod, sp, origins, keys, name in ((FT, FT.make_spec(d), range(2006, 2025), ["origin", "h", "key"], "혼인율"),
                                         (RT, RT.make_spec(sgg, sido), range(2007, 2025), ["origin", "h", "code"], "시군구")):
        nv = {**mod.FINAL, "s": 0.0}
        for label, p in (("현재 규칙", mod.FINAL), ("추세 상한 7%", {**mod.FINAL, "cap": 0.07}), ("추세 상한 3%", {**mod.FINAL, "cap": 0.03}),
                         ("추세 절반만 믿기 (s=0.5)", {**mod.FINAL, "s": 0.5})):
            rows.append({"모델": name, "규칙": label, **per(sp, p, nv, origins, keys)})
    t = pd.DataFrame(rows)
    return ["## 4. 약한 구간을 고칠 후보 (진단만, 채택 아님)", "",
            "숫자 = '마지막 값 그대로' 대비 개선률(%). 이 표는 전 기간을 보고 만든 것이라 채택 근거가 아니라 다음에 판정할 가설입니다.", "",
            md_table(t, {c: "{:+.1f}" for c in ["전체", "평상시", "코로나 급감", "반등기"]}), ""]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (le, r, est, ec), (lm, xm, mst, mc), (lr, xr, rst, rc) = epl_section(), marriage_section(), regional_section()
    summary = pd.DataFrame([
        {"모델": "EPL 경기", "시즌 수": len(est), "🏆": ec[1], "➖": ec[0], "❌": ec[-1]},
        {"모델": "연령별 혼인율", "시즌 수": len(mst), "🏆": mc[1], "➖": mc[0], "❌": mc[-1]},
        {"모델": "시군구 혼인 건수", "시즌 수": len(rst), "🏆": rc[1], "➖": rc[0], "❌": rc[-1]}])
    L = ["# 수석코치 빠른 진행: 시즌별 성적표와 약한 구간", "",
         "모델을 과거 시즌마다 처음부터 다시 돌려서, 그 시점에 알던 것만으로 예측하고 단순 기준선과 겨뤘습니다.",
         "🏆 우승 = 기준선보다 확실히 나음 (95% 하한 > 0) · ➖ 무승부 = 우연과 구분 안 됨 · ❌ 패배 = 확실히 나쁨.", "",
         "주의: 지금 규칙은 이 과거 시즌 일부를 보고 고른 것이라, 성적이 실제 미래보다 좋게 나올 수 있습니다 (특히 검증·시험 구간).", "",
         "## 요약", "", md_table(summary, {}), "",
         "### 수석코치 보고 (핵심)", "",
         "- **EPL**: 16시즌 중 15회 우승, 남은 1회는 진행 중인 이번 시즌. 시기·승격팀·무관중 어디서도 무너지지 않음. 약점은 **무승부와 이변**(약한 팀 승리) 경기: 기준선보다 각각 18%, 36% 나쁨. 우세 팀을 강하게 믿는 만큼 이변에 크게 짐. 다만 상대가 약한 기준선이라, 진짜 시험은 Opta·배당과의 비교.",
         "- **연령별 혼인율**: 우승 7 · 무 9 · 패 2. 코로나 급감기엔 강했지만(+21%) **반등기(2024~2025)에 2연패**(−35%, −39%). 한국 계열은 기준선과 비슷(+3.6%, 우연 범위), 30대 이상 연령도 이득 없음.",
         "- **시군구 혼인 건수**: 우승 9 · 패 9, 전 기간으로는 '마지막 값 그대로'보다 2.6% 나쁨. 튜닝 때 검증 구간(목표 2014~2018)이 마침 추세가 잘 맞는 시기였던 것. **검증 구간이 한 국면에만 걸쳐 있으면 규칙이 그 국면에 맞춰진다**는 교훈. 특히 2023년(−80%)은 코로나 급감 추세를 반등기에 그대로 이어 붙여 크게 졌음.",
         "- 공통 약점: **국면이 바뀌는 시점**(급감 → 반등). 추세 모델은 방향 전환을 못 봄. 아래 4절의 상한·추세 절반 규칙이 반등기 손실을 크게 줄이지만, 전 기간을 보고 고른 것이라 가설로만 등록.", ""] + le + lm + lr + fixes()
    (OUT / "report.md").write_text("\n".join(L), encoding="utf-8")
    pd.concat([est.assign(model="EPL"), mst.assign(model="혼인율"), rst.assign(model="시군구")]).to_csv(OUT / "seasons.csv", index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
