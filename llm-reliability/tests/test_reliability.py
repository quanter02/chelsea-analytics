"""스키마 뷰, 축소 추정, 라우터 규칙, 시뮬레이터 성질 검사."""
import numpy as np
import pandas as pd
import pytest

from llmrel import db, profiles as P, router as R, simulate as S


def _pred(i, model, answer, conf=None, ok=None, h=None):
    return dict(model_id=model, model_version="v1", prompt_version="p1", task_type="t", input_hash=h or f"h{i}",
                answer=answer, confidence=conf, evidence=None, evidence_ok=ok, cost_usd=0.01, latency_ms=100,
                created_at=f"2026-01-{i + 1:02d}T00:00:00")


@pytest.fixture
def tiny():
    con = db.connect()
    db.insert(con, "predictions", pd.DataFrame([
        _pred(0, "a", "R", 0.9, 1), _pred(1, "a", "M", 0.8, 0), _pred(2, "a", "ABSTAIN"),
        _pred(3, "a", "ABSTAIN"),                       # 정답 대기 중인 기권
    ]))
    db.insert(con, "outcomes", pd.DataFrame([
        dict(input_hash=f"h{i}", task_type="t", true_label=l, label_source="deploy_log", labeled_at="2026-02-01")
        for i, l in enumerate(["R", "R", "N"])]))
    db.insert(con, "loss_matrix", pd.DataFrame([("t", "R", "M", 50.0), ("t", "N", "ABSTAIN", 5.0)],
                                               columns=["task_type", "true_label", "answer", "loss_usd"]))
    return con


def test_views_split_outcomes(tiny):
    one = lambda sql: tiny.execute(sql).fetchone()[0]
    assert one("SELECT COUNT(*) FROM confident_errors") == 1
    assert one("SELECT loss_usd FROM confident_errors") == 50.0
    assert one("SELECT COUNT(*) FROM abstentions") == 2           # 정답 없는 기권도 포함
    assert one("SELECT COUNT(*) FROM pending") == 1
    assert one("SELECT COUNT(*) FROM scored WHERE outcome = 'correct'") == 1


def test_profile_rates_use_labeled_denominator(tiny):
    preds, scored = P.load(tiny)
    prof = P.outcome_profile(preds, scored).iloc[0]
    assert prof.n_labeled == 3 and prof.n_total == 4
    assert prof.correct_rate == pytest.approx(1 / 3) and prof.abstain_rate == pytest.approx(1 / 3)
    assert prof.pending_share == pytest.approx(0.25)


def test_eb_shrink_pulls_small_samples_toward_pool():
    k, n = np.array([0, 300, 310]), np.array([5, 1000, 1000])
    s = P.eb_shrink(k, n)
    assert s[0] > 0.05                                         # 5건 중 0건이라도 0%로 믿지 않음
    assert abs(s[1] - 0.30) < 0.01                             # 큰 표본은 거의 그대로


def test_router_accepts_or_escalates():
    W = dict(models=["a", "b"], truth=np.array(["R", "R", "M"]), when=np.zeros(3), index=None,
             answer=pd.DataFrame({"a": ["R", "M", "ABSTAIN"], "b": ["R", "R", "M"]}),
             confidence=pd.DataFrame({"a": [0.9, 0.4, np.nan], "b": [0.9, 0.9, 0.9]}),
             evidence_ok=pd.DataFrame({"a": [1, 1, np.nan], "b": [1, 1, 0]}),
             cost_usd=pd.DataFrame({"a": [1.0] * 3, "b": [2.0] * 3}))
    loss = {("R", "M"): 50.0}
    r = R.evaluate(W, R.Policy((("a", 0.5), ("b", 0.5)), require_evidence=True), loss, human_usd=10)
    # 1번: a 채택 / 2번: a 확신 부족 → b 채택(정답) / 3번: a 기권 → b 근거 실패 → 사람
    assert r["error_loss"] == 0 and r["human_share"] == pytest.approx(1 / 3)
    assert r["call_cost"] == pytest.approx((1 + 3 + 3) / 3)
    r2 = R.evaluate(W, R.Policy((("a", 0.0),)), loss, human_usd=10)
    assert r2["error_loss"] == pytest.approx(50 / 3)            # 2번 문항의 오답을 그대로 채택


def test_simulation_is_reproducible_and_overconfidence_shows_in_ece():
    p1, o1 = S.simulate(600, seed=3)
    p2, _ = S.simulate(600, seed=3)
    pd.testing.assert_frame_equal(p1, p2)
    con = db.connect()
    db.insert(con, "predictions", p1); db.insert(con, "outcomes", o1); db.insert(con, "loss_matrix", S.loss_matrix())
    _, scored = P.load(con)
    _, ece = P.calibration(scored)
    assert ece["mid-bold@v1"] > ece["mid-cautious@v1"]          # gamma < 1 인 모델이 과신


def test_online_static_keeps_policy_and_adaptive_flags_silent_drift():
    from llmrel import online
    p, o = S.simulate(1500, days=S.DRIFT_DAYS, seed=3, models=S.DRIFT_MODELS, label_all=True)
    con = db.connect()
    db.insert(con, "predictions", p); db.insert(con, "outcomes", o); db.insert(con, "loss_matrix", S.loss_matrix())
    _, scored = P.load(con)
    static, none = online.run(scored, S.HUMAN_REVIEW_USD, "static", end_day=180, step=14)
    assert static.policy.nunique() == 1 and none.empty
    _, alarms = online.run(scored, S.HUMAN_REVIEW_USD, "adaptive", end_day=180, step=7)
    hit = alarms[(alarms.model == "large@v2") & (alarms.direction == "악화")]
    assert len(hit) and hit.day.min() > 110                      # 성능 저하(110일) 이후에만 경보


def test_marginal_utility_of_reports_diminishes():
    from llmrel import extraction as X
    p, mus = X.PRIOR_TRUE, []
    for _ in range(5):                                          # 같은 사건에 '사실' 보고가 계속 쌓임
        mus.append(X.mu_report(p, 0.8, 1.0)); p = X.update(p, 0.8, True)
    assert all(a > b for a, b in zip(mus, mus[1:]))
    assert X.mu_report(0.7, 0.5, 1.0) == pytest.approx(0)       # 동전 던지기 수준 채널은 가치 0
    assert X.mu_discovery(0.8, 1.0) > X.mu_report(X.PRIOR_TRUE, 0.8, 1.0)   # 첫 발견 > 확인


def test_marginal_policy_beats_extract_everything():
    from llmrel import extraction as X
    ev, _, vids = X.simulate_feed(0)
    res = {p.name: X.run(ev, vids, p)[0] for p in X.POLICIES}
    assert res["한계효용 기준"]["total"] < res["전부 추출, 검증 없음"]["total"]
    assert res["한계효용 기준"]["extracted"] < res["사건당 앞 2개만 추출"]["extracted"]


def test_football_predictions_use_only_past_results():
    from llmrel import football as F
    teams = ["A", "B", "C", "D"]
    rows = [dict(match_id=str(i), league="X", date=f"2015-08-{i + 1:02d}", kickoff="15:00:00.000",
                 home=teams[i % 4], away=teams[(i + 1) % 4], hs=i % 3, as_=1, result="H" if i % 3 > 1 else "A" if i % 3 < 1 else "D")
            for i in range(12)]
    m = pd.DataFrame(rows); m["ts"] = pd.to_datetime(m.date + " 15:00:00")
    base = F.predict(m)
    m2 = m.copy(); m2.loc[11, ["hs", "as_", "result"]] = [9, 0, "H"]           # 마지막 경기 결과만 바꿈
    alt = F.predict(m2)
    early = lambda d: d[d.input_hash != "11"].reset_index(drop=True)
    pd.testing.assert_frame_equal(early(base), early(alt))                       # 앞선 예측은 그대로


def test_collect_normalizes_urls_and_extracts_numbers():
    from llmrel import collect as C
    assert C.canonical_url("https://mobile.newsis.com/view/NISX1") == C.canonical_url("https://www.newsis.com/view/NISX1")
    assert C.canonical_url("https://soranews24.com/2026/09/01/x/amp/") == C.canonical_url("https://soranews24.com/2026/09/01/x/")
    assert C.classify("https://www.mhlw.go.jp/a.pdf") == "공식" and C.classify("https://www.datanow.kr/x") == "2차 정리·블로그"
    ko = C.extract_claims("7월 출생아 수는 2만4275명으로 지난해 같은 달보다 2421명(11.1%) 증가했습니다.", "ko")
    got = {(c["indicator"], c["value"]) for c in ko}
    assert ("births", 24275) in got and ("births_yoy", 11.1) in got and ("births", 2421) not in got   # 증감분은 제외
    ja = C.extract_claims("2026年1~6月の出生数は、前年同期比0.8%増の34万2068人だった。", "ja")
    assert ("births", 342068) in {(c["indicator"], c["value"]) for c in ja}
    neg = C.extract_claims("The number of marriages fell 6.4 per cent year-on-year to 20,368 in May.", "en")
    assert ("marriages_yoy", -6.4) in {(c["indicator"], c["value"]) for c in neg}


def test_official_age_bands_and_ratio():
    import pandas as pd
    from llmrel.official import _band, _ratio
    assert _band("25~29세") == _band("25 - 29세") == _band("25～29歳") == "25-29"
    assert _band("19歳以下") == _band("15 - 19세") == "~19"
    assert _band("합계") is None and _band("85세이상") is None
    df = pd.DataFrame(dict(year=[2020] * 2, sex=["F"] * 2, kind=["t", "u"], age="30~34세", value=[200.0, 90.0]))
    r = _ratio(df, "age", "T", "b")
    assert r.value.iloc[0] == 45.0 and r.age_band.iloc[0] == "30-34"


def test_forecast_no_lookahead_and_scoring():
    import numpy as np
    import pandas as pd
    from llmrel import forecast as F
    years = np.arange(2000, 2021)
    rows = [dict(country=c, indicator="marriage_rate", sex="F", age_band=b, year=y, value=50 * (0.97 ** (y - 2000)) * (1 + i / 10))
            for c in ("KR", "JP") for i, b in enumerate(F.BANDS) for y in years]
    d = pd.DataFrame(rows)
    L = np.log(F.series_table(d))
    # 원점 이후 자료를 바꿔도 원점 예측은 같아야 한다 (미래 누설 없음)
    a = F.predict_all(L, 2012, 2)
    L2 = L.copy(); L2.loc[2013:] += 1.0
    b = F.predict_all(L2, 2012, 2)
    assert all(abs(a[k][m] - b[k][m]) < 1e-12 for k in a for m in a[k])
    # 매끈한 지수 감소는 추세 모델이 거의 정확히 맞힌다
    bt = F.backtest(L, range(2008, 2018), horizons=(1,))
    assert bt[bt.model == "drift"].ape.max() < 1e-6
    fc = pd.DataFrame([dict(country="KR", indicator="marriage_rate", sex="F", age_band="25-29", year=2020, forecast=10.0, lo=9.0, hi=11.0),
                       dict(country="KR", indicator="marriage_rate", sex="F", age_band="25-29", year=2031, forecast=10.0, lo=9.0, hi=11.0)])
    s = F.score_forecasts(fc, d)
    assert list(s.status) == ["resolved", "pending"]


def test_regional_ratio_decomposes():
    import numpy as np
    import pandas as pd
    from llmrel import regional as R
    rows = []
    for code, (m, f, um, uf) in {"00": (300, 250, 200, 140), "11010": (100, 110, 70, 75), "32510": (200, 140, 130, 65)}.items():
        for sex, tot, un in (("M", m, um), ("F", f, uf)):
            rows += [dict(year=2025, code=code, name=code, sex=sex, kind="total", age_band="30-34", value=tot),
                     dict(year=2025, code=code, name=code, sex=sex, kind="unmarried", age_band="30-34", value=un)]
    t = R.region_table(pd.DataFrame(rows), 2025, ["30-34"])
    assert np.allclose(t.unmarried_ratio, t.pop_ratio * t.mrate_ratio)
    d = R.decompose(t)
    assert abs(d["인구 성비 몫"] + d["미혼율 비 몫"] - 1) < 1e-9
    assert R.analysis_regions({"31010", "31011", "31003", "11010", "31"}) == ["11010", "31010"]


def test_preference_logit_recovers_known_effects():
    import itertools
    import numpy as np
    import pandas as pd
    from llmrel import preference as P
    true = {"income=800+": 2.0, "income=~199": -1.5, "emp=비정규직": -1.0}
    rows = []
    for age, emp, edu, inc in itertools.product(*P.LEVELS.values()):
        z = -0.2 + sum(true.get(f"{k}={v}", 0) for k, v in (("age", age), ("emp", emp), ("edu", edu), ("income", inc)))
        n = 10000
        rows.append(dict(age=age, emp=emp, edu=edu, income=inc, total=n, married=n / (1 + np.exp(-z))))
    c = pd.DataFrame(rows)
    b = P.fit_logit(c)
    for k, v in true.items():
        assert abs(b[k] - v) < 1e-3
    assert abs(b["const"] + 0.2) < 1e-3
    me = P.marginal_effects(c, b).set_index(["factor", "level"]).vs_ref_pp
    assert me["income", "800+"] > 0 > me["emp", "비정규직"]


def test_rules_floor_tradeoff_with_known_values():
    import pandas as pd
    from llmrel import rules as R
    d = R.compare(range(3), periods=36, rules=("O1", "O2", "O4"))
    s = d.groupby("rule").agg(net=("net", "mean"), avg=("avg", "mean"))
    # 기준선을 올리면 평균은 오르고 순가치는 줄어든다; 기준선 = 비용일 때 순가치 최대
    assert s.avg["O1"] < s.avg["O2"] < s.avg["O4"]
    assert s.net["O1"] > s.net["O2"] > s.net["O4"]
    assert R._units(2.0, 1.0) == 4 and R._units(0.5, 1.0) == 0


def test_factcheck_engine_verdicts():
    """숫자 검증기: 정답을 아는 세 글에서 판정이 기대와 같아야 하고, 실제 기사 요약에서 '틀림' 오판이 없어야 한다."""
    import json
    import shutil
    import subprocess
    from collections import Counter
    from pathlib import Path
    import pytest
    if not shutil.which("node"):
        pytest.skip("node 없음")
    d = Path(__file__).parent.parent / "factcheck"
    exp = json.loads((d / "expected.json").read_text(encoding="utf-8"))
    for name, want in exp.items():
        out = subprocess.run(["node", str(d / "run.js"), str(d / name), "--json"], capture_output=True, text=True, check=True).stdout
        got = [c["verdict"] for c in json.loads(out) if c["verdict"] != "대상 아님"]
        if isinstance(want, list):
            assert got == want, (name, got)
        else:
            cnt = Counter(got)
            assert all(cnt.get(k, 0) == v for k, v in want.items()), (name, cnt)


def test_tuning_accepts_real_gain_and_rejects_noise():
    import numpy as np
    from llmrel import tuning
    rng = np.random.default_rng(0)
    noise = {s: rng.normal(0, 1, 400) for s in ("val", "test")}

    def evaluate(p, split):
        # x 는 진짜 효과(0.3 → 손실 감소), y 는 효과 없음
        return 1.0 + noise[split] + 0.3 * (p["x"] == 1) * -1 + rng.normal(0, 0.01, 400) * p["y"]
    spec = tuning.Spec("test", {"x": [0, 1], "y": [0, 1, 2]}, {"x": 0, "y": 0}, evaluate)
    r = tuning.hill_climb(spec, mode="certain")
    assert r["final"]["x"] == 1 and r["final"]["y"] == 0
    assert r["test_gain"] > 0.2


def test_epl_names_and_probabilities():
    from llmrel import epl
    assert epl.norm("Aston Villa FC") == epl.norm("Aston Villa") == "Aston Villa"
    assert epl.norm("AFC Bournemouth") == "Bournemouth"
    p = epl.probs(1.5, 1.1, 0.1)
    assert abs(p.sum() - 1) < 1e-9 and p[0] > p[2]


def test_epl_xg_modes():
    import numpy as np
    from llmrel import epl
    df = epl.load()
    assert df.hxg.notna().sum() > 2900                      # fbref xG 2017/18~ 연결
    h = epl.hide_xg(df, ["2024-25"])
    assert h[h.season == "2024-25"].hxg.isna().all() and h[h.season == "2023-24"].hxg.notna().any()
    mode, p = epl.current_mode(df)
    assert (mode == "골 모드") == (p is epl.GOALS_MODE)


def test_understat_parsing():
    import json
    from llmrel import understat
    items = [{"id": "1", "isResult": True, "h": {"title": "Manchester United"}, "a": {"title": "Tottenham"},
              "goals": {"h": "2", "a": "1"}, "xG": {"h": "1.83", "a": "0.71"}, "datetime": "2026-10-10 19:30:00"},
             {"id": "2", "isResult": False, "h": {"title": "Leeds"}, "a": {"title": "Manchester United"},
              "goals": {"h": None, "a": None}, "xG": {"h": None, "a": None}, "datetime": "2026-10-18 14:00:00"}]
    esc = json.dumps(items).encode("unicode_escape").decode().replace('"', "\\x22")
    html = f"<script>var datesData = JSON.parse('{esc}');</script>"
    df = understat.parse_dates(understat.parse_html(html))
    assert len(df) == 1 and df.home[0] == "Manchester United" and df.away[0] == "Tottenham Hotspur"
    assert abs(df.hxg[0] - 1.83) < 1e-9 and df.date[0] == "2026-10-10"


def test_epl_cap_and_early_rules_default_off():
    from llmrel import epl
    df = epl.load()
    base = epl.run(df, **{k: v for k, v in epl.XG_MODE.items() if k not in ("cap", "e")})
    same = epl.run(df, **epl.XG_MODE)
    assert abs(epl.losses(base).mean() - epl.losses(same).mean()) < 1e-12   # cap=99, e=1 은 기존과 같음
    capped = epl.run(df, **{**epl.XG_MODE, "cap": 2.0})
    t = capped[(capped.date == "2026-10-10") & (capped.home == "Manchester United")].iloc[0]
    assert t.pH < same[(same.date == "2026-10-10") & (same.home == "Manchester United")].iloc[0].pH


def test_annual_cap_rules_default_off():
    import pandas as pd
    from pathlib import Path
    from llmrel import forecast_tune as FT, regional_tune as RT
    root = Path(__file__).parent.parent
    d = pd.read_csv(root / "data_official" / "kr_jp_marriage_official.csv"); d["age_band"] = d.age_band.astype(str)
    sp = FT.make_spec(d)
    old = {k: v for k, v in FT.FINAL.items() if k not in ("cap", "robust")}
    assert abs(sp.evaluate(old, "val").mean() - sp.evaluate(FT.FINAL, "val").mean()) < 1e-12
    assert sp.evaluate({**FT.FINAL, "cap": 0.03}, "live").mean() < sp.evaluate(FT.FINAL, "live").mean()   # 2025 반등에서 급변 제한이 덜 틀림
    sgg, sido = RT.load(); rs = RT.make_spec(sgg, sido)
    assert len(rs.evaluate(RT.FINAL, "live")) == sgg.shape[1]
