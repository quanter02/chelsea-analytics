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
