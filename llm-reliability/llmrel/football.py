"""실제 데이터 테스트: StatsBomb 공개 데이터의 실제 경기 결과로 신뢰도 프레임워크를 검증한다.

LLM 대신 경기 전 정보만 쓰는 예측 모델 5개가 승(H)/무(D)/패(A)와 확신도를 기록하고,
경기가 끝나면 실제 결과가 정답으로 붙는다. 판단 → 나중에 확정되는 정답이라는 구조가 같으므로
프로파일·보정·라우터·변화 감지를 실제 결과로 시험할 수 있다.

데이터: https://github.com/statsbomb/open-data (비상업적 이용, StatsBomb 출처 표기)
"""
from __future__ import annotations

import json
import math
import urllib.request
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/matches"
LEAGUES = {(2, 27): "EPL", (11, 27): "La Liga", (12, 27): "Serie A", (7, 27): "Ligue 1"}   # 2015/16, 전 경기 공개
CACHE = Path(__file__).resolve().parents[1] / "data_cache"
TASK = "match_result"
WRONG_LOSS, ABSTAIN_LOSS = 1.0, 0.5      # 틀리면 1, 판단 보류는 0.5 (동전 던지기보다 나은 경우에만 답할 가치)


def fetch_matches() -> pd.DataFrame:
    CACHE.mkdir(exist_ok=True)
    rows = []
    for (cid, sid), league in LEAGUES.items():
        f = CACHE / f"matches_{cid}_{sid}.json"
        if not f.exists():
            f.write_bytes(urllib.request.urlopen(f"{BASE}/{cid}/{sid}.json", timeout=60).read())
        for m in json.loads(f.read_text(encoding="utf-8")):
            hs, as_ = m["home_score"], m["away_score"]
            rows.append(dict(match_id=str(m["match_id"]), league=league, date=m["match_date"],
                             kickoff=m.get("kick_off") or "15:00:00.000",
                             home=m["home_team"]["home_team_name"], away=m["away_team"]["away_team_name"],
                             hs=hs, as_=as_, result="H" if hs > as_ else "A" if hs < as_ else "D"))
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df.date + " " + df.kickoff.str[:8])
    return df.sort_values(["ts", "match_id"]).reset_index(drop=True)


def _poisson_probs(lh: float, la: float, kmax: int = 9) -> tuple[float, float, float]:
    ph = [math.exp(-lh) * lh ** k / math.factorial(k) for k in range(kmax)]
    pa = [math.exp(-la) * la ** k / math.factorial(k) for k in range(kmax)]
    h = sum(ph[i] * pa[j] for i in range(kmax) for j in range(kmax) if i > j)
    d = sum(ph[i] * pa[i] for i in range(kmax))
    return h, d, 1 - h - d


def _elo_probs(rh: float, ra: float, hfa: float = 60.0) -> tuple[float, float, float]:
    e = 1 / (1 + 10 ** ((ra - rh - hfa) / 400))          # 홈팀 기대 득점(승=1, 무=0.5)
    d = max(0.18, 0.30 - 0.45 * abs(e - 0.5))
    h = max(e - d / 2, 0.02)
    return h, d, max(1 - h - d, 0.02)


def predict(matches: pd.DataFrame) -> pd.DataFrame:
    """각 경기 시작 전까지의 결과만 써서 예측 (미래 정보 누설 없음). 리그마다 시즌 첫 경기부터 학습."""
    elo = defaultdict(lambda: 1500.0)
    rates = defaultdict(lambda: np.array([0.46, 0.26, 0.28]) * 10)   # 리그별 H/D/A 빈도 (사전값 10경기분)
    gf, ga = defaultdict(lambda: deque(maxlen=6)), defaultdict(lambda: deque(maxlen=6))
    lg_goals = defaultdict(lambda: [2.6 * 10, 10])
    out = []
    for m in matches.itertuples():
        L = m.league
        r = rates[L] / rates[L].sum()
        eh, ed, ea = _elo_probs(elo[(L, m.home)], elo[(L, m.away)])
        avg = lg_goals[L][0] / lg_goals[L][1] / 2
        def att(t): return (sum(gf[(L, t)]) + 3 * avg) / (len(gf[(L, t)]) + 3) / avg     # 리그 평균 쪽으로 축소
        def dfn(t): return (sum(ga[(L, t)]) + 3 * avg) / (len(ga[(L, t)]) + 3) / avg
        lh = avg * 1.15 * att(m.home) * dfn(m.away); la = avg * 0.87 * att(m.away) * dfn(m.home)
        ph, pd_, pa = _poisson_probs(lh, la)
        preds = {
            "home-always":  (r, "H"),                                   # 항상 홈승, 확신도 = 리그 홈승률
            "elo":          ((eh, ed, ea), None),
            "elo-cautious": ((eh, ed, ea), None),                       # 최고 확률 < 0.5 이면 기권
            "elo-sharp":    ((eh, ed, ea), None),                       # 확신도를 일부러 부풀린 과신형
            "poisson-form": ((ph, pd_, pa), None),                      # 최근 6경기 득실 기반
        }
        for mid, (p, fixed) in preds.items():
            p = np.asarray(p, float); p = p / p.sum()
            k = "HDA".index(fixed) if fixed else int(np.argmax(p))
            ans, conf = "HDA"[k], float(p[k])
            if mid == "elo-sharp":
                conf = conf ** 0.5
            if mid == "elo-cautious" and conf < 0.5:
                ans, conf = "ABSTAIN", None
            out.append(dict(model_id=mid, model_version="v1", prompt_version="-", task_type=TASK, input_hash=m.match_id,
                            answer=ans, confidence=conf, evidence=None, evidence_ok=None if ans == "ABSTAIN" else 1,
                            cost_usd=0.0, latency_ms=0, created_at=m.ts.isoformat(), league=L))
        # 경기 후 상태 갱신
        s = {"H": 1.0, "D": 0.5, "A": 0.0}[m.result]
        e = 1 / (1 + 10 ** ((elo[(L, m.away)] - elo[(L, m.home)] - 60) / 400))
        elo[(L, m.home)] += 20 * (s - e); elo[(L, m.away)] -= 20 * (s - e)
        rates[L] = rates[L] + np.array([m.result == "H", m.result == "D", m.result == "A"], float)
        gf[(L, m.home)].append(m.hs); ga[(L, m.home)].append(m.as_)
        gf[(L, m.away)].append(m.as_); ga[(L, m.away)].append(m.hs)
        lg_goals[L][0] += m.hs + m.as_; lg_goals[L][1] += 1
    return pd.DataFrame(out)


def outcomes(matches: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(dict(input_hash=matches.match_id, task_type=TASK, true_label=matches.result, label_source="match_result",
                             labeled_at=(matches.ts + pd.Timedelta(hours=2)).dt.strftime("%Y-%m-%dT%H:%M:%S")))


def loss_matrix() -> pd.DataFrame:
    rows = [(TASK, t, a, 0.0 if t == a else WRONG_LOSS) for t in "HDA" for a in "HDA"]
    rows += [(TASK, t, "ABSTAIN", ABSTAIN_LOSS) for t in "HDA"]
    return pd.DataFrame(rows, columns=["task_type", "true_label", "answer", "loss_usd"])
