"""Extraction and aggregation checks for the text trend report (no network)."""
import json

import pandas as pd
import pytest

from trends import aggregate as A, extract as X, llm


@pytest.mark.parametrize("text,age", [
    ("저 28살인데 남친이 연락을 잘 안 해요", 28),
    ("29세 직장인입니다", 29),
    ("서른 앞두고 결혼 고민", 30),
    ("스물일곱 살 여자예요", 27),
    ("저는 20대 후반이고요", 28),
    ("96년생인데요", 30),          # post year 2026
    ("31F, need advice", 31),
])
def test_author_age(text, age):
    assert X.author_age(text, year=2026)[0] == age


@pytest.mark.parametrize("text", [
    "남친이 32살인데 철이 없어요",     # someone else's age
    "오빠는 35살이에요",
    "5살 차이 나는데 괜찮을까요",       # age gap, not an age
    "20대 남자들은 왜 그래요",          # decade without self-reference
])
def test_not_author_age(text):
    assert X.author_age(text, year=2026)[0] is None


def test_gender_is_explicit_only_by_default():
    assert X.author_gender("29살 여자입니다")[0] == "F"
    assert X.author_gender("저는 남자인데 질문이요")[0] == "M"
    assert X.author_gender("남친이랑 싸웠어요")[0] is None
    assert X.author_gender("남친이랑 싸웠어요", partner_cue=True) == ("F", "partner")
    assert X.author_gender("여자인데 여친이 있어요")[0] == "F"     # explicit beats the partner cue


def test_topics_multi_label():
    t = X.topics("연봉 차이 때문에 결혼 얘기하다 싸웠어요")
    assert {"경제력·돈", "결혼", "갈등·이별"} <= set(t)
    assert X.topics("오늘 날씨 맑음") == []


def test_writer_counted_once():
    df = pd.DataFrame({
        "text": ["x"] * 4,
        "author_hash": ["a", "a", "a", "b"],
        "age_band": ["25-29"] * 4, "gender": ["F"] * 4,
        "topics": [["경제력·돈"], ["경제력·돈"], ["결혼"], ["결혼"]],
        "sentiment": ["부정"] * 4,
    })
    w = A.writers(df)
    assert len(w) == 2 and w.loc["a", "topics"] == ["결혼", "경제력·돈"]
    tbl = A.share_table(w).set_index(["topic", "age_band"])
    assert tbl.loc[("결혼", "25-29"), "share"] == 1.0
    assert tbl.loc[("경제력·돈", "25-29"), "share"] == 0.5


def test_wilson_interval():
    lo, hi = A.wilson(50, 100)
    assert lo < 0.5 < hi and abs((lo + hi) / 2 - 0.5) < 1e-9
    assert all(pd.isna(v) for v in A.wilson(0, 0))  # NaN for empty bands


def test_llm_extract_uses_labels_and_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE", tmp_path / "cache.jsonl")
    calls = []

    class Block:
        type = "text"

        def __init__(self, text):
            self.text = text

    class Resp:
        stop_reason = "end_turn"

        def __init__(self, text):
            self.content = [Block(text)]

    class Fake:
        class beta:
            class messages:
                @staticmethod
                def create(**kw):
                    calls.append(kw)
                    ids = [int(s.split('"')[1]) for s in kw["messages"][0]["content"].split("<post id=")[1:]]
                    out = [{"id": i, "age": 29, "gender": "F", "topics": ["결혼"], "sentiment": "중립"} for i in ids]
                    return Resp(json.dumps({"results": out}))

    df = pd.DataFrame({"text": ["서른 앞두고 결혼 고민이에요", "회사 그만두고 싶다"]})
    out = llm.extract(df, client=Fake, verbose=False)
    assert list(out.age_band) == ["25-29", "25-29"] and out.topics[0] == ["결혼"]
    assert calls[0]["fallbacks"] == "default" and calls[0]["output_config"]["format"]["type"] == "json_schema"
    llm.extract(df, client=Fake, verbose=False)   # second run is served from the cache
    assert len(calls) == 1
