"""1주차 과제 채점. 정답 코드는 보여주지 않고 통과 여부와 힌트만 알려준다.

    python exercises/week1_v1/check.py     # llm-reliability 폴더에서 실행
"""
import sys
import types
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
# 캐시(.pyc)를 거치지 않고 매번 소스를 새로 읽는다: 고친 직후 다시 채점해도 옛 코드가 돌지 않게
me = types.ModuleType("my_v1")
src = HERE / "my_v1.py"
exec(compile(src.read_text(encoding="utf-8"), str(src), "exec"), me.__dict__)

ans = pd.read_csv(me.ANSWER, dtype={"code": str}).set_index("code")


def step1():
    s = me.last_year_counts()
    assert isinstance(s, pd.Series), "pd.Series를 반환해야 해요"
    assert (s.index.str.len() == 5).all(), "코드 5자리(시군구)만 남겨야 해요. code를 문자열(dtype=str)로 읽었나요?"
    assert s.get("11010") == 615, f"종로구(11010)는 615여야 해요 (지금 {s.get('11010')})"
    assert (s.reindex(ans.index) == ans.last_2025).all(), "243곳 중 일부 값이 달라요. 2025년만 골랐나요?"


def step2():
    d = me.half_year_sums()
    assert {2025, 2026} <= set(d.columns), "열 이름이 2025, 2026(숫자)이어야 해요"
    assert d.loc["11", 2025] > 0, "서울('11') 행이 있어야 해요"
    m = pd.read_csv(me.MONTHLY, dtype={"code": str})
    jan = m[(m.code == "11") & (m.month == 202501)].value.iloc[0]
    assert d.loc["11", 2025] > jan * 3, "1~6월 '합계'여야 해요 (한 달 값만 들어간 것 같아요)"
    full = m[(m.code == "11") & (m.month // 100 == 2025)].value.sum()
    assert d.loc["11", 2025] < full, "1~12월이 아니라 1~6월만 더해야 해요"


def step3():
    g = me.sido_growth()
    assert isinstance(g, pd.Series), "pd.Series를 반환해야 해요"
    assert 0.5 < g.get("11", 0) < 1.5, "배율은 1 근처 숫자여야 해요 (예 1.1). 퍼센트로 바꾸지 마세요"
    assert abs(round(615 * g["11"]) - ans.loc["11010", "pred_2026"]) <= 0, "서울 배율이 정답과 달라요. 2026 ÷ 2025 순서를 확인하세요"


def step4():
    p = me.predict()
    assert len(p) == len(ans), f"정답 파일의 {len(ans)}곳만 남겨야 해요 (지금 {len(p)}곳)"
    diff = (p.reindex(ans.index) - ans.pred_2026).abs()
    assert diff.max() == 0, f"{int((diff > 0).sum())}곳이 달라요 (가장 큰 차이 {int(diff.max())}건). 반올림(round)을 했나요?"


def step5():
    r = me.compare()
    assert r.get("n") == len(ans) and r.get("same") == len(ans) and r.get("max_diff") == 0, \
        f"n={len(ans)}, same={len(ans)}, max_diff=0이 나와야 해요 (지금 {r})"


ok = 0
for i, f in enumerate([step1, step2, step3, step4, step5], 1):
    try:
        f()
        print(f"✅ {i}단계 통과")
        ok += 1
    except NotImplementedError:
        print(f"⬜ {i}단계 아직 안 함 — README의 {i}단계 힌트를 보세요")
        break
    except AssertionError as e:
        print(f"❌ {i}단계: {e}")
        break
    except Exception as e:
        print(f"❌ {i}단계 실행 오류: {type(e).__name__}: {e}")
        break
print(f"\n{ok}/5 단계 완료" + (" — v1을 완전히 재현했어요! 🎉" if ok == 5 else ""))
sys.exit(0)
