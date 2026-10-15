"""EPL 아이디어 실험대 실행.

    python epl_idea.py             # 예시 아이디어 3개를 검증 구간에서 판정 → output/epl_ideas/report.md
    python epl_idea.py --test 이름  # 검증을 통과한 아이디어 하나만 시험 구간을 한 번 연다

새 아이디어 추가: llmrel/idea_lab.py 의 EXAMPLES 처럼 함수 하나 + 가설 한 줄을 만들어 evaluate()에 넘긴다.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

from llmrel import idea_lab

OUT = Path(__file__).parent / "output" / "epl_ideas"


def main():
    warnings.filterwarnings("ignore")
    ctx = idea_lab.context()
    if "--test" in sys.argv:
        name = sys.argv[sys.argv.index("--test") + 1]
        fn, hyp = idea_lab.EXAMPLES[name]
        r = idea_lab.evaluate(name, fn, hyp, ctx, open_test=True)
        print(f"{name}: 시험 구간 이득 {r['test']:+.5f} (95% {r['test_lo']:+.5f} ~ {r['test_hi']:+.5f}, {r['n_test']}경기)")
        return
    res = [idea_lab.evaluate(n, fn, hyp, ctx) for n, (fn, hyp) in idea_lab.EXAMPLES.items()]
    k = len(idea_lab.tried().drop_duplicates(["name", "code"]))
    rows = "\n".join(f"| {r['name']} | {r['hypothesis']} | {r['beta']:+.2f} | {r['val']:+.5f} | {r['val_lo']:+.5f} ~ {r['val_hi']:+.5f} | "
                     f"{'통과' if r['passed'] else '탈락'} |" for r in res)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.md").write_text(f"""# EPL 아이디어 실험대

시장 확률(마감 배당, 마진 제거)에 아이디어만큼 조정을 더해 시장보다 잘 맞히는지 본다.
β(아이디어 세기)는 학습 구간(2010/11~2020/21)에서만 고르고, 판정은 검증 구간(2021/22~2023/24, {res[0]['n_val']}경기).
이득 = 시장 RPS − 조정 RPS (양수면 아이디어가 도움). 95% 하한 > 0 이고 β ≠ 0 이면 통과.

| 아이디어 | 가설 | β | 검증 이득 | 95% 범위 | 판정 |
|---|---|---|---|---|---|
{rows}

- 지금까지 장부에 등록된 아이디어 시도: **{k}개** → 아무 효과 없어도 하나쯤 우연히 통과할 확률 ≈ {1 - 0.975 ** k:.0%}
- 시험 구간(2024/25~2025/26)은 통과한 아이디어만 `python epl_idea.py --test 이름` 으로 한 번 연다.
""", encoding="utf-8")
    print((OUT / "report.md").read_text())


if __name__ == "__main__":
    main()
