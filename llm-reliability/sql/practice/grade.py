"""내 풀이(my_answers.sql)를 정답(answers.sql)과 결과로 비교한다. 정답 파일은 열지 않아도 된다.

    python practice/grade.py              # 전부 채점
    python practice/grade.py 4 13         # 4번, 13번만
    python practice/grade.py --show 4     # 내 결과 앞부분을 출력 (정답은 보여주지 않음)

규칙
  * 파일 안에서 '-- Q01' 같은 줄이 문제의 시작. 그 아래 문장들을 순서대로 실행하고 마지막 SELECT 결과를 비교한다.
  * 정답에 ORDER BY 가 있으면 행 순서까지, 없으면 행 묶음만 본다. 열 이름은 보지 않고 열 순서와 값을 본다.
  * 숫자는 소수 넷째 자리까지 비교한다. 문제마다 실행 뒤 ROLLBACK 하므로 연습 테이블이 더러워지지 않는다.
"""
from __future__ import annotations

import re
import sys
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from load import connect, split_sql  # noqa: E402


def parse(path: Path) -> dict[int, list[str]]:
    blocks, cur = {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"--\s*Q(\d+)\b", line.strip())
        if m:
            cur = int(m.group(1)); blocks[cur] = []
        elif cur is not None:
            blocks[cur].append(line)
    return {q: split_sql("\n".join(lines)) for q, lines in blocks.items() if "\n".join(lines).strip()}


def norm(v):
    if isinstance(v, (int, float, Decimal)):
        return round(float(v), 4)
    if isinstance(v, str):
        return v.rstrip()
    return v


def top_level_order_by(sql: str) -> bool:
    """괄호 밖(바깥 쿼리)에 ORDER BY 가 있는가. OVER (ORDER BY ...) 나 서브쿼리 안의 것은 제외."""
    sql = re.sub(r"'[^']*'|--[^\n]*", "", sql)
    depth = 0
    for i, ch in enumerate(sql):
        depth += (ch == "(") - (ch == ")")
        if depth == 0 and re.match(r"(?is)order\s+by\b", sql[i:i + 12]) and (i == 0 or not sql[i - 1].isalnum()):
            return True
    return False


def run(cur, stmts: list[str]):
    rows, ordered = None, False
    for s in stmts:
        cur.execute(s)
        if cur.description:
            rows = [tuple(norm(v) for v in r) for r in cur.fetchall()]
            ordered = top_level_order_by(s)
    return rows, ordered


def main(argv: list[str]) -> None:
    show = "--show" in argv
    only = {int(a) for a in argv if a.isdigit()}
    key = parse(HERE / "answers.sql")
    mine_path = HERE / "my_answers.sql"
    mine = parse(mine_path) if mine_path.exists() else {}
    con = connect()
    cur = con.cursor()
    passed = 0
    for q in sorted(key):
        if only and q not in only:
            continue
        if q not in mine:
            print(f"Q{q:02d}  ·  아직 안 풂"); continue
        try:
            want, ordered = run(cur, key[q]); con.rollback()
            got, _ = run(cur, mine[q]); con.rollback()
        except Exception as e:  # noqa: BLE001
            con.rollback()
            print(f"Q{q:02d}  ❌ 실행 오류: {str(e).splitlines()[0]}"); continue
        if got is None:
            print(f"Q{q:02d}  ❌ SELECT 결과가 없음"); continue
        ok = got == want if ordered else sorted(map(repr, got)) == sorted(map(repr, want))
        hint = ""
        if not ok:
            if len(got) != len(want):
                hint = f" (행 수 {len(got)} — 정답 {len(want)}행)"
            elif got and want and len(got[0]) != len(want[0]):
                hint = f" (열 수 {len(got[0])} — 정답 {len(want[0])}열)"
            elif ordered and sorted(map(repr, got)) == sorted(map(repr, want)):
                hint = " (값은 맞고 순서가 다름)"
            else:
                hint = " (행 수는 맞고 값이 다름)"
        passed += ok
        print(f"Q{q:02d}  {'✅' if ok else '❌'}{hint}")
        if show:
            for r in got[:8]:
                print("      ", r)
    total = len([q for q in key if not only or q in only])
    print(f"\n{passed}/{total} 통과")


if __name__ == "__main__":
    main(sys.argv[1:])
