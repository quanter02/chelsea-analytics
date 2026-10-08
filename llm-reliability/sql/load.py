"""셈법 원자료(CSV·장부)를 Oracle 연습 DB에 넣는다.

    pip install oracledb pandas
    docker compose up -d          # sql/ 폴더에서
    python load.py                # 00_schema.sql 실행 → 데이터 적재 → 파이썬 결과와 대조

접속 정보는 환경변수로 바꿀 수 있다: ORA_USER, ORA_PASSWORD, ORA_DSN (기본 sembeop/sembeop@localhost:1521/FREEPDB1)
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import oracledb
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from llmrel.ledger import _canon  # noqa: E402  장부와 같은 직렬화로 body_json 을 만든다
from llmrel.regional import SIDO, analysis_regions, region_table  # noqa: E402


def connect() -> oracledb.Connection:
    return oracledb.connect(user=os.getenv("ORA_USER", "sembeop"), password=os.getenv("ORA_PASSWORD", "sembeop"),
                            dsn=os.getenv("ORA_DSN", "localhost:1521/FREEPDB1"))


def split_sql(text: str) -> list[str]:
    """';' 로 끝나는 문장과 '/' 한 줄로 끝나는 PL/SQL 블록을 나눈다 (sqlplus 와 같은 규칙)."""
    out, buf, plsql = [], [], False
    for line in text.splitlines():
        s = line.strip()
        if not buf and (not s or s.startswith("--")):
            continue
        if not buf and re.match(r"(?i)(BEGIN|DECLARE)\b", s):
            plsql = True
        if plsql and s == "/":
            out.append("\n".join(buf)); buf, plsql = [], False
            continue
        buf.append(line)
        if not plsql and re.sub(r"--.*$", "", s).rstrip().endswith(";"):
            out.append(re.sub(r";\s*(--.*)?$", "", "\n".join(buf).rstrip())); buf = []
    if "".join(buf).strip():
        out.append("\n".join(buf))
    return out


def region_rows(u: pd.DataFrame, m: pd.DataFrame, mo: pd.DataFrame) -> list[tuple]:
    names = {}
    for d in (mo, m, u):                       # 뒤에 오는 표(인구)의 이름을 우선
        names.update(dict(zip(d.code, d.name)))
    for c, n in SIDO.items():
        names.setdefault(c, n)
    codes = set(names)
    analysis = set(analysis_regions(set(u.code)))
    rows = []
    for c in sorted(codes, key=lambda x: (len(x), x)):
        n = names[c]
        if c == "00":
            parent, kind = None, "전국"
        elif len(c) == 2:
            parent, kind = "00", ("동읍면부" if c in ("03", "04", "05") else "기타" if c == "90" else "시도")
        elif c[2:] in ("003", "004", "005"):
            parent, kind = c[:2], "동읍면부"
        elif c[-1] != "0" and c[:4] + "0" in codes:
            parent, kind = c[:4] + "0", "일반구"
        else:
            parent, kind = c[:2], "시군구"
        if "변동전" in n:
            kind = "변동전"
        rows.append((c, n, parent, kind, "Y" if c in analysis else "N"))
    return rows


def ledger_rows() -> tuple[list[tuple], list[tuple]]:
    recs = [json.loads(line) for line in (ROOT / "ledger" / "predictions.jsonl").open(encoding="utf-8")]
    rows, last = [], {}
    for r in recs:
        body = {k: v for k, v in r.items() if k != "hash"}
        rows.append((r["seq"], pd.Timestamp(r["recorded_at"]).tz_convert(None).to_pydatetime(), r["kind"], r["key"],
                     r.get("model", ""), json.dumps(r["payload"], ensure_ascii=False), r.get("code_commit", ""),
                     r["prev"], r["hash"], _canon(body)))
        if r["kind"].startswith("nowcast_2026"):
            last.setdefault(r["key"], {}).update(r["payload"])   # 같은 지역은 마지막 줄이 이긴다
    now = [(k, v["pred_2026"], v.get("lo80"), v.get("hi80"), None) for k, v in sorted(last.items())]
    return rows, now


def main() -> None:
    u = pd.read_csv(ROOT / "data_regional/kr_unmarried_by_region.csv", dtype={"code": str})
    m = pd.read_csv(ROOT / "data_regional/kr_marriages_by_region.csv", dtype={"code": str})
    mo = pd.read_csv(ROOT / "data_monthly/kr_marriages_monthly_sido.csv", dtype={"code": str, "month": str})
    g = pd.read_csv(ROOT / "data_regional/kr_migration_by_region.csv", dtype={"mcode": str})
    led, now = ledger_rows()

    con = connect()
    cur = con.cursor()
    for stmt in split_sql((HERE / "00_schema.sql").read_text(encoding="utf-8")):
        cur.execute(stmt)

    loads = {
        "region": ("INSERT INTO region VALUES (:1, :2, :3, :4, :5)", region_rows(u, m, mo)),
        "population": ("INSERT INTO population VALUES (:1, :2, :3, :4, :5, :6)",
                       list(u[["year", "code", "sex", "kind", "age_band", "value"]].itertuples(index=False, name=None))),
        "marriage_year": ("INSERT INTO marriage_year VALUES (:1, :2, :3)",
                          list(m[["code", "year", "value"]].itertuples(index=False, name=None))),
        "marriage_month": ("INSERT INTO marriage_month VALUES (:1, :2, :3)",
                           list(mo[["code", "month", "value"]].itertuples(index=False, name=None))),
        "migration": ("INSERT INTO migration VALUES (:1, :2, :3, :4, :5, :6)",
                      list(g[["year", "mcode", "name", "sex", "age_band", "value"]].itertuples(index=False, name=None))),
        "ledger": ("INSERT INTO ledger VALUES (:1, :2, :3, :4, :5, :6, :7, :8, :9, :10)", led),
        "nowcast": ("INSERT INTO nowcast VALUES (:1, :2, :3, :4, :5)", now),
    }
    for table, (sql, rows) in loads.items():
        cur.executemany(sql, [tuple(x.item() if hasattr(x, "item") else x for x in r) for r in rows])
        print(f"{table:15s} {len(rows):6d}행")
    con.commit()

    # 파이썬 분석(llmrel/regional.py)과 같은 숫자가 나오는지 대조
    py = region_table(u, 2025).set_index("code").unmarried_ratio
    cur.execute("""
        SELECT p.code, SUM(CASE WHEN sex = 'M' AND kind = 'unmarried' THEN pop END)
                     / SUM(CASE WHEN sex = 'F' AND kind = 'unmarried' THEN pop END)
        FROM population p JOIN region r ON r.code = p.code
        WHERE p.yr = 2025 AND p.age_band IN ('25-29', '30-34', '35-39') AND (r.is_analysis = 'Y' OR r.code = '00')
        GROUP BY p.code""")
    sql = dict(cur.fetchall())
    diff = max(abs(float(sql[c]) - py[c]) for c in py.index)
    same = set(sql) == set(py.index) and diff < 1e-9
    print(f"\n미혼 성비 대조: SQL {len(sql)}곳 vs 파이썬 {len(py)}곳, 최대 차이 {diff:.1e} → {'일치 ✅' if same else '불일치 ❌'}")
    for c in ("00", "11140", "32590"):
        print(f"  {c} {float(sql[c]):.2f}")  # 전국 1.33 · 마포구 0.89 · 인제군 3.21
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
