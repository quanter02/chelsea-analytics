"""예측 장부: 결과가 나오기 전에 낸 예측을 고칠 수 없게 남긴다.

ledger/predictions.jsonl 에 한 줄씩 추가만 한다. 각 줄의 hash 는
  sha256( 앞 줄의 hash + 이 줄 내용(정렬된 JSON) )
이라서, 과거 줄을 하나라도 고치면 그 뒤 hash 가 전부 어긋난다 (verify 가 잡아냄).
git 커밋 시각과 GitHub 기록이 "이 예측은 이 날짜에 이미 있었다"는 바깥 증거가 된다.
같은 예측을 다시 기록하면(내용 동일) 건너뛰고, 내용이 바뀌면 새 줄로 남긴다 (예측을 고친 이력도 공개).

    python -m llmrel.ledger verify      # 체인 검사 + 마지막 hash 출력
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
PATH = ROOT / "ledger" / "predictions.jsonl"
GENESIS = "0" * 64


def _canon(rec: dict) -> str:
    return json.dumps(rec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(prev: str, body: dict) -> str:
    return hashlib.sha256((prev + _canon(body)).encode("utf-8")).hexdigest()


def read(path: Path = PATH) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def head(path: Path = PATH) -> str:
    rows = read(path)
    return rows[-1]["hash"] if rows else GENESIS


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def append(entries: list[dict], path: Path = PATH, now: str | None = None) -> int:
    """entries: {kind, key, payload, model, [evidence]} 의 목록. 추가한 줄 수를 돌려준다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = read(path)
    latest = {}
    for r in rows:
        latest[(r["kind"], r["key"])] = r["payload"]
    prev = rows[-1]["hash"] if rows else GENESIS
    seq = len(rows)
    now = now or dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    added = []
    for e in entries:
        k = (e["kind"], e["key"])
        if latest.get(k) == e["payload"]:
            continue                                        # 같은 예측은 다시 남기지 않음
        body = {"seq": seq, "recorded_at": now, "kind": e["kind"], "key": e["key"], "payload": e["payload"],
                "model": e.get("model", ""), "code_commit": git_head(), "evidence": e.get("evidence", ""), "prev": prev}
        h = _hash(prev, body)
        added.append({**body, "hash": h})
        latest[k], prev, seq = e["payload"], h, seq + 1
    if added:
        with path.open("a", encoding="utf-8") as f:
            for r in added:
                f.write(_canon(r) + "\n")
    return len(added)


def verify(path: Path = PATH) -> tuple[bool, int, str]:
    """(정상 여부, 검사한 줄 수, 마지막 hash 또는 처음 어긋난 줄 설명)."""
    prev = GENESIS
    rows = read(path)
    for i, r in enumerate(rows):
        body = {k: v for k, v in r.items() if k != "hash"}
        if r.get("prev") != prev or r.get("seq") != i or _hash(prev, body) != r.get("hash"):
            return False, i, f"{i}번째 줄이 어긋남 (seq={r.get('seq')})"
        prev = r["hash"]
    return True, len(rows), prev


if __name__ == "__main__":
    if sys.argv[1:] == ["verify"]:
        ok, n, msg = verify()
        print(("정상" if ok else "변조 의심"), f"{n}줄", msg)
        sys.exit(0 if ok else 1)
    print(__doc__)
