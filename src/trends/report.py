"""Self-contained HTML trend report (Korean): who talks about what, by age band, with uncertainty shown.

The table is the chart: each cell is the share of writers in that band who mention the topic, shaded on a
single-hue scale, with the 95% CI in the tooltip and in the text. Small bands are greyed out, not hidden.
"""
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

from cfc.report import CSS
from . import aggregate as A, extract as X

EXTRA_CSS = """
.heat td.c{position:relative;min-width:68px}
.heat td.c span{position:relative;z-index:1}
.heat td.c::before{content:"";position:absolute;inset:3px;border-radius:4px;background:var(--blue);opacity:var(--a);z-index:0}
.heat td.small{color:var(--muted)}
.heat td.small::before{background:var(--muted)}
.heat td.c:hover{outline:2px solid var(--ink);outline-offset:-2px}
.legend{font-size:13px;color:var(--muted);margin:6px 0 0}
.note{font-size:13.5px;color:var(--muted)}
"""


def _pct(x: float, d: int = 0) -> str:
    return "–" if pd.isna(x) else f"{x * 100:.{d}f}%"


def heat_table(tbl: pd.DataFrame) -> str:
    bands = list(dict.fromkeys(tbl.age_band))
    n = tbl.drop_duplicates("age_band").set_index("age_band").n
    head = "".join(f"<th>{b}세<br><small>n={n[b]:,}</small></th>" for b in bands)
    vmax = max(tbl.share.max(), 0.01)
    body = []
    for topic, sub in tbl.groupby("topic", sort=False):
        cells = []
        for _, r in sub.iterrows():
            a = 0.08 + 0.55 * (r.share / vmax)
            tip = (f"{topic} · {r.age_band}세: {_pct(r.share, 1)} (95% CI {_pct(r.lo, 1)}–{_pct(r.hi, 1)}), "
                   f"{r.k:,}/{r.n:,}명, 다른 연령대 대비 {r.vs_rest * 100:+.1f}%p")
            cls = "c small" if r.small else "c"
            cells.append(f'<td class="{cls}" style="--a:{a:.2f}" title="{html.escape(tip)}"><span>{_pct(r.share)}</span></td>')
        body.append(f"<tr><td>{html.escape(topic)}</td>{''.join(cells)}</tr>")
    return (f'<div class="tbl"><table class="heat"><thead><tr><th>주제</th>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>'
            f'<p class="legend">칸 = 해당 연령대 작성자 중 그 주제를 언급한 비율. 진할수록 높음. '
            f'회색 = 작성자 {A.MIN_N}명 미만이라 참고만. 마우스를 올리면 신뢰구간이 보입니다.</p>')


def standout_list(st: pd.DataFrame) -> str:
    if st.empty:
        return "<p>다른 연령대와 통계적으로 뚜렷하게 다른 칸이 없습니다 (|z| ≥ 2.5 기준). 표본을 늘려 보세요.</p>"
    items = []
    for _, r in st.iterrows():
        word = "더 많이" if r.vs_rest > 0 else "덜"
        items.append(f'<li class="{"weak" if r.vs_rest > 0 else "threat"}"><h4>{r.age_band}세는 「{html.escape(r.topic)}」를 {word} 이야기한다</h4>'
                     f'<p class="ev">{_pct(r.share, 1)} (95% CI {_pct(r.lo, 1)}–{_pct(r.hi, 1)}, {r.k:,}/{r.n:,}명) · '
                     f'다른 연령대 대비 {r.vs_rest * 100:+.1f}%p · z={r.z:.1f}</p></li>')
    return f'<ul class="findings">{"".join(items)}</ul>'


def baseline_section(path: Path | None) -> str:
    """Optional CSV of official survey numbers: age_band,item,share(0-1),source. Shown next to our numbers so
    readers can see whether the text data points the same way."""
    if not path or not Path(path).exists():
        return ("<p class='note'>공식 통계 기준선이 없습니다. 통계청 사회조사 등의 수치를 "
                "<code>data/trends/baseline.csv</code>(age_band,item,share,source)로 넣으면 여기에 나란히 표시됩니다. "
                "텍스트 데이터는 편향이 크므로 방향이 공식 통계와 맞는지 꼭 확인하세요.</p>")
    b = pd.read_csv(path)
    rows = "".join(f"<tr><td>{html.escape(str(r.item))}</td><td>{html.escape(str(r.age_band))}</td>"
                   f"<td>{_pct(r.share, 1)}</td><td>{html.escape(str(r.source))}</td></tr>" for _, r in b.iterrows())
    return (f'<div class="tbl"><table><thead><tr><th>항목</th><th>연령대</th><th>비율</th><th>출처</th></tr></thead>'
            f"<tbody>{rows}</tbody></table></div>")


def build(df: pd.DataFrame, out: Path, title: str = "2030 여성 연애·소비 트렌드", gender: str | None = "F",
          sources: str = "", baseline: Path | None = None, extractor: str = "rule") -> Path:
    w = A.writers(df, gender=gender)
    if w.empty:
        raise SystemExit("나이대(와 성별)를 알 수 있는 작성자가 없습니다. 데이터를 더 모으거나 --gender all 로 실행하세요.")
    tbl = A.share_table(w)
    st = A.standouts(tbl)
    cov = A.coverage(df)
    who = {"F": "여성", "M": "남성", None: "전체"}[gender]
    neg = w.groupby("age_band").neg_share.mean()
    neg_rows = "".join(f"<tr><td>{b}세</td><td>{_pct(v, 1)}</td></tr>" for b, v in neg.items())
    funnel = [("수집한 글", cov["posts"]), ("나이 확인", cov["with_age"]), ("성별 확인", cov["with_gender"]),
              (f"분석 대상 작성자 ({who})", len(w))]

    page = f"""<meta charset="utf-8">
<title>{html.escape(title)}</title>
<meta name="description" content="공개 텍스트에서 작성자의 나이·성별·주제를 추출해 연령대별 관심사 분포를 신뢰구간과 함께 보여주는 리포트">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>{CSS}{EXTRA_CSS}</style>
<div class="wrap">
<header class="mast">
<p class="kicker">Text trend report · {html.escape(sources) or "public text"}</p>
<h1>{html.escape(title)}</h1>
<p class="sub">스스로 나이와 성별을 밝힌 작성자 {len(w):,}명의 글에서 무엇을 이야기하는지 연령대별로 셌습니다.
비율은 "그 연령대 전체"가 아니라 "이 데이터에 글을 쓴 사람"의 비율입니다.</p>
</header>
<div class="summary">{"".join(f"<div><b>{v:,}</b><span>{k}</span></div>" for k, v in funnel)}</div>

<section class="part"><p class="eyebrow">01 · 핵심 차이</p><h2>연령대별로 뚜렷하게 다른 것</h2>
{standout_list(st)}</section>

<section class="part"><p class="eyebrow">02 · 전체 표</p><h2>주제 × 연령대</h2>
{heat_table(tbl)}
<div class="tables"><div class="tbl"><table><caption>부정적인 글의 비율</caption>
<thead><tr><th>연령대</th><th>작성자 평균</th></tr></thead><tbody>{neg_rows}</tbody></table></div></div>
</section>

<section class="part"><p class="eyebrow">03 · 기준선</p><h2>공식 통계와 비교</h2>{baseline_section(baseline)}</section>

<section class="part method"><p class="eyebrow">04 · 방법과 한계</p><h2>이 숫자를 읽는 법</h2>
<ul>
<li><b>추출:</b> {"Claude가 글마다 작성자 본인의 나이·성별·주제를 라벨링" if extractor == "llm" else "규칙 기반. 본인 소개(“28살인데”, “29살 여자입니다”)만 인정하고 남친·친구 등 다른 사람의 나이는 제외"}. 성별은 본인이 밝힌 경우만 셉니다.</li>
<li><b>단위는 작성자:</b> 한 사람이 같은 주제로 글을 여러 번 써도 1명으로 셉니다.</li>
<li><b>신뢰구간:</b> Wilson 95%. 칸 {len(tbl)}개를 동시에 비교하므로 우연한 차이를 줄이려고 |z| ≥ 2.5만 핵심 차이로 올렸습니다.</li>
<li><b>선택 편향:</b> 글을 쓰는 사람은 전체를 대표하지 않습니다. 특히 고민 글은 문제가 있는 사람이 씁니다. 나이·성별을 밝히는 사람은 더 그렇습니다.</li>
<li><b>개인에게 적용 금지:</b> 집단 비율이지 개인 예측이 아닙니다. 같은 연령대 안의 차이가 연령대 사이의 차이보다 큽니다.</li>
<li><b>개인정보:</b> 작성자 이름·ID는 저장하지 않고 해시로만 중복을 셉니다. 원문은 리포트에 싣지 않습니다.</li>
</ul></section>
<footer>주제 사전: {html.escape(", ".join(X.TOPICS))}</footer>
</div>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out
