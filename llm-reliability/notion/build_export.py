"""노션 가져오기 패키지 생성.

    python notion/build_export.py      # → notion/export/한계효용 신뢰도 엔진.zip

노션에서 '설정 → 가져오기 → Markdown & CSV'로 zip 파일을 올리면
페이지 계층과 데이터베이스(CSV)가 그대로 만들어집니다.
노션 커넥터가 연결된 세션에서는 같은 내용(notion/export/ 아래 파일)을 그대로 노션에 쓰면 됩니다.
"""
from __future__ import annotations

import csv
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notion" / "export"
TITLE = "한계효용 신뢰도 엔진"
WEB = "https://claude.ai/artifact/JvfGqAeSGNjLcNFP1eiCg8"
REPO = "https://github.com/quanter02/chelsea-analytics/tree/claude/adoring-dijkstra-jshoun/llm-reliability"

# ── 실험 기록 ─────────────────────────────────────────────────────────────
EXPERIMENTS = [
    dict(이름="성격 점수: 16유형 vs 연속 점수", 단계="효용 추정", 데이터="가상", 날짜="2026-10-03",
         질문="성격 정보를 유형으로 나눌까, 연속 점수로 쓸까",
         핵심결과="연속 점수 통합 모형의 95% 신뢰구간이 모든 구간에서 약 2.7배 좁음. 텍스트 추출 점수만 쓰면 효과가 0.12 → 0.044로 축소, 설문 옆 보조로 넣으면 0.105",
         한계="통합 모형의 함수 형태가 진짜와 일치하도록 설정(유리한 조건)", 코드="early_sims/01_personality_pooling.py", 그림="초기 시뮬레이션/utility_sim.png", 상태="완료"),
    dict(이름="망각 계수와 측정 버전 교체", 단계="지속 업데이트", 데이터="가상", 날짜="2026-10-03",
         질문="데이터가 매주 쌓일 때 무엇이 달라지나",
         핵심결과="선호 변화 후 누적 방식은 26주 뒤에도 17% 어긋남, 망각 계수는 약 11주 만에 10% 이내. 텍스트 추출 버전 교체 시 가짜 변화 +6.6%(망각)·+3.4%(누적), 버전 보정 시 +0.1%",
         한계="급변 1회만 시험", 코드="early_sims/02_forgetting_and_version_change.py", 그림="초기 시뮬레이션/utility_time_sim.png", 상태="완료"),
    dict(이름="업데이트 빈도 vs 정보량 vs 변화 감지", 단계="지속 업데이트", 데이터="가상", 날짜="2026-10-03",
         질문="점진적 업데이트를 빨리 돌리면 한계가 줄어드나",
         핵심결과="주 1회와 하루 1회 모두 적응 10.9주로 같음. 데이터 2배는 평소 오차만 개선. 변화 감지형 적응 망각은 적응 0.9주, 헛경보 실행당 0.07회",
         한계="감지 기준값은 임의 설정", 코드="early_sims/03_update_speed_vs_information.py", 그림="초기 시뮬레이션/utility_speed_sim.png", 상태="완료"),
    dict(이름="LLM 원인 분류 손익분기점", 단계="LLM 판단", 데이터="가상", 날짜="2026-10-03",
         질문="LLM이 변화 원인을 몇 % 맞혀야 이득이고, 언제 호출해야 하나",
         핵심결과="손익분기 정확도: 경보 시 약 59%, 애매할 때만 약 66%, 사건 기록 시 약 43%, 매주 40% 이하. 사건 기록 시 호출(16.9회, 오차 1.96%)이 매주 호출(104회, 1.90%)과 거의 같음",
         한계="LLM 오답의 절반은 '변화 없음'이라는 무해한 답이라고 가정", 코드="early_sims/04_llm_cause_breakeven.py", 그림="초기 시뮬레이션/utility_llm_sim.png", 상태="완료"),
    dict(이름="오답 구성 DB와 라우터", 단계="신뢰도 프로필", 데이터="가상", 날짜="2026-10-03",
         질문="확신 오답과 기권을 나눠 기록하면 무엇을 알 수 있나",
         핵심결과="기권형 모델 기대 손실 $6.00 < 과신형 $9.24 (능력치는 과신형이 더 높음). 최적 라우팅 정책 판단 1건당 $2.32 vs 대형 모델 단독 $4.82 vs 전부 사람 $5.00",
         한계="모든 모델이 모든 문항에 답했다고 가정", 코드="run_demo.py", 그림="결과 리포트/router.png", 상태="완료"),
    dict(이름="온라인 갱신 (조용한 성능 저하·신규 모델)", 단계="지속 업데이트", 데이터="가상", 날짜="2026-10-03",
         질문="새 데이터가 매주 들어올 때 정책을 어떻게 갱신하나",
         핵심결과="판단 1건당: 한 번만 선택 $4.24, 매주 재선택 $2.79, 망각+변화 감지 $2.78. 성능 저하 구간은 $4.17 → $3.98. 경보 첫 시점 중앙값 128일(저하 18일 후)",
         한계="정답 확정 지연(평균 14일)이 감지 속도의 하한", 코드="online_demo.py", 그림="결과 리포트/online.png", 상태="완료"),
    dict(이름="한계효용 기반 추출", 단계="추출", 데이터="가상", 날짜="2026-10-03",
         질문="유튜브 정리 영상을 어디까지 읽고 무엇을 검증할까",
         핵심결과="영상의 31%만 읽고 60일 총비용 $41 (전부 추출 $79). 같은 사건은 2~3번째 영상부터 가치 급감. 채널은 5번째부터 추가 가치 거의 0",
         한계="채널 신뢰도와 복제 관계를 안다고 가정", 코드="extraction_demo.py", 그림="결과 리포트/extraction.png", 상태="완료"),
    dict(이름="실제 데이터 검증 (실제 경기 1,517개)", 단계="실제 검증", 데이터="실제", 날짜="2026-10-03",
         질문="실제 데이터에서 이 프레임워크로 모델을 얼마나 정확히 평가할 수 있나",
         핵심결과="정답률·기대 손실 95% 구간: 100건 ±0.09, 500건 ±0.045, 1,517건 ±0.027. 손실 차이 0.03 이상은 250~500건이면 구분. 적은 표본에서 ECE 2.5배 부풀려짐. 라우터는 실제 데이터에서 과적합(단순 모델보다 낫지 않음). 헛경보 0건",
         한계="LLM 대신 통계 예측 모델로 시험", 코드="real_data_demo.py", 그림="결과 리포트/realdata.png", 상태="완료"),
    dict(이름="한·일 결혼 지표 (2026 최신 보도)", 단계="실제 적용", 데이터="실제", 날짜="2026-10-04",
         질문="한국·일본 여성의 연령대별 결혼 지표 차이와 변화 방향",
         핵심결과="2025년 혼인: 한국 24만 326건(+8.1%, 2019년 수준 회복) vs 일본 48만 9,119건(+0.8%, 2019년보다 18% 적음). 합계출산율 0.80(↑) vs 1.14(↓). 한국 여성 30~34세 혼인 +13.2%. '반등은 일시적이지 않다' 믿음 0.50 → 0.88",
         한계="일본 원문 차단으로 보도 일치 수치만 사용. 연령대별 미혼율 비교 미완료", 코드="data_claims/kr_jp_marriage_2026.csv", 그림="", 상태="진행 중"),
]

SOURCES = [
    ("StatsBomb 공개 데이터", "경기 결과", "접근 가능", "실제 데이터 검증에 사용", "https://github.com/statsbomb/open-data"),
    ("Our World in Data datasets (GitHub)", "공식 통계 모음", "접근 가능", "2017년까지, 연령대 구분 없음", "https://github.com/owid/owid-datasets"),
    ("KOSIS 국가통계포털", "한국 공식 통계", "연결됨 · 인증키 필요", "무료 인증키 발급 필요", "https://kosis.kr"),
    ("국가데이터처 보도자료", "한국 공식 발표", "검색으로 확인", "", "https://mods.go.kr"),
    ("e-Stat 政府統計の総合窓口", "일본 공식 통계", "차단", "네트워크 허용 + 앱 ID 필요", "https://www.e-stat.go.jp"),
    ("厚生労働省 人口動態統計", "일본 공식 발표", "차단", "네트워크 허용 필요", "https://www.mhlw.go.jp"),
    ("国立社会保障・人口問題研究所 출생동향기본조사", "일본 공식 조사", "차단", "네트워크 허용 필요", "https://www.ipss.go.jp"),
    ("YouTube RSS / Data API", "영상", "차단", "네트워크 허용 + API 키 필요", "https://www.youtube.com"),
    ("Anthropic API", "LLM 판단", "키 없음", "API 키를 환경 변수로 넣으면 사용 가능", "https://docs.anthropic.com"),
    ("Notion", "정리", "커넥터 미연결 · API 차단", "claude.ai 커넥터 연결 후 새 세션", "https://claude.ai/customize/connectors"),
]

TODO = [
    ("노션 커넥터 연결 후 새 세션에서 직접 동기화", "claude.ai 커넥터 설정에서 Notion 연결", "높음", "대기"),
    ("일본 연령대별 혼인·미혼율 수집", "e-Stat 도메인 허용 + 앱 ID", "높음", "막힘"),
    ("한국 연령대별 미혼율 시계열 수집", "KOSIS 인증키", "높음", "막힘"),
    ("유튜브 정리 채널 실제 연결", "도메인 허용 + YouTube API 키 + 채널 목록", "중간", "막힘"),
    ("실제 LLM 판단으로 신뢰도 프로필 측정", "Anthropic API 키", "중간", "막힘"),
    ("주장 테이블에 검증 경로 태그 추가 (결과/원문/재현/확인 불가)", "없음", "중간", "할 일"),
    ("라우터 후보 수를 표본 크기에 맞게 제한", "없음 (실제 데이터 검증의 교훈)", "중간", "할 일"),
    ("서서히 변하는 변화에서 변화 감지 성능 시험", "없음", "낮음", "할 일"),
    ("2027년 3월 한국 2026년 혼인 통계로 믿음 갱신", "발표 대기", "낮음", "예정"),
]

PRINCIPLES = """# 설계 원칙

이 대화에서 시뮬레이션과 실제 데이터로 확인한 원칙입니다.

## 정보를 고르는 기준
- **의미 있는 정보 = 내 판단을 바꿀 수 있는 정보.** 판단을 바꾸지 못하면 아무리 정확해도 가치가 0입니다.
- 다음 한 단위(영상 1개, 검증 1건, 채널 1개)의 **한계효용이 비용보다 클 때만** 처리합니다. 가치 = 중요도 × p(1−p)의 기대 감소.
- 같은 사건에 대한 정보는 쌓일수록 가치가 줄고, 다른 출처를 베낀 정보는 가치가 0입니다.

## 검증 경로
| 정보 종류 | 검증 방법 |
|---|---|
| 나중에 결과가 나오는 주장 | 기록해 두고 결과와 대조 |
| 원문이 있는 사실 | 공식 발표·문서와 대조 |
| 재현할 수 있는 방법 | 직접 해보기 |
| 해석·의견 | 직접 검증 불가 → 가중치를 낮추고 독립 출처 여럿이 일치할 때만 참고 |

코드와 자동화는 검증을 대신하지 않고, 검증을 많이·꾸준히·기록이 남게 해줍니다. 분석을 거친 결과도 실제 결과와 대조해야 합니다.

## 측정
- 성격 같은 특성은 유형으로 나누지 말고 연속 점수로, 그룹을 쪼개지 말고 통합 모형으로 추정합니다.
- 텍스트에서 추출한 점수는 정확도가 낮으므로 설문 같은 기준 데이터 옆의 보조 변수로만 씁니다.
- 측정 도구(추출 모델, LLM, 설문 문항)가 바뀌면 같은 기준 자료로 새·옛 버전을 비교해 보정합니다.

## 업데이트
- 계산을 자주 돌리는 것은 한계를 줄이지 못합니다. 정보량을 늘리거나, 변화를 감지해 과거를 버려야 합니다.
- 망각 계수는 실제 변화도 가짜 변화도 빨리 따라갑니다. 버전 보정과 세트로 씁니다.
- 정답이 늦게 확정되면 그 지연이 변화 감지 속도의 하한입니다.

## 모델 판단
- 정확도보다 오답 구성(확신 오답 vs 기권)을 봅니다. 모를 때 기권하는 모델이 손실이 작을 수 있습니다.
- LLM은 숫자에서 변화를 감지하는 일보다 변화의 원인을 판단하는 일에 씁니다. 경보가 울릴 때와 외부 사건이 기록될 때 호출합니다.
- 실제 데이터에서는 정책 후보가 많을수록 과적합됩니다. 표본이 적으면 단순한 정책부터 씁니다.
- 보정 오차(ECE)는 표본이 적으면 부풀려집니다. 항상 표본 수와 함께 봅니다.
- 모델 비교는 판단 수백 건부터 의미가 있고, 차이가 작으면 수천 건이 필요합니다.
"""


def write_csv(path: Path, header: list[str], rows: list) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:      # BOM: 노션·엑셀에서 한글이 깨지지 않게
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def report_page(src: Path, title: str, note: str, images_dir: Path) -> str:
    """output/*/report.md 를 노션 페이지로: 제목 교체, 이미지는 같은 폴더로 복사."""
    text = src.read_text(encoding="utf-8").split("\n", 1)[1]
    for img in src.parent.glob("*.png"):
        shutil.copy(img, images_dir / img.name)
    return f"# {title}\n\n> {note}\n\n{text}"


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    base = OUT / TITLE
    res = base / "결과 리포트"
    early = base / "초기 시뮬레이션"
    for d in (res, early):
        d.mkdir(parents=True)

    (OUT / f"{TITLE}.md").write_text(f"""# {TITLE}

"한계효용으로 평균효용을 구할 수 있을까"에서 출발해, 넘치는 정보 중 무엇을 읽고 어떤 AI 판단을 믿을지 정하는 시스템까지 설계했습니다.

- 웹 화면: {WEB}
- 코드: {REPO}
- 마지막 갱신: 2026-10-04

## 전체 흐름
1. **수집**: 정리 잘하는 유튜버의 새 영상, 배포 로그, 뉴스, 공식 통계
2. **한계효용 추출**: 다음 1개를 읽는 가치가 비용보다 클 때만 읽음
3. **판단 DB**: 모든 예측을 기록하고 정답은 나중에 붙임 (확신 오답·기권은 뷰)
4. **신뢰도 프로필**: 정답·확신 오답·기권, 보정 오차, 기대 손실
5. **라우터**: 어떤 모델을 언제 부르고 언제 사람에게 넘길지
6. **온라인 갱신**: 망각 계수와 변화 감지로 매주 다시 학습

## 이 페이지 안에 있는 것
- 설계 원칙
- 실험 기록 (데이터베이스)
- 주장 (데이터베이스): 기사·영상에서 뽑은 주장과 검증 상태
- 데이터 출처 (데이터베이스): 접근 가능 여부와 필요한 것
- 다음 할 일 (데이터베이스)
- 결과 리포트, 초기 시뮬레이션, DB 스키마와 코드 구조
""", encoding="utf-8")

    (base / "설계 원칙.md").write_text(PRINCIPLES, encoding="utf-8")

    keys = ["이름", "단계", "데이터", "날짜", "질문", "핵심결과", "한계", "코드", "그림", "상태"]
    write_csv(base / "실험 기록.csv", ["이름", "단계", "데이터", "날짜", "질문", "핵심 결과", "한계", "코드", "그림", "상태"],
              [[e[k] for k in keys] for e in EXPERIMENTS])

    with (ROOT / "data_claims" / "kr_jp_marriage_2026.csv").open(encoding="utf-8") as f:
        claims = list(csv.DictReader(f))
    write_csv(base / "주장.csv", ["주장", "국가", "지표", "기간", "값", "단위", "출처", "출처 종류", "검증 상태", "결정", "메모", "링크", "주제"],
              [[f"{c['country']} {c['indicator']} ({c['period']})", c["country"], c["indicator"], c["period"], c["value"], c["unit"],
                c["source"], c["source_type"], c["verification"], c["decision"], c["note"], c["source_url"], "한·일 결혼"] for c in claims])
    write_csv(base / "데이터 출처.csv", ["출처", "종류", "접근 상태", "필요한 것 · 메모", "링크"], SOURCES)
    write_csv(base / "다음 할 일.csv", ["할 일", "필요한 것", "우선순위", "상태"], TODO)

    pages = [
        ("output/report.md", "모델 신뢰도와 라우터 (가상)", "가상 판단 3,000건, 모델 4종. 실행: python run_demo.py"),
        ("output/online/report.md", "온라인 갱신 (가상)", "240일 시나리오, 시드 8개. 실행: python online_demo.py"),
        ("output/extraction/report.md", "한계효용 기반 추출 (가상)", "60일, 사건 300개, 채널 12개. 실행: python extraction_demo.py"),
        ("output/realdata/report.md", "실제 데이터 검증", "StatsBomb 실제 경기 1,517개. 실행: python real_data_demo.py"),
    ]
    for src, title, note in pages:
        (res / f"{title}.md").write_text(report_page(ROOT / src, title, note, res), encoding="utf-8")

    for img in (ROOT / "early_sims" / "figures").glob("*.png"):
        shutil.copy(img, early / img.name)
    early_md = (ROOT / "early_sims" / "README.md").read_text(encoding="utf-8")
    early_md += "\n" + "\n".join(f"![{p.stem}]({p.name})" for p in sorted(early.glob("*.png"))) + "\n"
    (base / "초기 시뮬레이션.md").write_text(early_md, encoding="utf-8")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    start = readme.index("## 구조"); end = readme.index("## 계산하는 지표")
    (base / "DB 스키마와 코드 구조.md").write_text("# DB 스키마와 코드 구조\n\n" + readme[start:end], encoding="utf-8")

    zpath = OUT / f"{TITLE}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file() and p != zpath:
                z.write(p, p.relative_to(OUT))
    print(f"{zpath}  ({zpath.stat().st_size // 1024} KB)")
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.suffix != ".zip":
            print("  ", p.relative_to(OUT))


if __name__ == "__main__":
    main()
