"""새 규칙 후보를 채택된 최종 규칙 위에 시험: 검증·시험·최신(튜닝에 안 쓴 마지막 해) 세 구간에서 비교.

    python rule_check.py    # → output/tuning/rule_check.md

판정: 검증 구간에서 '확실한 개선'(95% 하한 > 0)일 때만 채택. 시험·최신 구간 결과는 가설로 기록만 한다.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from llmrel import forecast_tune as FT, regional_tune as RT, tuning
from llmrel.report import md_table

ROOT = Path(__file__).parent
OUT = ROOT / "output" / "tuning"
SPLITS = ("val", "test", "live")


def check(mod, spec) -> tuple[pd.DataFrame, dict]:
    spec.start = mod.FINAL
    r = tuning.hill_climb(spec, mode="certain", max_steps=25)
    base = {s: spec.evaluate(mod.FINAL, s) for s in SPLITS}
    rows = [dict(규칙="현재 (최종 규칙)", **{s: f"{base[s].mean():.4f}" for s in SPLITS})]
    for p, vals in (("cap", mod.GRID["cap"][1:]), ("robust", [1])):
        for v in vals:
            c = {**mod.FINAL, p: v}; row = dict(규칙=f"{p}={v}")
            for s in SPLITS:
                loss = spec.evaluate(c, s); m, lo, hi = tuning.paired_ci(base[s], loss)
                mark = "" if abs(m) < 5e-5 else " ✔" if -hi > 0 else (" ✘" if -lo < 0 else "")
                row[s] = f"{loss.mean():.4f} ({-m:+.4f}, {-hi:+.4f}~{-lo:+.4f}){mark}"
            rows.append(row)
    n = {s: len(base[s]) for s in SPLITS}
    return pd.DataFrame(rows).rename(columns={"val": f"검증 (n={n['val']})", "test": f"시험 (n={n['test']})", "live": f"최신 (n={n['live']})"}), r


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(ROOT / "data_official" / "kr_jp_marriage_official.csv"); d["age_band"] = d.age_band.astype(str)
    sgg, sido = RT.load()
    L = ["# 다른 모델들도 같은 방식으로 점검: 한 해 급변 제한(cap)·중앙값 추세(robust)", "",
         "EPL에서 시험한 '한 경기 영향 제한'과 같은 생각을 연간 예측 두 모델에 넣었습니다.", "",
         "- **cap**: 예측에 쓰는 한 해 추세(로그 변화)의 상한. 0.07 ≈ 연 7%.",
         "- **robust**: 추세를 최근 k년 변화의 평균 대신 중앙값으로.",
         "- 출발점은 지난번 채택된 최종 규칙. 판정은 EPL과 같음: **검증 구간에서 확실히 좋아질 때만 채택**, 시험·최신 구간은 기록만.",
         "- 칸: 평균 손실 (개선, 개선 95% 구간). ✔ 확실한 개선, ✘ 확실한 악화. 손실은 |로그 오차| (0.01 ≈ 1%).", ""]
    for mod, spec, live in ((FT, FT.make_spec(d), "원점 2022~2024 → 2025 (한국 10계열, 일본 2025 미공표)"),
                            (RT, RT.make_spec(sgg, sido), "원점 2024 → 2025 (1년 뒤, 243개 시군구)")):
        tab, r = check(mod, spec)
        L += [f"## {spec.name}", "", f"최신 구간: {live}", "", md_table(tab, {}), "",
              f"- 실험대 결과: 채택 {int(r['history'].accepted.sum()) - 1}개 → 규칙 그대로 ({', '.join(f'{k}={v}' for k, v in r['final'].items())})", ""]
    L += ["## 해석", "",
          "- **두 모델 모두 채택 0개.** 검증 구간(혼인율 2008~2017, 시군구 2013~2018)은 변화가 완만해서 상한이 거의 걸리지 않습니다. 그래서 검증은 cap에 대해 '좋다/나쁘다'를 말해주지 못합니다.",
          "- EPL과 다른 점: EPL cap은 지난 시즌 검증에서 **나빠졌고**, 여기서는 검증이 **조용하고**(차이 없음) 시험·최신이 **좋아집니다**.",
          "- 혼인율: 한국 2020~2022 급감 뒤 2024~2025 반등(25~39세 연 +9~19%)을 추세 모델이 '계속 감소'로 내다봐서 최신 오차가 큽니다 (0.229 ≈ 23%). cap 0.07은 시험·최신 모두 확실히 개선, 더 낮은 cap은 최신만 크게 개선.",
          "- 시군구: 코로나 급감·반등 시기(시험)에 cap 0.07이 확실히 개선(손실 −7.5%). 2025년 1년 뒤 예측(최신)은 방향은 같지만 우연 범위.",
          "- robust(중앙값 추세): 혼인율 검증에서 확실히 나빠짐, 시군구는 k=2라 평균과 같아 효과 없음 → 폐기.",
          "- 결정: **채택하지 않음.** 시험 구간은 이미 열어 본 데이터라 그걸 보고 규칙을 고르면 안 됩니다. cap 0.07을 **사전 등록 가설**로 남기고, 2026년 통계(2027년 발표)로 판정합니다: 그 해 cap 0.07이 확실히 좋으면 채택.", ""]
    (OUT / "rule_check.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
