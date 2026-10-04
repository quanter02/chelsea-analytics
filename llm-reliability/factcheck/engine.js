/* 숫자 검증기 엔진 (브라우저·node 공용).
 *
 * 1) 문장 나누기 → 나라·연도는 앞 문장에서 이어받음
 * 2) 숫자 + 단위를 찾고, 그 앞의 가장 가까운 지표 단어에 붙임
 * 3) 정답 저장소와 대조해 판정
 *      일치 / 근사 일치(허용 오차 이내) / 기준 다름(잠정·확정 등) / 연도 혼동 / 나라 혼동 / 틀림 / 확인 불가(공표 전) / 대상 아님
 *    허용 오차(기본 수준 값 1%, 증감률 0.5%p) 안이면 오류로 표시하지 않는다
 * 4) 제어기: 대조할 가치가 없는 문장(나라·지표를 모름)은 조회하지 않고 '대상 아님'으로 기권
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.FactCheck = factory();
})(typeof self !== "undefined" ? self : this, function () {
  const IND = [ // [지표, 정규식] — 앞에 있을수록 우선 (긴 표현 먼저)
    ["fm_age", /평균\s*초혼\s*연령|초혼\s*연령|결혼\s*연령|혼인\s*연령|first[- ]marriage age/gi],
    ["cmr", /조혼인율|인구\s*천\s*명당\s*혼인|crude marriage rate/gi],
    ["tfr", /합계\s*출산율|total fertility rate|fertility rate|TFR|合計特殊出生率/gi],
    ["births", /출생아\s*수|출생아|출생\s*건수|신생아|태어난\s*아이|출생|births?/gi],
    ["deaths", /사망자\s*수|사망자|사망|deaths?/gi],
    ["divorces", /이혼\s*건수|이혼|divorces?/gi],
    ["marriages", /혼인\s*건수|결혼\s*건수|혼인|결혼|marriages?/gi],
  ];
  const NAME = { births: "출생아 수", deaths: "사망자 수", marriages: "혼인 건수", divorces: "이혼 건수", tfr: "합계출산율",
                 cmr: "조혼인율", fm_age_F: "여성 평균 초혼연령", fm_age_M: "남성 평균 초혼연령" };
  const UNIT = { births: "명", deaths: "명", marriages: "건", divorces: "건", tfr: "명", cmr: "건(천 명당)", fm_age_F: "세", fm_age_M: "세" };
  const KR = /한국|국내|우리나라|대한민국|통계청|Korea/i, JP = /일본|日本|Japan|후생노동성/i;

  // "24만 326", "24만", "25만 4천", "240,326", "0.72", "1,234.5"
  const NUM = /(\d+(?:\.\d+)?)\s*만\s*(?:(\d+)\s*천\s*(\d+)?|(\d,\d{3}|\d{1,4}))?|(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*천?/g;

  function parseNum(m) {
    if (m[1] !== undefined) {                                   // 만 단위
      const man = parseFloat(m[1]) * 10000;
      if (m[2] !== undefined) {                                 // 25만 4천 (300)
        const v = man + parseInt(m[2]) * 1000 + (m[3] ? parseInt(m[3]) : 0);
        return { v, step: m[3] ? 1 : 1000 };
      }
      if (m[4] !== undefined) return { v: man + parseInt(m[4].replace(",", "")), step: 1 };
      const dec = (m[1].split(".")[1] || "").length;
      return { v: man, step: 10000 / Math.pow(10, dec) };
    }
    const s = m[5].replace(/,/g, "");
    let v = parseFloat(s), dec = (s.split(".")[1] || "").length;
    let step = Math.pow(10, -dec);
    if (!dec && s.length >= 4) { const z = s.match(/0+$/); if (z) step = Math.pow(10, Math.min(z[0].length, s.length - 1)); }
    if (/천\s*$/.test(m[0])) { v *= 1000; step *= 1000; }
    return { v, step };
  }

  function sentences(para) {
    return para.split(/(?<=[.!?。])\s+|\n/).map((s) => s.trim()).filter(Boolean);
  }

  function extract(text) {
    const out = [];
    let country = null, year = null, paraPeriod = false;
    var PERIOD = new RegExp([
      "\\d{1,2}\\s*월(?!\\s*\\d{1,2}\\s*일)", "분기", "상반기", "하반기", "반기", "월간", "누적", "올해\\s*들어",
      "January|February|March|April|May|June|July|August|September|October|November|December", "first half|first six months|quarter|monthly",
      "\\d+\\s*月(?!\\s*\\d+\\s*日)", "上半期|下半期|四半期"].join("|"), "i");
    const BLOCK = /비중|비율|구성비|점유율|share|proportion|割合/i;
    const units = [];
    for (const para of text.replace(/\r/g, "").split(/\n\s*\n/)) {  // 빈 줄 = 문단 경계: 나라·연도를 새로 읽는다
      units.push({ reset: true, para });
      for (const s of sentences(para)) units.push({ sent: s });
    }
    for (const u of units) {
      if (u.reset) { country = null; year = null; paraPeriod = PERIOD.test(u.para); continue; }
      const sent = u.sent;
      if (/[?？]\s*$/.test(sent)) continue;                    // 질문은 주장이 아님
      const kr = KR.test(sent), jp = JP.test(sent);
      if (kr && !jp) country = "KR"; else if (jp && !kr) country = "JP";
      const ys = [...sent.matchAll(/(19[5-9]\d|20[0-4]\d)\s*년(?!\s*(이후|이래|대비|만에|보다|에\s*비해|부터|과\s*비교|래|\d{1,2}\s*월\s*\d{1,2}\s*일))/g)];
      const yearAt = (pos) => { const b = ys.filter((y) => y.index < pos).pop(); return b ? parseInt(b[1]) : (ys.length && ys[0].index > pos ? null : undefined); };
      const carried = year;
      if (ys.length) year = parseInt(ys[ys.length - 1][1]);
      const sentHangul = /[가-힣]/.test(sent);
      // 지표 위치
      const hits = [];
      for (const [ind, rx] of IND) for (const m of sent.matchAll(rx)) {
        if (!hits.some((h) => m.index >= h.at && m.index < h.end)) hits.push({ ind, at: m.index, end: m.index + m[0].length });
      }
      hits.sort((a, b) => a.at - b.at);
      for (const m of sent.matchAll(NUM)) {
        const at = m.index, raw = m[0].trim(), after = sent.slice(at + m[0].length, at + m[0].length + 8);
        const before = sent.slice(Math.max(0, at - 12), at);
        if (/^\s*년/.test(after) || /^\d{4}$/.test(raw) && /^\s*(년|\))/.test(after)) continue;   // 연도
        if (/^\s*(월|분기|일|위|개|곳|배|시|호|만에|년)/.test(after)) continue;
        if (/^(19|20)\d{2}$/.test(raw)) continue;                // 단위 없는 연도형 숫자
        const tail = sent.slice(at + m[0].length, at + m[0].length + 14);
        const prev = hits.filter((h) => h.end <= at).pop();
        if (!prev) continue;
        if (BLOCK.test(sent.slice(prev.end, at))) continue;          // '산모 비중 37%' 같은 비율은 지표 값이 아님
        const win = sent.slice(Math.max(0, prev.at - 25), Math.min(sent.length, at + m[0].length + 30));
        const ownYear = ys.length > 0;
        const subannual = (PERIOD.test(win) || (paraPeriod && !ownYear)) && !/연간|한\s*해\s*전체|annual/i.test(win);
        const { v, step } = parseNum(m);
        const isPct = /^\s*(%|퍼센트|％|percent|per\s*cent)/i.test(after);
        const isPp = /^\s*%p|^\s*%포인트|^\s*퍼센트포인트/.test(after);
        if (isPp) continue;
        const changeWord = /증가|감소|늘|줄|상승|하락|올랐|떨어|급증|급감|↑|↓|rose|fell|up|down|increase|decrease|増|減/i;
        if (!isPct && /^\s*(명|건|쌍)?\s*(이|가)?\s*(늘|증가|감소|줄|많|적|↑|↓|더)/.test(tail)) continue;  // 증감폭
        if (!isPct && /(보다|대비|전년|에\s*비해|\bby)\s*$/i.test(before)) continue;
        if (isPct && !changeWord.test(tail) && !/[+−-]\s*$/.test(before)) continue;            // 비율(구성비 등)은 증감률이 아님
        const neg = /감소|줄|하락|떨어|fell|declin|減|−|-\s*$/.test(sent.slice(at, at + m[0].length + 16)) || /[−-]\s*$/.test(before);
        let ind = prev.ind;
        if (ind === "fm_age") {
          const sx = [...sent.slice(0, at).matchAll(/(여성|아내|여자|女)|(남성|남편|남자|男)/g)].pop();
          ind = sx && sx[2] ? "fm_age_M" : "fm_age_F";
        }
        const th = /^\s*(명|건|쌍)?\s*(을|를|이|가)?\s*(넘|초과|웃돌|돌파|이상)/.test(tail) ? ">"
                 : /^\s*(명|건|쌍)?\s*(을|를|이|가)?\s*(아래|미만|이하|밑|밑돌)/.test(tail) ? "<" : null;
        const kind = isPct ? "yoy" : th ? "threshold" : "level";
        let yr = yearAt(at); if (yr === undefined) yr = carried; if (yr === null) yr = ys.length ? parseInt(ys[0][1]) : carried;
        // 단위 상식 검사 (단위가 안 맞는 숫자는 그 지표의 값이 아님)
        if (kind !== "yoy") {
          if ((ind === "tfr") && !(v > 0.3 && v < 3)) continue;
          if ((ind === "cmr") && !(v > 0.5 && v < 20)) continue;
          if (ind.startsWith("fm_age") && !(v >= 18 && v <= 45)) continue;
          if (["births", "deaths", "marriages", "divorces"].includes(ind) && v < 1000) continue;
        } else if (Math.abs(v) > 60) continue;
        let ctry = country, assumed = false;
        if (!ctry && sentHangul) { ctry = "KR"; assumed = true; }
        out.push({ sentence: sent, raw: raw + (isPct ? "%" : ""), country: ctry, assumed, subannual, year: yr, indicator: ind, kind, op: th,
                   value: kind === "yoy" && neg ? -v : v, step: kind === "yoy" ? Math.max(step, 0.1) : step });
      }
    }
    return out;
  }

  function index(ref) {
    const ix = {};
    for (const r of ref) (ix[`${r.country}|${r.indicator}`] ||= []).push(r);
    return ix;
  }

  const fmt = (v, ind, kind) => kind === "yoy" ? `${v > 0 ? "+" : ""}${v.toFixed(1)}%`
    : ind === "tfr" ? v.toFixed(v < 1 ? 3 : 2) + "명" : ind === "cmr" ? v.toFixed(1) + "건" : ind.startsWith("fm_age") ? v.toFixed(1) + "세"
    : Math.round(v).toLocaleString("ko-KR") + UNIT[ind];

  function valueAt(rows, year, basisPref) {
    const xs = rows.filter((r) => r.year === year);
    return xs.find((r) => r.basis.startsWith(basisPref)) || xs[0];
  }

  function check(claims, ref, opt = { tol: 0.01, tolPp: 0.5 }) {
    const ix = index(ref);
    return claims.map((c) => {
      const res = { ...c, name: NAME[c.indicator], verdict: "", detail: "", correct: null, source: "" };
      if (!c.country) return { ...res, verdict: "대상 아님", detail: "어느 나라 수치인지 문장에서 알 수 없어 조회하지 않음" };
      if (c.subannual) return { ...res, verdict: "대상 아님", detail: "월·분기·누적 값으로 보임. 저장소는 연간 값이라 판정하지 않음 (틀렸다고 단정하지 않기)" };
      const rows = ix[`${c.country}|${c.indicator}`];
      if (!rows) return { ...res, verdict: "대상 아님", detail: "정답 저장소에 없는 지표" };
      const years = [...new Set(rows.map((r) => r.year))].sort((a, b) => a - b);
      const latest = years[years.length - 1];
      const y = c.year ?? latest;
      const tol = (truth) => c.step / 2 + Math.abs(truth) * 1e-9;
      const levelOf = (yr) => rows.filter((r) => r.year === yr);
      const yoyOf = (yr, basis) => {
        const a = valueAt(rows, yr, basis), b = valueAt(rows, yr - 1, "확정");
        return a && b ? { v: (a.value / b.value - 1) * 100, r: a } : null;
      };
      const candidates = (yr) => c.kind !== "yoy" ? levelOf(yr).map((r) => ({ v: r.value, r }))
        : [...new Set(levelOf(yr).map((r) => r.basis))].map((bs) => yoyOf(yr, bs)).filter(Boolean);
      const near = (xs) => xs.find((x) => Math.abs(x.v - c.value) <= tol(x.v));
      if (!c.year) res.detail = `연도가 없어 최신(${latest}년)으로 봄. `;
      if (c.assumed) res.detail = "나라 언급이 없어 한국으로 봄. " + res.detail;
      const here = candidates(y);
      if (!here.length) {
        const other = near(candidates(latest));
        const lv = candidates(latest)[0];
        return { ...res, verdict: "확인 불가",
                 detail: res.detail + (other ? `${y}년 값은 아직 공표 전. 이 숫자는 ${latest}년 값(${fmt(other.v, c.indicator, c.kind)})과 같아, 최신 값을 ${y}년으로 쓴 것일 수 있음`
                   : `${y}년 값은 아직 공표 전. 최신 ${latest}년 = ${lv ? fmt(lv.v, c.indicator, c.kind) : "—"}`),
                 correct: lv ? lv.v : null, source: lv ? lv.r.source : "" };
      }
      if (c.kind === "threshold") {
        const m0 = here.find((x) => x.r.basis.startsWith("확정")) || here[0];
        const ok = c.op === ">" ? m0.v > c.value : m0.v < c.value;
        return { ...res, verdict: ok ? "일치" : "틀림", correct: m0.v, source: m0.r.source,
                 detail: res.detail + `기준선 주장 (${c.op === ">" ? "넘음" : "아래"}): 공식 ${y}년 ${fmt(m0.v, c.indicator, "level")} → ${ok ? "맞음" : "아님"}` };
      }
      const hit = near(here);
      const main = here.find((x) => x.r.basis.startsWith("확정")) || here[0];
      // 허용 오차: 수준 값은 상대 오차(기본 1%), 증감률은 %p (기본 0.5%p)
      const gap = c.kind === "yoy" ? Math.abs(c.value - main.v) : Math.abs(c.value / main.v - 1);
      const within = c.kind === "yoy" ? gap <= opt.tolPp : gap <= opt.tol;
      const gapTxt = c.kind === "yoy" ? `${gap.toFixed(1)}%p` : `${(gap * 100).toFixed(gap < 0.01 ? 2 : 1)}%`;
      const tolTxt = c.kind === "yoy" ? `${opt.tolPp}%p` : `${(opt.tol * 100).toFixed(1).replace(/\.0$/, "")}%`;
      if (hit) {
        const hasFinal = here.some((x) => x.r.basis.startsWith("확정"));
        if (hit.r.basis.startsWith("확정") || !hasFinal)
          return { ...res, verdict: "일치", correct: main.v, source: hit.r.source,
                   detail: res.detail + `${y}년 ${fmt(hit.v, c.indicator, c.kind)} (${hit.r.basis}${hasFinal ? "" : ", 확정치 나오면 다시 확인"})` };
        return { ...res, verdict: within ? "일치" : "기준 다름", correct: main.v, source: hit.r.source,
                 detail: res.detail + `${hit.r.basis} 값과 같음. 확정치 ${fmt(main.v, c.indicator, c.kind)} 대비 ${gapTxt} 차이` + (within ? ` (허용 ${tolTxt} 이내)` : "") };
      }
      if (within) {                                              // 반올림·대략적 표현: 오류로 보지 않음
        const sameOther = [-1, 1].map((dy) => ({ dy, o: near(candidates(y + dy)) })).find((x) => x.o);
        return { ...res, verdict: "근사 일치", correct: main.v, source: main.r.source,
                 detail: res.detail + `공식 ${y}년 ${fmt(main.v, c.indicator, c.kind)} 대비 ${gapTxt} 차이 (허용 ${tolTxt} 이내)` +
                   (sameOther ? `. 참고: ${y + sameOther.dy}년 값과 정확히 같음` : "") };
      }
      for (const dy of [-1, 1, -2]) {                            // 다른 해의 값을 말했나
        const o = near(candidates(y + dy));
        if (o) return { ...res, verdict: "연도 혼동", correct: main.v, source: main.r.source,
                        detail: res.detail + `${y + dy}년 값(${fmt(o.v, c.indicator, c.kind)})과 같음. ${y}년은 ${fmt(main.v, c.indicator, c.kind)} (${gapTxt} 차이)` };
      }
      const other = ix[`${c.country === "KR" ? "JP" : "KR"}|${c.indicator}`];
      if (other) {
        const o = other.filter((r) => r.year === y).find((r) => c.kind === "level" && Math.abs(r.value - c.value) <= tol(r.value));
        if (o) return { ...res, verdict: "나라 혼동", correct: main.v, source: main.r.source,
                        detail: res.detail + `${o.country === "KR" ? "한국" : "일본"} 값과 같음. ${c.country === "KR" ? "한국" : "일본"} ${y}년은 ${fmt(main.v, c.indicator, c.kind)}` };
      }
      return { ...res, verdict: "틀림", correct: main.v, source: main.r.source,
               detail: res.detail + `공식 ${y}년 ${fmt(main.v, c.indicator, c.kind)} (${main.r.basis}), 차이 ${gapTxt} (허용 ${tolTxt} 초과)` };
    });
  }

  const DEFAULT = { tol: 0.01, tolPp: 0.5 };
  function run(text, ref, opt) {
    const r = check(extract(text), ref, { ...DEFAULT, ...(opt || {}) });
    const count = (v) => r.filter((x) => x.verdict === v).length;
    return { claims: r, summary: { total: r.length, ok: count("일치"), near: count("근사 일치"), basis: count("기준 다름"), year: count("연도 혼동"),
             country: count("나라 혼동"), wrong: count("틀림"), unknown: count("확인 불가"), skipped: count("대상 아님") }, fmt };
  }

  return { extract, check, run, fmt, NAME };
});
