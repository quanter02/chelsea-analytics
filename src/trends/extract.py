"""Rule-based extraction: one Korean post → author age, author gender, topics, sentiment.

Only self-descriptions count ("저 28살인데", "29살 여자입니다"). Ages of other people ("남친이 32살") are
skipped, because the report is about the writer. Gender is explicit-only by default; `partner_cue=True` also
reads "남친/남편 → F", "여친/아내 → M", which is a guess and is labelled as one in the output.
"""
from __future__ import annotations

import re

import pandas as pd

OTHERS = r"(남친|여친|남자\s?친구|여자\s?친구|남편|아내|와이프|신랑|그\s?사람|그분|상대(방)?|오빠|언니|형|누나|친구|엄마|아빠|동생|선배|후배|썸남|썸녀|애인)"
SELF = r"(저는|전|저|제가|나는|난|내가|본인|저희)"
NATIVE_TENS = {"스물": 20, "서른": 30, "마흔": 40}
NATIVE_UNITS = {"한": 1, "하나": 1, "두": 2, "둘": 2, "세": 3, "셋": 3, "네": 4, "넷": 4, "다섯": 5,
                "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9}
DECADE_POS = {"초반": 2, "중반": 5, "후반": 8, None: 5}

RE_NUM = re.compile(r"(?<!\d)([1-5]\d)\s*(살|세)(?!대|\s*(차이|연상|연하|어린|많|위|아래|때))")
RE_NATIVE = re.compile(r"(스물|서른|마흔)\s*(하나|한|두|둘|세|셋|네|넷|다섯|여섯|일곱|여덟|아홉)?\s*(살|넘|초반|중반|후반|앞두|인데|이에요|입니다)")
RE_DECADE = re.compile(r"([2-4])0\s*대\s*(초반|중반|후반)?")
RE_BORN = re.compile(r"(?<!\d)(\d{2})\s*년생")
RE_TAG = re.compile(r"(?<![\w\d])([1-5]\d)\s*([FMfm여남])(?![\w])")

F_EXPLICIT = re.compile(r"(여자|여성|여자\s?직장인|여대생|女)\s*(인데|이고|입니다|이에요|예요|에요|임|이라|라서|로서)|"
                        r"(저는|전|난|나는)\s*여자|(\d{2}|20대|30대)\s*(살\s*)?여자\s*(인데|입니다|예요|에요|임)")
M_EXPLICIT = re.compile(r"(남자|남성|남자\s?직장인|남대생|男)\s*(인데|이고|입니다|이에요|예요|에요|임|이라|라서|로서)|"
                        r"(저는|전|난|나는)\s*남자|(\d{2}|20대|30대)\s*(살\s*)?남자\s*(인데|입니다|예요|에요|임)")
F_PARTNER = re.compile(r"(남친|남자\s?친구|남편|신랑|예비\s?신랑)")
M_PARTNER = re.compile(r"(여친|여자\s?친구|아내|와이프|예비\s?신부)")

# Topic lexicon (regex). Multi-label: a post can hit several topics.
TOPICS = {
    "연락·관심": r"연락|카톡|답장|읽씹|안\s?읽씹|전화\s?(안|를)|선톡",
    "만남·시작": r"소개팅|썸|고백|첫\s?만남|첫\s?데이트|데이트\s?신청|어플|소개\s?받|번호\s?(따|물어)",
    "갈등·이별": r"싸웠|싸움|헤어지|헤어졌|이별|권태기|환승|바람\s?(피|폈)|차였|잠수",
    "결혼": r"결혼|상견례|예식|혼수|프로포즈|청혼|웨딩|예비\s?(신랑|신부)",
    "경제력·돈": r"연봉|월급|돈|집값|전세|월세|대출|재산|모아\s?둔|경제력|더치|빚|자산|청약",
    "출산·육아": r"출산|임신|육아|아기|아이를|아이가|아이\s?낳|애\s?낳|딩크|난임",
    "외모": r"외모|얼굴|몸매|키가|키\s?\d|다이어트|잘생|예쁘|이쁘|성형|패션",
    "성격·가치관": r"성격|가치관|배려|다정|존중|종교|정치|대화가\s?(잘|안)|말투|자존감",
    "커리어": r"이직|취업|취준|회사|퇴사|승진|커리어|자격증|공무원",
    "소비": r"쇼핑|명품|소비|할부|지름|샀어|샀다|구매|할인",
}
TOPIC_RE = {k: re.compile(v) for k, v in TOPICS.items()}
POS = re.compile(r"좋아|좋은|좋았|행복|설레|고마|감사|만족|사랑|다행|최고|기뻐|편해|든든")
NEG = re.compile(r"힘들|불안|서운|짜증|화나|화가|우울|지치|지쳐|싫어|싫은|걱정|외롭|답답|속상|억울|후회|무서")

BANDS = [(20, 24, "20-24"), (25, 29, "25-29"), (30, 34, "30-34"), (35, 39, "35-39")]


def _about_other(text: str, start: int) -> bool:
    """An age right after '남친이', '오빠는' etc. describes someone else."""
    before = text[max(0, start - 8):start]
    return re.search(OTHERS + r"\s*(이|가|은|는|도|의)?\s*$", before) is not None


def author_age(text: str, year: int | None = None) -> tuple[int | None, str]:
    """(age, how) for the writer. how: number / native / decade / born / tag / '' (not found)."""
    for m in RE_TAG.finditer(text):           # "28F", "31남"
        return int(m.group(1)), "tag"
    for m in RE_NUM.finditer(text):
        if not _about_other(text, m.start()):
            return int(m.group(1)), "number"
    for m in RE_NATIVE.finditer(text):
        if not _about_other(text, m.start()):
            return NATIVE_TENS[m.group(1)] + NATIVE_UNITS.get(m.group(2), 0), "native"
    if year:
        for m in RE_BORN.finditer(text):
            if not _about_other(text, m.start()):
                yy = int(m.group(1))
                born = 2000 + yy if yy <= year % 100 else 1900 + yy
                return year - born, "born"
    for m in RE_DECADE.finditer(text):         # decades only with a self-reference nearby: "저는 20대 후반"
        near = text[max(0, m.start() - 6):m.end() + 8]
        after = text[m.end():m.end() + 12]
        if re.search(SELF, near) or re.match(r"\s*((여자|남자|직장인|여성|남성)\s*)?(인데|이고|입니다|이에요|예요|에요|임)", after):
            return int(m.group(1)) * 10 + DECADE_POS[m.group(2)], "decade"
    return None, ""


def author_gender(text: str, partner_cue: bool = False) -> tuple[str | None, str]:
    """('F'|'M'|None, 'explicit'|'tag'|'partner'|'')."""
    t = RE_TAG.search(text)
    if t:
        return ("F" if t.group(2) in "Ff여" else "M"), "tag"
    f, m = bool(F_EXPLICIT.search(text)), bool(M_EXPLICIT.search(text))
    if f != m:
        return ("F" if f else "M"), "explicit"
    if partner_cue and not (f or m):
        fp, mp = bool(F_PARTNER.search(text)), bool(M_PARTNER.search(text))
        if fp != mp:
            return ("F" if fp else "M"), "partner"
    return None, ""


def topics(text: str) -> list[str]:
    return [k for k, r in TOPIC_RE.items() if r.search(text)]


def sentiment(text: str) -> str:
    p, n = len(POS.findall(text)), len(NEG.findall(text))
    return "긍정" if p > n else "부정" if n > p else "중립"


def age_band(age) -> str | None:
    if age is None or pd.isna(age):
        return None
    for lo, hi, name in BANDS:
        if lo <= age <= hi:
            return name
    return None


def extract(df: pd.DataFrame, partner_cue: bool = False) -> pd.DataFrame:
    """df needs `text`; `created_at` (for '96년생') is optional. Adds age, age_how, age_band, gender,
    gender_how, topics, sentiment."""
    years = pd.to_datetime(df.get("created_at"), errors="coerce", utc=True) if "created_at" in df else None
    rows = []
    for i, text in enumerate(df["text"].fillna("").astype(str)):
        y = int(years.iloc[i].year) if years is not None and pd.notna(years.iloc[i]) else None
        age, how = author_age(text, y)
        g, ghow = author_gender(text, partner_cue)
        rows.append({"age": age, "age_how": how, "age_band": age_band(age), "gender": g, "gender_how": ghow,
                     "topics": topics(text), "sentiment": sentiment(text)})
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
