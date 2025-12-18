#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KSS .lab 일괄 정리 스크립트
- 대상: /home/tts/rvc_project/FastSpeech2/KSS/1,2,3,4/**.lab
- 동작: 영어/숫자/자모/이상한 기호 제거 후 '첫 번째 한글 문장'만 남김
- 백업: 원본을 mirror 구조로 backup 디렉터리에 저장
- 로그: 변경/스킵/에러를 콘솔 요약 출력

필요시 옵션만 바꿔서 사용.
"""

import re
import sys
import shutil
import unicodedata
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple


# ===== 설정 =====
ROOT = Path("/home/tts/rvc_project/FastSpeech2/KSS")   # KSS 루트
SUBDIRS = ["1", "2", "3", "4"]                         # 처리할 하위 폴더
KEEP_ONLY_FIRST_SENTENCE = True                        # True면 첫 한글 문장만, False면 한글만 남긴 전체 문장
MAKE_BACKUP = True                                     # 원본 백업 여부
BACKUP_DIR = ROOT.parent / f"KSS_lab_backup_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
ENCODING = "utf-8"

# 한글 음절 범위, 자모 범위
HANGUL_SYLLABLE = (0xAC00, 0xD7A3)
JAMO_BLOCKS = [(0x1100, 0x11FF), (0x3130, 0x318F)]

# 정규식 미리 컴파일
RE_ENGLISH = re.compile(r"[A-Za-z]+")                     # 영어
RE_NUMBER = re.compile(r"\d+([.,]\d+)?")                  # 정수/소수
RE_JAMO = re.compile(r"[\u1100-\u11FF\u3130-\u318F]+")    # 자모
RE_WEIRD_QUOTES = [
    (re.compile(r"[“”]"), '"'),
    (re.compile(r"[‘’]"), "'"),
    (re.compile(r"…"), "..."),
]
# 한글/공백/일부 문장부호만 남기기(쉼표는 문장 내부 용도로 유지)
RE_KEEP = re.compile(r"[^가-힣\s,\.!\?]")

# 첫 번째 한글 문장 캡처: 한글/공백/쉼표가 나오다 마침표/느낌표/물음표로 끝나는 패턴
RE_FIRST_SENT = re.compile(r"([가-힣\s,]+[\.!\?])")

def is_hangul_syllable(ch: str) -> bool:
    cp = ord(ch)
    return HANGUL_SYLLABLE[0] <= cp <= HANGUL_SYLLABLE[1]

def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)

def clean_line(text: str) -> str:
    t = nfc(text)
    # 특수 따옴표, 말줄임 교정
    for pat, repl in RE_WEIRD_QUOTES:
        t = pat.sub(repl, t)
    # 영어/숫자/자모 제거
    t = RE_ENGLISH.sub(" ", t)
    t = RE_NUMBER.sub(" ", t)
    t = RE_JAMO.sub(" ", t)
    # 허용 문자 외 제거
    t = RE_KEEP.sub(" ", t)
    # 공백 정리
    t = re.sub(r"\s+", " ", t).strip()
    return t

def extract_first_korean_sentence(t: str) -> str:
    m = RE_FIRST_SENT.search(t)
    if m:
        return m.group(1).strip()
    # 첫 문장을 못 찾으면 '한글이 많은 연속 구간' 히ュー리스틱
    longest = ""
    cur = []
    for ch in t:
        if is_hangul_syllable(ch) or ch in " ,":
            cur.append(ch)
        else:
            if len(cur) > len(longest):
                longest = "".join(cur).strip()
            cur = []
    if len(cur) > len(longest):
        longest = "".join(cur).strip()
    return longest

def process_lab_file(lab_path: Path, backup_root: Optional[Path]) -> Tuple[bool, str]:
    try:
        original = lab_path.read_text(encoding=ENCODING, errors="replace")
    except Exception as e:
        return False, f"READ_FAIL: {lab_path} ({e})"

    cleaned_base = clean_line(original)
    if not cleaned_base:
        return False, f"EMPTY_AFTER_CLEAN: {lab_path}"

    if KEEP_ONLY_FIRST_SENTENCE:
        cleaned = extract_first_korean_sentence(cleaned_base)
        if not cleaned:
            # 비상 플랜: 그래도 아무것도 못 뽑으면 최대 80자만
            cleaned = cleaned_base[:80].strip()
    else:
        cleaned = cleaned_base

    # 내용이 실제로 변했는지 비교(양끝 공백 무시)
    if nfc(original).strip() == cleaned.strip():
        # 변화 없는 파일은 스킵
        return False, f"UNCHANGED: {lab_path}"

    # 백업
    if MAKE_BACKUP and backup_root is not None:
        rel = lab_path.relative_to(ROOT)
        dst = backup_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(lab_path, dst)

    # 저장
    try:
        lab_path.write_text(cleaned + "\n", encoding=ENCODING)
    except Exception as e:
        return False, f"WRITE_FAIL: {lab_path} ({e})"

    return True, f"OK: {lab_path}"

def main():
    # 대상 .lab 수집
    targets = []
    for sd in SUBDIRS:
        d = ROOT / sd
        if d.is_dir():
            targets.extend(d.rglob("*.lab"))
    targets = sorted(set(targets))

    if not targets:
        print("NO_LAB_FILES_FOUND")
        sys.exit(1)

    if MAKE_BACKUP:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    changed = 0
    unchanged = 0
    failed = 0
    emptied = 0
    msgs = []

    for p in targets:
        ok, msg = process_lab_file(p, BACKUP_DIR if MAKE_BACKUP else None)
        msgs.append(msg)
        if ok:
            changed += 1
        else:
            if msg.startswith("UNCHANGED"):
                unchanged += 1
            elif msg.startswith("EMPTY_AFTER_CLEAN"):
                emptied += 1
            else:
                failed += 1

    # 요약
    print("==== SUMMARY ====")
    print(f"Total .lab:     {len(targets)}")
    print(f"Changed:        {changed}")
    print(f"Unchanged:      {unchanged}")
    print(f"Empty after:    {emptied}")
    print(f"Failed:         {failed}")
    print(f"Backup dir:     {str(BACKUP_DIR) if MAKE_BACKUP else 'DISABLED'}")
    print("=================")

    # 변경/실패 로그 앞부분만 출력
    for m in msgs[:50]:
        print(m)

if __name__ == "__main__":
    main()
