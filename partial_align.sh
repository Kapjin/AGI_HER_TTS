#!/usr/bin/env bash

BASE=/home/tts/rvc_project/FastSpeech2
CORPUS=$BASE/KSS
ALIGN_OUT=$BASE/KSS_aligned
TEMP_CORPUS=$BASE/KSS_partial
TEMP_ALIGNED=$BASE/KSS_partial_aligned
NOGRID=$BASE/nogrid.txt

rm -rf "$TEMP_CORPUS" "$TEMP_ALIGNED"
mkdir -p "$TEMP_CORPUS"
mkdir -p "$TEMP_ALIGNED"   # ★ 추가: 로그/정렬 결과 폴더 미리 만들기

while read -r rel; do
  [ -z "$rel" ] && continue
  case "$rel" in
    \#*) continue ;;
  esac

  rel=$(echo "$rel" | tr -d '\r')
  base=${rel%.wav}
  folder=${base%%_*}

  wav_src="$CORPUS/$folder/${base}.wav"
  lab_src="$CORPUS/$folder/${base}.lab"

  echo "처리 중: rel='$rel' → folder='$folder'"
  echo "  WAV 후보: $wav_src"
  echo "  LAB 후보: $lab_src"

  if [ ! -f "$wav_src" ]; then
    echo "  [경고] WAV 없음: $wav_src"
    continue
  fi

  mkdir -p "$TEMP_CORPUS/$folder"
  ln -s "$wav_src" "$TEMP_CORPUS/$folder/"

  if [ -f "$lab_src" ]; then
    ln -s "$lab_src" "$TEMP_CORPUS/$folder/"
  else
    echo "  [경고] LAB 없음: $lab_src"
  fi

  echo
done < "$NOGRID"

echo "=== 부분 코퍼스에 대해 MFA align 실행 ==="

# 이제 TEMP_ALIGNED가 존재하므로 로그 파일이 정상적으로 생성됨
mfa align \
  "$TEMP_CORPUS" \
  korean_mfa \
  korean_mfa \
  "$TEMP_ALIGNED" \
  --clean \
  1> /dev/null \
  2> "$TEMP_ALIGNED/mfa.log"

echo "=== TextGrid 병합 ==="

find "$TEMP_ALIGNED" -name "*.TextGrid" | while read -r tg; do
  relpath=${tg#"$TEMP_ALIGNED"/}
  mkdir -p "$ALIGN_OUT/$(dirname "$relpath")"
  cp -n "$tg" "$ALIGN_OUT/$relpath"
done

echo
echo "=== MFA 로그 (마지막 30줄 표시) ==="
tail -n 30 "$TEMP_ALIGNED/mfa.log"
