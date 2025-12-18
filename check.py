import os
import numpy as np
import csv

# 현재 작업 위치: ~/rvc_project/FastSpeech2
duration_dir = "preprocessed_data/KSS/duration"
mel_dir      = "preprocessed_data/KSS/mel"

duration_files = sorted([f for f in os.listdir(duration_dir) if f.endswith(".npy")])

error_count = 0
checked = 0

# 전체 통계용
all_duration_sums = []
all_mel_lens = []
all_diffs = []  # mel_len - duration_sum

# CSV 기록용
csv_path = "duration_mel_check_kss.csv"
csv_rows = [["file", "duration_sum", "mel_len", "diff(mel_len-duration_sum)"]]

for dur_f in duration_files:
    mel_f = dur_f.replace("duration", "mel")
    mel_path = os.path.join(mel_dir, mel_f)

    if not os.path.exists(mel_path):
        print(f"[Missing mel] {mel_f} (for duration {dur_f})")
        error_count += 1
        continue

    dur = np.load(os.path.join(duration_dir, dur_f))
    mel = np.load(mel_path)

    duration_sum = int(dur.sum())
    mel_len = int(mel.shape[0])
    diff = mel_len - duration_sum

    checked += 1
    all_duration_sums.append(duration_sum)
    all_mel_lens.append(mel_len)
    all_diffs.append(diff)

    csv_rows.append([dur_f, duration_sum, mel_len, diff])

    # 개별 파일 중간 결과 출력
    print(f"{dur_f} | duration_sum={duration_sum}, mel_len={mel_len}, diff={diff}")

    if diff != 0:
        print(f"  -> [Mismatch]")
        error_count += 1

# CSV로 저장
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(csv_rows)

print("\n================ Summary ================")
print(f"Checked {checked} file pairs.")
print(f"Mismatched (or missing mel) count: {error_count}")

if checked > 0:
    total_duration = sum(all_duration_sums)
    total_mel_len = sum(all_mel_lens)

    print(f"\n[Total]")
    print(f"  Sum of all duration_sums = {total_duration}")
    print(f"  Sum of all mel_lens      = {total_mel_len}")
    print(f"  Total diff (mel - dur)   = {total_mel_len - total_duration}")

    diffs = np.array(all_diffs)
    print("\n[Diff statistics per file]  (mel_len - duration_sum)")
    print(f"  min diff = {diffs.min()}")
    print(f"  max diff = {diffs.max()}")
    print(f"  mean diff = {diffs.mean():.4f}")

print(f"\nCSV saved to: {csv_path}")
