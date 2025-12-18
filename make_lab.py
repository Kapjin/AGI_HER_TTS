# scripts/kss_make_lab.py (대충 이런 식)
import os, re, codecs
root = "./KSS"
script = os.path.join(root, "transcript.v.1.4.txt")
# transcript 라인 예: 1/1/1.wav|어쩌구 저쩌구
norm = lambda s: re.sub(r"[^\w\s가-힣,.\-?!…~:%/()·]", " ", s).strip()
with codecs.open(script, "r", "utf-8") as f:
    for line in f:
        utt, text = line.strip().split("|", 1)
        lab_path = os.path.join(root, utt.replace(".wav", ".lab"))
        os.makedirs(os.path.dirname(lab_path), exist_ok=True)
        with codecs.open(lab_path, "w", "utf-8") as g:
            g.write(norm(text))
