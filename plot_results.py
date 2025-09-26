import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

OUTPUT_DIR = r"C:\Users\user\PycharmProjects\emotion_subtitle_improve\sample_test_1"
CSV_PATH   = os.path.join(OUTPUT_DIR, "words_emotion_pitch.csv")

df = pd.read_csv(CSV_PATH)

# 안전 처리
for col in ["emo_label", "pitch_label", "speaker"]:
    if col not in df.columns: df[col] = ""
df["emo_entropy"] = pd.to_numeric(df.get("emo_entropy", np.nan), errors="coerce")

# 1) 감정 분포 막대그래프 (최상위 라벨 개수)
emo_counts = df["emo_label"].value_counts().sort_values(ascending=False)

plt.figure()
emo_counts.plot(kind="bar")
plt.ylabel("Count")
plt.title("Emotion Distribution (Top label per word)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "count_emotion_bar.png"), dpi=150)
plt.close()

# 2) 화자별 감정 스택 막대
speakers = [s if isinstance(s,str) and s.strip() else "unknown" for s in df["speaker"]]
df["speaker_clean"] = speakers
emo_vals = sorted([e for e in df["emo_label"].unique() if isinstance(e,str)])
spk_vals = list(df["speaker_clean"].value_counts().index)

# 누적행렬 만들기 (rows=emotion, cols=speaker)
mat = np.zeros((len(emo_vals), len(spk_vals)), dtype=int)
spk_index = {s:i for i,s in enumerate(spk_vals)}
emo_index = {e:i for i,e in enumerate(emo_vals)}
for _, r in df.iterrows():
    e = r["emo_label"]; s = r["speaker_clean"]
    if e in emo_index and s in spk_index:
        mat[emo_index[e], spk_index[s]] += 1

bottom = np.zeros(len(spk_vals))
plt.figure()
for i,e in enumerate(emo_vals):
    plt.bar(spk_vals, mat[i], bottom=bottom, label=e)
    bottom += mat[i]
plt.ylabel("Count")
plt.title("Emotion by Speaker (stacked)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "count_emotion_by_speaker_stacked.png"), dpi=150)
plt.close()

# 3) 피치 분포 막대 (low/mid/high)
pitch_order = ["low","mid","high"]
pitch_counts = [int((df["pitch_label"]==p).sum()) for p in pitch_order]

plt.figure()
plt.bar(pitch_order, pitch_counts)
plt.ylabel("Count")
plt.title("Pitch Category Distribution")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "count_pitch_bar.png"), dpi=150)
plt.close()

# 4) 엔트로피 분포 히스토그램 (감정)
valid_entropy = df["emo_entropy"].dropna()
plt.figure()
plt.hist(valid_entropy, bins=20)
plt.xlabel("Emotion Entropy")
plt.ylabel("Count of words")
plt.title("Emotion Entropy Distribution")
# 임계선(예시): 엔트로피 1.2 이상을 '불확실'로 간주
thr = 1.2
plt.axvline(thr, linestyle="--")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hist_emo_entropy.png"), dpi=150)
plt.close()
