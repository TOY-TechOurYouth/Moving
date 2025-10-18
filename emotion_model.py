import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import librosa
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

def ensure_mono_sr(y, sr, target_sr=16000):
    if y.ndim == 2:
        y = librosa.to_mono(y.T)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return y, sr

def _get_sampling_rate(fe) -> int:
    fe_like = getattr(fe, "feature_extractor", fe)
    return getattr(fe_like, "sampling_rate", 16000)

def extract_word_centered_segment(y: np.ndarray, sr: int,
                                  word_start: float, word_end: float,
                                  target_duration: float = 5.0) -> np.ndarray:
    word_center = (word_start + word_end) / 2.0
    word_center_sample = int(word_center * sr)
    half_len = int(target_duration * sr / 2)
    start_sample = word_center_sample - half_len
    end_sample = word_center_sample + half_len
    audio_len = len(y)

    if start_sample < 0:
        segment = y[:int(target_duration * sr)]
        if len(segment) < int(target_duration * sr):
            padded = np.zeros(int(target_duration * sr), dtype=y.dtype)
            padded[:len(segment)] = segment
            return padded
        return segment
    elif end_sample > audio_len:
        segment = y[-int(target_duration * sr):]
        if len(segment) < int(target_duration * sr):
            padded = np.zeros(int(target_duration * sr), dtype=y.dtype)
            padded[:len(segment)] = segment
            return padded
        return segment
    else:
        return y[start_sample:end_sample]

def emotion_probs_per_words(y: np.ndarray, sr: int, words_df: pd.DataFrame,
                            model_id: str,
                            target_duration: float = 5.0,
                            batch_size: int = 32) -> pd.DataFrame:
    print(f"🧠 단어 중심 {target_duration}초 구간 감정 분석 시작...")

    fe = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModelForAudioClassification.from_pretrained(model_id)
    model.eval()

    target_sr = _get_sampling_rate(fe)
    required_samples = int(target_sr * target_duration)
    y, sr = ensure_mono_sr(y, sr, target_sr=target_sr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    emotion_labels = [model.config.id2label[i] for i in range(model.config.num_labels)]

    total_words = len(words_df)
    batch_segments, batch_indices = [], []
    all_probs = []

    for idx, row in words_df.iterrows():
        seg = extract_word_centered_segment(y, sr, row['start'], row['end'], target_duration)
        if len(seg) < required_samples:
            padded = np.zeros(required_samples, dtype=np.float32)
            padded[:len(seg)] = seg
            seg = padded
        elif len(seg) > required_samples:
            seg = seg[:required_samples]

        batch_segments.append(seg)
        batch_indices.append(idx)

        if len(batch_segments) == batch_size or idx == words_df.index[-1]:
            inputs = fe(batch_segments, sampling_rate=target_sr, return_tensors="pt")
            if isinstance(inputs, dict) and "attention_mask" in inputs:
                inputs.pop("attention_mask", None)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = model(**inputs).logits
                probs = F.softmax(logits, dim=-1).detach().cpu().numpy()

            all_probs.extend(list(probs))
            batch_segments, batch_indices = [], []

    probs_array = np.asarray(all_probs, dtype=float)  # shape: (N_words, N_emotions)

    # 🔻 여기서부터 "압축 컬럼"만 생성
    # emo_probs: {"anger":0.12,"fear":0.03,...} 같은 JSON 문자열
    emo_probs_json = []
    emo_top_labels = []
    emo_entropy = []

    for pv in probs_array:
        # dict(label->prob)
        d = {emotion_labels[i]: float(pv[i]) for i in range(len(emotion_labels))}
        emo_probs_json.append(json.dumps(d, ensure_ascii=False))
        emo_top_labels.append(emotion_labels[int(np.argmax(pv))])
        # Shannon entropy (ln)
        ent = float(-(pv * np.log(pv + 1e-9)).sum())
        emo_entropy.append(ent)

    # ✅ 최종 컬럼만 추가 (기존 per-label 확률/confidence 컬럼 생성 안 함)
    words_df["emo_label"] = emo_top_labels
    words_df["emo_entropy"] = emo_entropy
    words_df["emo_probs"] = emo_probs_json

    print(f"✅ 단어별 감정 분석 완료: {total_words}개 단어")
    return words_df