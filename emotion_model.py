import numpy as np
import torch
import torch.nn.functional as F
import librosa
from dataclasses import dataclass
from typing import List
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

# ---------------------------
# 감정 프레임 인퍼런스(기존 유지)
# ---------------------------
@dataclass
class EmotionFrames:
    times: np.ndarray   # 각 프레임 중앙 시간
    probs: np.ndarray   # (N, C)
    rms: np.ndarray     # RMS 가중치
    labels: List[str]   # 클래스명

# (로컬 유틸) 오디오를 단일 채널(mono) & 지정된 샘플링레이트로 변환
def ensure_mono_sr(y, sr, target_sr=16000):
    if y.ndim == 2:
        y = librosa.to_mono(y.T)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return y, sr

# (로컬 유틸) 프레임 RMS
def rms_per_frame(frames: np.ndarray) -> np.ndarray:
    return np.sqrt((frames**2).mean(axis=1)) if len(frames) else np.array([])

# 오디오를 win_s, hop_s 단위로 프레임 분할
def frame_audio(y: np.ndarray, sr: int, win_s: float, hop_s: float):
    win = int(win_s * sr)
    hop = int(hop_s * sr)
    frames, centers = [], []
    for start in range(0, max(1, len(y) - win + 1), hop):
        end = start + win
        if end > len(y):
            break
        frames.append(y[start:end])
        centers.append((start + end) / 2 / sr)
    frames = np.stack(frames) if frames else np.empty((0,))
    centers = np.array(centers, dtype=float)
    rms = rms_per_frame(frames)
    return frames, centers, rms

def _get_sampling_rate(fe) -> int:
    fe_like = getattr(fe, "feature_extractor", fe)
    return getattr(fe_like, "sampling_rate", 16000)

# 감정 모델을 사용해 프레임별 감정 확률 계산
def emotion_frame_probs(y: np.ndarray, sr: int, model_id: str,
                        win_s: float = 0.5, hop_s: float = 0.25) -> EmotionFrames:
    fe = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModelForAudioClassification.from_pretrained(model_id)
    model.eval()

    target_sr = _get_sampling_rate(fe)
    required_len = int(target_sr * 30.0)  # Whisper 입력 고정 30초
    y, sr = ensure_mono_sr(y, sr, target_sr=target_sr)

    frames, centers, rms = frame_audio(y, sr, win_s, hop_s)
    if frames.size == 0:
        num_labels = getattr(model.config, "num_labels", 0)
        return EmotionFrames(np.array([]), np.zeros((0, num_labels)), np.array([]), [])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    all_probs = []
    BS = 64

    for i in range(0, len(frames), BS):
        batch_raw = frames[i:i+BS]
        # 고정 30초 패딩
        padded = []
        for f in batch_raw:
            f = np.asarray(f, dtype=np.float32)
            if len(f) >= required_len:
                padded.append(f[:required_len])
            else:
                pad = np.zeros(required_len, dtype=np.float32)
                pad[:len(f)] = f
                padded.append(pad)

        inputs = fe(padded, sampling_rate=sr, return_tensors="pt")
        if isinstance(inputs, dict) and "attention_mask" in inputs:
            inputs.pop("attention_mask", None)

        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
            all_probs.append(probs)

    probs = np.concatenate(all_probs, axis=0)
    labels = [model.config.id2label[i] for i in range(probs.shape[1])]
    return EmotionFrames(centers, probs, rms, labels)
