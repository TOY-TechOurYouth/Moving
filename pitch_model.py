import numpy as np
import pandas as pd
import librosa
from dataclasses import dataclass
from typing import List, Optional

# ---------------------------
# 피치: F0 -> 세미톤 -> 소프트 확률 (화자별 기준치)
# ---------------------------

# 피치 분석 결과 저장용 데이터 클래스
@dataclass
class PitchFrames:
    times: np.ndarray           # 각 프레임의 중앙 시간
    probs: np.ndarray           # 각 프레임의 피치 softmax 확률 (N x 3 : [low, mid, high])
    entropy: np.ndarray         # 각 프레임의 확률 엔트로피 (불확실성 지표)
    f0_med_per_speaker: dict    # 화자별 F0 중앙값 {speaker: median_f0}
    st_values: np.ndarray       # 각 프레임의 세미톤 값 (상대 피치)
    frame_speakers: List[str]   # 각 프레임에 할당된 화자

# 오디오를 win_s, hop_s 단위로 프레임 분할 (로컬 복사본)
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
    return frames, centers, None

# 특정 시각 t에서 활성 화자 라벨 찾기 (로컬 복사본)
def assign_speaker_at_time(t: float, diar_view: pd.DataFrame, default="unknown") -> str:
    hits = diar_view[(diar_view["start"] <= t) & (diar_view["end"] >= t)]
    if hits.empty:
        return default
    lens = (hits["end"] - hits["start"]).to_numpy()
    return str(hits.iloc[int(lens.argmax())]["speaker"])

# 프레임 단위로 화자 라벨 매핑 (로컬 복사본)
def assign_speaker_to_frames(frame_times: np.ndarray, diar_view: pd.DataFrame) -> List[str]:
    return [assign_speaker_at_time(float(t), diar_view) for t in frame_times]

# 각 프레임별 기본 주파수(F0) 추정
def estimate_f0_per_frame(frames: np.ndarray, sr: int) -> np.ndarray:
    f0_all = []
    for f in frames:
        try:
            # librosa.pyin: F0 추정 (범위: 50Hz ~ 600Hz)
            f0, _, _ = librosa.pyin(
                f, fmin=50, fmax=600, sr=sr,
                frame_length=2048, hop_length=512
            )
            f0_val = np.nanmedian(f0)  # NaN 제외한 중앙값
        except Exception:
            f0_val = np.nan
        # 값이 있으면 [50, 600] 범위로 clip
        if not np.isnan(f0_val):
            f0_val = float(np.clip(f0_val, 50, 600))
        f0_all.append(f0_val)
    return np.array(f0_all, dtype=float)

# 피치 소프트 확률 계산 (화자별 기준 F0 반영)
def pitch_soft_probs_with_speakers(
    y: np.ndarray, sr: int,
    diar_view: Optional[pd.DataFrame] = None,   # 화자 정보 (start, end, speaker)
    win_s: float = 0.5, hop_s: float = 0.25,    # 프레임 길이/홉 (초)
    delta_st: float = 2.0, sigma: float = 1.2   # 분류 경계 및 분산
) -> PitchFrames:
    # 오디오를 프레임 단위로 분할
    frames, centers, _ = frame_audio(y, sr, win_s, hop_s)
    if frames.size == 0:
        # 프레임이 없을 경우 빈 구조 반환
        return PitchFrames(np.array([]), np.zeros((0,3)), np.array([]), {}, np.array([]), [])

    # 각 프레임별 F0 추정
    f0_all = estimate_f0_per_frame(frames, sr)

    # 프레임별 화자 할당 (다이어리제이션 결과 활용)
    if diar_view is not None and not diar_view.empty:
        frame_speakers = assign_speaker_to_frames(centers, diar_view)
    else:
        frame_speakers = ["unknown"] * len(centers)

    # 화자별 F0 중앙값 계산
    f0_med_per_speaker = {}
    fs = np.array(frame_speakers, dtype=object)
    for spk in set(frame_speakers):
        vals = f0_all[(fs == spk) & (~np.isnan(f0_all))]
        if len(vals) > 0:
            f0_med_per_speaker[spk] = float(np.median(vals))

    # 모든 화자를 통틀어 global median (백업용)
    global_med = float(np.median(f0_all[~np.isnan(f0_all)])) if np.any(~np.isnan(f0_all)) else np.nan

    # 각 프레임의 상대 세미톤 계산
    st = np.zeros_like(centers, dtype=float)
    for i, (f0, spk) in enumerate(zip(f0_all, frame_speakers)):
        base = f0_med_per_speaker.get(spk, global_med)  # 화자별 중앙값 기준
        if np.isnan(f0) or not np.isfinite(base):
            st[i] = 0.0  # 값이 없으면 0
        else:
            # 세미톤 변환: 12 * log2(f0 / 기준값)
            st[i] = 12.0 * np.log2(max(f0, 1e-6) / base)

    # 소프트 분류 (low, mid, high)
    # 중심점: -delta_st, 0, +delta_st
    centers_st = np.array([-delta_st, 0.0, +delta_st])[:, None]

    # 각 세미톤 값에 대해 가우시안 분포 점수 계산
    scores = np.exp(-0.5 * ((st[None, :] - centers_st) / sigma) ** 2)

    # 정규화하여 확률로 변환 (softmax와 유사)
    probs = (scores / (scores.sum(axis=0, keepdims=True) + 1e-9)).T

    # 엔트로피 계산 (불확실성 정도)
    ent = -np.sum(probs * np.log(probs + 1e-9), axis=1)

    # 최종 결과 구조체 반환
    return PitchFrames(centers, probs, ent, f0_med_per_speaker, st, frame_speakers)
