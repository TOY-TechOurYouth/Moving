# filename: full_emotion_pitch_with_diarization.py
# --------------------------------------------------
# 입력 오디오 -> (선택)보컬 분리 -> Whisper 단어 타임스탬프
#           -> (선택)화자 분리(pyannote) -> 프레임 감정 확률(W2V2 SER)
#           -> 피치(F0->세미톤->소프트 확률, 화자별 기준치) -> 단어 구간 집계
#           -> CSV 저장(start, end, word, speaker, emo_*, pitch_*)
# --------------------------------------------------
import sys

INPUT_AUDIO = r"C:\Users\user\PycharmProjects\emotion_subtitle_improve\sample.wav"   # ← 입력 오디오 경로
OUTPUT_DIR  = r"C:\Users\user\PycharmProjects\emotion_subtitle_improve\sample_test_1"      # ← 출력 폴더
LANG        = "en"                           # Whisper 강제 언어 (예: "en", "ko")
WORDS_CSV   = None                           # Whisper 대신 쓸 단어 CSV(start,end,word). 없으면 None

USE_VOCAL_SEPARATION = True                  # Spleeter/Demucs가 있으면 보컬 분리 사용
USE_WHISPER          = True                  # openai-whisper 설치 시 단어 타임스탬프 자동 추출
USE_DIARIZATION      = True                  # pyannote.audio 설치 + HF_TOKEN 필요

# 모델 ID
EMO_MODEL_ID          = "firdhokk/speech-emotion-recognition"         # 7-class SER
DIARIZATION_MODEL_ID  = "pyannote/speaker-diarization-3.1"            # pyannote diarization pipeline

# 프레이밍 파라미터 (감정/피치 공통 그리드)
WIN_S = 0.5        # 윈도 길이(초)
HOP_S = 0.25       # 홉(초)
WORD_PAD_S = 0.05  # 단어 경계 패딩(±초)

# 피치 소프트 분류(세미톤) 파라미터
PITCH_DELTA_ST = 2.0   # class centers [-Δ, 0, +Δ] st
PITCH_SIGMA    = 1.2   # Gaussian sigma (st)

HF_TOKEN = "HF_TOKEN_REDACTED"  # ← 여기에 토큰 입력

# ==================================================

import os, json, math
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
import torch.nn.functional as F
import subprocess, shutil
from pathlib import Path

from transformers import AutoModelForAudioClassification # 오디오 분류용 사전학습 모델
from transformers import AutoFeatureExtractor as AudioProcessorClass # 오디오 전처리기(파형 -> 모델 입력 텐서)

from pydub import AudioSegment # 오디오 포맷 변환/자르기/합치기 등에 유용
AudioSegment.converter = r"C:\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"

def preview(df: pd.DataFrame, name: str, n: int = 10):
    try:
        print(f"📝 {name} 미리보기 (top {n})")
        print(df.head(n).to_string(index=False))
    except Exception as e:
        print(f"(미리보기 실패: {e})")

# ---------------------------
# 유틸
# ---------------------------

# 안전한 디렉터리 생성
def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)

# 오디오 파형 y와 샘플레이트 sr을 받아서 모노로 변환 후 16kHz로 리샘플
def ensure_mono_16k(y, sr, target_sr=16000):
    # 채널 차원이 2이면 스테레오로 간주하고 모노로 변환
    if y.ndim == 2:
        y = librosa.to_mono(y.T)
    # 샘플레이트가 목표와 다르면 리샘플
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return y, sr

# CSV로 저장하고 경로를 로그로 남김
def write_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"💾 Saved: {path}")

# 오디오 로드
def load_audio(path: str):
    y, sr = librosa.load(path, sr=None, mono=True)
    return y, sr

# 프레임 묶음(2D 배열)에 대해 각 프레임의 RMS를 계산 -> 단어 구간 집계에서 활용
def rms_per_frame(frames: np.ndarray) -> np.ndarray:
    return np.sqrt((frames**2).mean(axis=1)) if len(frames) else np.array([])

# ---------------------------
# 보컬 분리
# ---------------------------
def separate_vocals_with_demucs(wav_path: str, session_dir: str) -> str:
    """
    입력 WAV에서 보컬만 Demucs로 분리해 session_dir/separation/vocals.wav로 저장.
    실패하면 예외 대신 원본 경로를 반환(폴백)하도록 안전하게 처리.
    """
    from pathlib import Path
    import subprocess, shutil, sys
    import torch

    # 입력 경로/세션 디렉토리 절대경로화
    wav_path    = str(Path(wav_path).resolve())
    session_dir = Path(session_dir).resolve()
    # 출력 루트
    out_root    = session_dir / "separation" / "demucs_out"
    out_root.mkdir(parents=True, exist_ok=True)

    # GPU 가능 여부에 따라 장치 자동 선택
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cmd = [
        sys.executable, "-m", "demucs.separate",
        "-n", "htdemucs",
        "--two-stems", "vocals",
        "-d", device,                # auto(cu/cpu)
        "-o", str(out_root),
        wav_path
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore"
    )

    # 실행 실패 시 처리
    if proc.returncode != 0:
        print("❌ Demucs 실패")
        print("── stdout ──")
        print(proc.stdout.strip())
        print("── stderr ──")
        print(proc.stderr.strip())
        print("⚠️ 보컬 분리 없이 원본으로 계속 진행합니다.")
        return wav_path  # 폴백

    # 실행 성공했지만 vocal.wav를 못 찾은 경우
    # 버전/플랫폼별 폴더 차이를 허용: 어디든 vocals.wav만 찾아옴
    candidates = list(out_root.rglob("vocals.wav"))
    if not candidates:
        print("❌ Demucs 출력 파일(vocals.wav)을 찾지 못했습니다.")
        print("── stdout ──")
        print(proc.stdout.strip())
        print("── stderr ──")
        print(proc.stderr.strip())
        print("⚠️ 보컬 분리 없이 원본으로 계속 진행합니다.")
        return wav_path  # 폴백

    # 정상적으로 찾은 경우 -> 첫 번째 vocals.wav 사용
    demucs_vocals = candidates[0]
    fixed = session_dir / "separation" / "vocals.wav"
    fixed.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(demucs_vocals, fixed)
    # 최종 반환: 분리된 vocals.wav의 경로
    return str(fixed)

# ---------------------------
# Whisper 단어 타임스탬프
# ---------------------------
def whisper_word_timestamps(audio_path: str, language: Optional[str] = "en") -> pd.DataFrame:
    import whisper  # pip install openai-whisper
    print("🔤 Whisper(small) 단어 타임스탬프 추출...")
    # "tiny", "base", "small", "medium", "large" 등 크기별 모델 존재 -> 크기 클수록 정확도는 높아지지만 속도가 느려짐
    model = whisper.load_model("small")
    result = model.transcribe(audio_path, language=language, word_timestamps=True, verbose=False)
    words = []
    # Whisper 출력 구조
    for seg in result.get("segments", []): # result["segments"] = 문장/구간 단위 세그먼트 리스트
        for w in seg.get("words", []): # seg["words"] = 각 세그먼트의 단어별 정보 (start, end, word)
            words.append({"start": w["start"], "end": w["end"], "word": w["word"].strip()})
    df = pd.DataFrame(words)
    if df.empty:
        raise RuntimeError("Whisper가 단어를 찾지 못했습니다.")
    return df

# ---------------------------
# 화자 분리(다이어라이제이션)
# ---------------------------

# 하나의 화자 구간을 담는 데이터 구조 정의
@dataclass
class DiarizationSeg:
    start: float
    end: float
    speaker: str

def run_diarization(audio_path: str, model_id=DIARIZATION_MODEL_ID) -> List[DiarizationSeg]:
    """
    pyannote.audio Pipeline 사용. 전역 상수 HF_TOKEN 사용.
    반환: [DiarizationSeg(...)] 리스트
    """
    print("👥 화자 분리 실행 중 (pyannote)...")
    from pyannote.audio import Pipeline

    # Hugging Face 토큰 필수
    if not HF_TOKEN or not isinstance(HF_TOKEN, str):
        raise RuntimeError("HF_TOKEN이 비어 있습니다. 유효한 HuggingFace 토큰을 넣어주세요.")

    # 환경변수 대신 하드코딩된 토큰 사용
    pipeline = Pipeline.from_pretrained(model_id, use_auth_token=HF_TOKEN)

    # 오디오에 대해 diarization 실행
    diar = pipeline(audio_path)  # Annotation
    segs: List[DiarizationSeg] = []
    for turn, _, speaker in diar.itertracks(yield_label=True):
        segs.append(DiarizationSeg(start=float(turn.start), end=float(turn.end), speaker=str(speaker)))
    # 시작 시간 기준으로 정렬
    segs.sort(key=lambda s: s.start)
    print(f"👥 화자 세그먼트 {len(segs)}개")
    return segs

# 화자 세그먼트 리스트 -> dataFrame 변환
def diar_to_dataframe(segs: List[DiarizationSeg]) -> pd.DataFrame:
    return pd.DataFrame([{"start": s.start, "end": s.end, "speaker": s.speaker} for s in segs])

# 특정 시각 t에 해당하는 화자 ID를 반환
def assign_speaker_at_time(t: float, diar_df: pd.DataFrame, default="unknown") -> str:
    # t가 포함되는 첫 세그먼트의 speaker 반환 (겹치면 가장 긴 세그먼트 우선)
    hits = diar_df[(diar_df["start"] <= t) & (diar_df["end"] >= t)]
    if hits.empty:
        return default
    # 가장 긴 세그먼트
    lens = (hits["end"] - hits["start"]).to_numpy()
    return hits.iloc[int(lens.argmax())]["speaker"]

# 여러 프레임 시각 배열에 대해 각 프레임이 속하는 화자를 리스트로 반환
def assign_speaker_to_frames(frame_times: np.ndarray, diar_df: pd.DataFrame) -> List[str]:
    return [assign_speaker_at_time(float(t), diar_df) for t in frame_times]

# words_with_speaker.csv
def annotate_words_with_speaker(words_df: pd.DataFrame, diar_df: pd.DataFrame) -> pd.DataFrame:
    """
    각 단어에 스피커 라벨 부여: 단어 중앙 시각 기준으로 매핑(간단/견고)
    """
    speakers = []
    for _, w in words_df.iterrows():
        mid = (float(w["start"]) + float(w["end"])) / 2.0
        speakers.append(assign_speaker_at_time(mid, diar_df))
    out = words_df.copy()
    out["speaker"] = speakers
    return out

# ---------------------------
# 감정: 프레임 인퍼런스
# ---------------------------
# 프레임 단위 감정 분석 결과를 담는 데이터 구조
@dataclass
class EmotionFrames:
    times: np.ndarray   # 각 프레임의 중앙 시간
    probs: np.ndarray   # 감정 확률 배열 (프레임 수 X 감정 클래스 수)
    rms: np.ndarray     # 프레임별 RMS 에너지 (가중치로 사용 가능)
    labels: List[str]   # 감정 클래스 이름 리스트

# 긴 오디오 파형을 일정한 길이로 잘라 프레임 만듦
def frame_audio(y: np.ndarray, sr: int, win_s=WIN_S, hop_s=HOP_S):
    win = int(win_s * sr) # 윈도 길이를 샘플 단위로 변환
    hop = int(hop_s * sr) # 홉 크기를 샘플 단위로 변환
    frames, centers = [], []
    # 0부터 len(y)-win 까지 hop씩 증가하여 프레임 생성
    for start in range(0, max(1, len(y)-win+1), hop):
        end = start + win
        if end > len(y): break # 오디오 끝 넘으면 중단
        frames.append(y[start:end])
        # 프레임의 중앙 위치를 시간(초)로 저장
        centers.append((start + end) / 2 / sr)
    # 리스트를 numpy 배열로 변환
    frames = np.stack(frames) if frames else np.empty((0,))
    centers = np.array(centers, dtype=float)
    rms = rms_per_frame(frames) # 프레임별 RMS 계산
    return frames, centers, rms

# 오디오 파형을 프레임 단위로 잘라서 각 프레임별 감정 확률 계산
def emotion_frame_probs(y: np.ndarray, sr: int, model_id=EMO_MODEL_ID) -> EmotionFrames:
    # 모델에 맞는 전처리기(processor)와 모델 불러오기
    proc = AudioProcessorClass.from_pretrained(model_id)
    model = AutoModelForAudioClassification.from_pretrained(model_id)
    model.eval() # 평가 모드 (dropout 등 비활성화)

    # 오디오를 모노/16k로 맞춤 (모델 요구 SR로 변환)
    y, sr = ensure_mono_16k(y, sr, target_sr=getattr(proc, "sampling_rate", 16000))
    # 오디오를 프레임 단위로 분할
    frames, centers, rms = frame_audio(y, sr, WIN_S, HOP_S)
    # 프레임이 없으면 빈 EmotionFrames 반환
    if frames.size == 0:
        return EmotionFrames(np.array([]), np.zeros((0, model.config.num_labels)), np.array([]), [])

    # 전처리기로 파형 -> 모델 입력 텐서 변환
    inputs = proc(frames.tolist(), sampling_rate=sr, return_tensors="pt", padding=True)
    # 모델 추론(no_grad: 그래디언트 계산 비활성화 -> 메모리/속도 커짐)
    with torch.no_grad():
        logits = model(**{k: v for k, v in inputs.items() if k in ["input_values", "attention_mask"]}).logits
        # 소프트맥스로 확률화
        probs = F.softmax(logits, dim=-1).cpu().numpy()
    # id2label 매핑으로 클래스 이름 추출
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    return EmotionFrames(centers, probs, rms, labels)


# ---------------------------
# 피치: F0 -> 세미톤 -> 소프트 확률 (화자별 기준치)
# ---------------------------
@dataclass
class PitchFrames:
    times: np.ndarray           # 각 프레임의 중앙 시간(초)
    probs: np.ndarray           # (N,3) [low, mid, high]
    entropy: np.ndarray         # 각 프레임 확률분포의 엔트로피(불확실성 지표)
    f0_med_per_speaker: dict    # 화자별 F0 중앙값
    st_values: np.ndarray       # 각 프레임의 세미톤 값(기준치 대비 상대 피치)
    frame_speakers: List[str]   # 각 프레임에 매핑된 화자 라벨

# 각 프레임 파형에 대해 기본주파수(F0) 추정 후, 중앙값 계산
def estimate_f0_per_frame(frames: np.ndarray, sr: int) -> np.ndarray:
    """각 프레임에서 pyin으로 F0 대표값(중앙값) 추정"""
    f0_all = []
    for f in frames:
        try:
            # pyin은 프레임 내부를 다시 세부 프레임으로 나눠 F0 시퀀스를 추정
            f0, _, _ = librosa.pyin(f, fmin=50, fmax=600, sr=sr, frame_length=2048, hop_length=512)
            f0_val = np.nanmedian(f0) # 대표값: 중앙값(노이즈에 강함)
        except Exception:
            f0_val = np.nan
        if not np.isnan(f0_val):
            f0_val = float(np.clip(f0_val, 50, 600)) # 안전 범위 클리핑
        f0_all.append(f0_val)
    return np.array(f0_all, dtype=float)

# 피치 결과 출력
def pitch_soft_probs_with_speakers(y: np.ndarray, sr: int,
                                   diar_df: Optional[pd.DataFrame] = None,
                                   win_s=WIN_S, hop_s=HOP_S,
                                   delta_st=PITCH_DELTA_ST, sigma=PITCH_SIGMA) -> PitchFrames:
    # 프레임 만들기 (감정 쪽과 동일한 윈도/홉 사용)
    frames, centers, _ = frame_audio(y, sr, win_s, hop_s)
    if frames.size == 0:
        return PitchFrames(np.array([]), np.zeros((0,3)), np.array([]), {}, np.array([]), [])

    # 1) 프레임별 F0
    f0_all = estimate_f0_per_frame(frames, sr)

    # 2) 프레임별 speaker 라벨 (없으면 'unknown')
    if diar_df is not None and not diar_df.empty:
        frame_speakers = assign_speaker_to_frames(centers, diar_df)
    else:
        frame_speakers = ["unknown"] * len(centers)

    # 3) 화자별 기준 F0 중앙값 계산 -> 같은 화자의 프레임 F0들 중 NaN를 제외하고 median
    f0_med_per_speaker = {}
    for spk in set(frame_speakers):
        vals = f0_all[(np.array(frame_speakers) == spk) & (~np.isnan(f0_all))]
        if len(vals) > 0:
            f0_med_per_speaker[spk] = float(np.median(vals))
    # 글로벌 백업 -> 화자별 값이 없을 때 사용
    global_med = float(np.median(f0_all[~np.isnan(f0_all)])) if np.any(~np.isnan(f0_all)) else np.nan

    # 4) 세미톤 변환(화자별 기준치를 사용) -> base는 해당 화자의 median F0
    st = np.zeros_like(centers, dtype=float)
    for i, (t, f0, spk) in enumerate(zip(centers, f0_all, frame_speakers)):
        base = f0_med_per_speaker.get(spk, global_med)
        if np.isnan(f0) or not np.isfinite(base):
            st[i] = 0.0  # 무성/기준치 없음 → Mid로 수렴
        else:
            st[i] = 12.0 * np.log2(max(f0, 1e-6) / base)

    # 5) 가우시안 점수 -> 소프트맥스 확률
    # 3개의 중심치: [-Δ, 0, +Δ] semitone (Δ=delta_st)
    centers_st = np.array([-delta_st, 0.0, +delta_st])[:, None]  # (3,1)
    # 각 중심에 대해 가우시안 점수
    scores = np.exp(-0.5 * ((st[None, :] - centers_st) / sigma) ** 2)  # (3, N)
    # 프레임마다 3개 점수를 정규화(소프트맥스) -> 확률 벡터
    probs = (scores / (scores.sum(axis=0, keepdims=True) + 1e-9)).T      # (N,3)

    # 불확실성 측정: 엔트로피
    ent = -np.sum(probs * np.log(probs + 1e-9), axis=1)

    # 최종 패키징
    return PitchFrames(centers, probs, ent, f0_med_per_speaker, st, frame_speakers)

# ---------------------------
# 단어 구간 집계 (가중 평균 + 최대값 + 엔트로피)
# ---------------------------
# 단어별 시간 구간(start, end)에 포함된느 감정, 피치 프레임 확률을 모아 요약하고 저장
def aggregate_over_words(words_df: pd.DataFrame,
                         emo: EmotionFrames,
                         pitch: PitchFrames,
                         diar_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    반환 컬럼:
      start, end, word, speaker,
      emo_label, emo_entropy, emo_probs(json),
      pitch_label, pitch_entropy, pitch_probs(json)
    """
    out_rows = []
    # 감정 라벨 이름
    emo_labels = emo.labels if emo.labels else [f"class_{i}" for i in range(emo.probs.shape[1])]
    # 피치 라벨 이름
    pitch_labels = ["low", "mid", "high"]

    # 프레임 → 빠른 인덱싱용
    etimes, eprobs, erms = emo.times, emo.probs, emo.rms
    ptimes, pprobs = pitch.times, pitch.probs

    for _, w in words_df.iterrows():
        # 단어 경계를 약간 확장(WORD_PAD_S만큼 앞뒤 패딩) -> 경계 누락 방지
        t0 = float(w["start"]) - WORD_PAD_S
        t1 = float(w["end"]) + WORD_PAD_S

        # 화자 라벨: 단어 중앙으로 결정(다이어라이제이션이 있으면)
        if diar_df is not None and not diar_df.empty:
            speaker = assign_speaker_at_time((t0+t1)/2.0, diar_df)
        else:
            speaker = "unknown"

        # 감정 집계
        if etimes.size:
            # 단어 구간[t0, t1]에 포함되는 감정 프레임만 선택
            mask_e = (etimes >= t0) & (etimes <= t1)
            Pe = eprobs[mask_e] # (num_frames_in_word, K)  K=감정 클래스 수
            We = erms[mask_e]   # (num_frames_in_word,)    RMS: 에너지 기반 가중치
            if Pe.shape[0] > 0:
                # 가중치 합이 0에 가깝다면 균등 가중치로 대체
                if We.sum() <= 1e-12:
                    We = np.ones_like(We) / len(We) # 정규화(합=1)
                else:
                    We = We / We.sum()
                # 가중 평균: 각 프레임 확률에 RMS 가중치를 곱해 합산 -> 단어별 대표 확률 벡터
                emo_mean = (Pe * We[:, None]).sum(axis=0)     # (K,)
                # 프레임별 확률 중 최대값
                emo_max  = Pe.max(axis=0)
                # 엔트로피: 불확실성 지표(높을수록 분포가 퍼져있음)
                emo_entropy = float(-(emo_mean * np.log(emo_mean + 1e-9)).sum())
                # 최상위 감정 라벨
                emo_top = emo_labels[int(np.argmax(emo_mean))]
                emo_probs_dict = {emo_labels[i]: float(emo_mean[i]) for i in range(len(emo_mean))}
            else:
                # 해당 구간에 감정 프레임이 없으면 결측 처리
                emo_entropy = float(np.nan); emo_top = ""
                emo_probs_dict = {}
        else:
            # 감정 프레임 자체가 없을 때
            emo_entropy = float(np.nan); emo_top = ""
            emo_probs_dict = {}

        # 피치 집계
        if ptimes.size:
            # 단어 구간[t0, t1]에 포함되는 피치 프레임만 선택
            mask_p = (ptimes >= t0) & (ptimes <= t1)
            Pp = pprobs[mask_p]
            if Pp.shape[0] > 0:
                # 피치는 기본적으로 균등 가중 평균(원한다면 RMS/voiced 가중치로 확장 가능)
                Wp = np.ones((Pp.shape[0],), dtype=float) / Pp.shape[0]
                # 단어 구간의 대표 피치 확률
                pitch_mean = (Pp * Wp[:, None]).sum(axis=0)
                # 프레임별 최대값
                pitch_max  = Pp.max(axis=0)
                # 엔트로피
                pitch_entropy = float(-(pitch_mean * np.log(pitch_mean + 1e-9)).sum())
                # 최상위 피치 라벨
                pitch_top = pitch_labels[int(np.argmax(pitch_mean))]
                pitch_probs_dict = {pitch_labels[i]: float(pitch_mean[i]) for i in range(3)}
            else:
                pitch_entropy = float(np.nan); pitch_top = ""
                pitch_probs_dict = {}
        else:
            pitch_entropy = float(np.nan); pitch_top = ""
            pitch_probs_dict = {}

        out_rows.append({
            "start": float(w["start"]),
            "end": float(w["end"]),
            "word": str(w["word"]),
            "speaker": speaker,
            # 감정
            "emo_label": emo_top,
            "emo_entropy": emo_entropy,
            "emo_probs": json.dumps(emo_probs_dict, ensure_ascii=False),
            # 피치
            "pitch_label": pitch_top,
            "pitch_entropy": pitch_entropy,
            "pitch_probs": json.dumps(pitch_probs_dict, ensure_ascii=False),
        })

    return pd.DataFrame(out_rows)

# ---------------------------
# 메인 파이프라인
# ---------------------------
def main():
    safe_mkdir(OUTPUT_DIR)

    # 1) 입력 로드 & (선택) 보컬 분리
    src_wav = INPUT_AUDIO
    print(f"🎵 Input: {src_wav}")
    use_path = src_wav
    if USE_VOCAL_SEPARATION:
        # Demucs를 이용해 보컬만 추출
        use_path = separate_vocals_with_demucs(src_wav, OUTPUT_DIR)
    print(f"🎤 분석 오디오: {use_path}")

    # 2) 단어 타임스탬프
    if USE_WHISPER:
        try:
            # Whisper로 오디오에서 단어 단위 (start, end, word) 추출
            words_df = whisper_word_timestamps(use_path, language=LANG)
        except Exception as e:
            print(f"⚠️ Whisper 실패: {e}")
            if WORDS_CSV and os.path.exists(WORDS_CSV):
                words_df = pd.read_csv(WORDS_CSV)
                assert {"start","end","word"}.issubset(words_df.columns)
            else:
                raise
    else:
        # USE_WHISPER=False → 반드시 WORDS_CSV 제공해야 함
        if WORDS_CSV and os.path.exists(WORDS_CSV):
            words_df = pd.read_csv(WORDS_CSV)
            assert {"start","end","word"}.issubset(words_df.columns)
        else:
            raise RuntimeError("USE_WHISPER=False 인 경우 WORDS_CSV 경로가 필요합니다.")

    # Whisper 결과 CSV 저장 및 미리보기
    words_raw_csv = os.path.join(OUTPUT_DIR, "words_raw.csv")
    write_csv(words_df, words_raw_csv)  # ① Whisper 원본 단어 CSV 저장
    preview(words_df, "Whisper 단어")  # ② 콘솔 미리보기
    print(f"📄 단어 파일 저장: {words_raw_csv}")  # ③ 저장 경로 로그

    # 3) 화자 분리(선택)
    diar_df = pd.DataFrame(columns=["start","end","speaker"])
    if USE_DIARIZATION:
        try:
            # pyannote.audio 모델 실행 → 화자 구간 리스트
            segs = run_diarization(use_path, DIARIZATION_MODEL_ID)
            diar_df = diar_to_dataframe(segs)
            diar_csv = os.path.join(OUTPUT_DIR, "diarization_segments.csv")
            write_csv(diar_df, diar_csv)
            # 단어에 speaker 태깅(중앙시각 기준)
            words_df = annotate_words_with_speaker(words_df, diar_df)
        except Exception as e:
            print(f"⚠️ 화자 분리 실패(계속 진행): {e}")
            diar_df = pd.DataFrame(columns=["start","end","speaker"])
            words_df["speaker"] = "unknown"
    else:
        # 화자 분리 기능 비활성화 → speaker="unknown"
        words_df["speaker"] = "unknown"

    # 단어 + 화자 CSV 저장
    words_df = annotate_words_with_speaker(words_df, diar_df)
    words_spk_csv = os.path.join(OUTPUT_DIR, "words_with_speaker.csv")
    write_csv(words_df, words_spk_csv)
    preview(words_df, "단어+화자 라벨")
    print(f"📄 단어+화자 파일 저장: {words_spk_csv}")

    # 4) 오디오 로드 (모노/16k)
    y, sr = load_audio(use_path)
    y, sr = ensure_mono_16k(y, sr)

    # 5) 감정: 프레임 인퍼런스
    print("🧠 Emotion frame inference...")
    emo = emotion_frame_probs(y, sr, EMO_MODEL_ID)

    # 6) 피치: 화자별 기준치로 소프트 확률
    print("🎼 Pitch soft-prob (speaker-aware)...")
    pitch = pitch_soft_probs_with_speakers(y, sr, diar_df, WIN_S, HOP_S, PITCH_DELTA_ST, PITCH_SIGMA)

    # 7) 단어 구간 집계
    print("🧮 Aggregating per word...")
    out_df = aggregate_over_words(words_df, emo, pitch, diar_df)

    # 8) 저장
    out_csv = os.path.join(OUTPUT_DIR, "words_emotion_pitch.csv")
    write_csv(out_df, out_csv)
    print("✅ Done.")


if __name__ == "__main__":
    main()
