# filename: full_emotion_pitch_with_diarization.py
# --------------------------------------------------
# 입력 오디오 -> (선택)보컬 분리 -> Whisper 단어 타임스탬프
#           -> (선택)화자 분리(pyannote) -> 프레임 감정 확률(W2V2 SER)
#           -> 피치(F0->세미톤->소프트 확률, 화자별 기준치) -> 단어 구간 집계
#           -> CSV 저장(start, end, word, speaker, emo_*, pitch_*)
# --------------------------------------------------
import sys

# ===== 1) 여기만 채우면 바로 실행됩니다! =====
INPUT_AUDIO = r"C:\Users\user\PycharmProjects\emotion_subtitle_improve\sample.wav"   # ← 입력 오디오 경로
OUTPUT_DIR  = r"C:\Users\user\PycharmProjects\emotion_subtitle_improve\sample_test_2"      # ← 출력 폴더
LANG        = "en"                           # Whisper 강제 언어 (예: "en", "ko")
WORDS_CSV   = None                           # Whisper 대신 쓸 단어 CSV(start,end,word). 없으면 None

USE_VOCAL_SEPARATION = True                  # Spleeter/Demucs가 있으면 보컬 분리 사용
USE_WHISPER          = True                  # openai-whisper 설치 시 단어 타임스탬프 자동 추출
USE_DIARIZATION      = True                  # pyannote.audio 설치 + HF_TOKEN 필요

# 모델 ID
EMO_MODEL_ID          = "superb/hubert-large-superb-er"         # 7-class SER
DIARIZATION_MODEL_ID  = "pyannote/speaker-diarization-3.1"            # pyannote diarization pipeline

# 프레이밍 파라미터 (감정/피치 공통 그리드)
WIN_S = 0.5        # 윈도 길이(초)
HOP_S = 0.25       # 홉(초)
WORD_PAD_S = 0.05  # 단어 경계 패딩(±초)

# 피치 소프트 분류(세미톤) 파라미터
PITCH_DELTA_ST = 2.0   # class centers [-Δ, 0, +Δ] st
PITCH_SIGMA    = 1.2   # Gaussian sigma (st)

HF_TOKEN = "HF_TOKEN_REDACTED"  # ← 여기에 본인 토큰

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

from transformers import AutoModelForAudioClassification
from transformers import AutoFeatureExtractor as AudioProcessorClass

from pydub import AudioSegment

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
def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)

def ensure_mono_16k(y, sr, target_sr=16000):
    if y.ndim == 2:
        y = librosa.to_mono(y.T)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return y, sr

def write_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"💾 Saved: {path}")

def load_audio(path: str):
    y, sr = librosa.load(path, sr=None, mono=True)
    return y, sr

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

    wav_path    = str(Path(wav_path).resolve())
    session_dir = Path(session_dir).resolve()
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

    if proc.returncode != 0:
        print("❌ Demucs 실패")
        print("── stdout ──")
        print(proc.stdout.strip())
        print("── stderr ──")
        print(proc.stderr.strip())
        print("⚠️ 보컬 분리 없이 원본으로 계속 진행합니다.")
        return wav_path  # 폴백

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

    demucs_vocals = candidates[0]
    fixed = session_dir / "separation" / "vocals.wav"
    fixed.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(demucs_vocals, fixed)
    return str(fixed)

# ---------------------------
# Whisper 단어 타임스탬프
# ---------------------------
def whisper_word_timestamps(audio_path: str, language: Optional[str] = "en") -> pd.DataFrame:
    import whisper  # pip install openai-whisper
    print("🔤 Whisper(small) 단어 타임스탬프 추출...")
    model = whisper.load_model("small")
    result = model.transcribe(audio_path, language=language, word_timestamps=True, verbose=False)
    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            words.append({"start": w["start"], "end": w["end"], "word": w["word"].strip()})
    df = pd.DataFrame(words)
    if df.empty:
        raise RuntimeError("Whisper가 단어를 찾지 못했습니다.")
    return df

# ---------------------------
# 화자 분리(다이어라이제이션)
# ---------------------------
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

    if not HF_TOKEN or not isinstance(HF_TOKEN, str):
        raise RuntimeError("HF_TOKEN이 비어 있습니다. 유효한 HuggingFace 토큰을 넣어주세요.")

    # 환경변수 대신 하드코딩된 토큰 사용
    pipeline = Pipeline.from_pretrained(model_id, use_auth_token=HF_TOKEN)

    diar = pipeline(audio_path)  # Annotation
    segs: List[DiarizationSeg] = []
    for turn, _, speaker in diar.itertracks(yield_label=True):
        segs.append(DiarizationSeg(start=float(turn.start), end=float(turn.end), speaker=str(speaker)))
    segs.sort(key=lambda s: s.start)
    print(f"👥 화자 세그먼트 {len(segs)}개")
    return segs

def diar_to_dataframe(segs: List[DiarizationSeg]) -> pd.DataFrame:
    return pd.DataFrame([{"start": s.start, "end": s.end, "speaker": s.speaker} for s in segs])

def assign_speaker_at_time(t: float, diar_df: pd.DataFrame, default="unknown") -> str:
    # t가 포함되는 첫 세그먼트의 speaker 반환 (겹치면 가장 긴 세그먼트 우선)
    hits = diar_df[(diar_df["start"] <= t) & (diar_df["end"] >= t)]
    if hits.empty:
        return default
    # 가장 긴 세그먼트
    lens = (hits["end"] - hits["start"]).to_numpy()
    return hits.iloc[int(lens.argmax())]["speaker"]

def assign_speaker_to_frames(frame_times: np.ndarray, diar_df: pd.DataFrame) -> List[str]:
    return [assign_speaker_at_time(float(t), diar_df) for t in frame_times]

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
@dataclass
class EmotionFrames:
    times: np.ndarray
    probs: np.ndarray
    rms: np.ndarray
    labels: List[str]

def frame_audio(y: np.ndarray, sr: int, win_s=WIN_S, hop_s=HOP_S):
    win = int(win_s * sr); hop = int(hop_s * sr)
    frames, centers = [], []
    for start in range(0, max(1, len(y)-win+1), hop):
        end = start + win
        if end > len(y): break
        frames.append(y[start:end])
        centers.append((start + end) / 2 / sr)
    frames = np.stack(frames) if frames else np.empty((0,))
    centers = np.array(centers, dtype=float)
    rms = rms_per_frame(frames)
    return frames, centers, rms

def emotion_frame_probs(y: np.ndarray, sr: int, model_id=EMO_MODEL_ID) -> EmotionFrames:
    # Processor (AutoProcessor or AutoFeatureExtractor)
    proc = AudioProcessorClass.from_pretrained(model_id)
    model = AutoModelForAudioClassification.from_pretrained(model_id)
    model.eval()

    y, sr = ensure_mono_16k(y, sr, target_sr=getattr(proc, "sampling_rate", 16000))
    frames, centers, rms = frame_audio(y, sr, WIN_S, HOP_S)
    if frames.size == 0:
        return EmotionFrames(np.array([]), np.zeros((0, model.config.num_labels)), np.array([]), [])

    inputs = proc(frames.tolist(), sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**{k: v for k, v in inputs.items() if k in ["input_values", "attention_mask"]}).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    return EmotionFrames(centers, probs, rms, labels)


# ---------------------------
# 피치: F0 -> 세미톤 -> 소프트 확률 (화자별 기준치)
# ---------------------------
@dataclass
class PitchFrames:
    times: np.ndarray
    probs: np.ndarray      # (N,3) [low, mid, high]
    entropy: np.ndarray
    f0_med_per_speaker: dict
    st_values: np.ndarray
    frame_speakers: List[str]

def estimate_f0_per_frame(frames: np.ndarray, sr: int) -> np.ndarray:
    """각 프레임에서 pyin으로 F0 대표값(중앙값) 추정"""
    f0_all = []
    for f in frames:
        try:
            f0, _, _ = librosa.pyin(f, fmin=50, fmax=600, sr=sr, frame_length=2048, hop_length=512)
            f0_val = np.nanmedian(f0)
        except Exception:
            f0_val = np.nan
        if not np.isnan(f0_val):
            f0_val = float(np.clip(f0_val, 50, 600))
        f0_all.append(f0_val)
    return np.array(f0_all, dtype=float)

def pitch_soft_probs_with_speakers(y: np.ndarray, sr: int,
                                   diar_df: Optional[pd.DataFrame] = None,
                                   win_s=WIN_S, hop_s=HOP_S,
                                   delta_st=PITCH_DELTA_ST, sigma=PITCH_SIGMA) -> PitchFrames:
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

    # 3) 화자별 F0 중앙값
    f0_med_per_speaker = {}
    for spk in set(frame_speakers):
        vals = f0_all[(np.array(frame_speakers) == spk) & (~np.isnan(f0_all))]
        if len(vals) > 0:
            f0_med_per_speaker[spk] = float(np.median(vals))
    # 글로벌 백업
    global_med = float(np.median(f0_all[~np.isnan(f0_all)])) if np.any(~np.isnan(f0_all)) else np.nan

    # 4) 세미톤 변환(화자별 기준치를 사용)
    st = np.zeros_like(centers, dtype=float)
    for i, (t, f0, spk) in enumerate(zip(centers, f0_all, frame_speakers)):
        base = f0_med_per_speaker.get(spk, global_med)
        if np.isnan(f0) or not np.isfinite(base):
            st[i] = 0.0  # 무성/기준치 없음 → Mid로 수렴
        else:
            st[i] = 12.0 * np.log2(max(f0, 1e-6) / base)

    # 5) 가우시안 점수 -> 소프트맥스 확률
    centers_st = np.array([-delta_st, 0.0, +delta_st])[:, None]  # (3,1)
    scores = np.exp(-0.5 * ((st[None, :] - centers_st) / sigma) ** 2)  # (3, N)
    probs = (scores / (scores.sum(axis=0, keepdims=True) + 1e-9)).T      # (N,3)

    ent = -np.sum(probs * np.log(probs + 1e-9), axis=1)

    return PitchFrames(centers, probs, ent, f0_med_per_speaker, st, frame_speakers)


# ---------------------------
# 단어 구간 집계 (가중 평균 + 최대값 + 엔트로피)
# ---------------------------
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
    emo_labels = emo.labels if emo.labels else [f"class_{i}" for i in range(emo.probs.shape[1])]
    pitch_labels = ["low", "mid", "high"]

    # 프레임 → 빠른 인덱싱용
    etimes, eprobs, erms = emo.times, emo.probs, emo.rms
    ptimes, pprobs = pitch.times, pitch.probs

    for _, w in words_df.iterrows():
        t0 = float(w["start"]) - WORD_PAD_S
        t1 = float(w["end"]) + WORD_PAD_S

        # speaker: 단어 중앙으로 결정(다이어라이제이션이 있으면)
        if diar_df is not None and not diar_df.empty:
            speaker = assign_speaker_at_time((t0+t1)/2.0, diar_df)
        else:
            speaker = "unknown"

        # 감정 집계
        if etimes.size:
            mask_e = (etimes >= t0) & (etimes <= t1)
            Pe = eprobs[mask_e]
            We = erms[mask_e]
            if Pe.shape[0] > 0:
                if We.sum() <= 1e-12:
                    We = np.ones_like(We) / len(We)
                else:
                    We = We / We.sum()
                emo_mean = (Pe * We[:, None]).sum(axis=0)     # (K,)
                emo_max  = Pe.max(axis=0)
                emo_entropy = float(-(emo_mean * np.log(emo_mean + 1e-9)).sum())
                emo_top = emo_labels[int(np.argmax(emo_mean))]
                emo_probs_dict = {emo_labels[i]: float(emo_mean[i]) for i in range(len(emo_mean))}
            else:
                emo_entropy = float(np.nan); emo_top = ""
                emo_probs_dict = {}
        else:
            emo_entropy = float(np.nan); emo_top = ""
            emo_probs_dict = {}

        # 피치 집계
        if ptimes.size:
            mask_p = (ptimes >= t0) & (ptimes <= t1)
            Pp = pprobs[mask_p]
            if Pp.shape[0] > 0:
                Wp = np.ones((Pp.shape[0],), dtype=float) / Pp.shape[0]
                pitch_mean = (Pp * Wp[:, None]).sum(axis=0)
                pitch_max  = Pp.max(axis=0)
                pitch_entropy = float(-(pitch_mean * np.log(pitch_mean + 1e-9)).sum())
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
        use_path = separate_vocals_with_demucs(src_wav, OUTPUT_DIR)
    print(f"🎤 분석 오디오: {use_path}")

    # 2) 단어 타임스탬프
    if USE_WHISPER:
        try:
            words_df = whisper_word_timestamps(use_path, language=LANG)
        except Exception as e:
            print(f"⚠️ Whisper 실패: {e}")
            if WORDS_CSV and os.path.exists(WORDS_CSV):
                words_df = pd.read_csv(WORDS_CSV)
                assert {"start","end","word"}.issubset(words_df.columns)
            else:
                raise
    else:
        if WORDS_CSV and os.path.exists(WORDS_CSV):
            words_df = pd.read_csv(WORDS_CSV)
            assert {"start","end","word"}.issubset(words_df.columns)
        else:
            raise RuntimeError("USE_WHISPER=False 인 경우 WORDS_CSV 경로가 필요합니다.")

    words_raw_csv = os.path.join(OUTPUT_DIR, "words_raw.csv")
    write_csv(words_df, words_raw_csv)  # ① Whisper 원본 단어 CSV 저장
    preview(words_df, "Whisper 단어")  # ② 콘솔 미리보기
    print(f"📄 단어 파일 저장: {words_raw_csv}")  # ③ 저장 경로 로그

    # 3) 화자 분리(선택)
    diar_df = pd.DataFrame(columns=["start","end","speaker"])
    if USE_DIARIZATION:
        try:
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
        words_df["speaker"] = "unknown"

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
