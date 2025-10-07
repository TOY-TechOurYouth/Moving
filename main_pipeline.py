# main_pipeline.py — per-word pipeline (간단판)
# - 프레임 분할 ❌, 화자 분리(pyannote diarization) ❌, 단어 구간 집계 ❌
# - Whisper로 단어 타임스탬프 → pitch_model: (하이브리드) 화자라벨+피치 → emotion_model: 감정
# - 최종: 단어별 결과 CSV 1개

import os, sys, json, shutil, subprocess
from pathlib import Path
from typing import Optional

import pandas as pd
import librosa
import torch

# 🔐 Hugging Face 토큰 불러오기 (기존 방식 유지)
# config/secrets.local.json 에 저장된 HF_TOKEN 값을 읽어서 환경변수에 등록
try:
    with open("config/secrets.local.json", "r", encoding="utf-8") as f:
        secrets = json.load(f)
    os.environ["HF_TOKEN"] = secrets["HF_TOKEN"]
    print("🔑 HF_TOKEN loaded from secrets.local.json")
except Exception as e:
    print(f"⚠️ HF_TOKEN 불러오기 실패: {e}")

# ===================== 사용자 설정 =====================
INPUT_AUDIO = r"C:\Users\user\PycharmProjects\emotion_subtitle_improve\sample.wav"
OUTPUT_DIR  = r"C:\Users\user\PycharmProjects\emotion_subtitle_improve\sample_test_per_word"
LANG        = "en"       # Whisper 강제언어 ("ko" 가능)
WORDS_CSV   = None       # 외부 words CSV 사용할 때만 경로 지정

USE_WHISPER          = True   # Whisper 사용
USE_VOCAL_SEPARATION = True   # Demucs 보컬 분리 사용

# 감정 / 피치 파라미터
EMO_MODEL_ID   = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
PITCH_DELTA_ST = 2.0    # 세미톤 경계(±Δ)
PITCH_SIGMA    = 1.2    # 가우시안 분산

# 하이브리드(피치특징+임베딩) 클러스터링 설정
CLUSTER_DISTANCE_THRESHOLD = 1.2
WORD_PITCH_FMIN = 50
WORD_PITCH_FMAX = 600
WORD_MIN_VOICED = 5      # 단어 내 유효 f0 최소 길이

# ===================== 외부 모듈 (필수) =====================
# pitch_model.py 에 다음 함수가 구현되어 있어야 합니다:
#   - label_speakers_by_word_hybrid(y, sr, words_df, audio_path, distance_threshold, fmin, fmax, min_voiced)
#   - classify_pitch_per_word(words_labeled, speaker_baselines, delta_st, sigma)
# emotion_model.py 에 다음 함수가 구현되어 있어야 합니다:
#   - emotion_probs_per_words(y, sr, words_df, model_id, pad_s=0.05, target_len_sec=30.0, batch_size=64)
from pitch_model import label_speakers_by_word_hybrid, classify_pitch_per_word
from emotion_model import emotion_probs_per_words

# ===================== 유틸 =====================
def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)

def write_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"💾 Saved: {path}")

def load_audio(path: str):
    y, sr = librosa.load(path, sr=None, mono=True)
    return y, sr

def preview(df: pd.DataFrame, name: str, n: int = 10):
    try:
        print(f"📝 {name} (top {n})")
        print(df.head(n).to_string(index=False))
    except Exception:
        pass

# ===================== 보컬 분리 (Demucs) =====================
def separate_vocals_with_demucs(wav_path: str, session_dir: str) -> str:
    wav_path    = str(Path(wav_path).resolve())
    session_dir = Path(session_dir).resolve()
    out_root    = session_dir / "separation" / "demucs_out"
    out_root.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cmd = [
        sys.executable, "-m", "demucs.separate",
        "-n", "htdemucs",
        "--two-stems", "vocals",
        "-d", device,
        "-o", str(out_root),
        wav_path
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if proc.returncode != 0:
        print("❌ Demucs 실패 — 원본으로 진행")
        return wav_path

    candidates = list(out_root.rglob("vocals.wav"))
    if not candidates:
        print("❌ Demucs 출력 없음 — 원본으로 진행")
        return wav_path

    demucs_vocals = candidates[0]
    fixed = session_dir / "separation" / "vocals.wav"
    fixed.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(demucs_vocals, fixed)
    return str(fixed)

# ===================== Whisper 단어 타임스탬프 =====================
def whisper_word_timestamps(audio_path: str, language: Optional[str] = "en") -> pd.DataFrame:
    import whisper  # pip install openai-whisper
    print("🔤 Whisper(small) 단어 타임스탬프 추출...")
    model = whisper.load_model("small")
    result = model.transcribe(audio_path, language=language, word_timestamps=True, verbose=False)
    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            words.append({
                "start": float(w["start"]),
                "end": float(w["end"]),
                "word": w["word"].strip()
            })
    df = pd.DataFrame(words, columns=["start","end","word"])
    if df.empty:
        raise RuntimeError("Whisper가 단어를 찾지 못했습니다.")
    return df

# ===================== 메인 =====================
def main():
    safe_mkdir(OUTPUT_DIR)

    # 1) 입력 로드 & (선택) 보컬 분리
    src_wav = INPUT_AUDIO
    print(f"🎵 Input: {src_wav}")
    use_path = separate_vocals_with_demucs(src_wav, OUTPUT_DIR) if USE_VOCAL_SEPARATION else src_wav
    print(f"🎤 분석 오디오: {use_path}")

    # 2) 단어 타임스탬프 (Whisper or 외부 CSV)
    if USE_WHISPER:
        words_df = whisper_word_timestamps(use_path, language=LANG)
    else:
        if WORDS_CSV and os.path.exists(WORDS_CSV):
            words_df = pd.read_csv(WORDS_CSV)
            assert {"start","end","word"}.issubset(words_df.columns)
        else:
            raise RuntimeError("USE_WHISPER=False 인 경우 WORDS_CSV 경로가 필요합니다.")

    write_csv(words_df, os.path.join(OUTPUT_DIR, "words_raw.csv"))
    preview(words_df, "Whisper 단어")

    # 3) 오디오 로드 (한 번만)
    y, sr = load_audio(use_path)

    # 4) 피치: 단어 단위 화자 라벨링 + 화자별 기준 F0 → 단어별 피치 확률/라벨
    print("🎼 피치(단어 단위): 하이브리드 라벨링 + 분류")
    words_labeled, speaker_baselines = label_speakers_by_word_hybrid(
        y, sr, words_df, audio_path=use_path,
        distance_threshold=CLUSTER_DISTANCE_THRESHOLD,
        fmin=WORD_PITCH_FMIN, fmax=WORD_PITCH_FMAX, min_voiced=WORD_MIN_VOICED
    )
    write_csv(words_labeled, os.path.join(OUTPUT_DIR, "words_with_speaker.csv"))

    words_pitch = classify_pitch_per_word(
        words_labeled, speaker_baselines,
        delta_st=PITCH_DELTA_ST, sigma=PITCH_SIGMA
    )

    # 5) 감정: 단어 단위 확률/라벨
    print("🧠 감정(단어 단위): 배치 추론")
    words_full = emotion_probs_per_words(
        y, sr, words_pitch,
        model_id=EMO_MODEL_ID,
        pad_s=0.05, target_len_sec=30.0, batch_size=64
    )

    # 6) 최종 저장
    out_csv = os.path.join(OUTPUT_DIR, "words_emotion_pitch.csv")
    write_csv(words_full, out_csv)
    preview(words_full, "최종 단어별 결과")
    print("✅ Done (per-word).")

if __name__ == "__main__":
    main()
