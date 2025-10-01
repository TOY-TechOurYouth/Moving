import os, sys, json, shutil, subprocess
from dataclasses import dataclass
from typing import Optional, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import torch

# 외부 모듈(분리한 파일)에서 가져오기
from emotion_model import EmotionFrames, emotion_frame_probs
from pitch_model import PitchFrames, pitch_soft_probs_with_speakers

# ===================== 사용자 설정 =====================
INPUT_AUDIO = r"C:\Users\user\PycharmProjects\emotion_subtitle_improve\sample.wav"
OUTPUT_DIR  = r"C:\Users\user\PycharmProjects\emotion_subtitle_improve\sample_test_1-2"
LANG        = "en"        # whisper 단어 추출 강제 언어 ("ko" 가능)
WORDS_CSV   = None        # 미사용 시 None, 사용 시 (start,end,word) words.raw.csv 생성

USE_WHISPER          = True # Whisper 사용 여부
USE_DIARIZATION      = True # 화자 분리 사용 여부(pynnote)
USE_VOCAL_SEPARATION = True # 보컬 분리 사용 여부(Demusc)

# 화자 라벨링 방식 토글
# True  -> 시간 흐름에서 화자가 바뀔 때마다 0,1,2,3... (동일 화자 재등장도 새 번호)
# False -> pyannote의 클러스터 라벨(SPEAKER_00 등)을 보존 -> 처음 방식
SEQUENTIAL_SPEAKER_LABELS = True

# 정확도 91% 감정 모델
EMO_MODEL_ID         = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
# 화자 분리 모델
DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-3.1"

# 프레이밍
WIN_S = 0.5     # 프레임 윈도우 크기(초 단위)
HOP_S = 0.25    # 프레임 이동 간격(초 단위)

# 피치 소프트 분류(세미톤)
PITCH_DELTA_ST = 2.0    # 기준 음정에서 몇 세미톤 떨어졌는지를 low/mid/high로 구분할지 기준
PITCH_SIGMA    = 1.2    # 확률 분포 계산 시 가우시안 분산 값

# pyannote diarization 모델을 사용하기 위한 HuggingFace API 토큰
HF_TOKEN = "HF_TOKEN_REDACTED"
# ======================================================

# ---------------------------
# 유틸 / IO
# ---------------------------
# 지정한 경로 p에 디렉토리를 안전하게 생성
def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)

# CSV 파일 저장 함수
def write_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"💾 Saved: {path}")

# librosa를 이용해 오디오 파일 로드
def load_audio(path: str):
    y, sr = librosa.load(path, sr=None, mono=True)
    return y, sr

# 콘솔에 간단히 미리보기 출력 함수
def preview(df: pd.DataFrame, name: str, n: int = 10):
    try:
        print(f"📝 {name} 미리보기 (top {n})")
        print(df.head(n).to_string(index=False))
    except Exception as e:
        print(f"(미리보기 실패: {e})")

# ---------------------------
# 보컬 분리
# ---------------------------
def separate_vocals_with_demucs(wav_path: str, session_dir: str) -> str:
    # 입력 오디오 경로와 세션 디렉토리 경로를 절대 경로로 반환
    wav_path    = str(Path(wav_path).resolve())
    session_dir = Path(session_dir).resolve()
    # 분리 결과 저장할 디렉토리
    out_root    = session_dir / "separation" / "demucs_out"
    out_root.mkdir(parents=True, exist_ok=True)

    # GPU가 있으면 cuda, 없으면 cpu 사용
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Demucs 실행 명령어 구성
    cmd = [
        sys.executable, "-m", "demucs.separate",
        "-n", "htdemucs",
        "--two-stems", "vocals",
        "-d", device,
        "-o", str(out_root),
        wav_path
    ]
    # subprocess로 외부 명령어 실행
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    # 만약 demucs 실행 실패 -> 원본 오디오 경로 반환
    if proc.returncode != 0:
        print("❌ Demucs 실패 — 원본으로 진행")
        return wav_path

    # 결과 디렉토리 안에서 'vocals.wav' 파일 찾기
    candidates = list(out_root.rglob("vocals.wav"))
    if not candidates: # 결과 파일이 없으면 원본 사용
        print("❌ Demucs 출력 없음 — 원본으로 진행")
        return wav_path

    demucs_vocals = candidates[0]
    fixed = session_dir / "separation" / "vocals.wav"
    fixed.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(demucs_vocals, fixed)

    # 최종적으로 보컬 오디오 파일 경로 반환
    return str(fixed)

# ---------------------------
# Whisper 단어 타임스탬프
# ---------------------------
def whisper_word_timestamps(audio_path: str, language: Optional[str] = "en") -> pd.DataFrame:
    import whisper  # pip install openai-whisper
    print("🔤 Whisper(small) 단어 타임스탬프 추출...")
    # whisper small 모델 로드
    model = whisper.load_model("small")
    # 오디오 파일을 Whisper로 변환
    result = model.transcribe(audio_path, language=language, word_timestamps=True, verbose=False)
    words = [] # 단어별 정보를 저장할 리스트
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            words.append({
                "start": float(w["start"]),  # 단어 시작 시간
                "end": float(w["end"]),      # 단어 끝 시간
                "word": w["word"].strip()    # 단어 텍스트
            })
    df = pd.DataFrame(words, columns=["start","end","word"])
    if df.empty:
        raise RuntimeError("Whisper가 단어를 찾지 못했습니다.")
    return df

# ---------------------------
# 화자 분리(다이어라이제이션)
# ---------------------------

@dataclass
class DiarizationSeg:
    start: float  # 구간 시작 시간
    end: float    # 구간 끝 시간
    speaker: str  # 원본 라벨 (SPEAKER_00 등)

# pyannote를 이용해 오디오에서 화자 구간 추출
def run_diarization(audio_path: str, model_id=DIARIZATION_MODEL_ID) -> List[DiarizationSeg]:
    print("👥 화자 분리 실행 중 (pyannote)...")
    from pyannote.audio import Pipeline

    # HuggingFace 토큰이 없으면 오류
    if not HF_TOKEN or not isinstance(HF_TOKEN, str):
        raise RuntimeError("HF_TOKEN이 비어 있습니다. 유효한 HuggingFace 토큰을 넣어주세요.")

    # pyannote diarization 파이프라인 로드
    pipeline = Pipeline.from_pretrained(model_id, use_auth_token=HF_TOKEN)
    # 오디오 파일에 대해 화자 분리 수행 -> Annotation 객체 반환
    diar = pipeline(audio_path)
    segs: List[DiarizationSeg] = []
    for turn, _, speaker in diar.itertracks(yield_label=True):
        segs.append(DiarizationSeg(start=float(turn.start), end=float(turn.end), speaker=str(speaker)))
    # 시작 시간 기준으로 정렬
    segs.sort(key=lambda s: s.start)
    # 전체 화자 리스트 출력
    uniq = sorted({s.speaker for s in segs})
    print(f"🧪 unique speakers from diarization: {uniq} (count={len(uniq)})")
    print(f"👥 화자 세그먼트 {len(segs)}개")
    return segs

# DiarizationSeg 리스트를 pandas DataFrame으로 변환
def diar_to_dataframe(segs: List[DiarizationSeg]) -> pd.DataFrame:
    return pd.DataFrame([{"start": s.start, "end": s.end, "speaker": s.speaker} for s in segs])

# 화자 라벨을 "시간 순차 라벨"로 바꿔주는 함수 -> 새로운 개선 방법
def _assign_label_seq_over_time(diar_df: pd.DataFrame) -> pd.DataFrame:
    """
    시간 흐름에서 '화자가 바뀔 때마다' 0,1,2,3...를 부여.
    동일 화자가 다시 등장해도 새 번호를 부여하는 규칙.
    """
    if diar_df.empty:
        return diar_df.copy()
    # 시간 기준 정렬
    diar_df = diar_df.sort_values(["start", "end"]).reset_index(drop=True)
    seq_labels = []
    last_raw = None
    counter = 0
    # 각 세그먼트를 순서대로 순회
    for _, row in diar_df.iterrows():
        raw = row["speaker"]
        # 직전 화자와 다르면 새로운 번호 부여
        if raw != last_raw:
            label = str(counter)
            counter += 1
            last_raw = raw
        seq_labels.append(label)
    # speaker_seq 컬럼 추가
    out = diar_df.copy()
    out["speaker_seq"] = seq_labels
    return out

# 실제로 downstream에서 사용할 speaker 컬럼 뷰 생성
def make_active_diar_view(diar_df: pd.DataFrame, use_sequential: bool) -> pd.DataFrame:
    """
    downstream에서 참조할 통일된 컬럼(speaker)을 생성해 반환.
    - use_sequential=True  -> speaker <- speaker_seq
    - use_sequential=False -> speaker 그대로
    """
    if diar_df.empty:
        return diar_df.copy()
    if use_sequential:
        if "speaker_seq" not in diar_df.columns:
            diar_df = _assign_label_seq_over_time(diar_df)
        # speaker_seq를 speaker로 바꿔서 반환
        view = diar_df[["start", "end", "speaker_seq"]].rename(columns={"speaker_seq": "speaker"}).copy()
    else:
        # 원본 라벨 그대로 반환
        view = diar_df[["start", "end", "speaker"]].copy()
    return view

# 특정 시각 t에서 활성 화자 라벨 찾기
def assign_speaker_at_time(t: float, diar_view: pd.DataFrame, default="unknown") -> str:
    # t가 속한 화자 구간 찾기
    hits = diar_view[(diar_view["start"] <= t) & (diar_view["end"] >= t)]
    if hits.empty:
        return default
    # 겹치는 구간이 여러 개일 경우, 길이가 가장 긴 구간 선택
    lens = (hits["end"] - hits["start"]).to_numpy()
    return str(hits.iloc[int(lens.argmax())]["speaker"])

# 프레임 단위로 화자 라벨 매핑
def assign_speaker_to_frames(frame_times: np.ndarray, diar_view: pd.DataFrame) -> List[str]:
    return [assign_speaker_at_time(float(t), diar_view) for t in frame_times]

# 단어별로 화자 라벨 부여
def annotate_words_with_speaker(words_df: pd.DataFrame, diar_view: pd.DataFrame) -> pd.DataFrame:
    speakers = []
    for _, w in words_df.iterrows():
        # 단어의 중간 시점을 기준으로 화자 찾기
        mid = (float(w["start"]) + float(w["end"])) / 2.0
        speakers.append(assign_speaker_at_time(mid, diar_view))
    out = words_df.copy()
    out["speaker"] = speakers
    return out

# ---------------------------
# 단어 구간 집계 — 화자값 '그대로' 보존(순차 라벨)
# ---------------------------
def aggregate_over_words(words_seq_df: pd.DataFrame,
                         emo: EmotionFrames,
                         pitch: PitchFrames,
                         pad_s: float = 0.05) -> pd.DataFrame:
    out_rows = []
    emo_labels = emo.labels if emo.labels else [f"class_{i}" for i in range(emo.probs.shape[1])]
    pitch_labels = ["low", "mid", "high"]

    etimes, eprobs, erms = emo.times, emo.probs, emo.rms
    ptimes, pprobs = pitch.times, pitch.probs

    for _, w in words_seq_df.iterrows():
        t0 = float(w["start"]) - pad_s
        t1 = float(w["end"]) + pad_s

        # 🔒 화자: 순차 라벨 컬럼 speaker를 그대로 사용
        speaker_seq = str(w["speaker"]) if "speaker" in w and pd.notna(w["speaker"]) else "unknown"

        # 감정 집계 (RMS 가중 평균)
        if etimes.size:
            mask_e = (etimes >= t0) & (etimes <= t1)
            Pe = eprobs[mask_e]
            We = erms[mask_e]
            if Pe.shape[0] > 0:
                if We.sum() <= 1e-12:
                    We = np.ones_like(We) / len(We)
                else:
                    We = We / We.sum()
                emo_mean = (Pe * We[:, None]).sum(axis=0)
                emo_entropy = float(-(emo_mean * np.log(emo_mean + 1e-9)).sum())
                emo_top = emo_labels[int(np.argmax(emo_mean))]
                emo_probs_dict = {emo_labels[i]: float(emo_mean[i]) for i in range(len(emo_mean))}
            else:
                emo_entropy = float(np.nan); emo_top = ""; emo_probs_dict = {}
        else:
            emo_entropy = float(np.nan); emo_top = ""; emo_probs_dict = {}

        # 피치 집계 (평균)
        if ptimes.size:
            mask_p = (ptimes >= t0) & (ptimes <= t1)
            Pp = pprobs[mask_p]
            if Pp.shape[0] > 0:
                Wp = np.ones((Pp.shape[0],), dtype=float) / Pp.shape[0]
                pitch_mean = (Pp * Wp[:, None]).sum(axis=0)
                pitch_entropy = float(-(pitch_mean * np.log(pitch_mean + 1e-9)).sum())
                pitch_top = pitch_labels[int(np.argmax(pitch_mean))]
                pitch_probs_dict = {pitch_labels[i]: float(pitch_mean[i]) for i in range(3)}
            else:
                pitch_entropy = float(np.nan); pitch_top = ""; pitch_probs_dict = {}
        else:
            pitch_entropy = float(np.nan); pitch_top = ""; pitch_probs_dict = {}

        out_rows.append({
            "start": float(w["start"]),
            "end": float(w["end"]),
            "word": str(w["word"]),
            "speaker": speaker_seq,                      # ← 순차 라벨
            "emo_label": emo_top,
            "emo_entropy": emo_entropy,
            "emo_probs": json.dumps(emo_probs_dict, ensure_ascii=False),
            "pitch_label": pitch_top,
            "pitch_entropy": pitch_entropy,
            "pitch_probs": json.dumps(pitch_probs_dict, ensure_ascii=False),
        })

    return pd.DataFrame(out_rows)

# ---------------------------
# 메인
# ---------------------------
def main():
    safe_mkdir(OUTPUT_DIR)

    # 1) 입력 로드 & 보컬 분리
    src_wav = INPUT_AUDIO
    print(f"🎵 Input: {src_wav}")
    use_path = separate_vocals_with_demucs(src_wav, OUTPUT_DIR) if USE_VOCAL_SEPARATION else src_wav
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
    write_csv(words_df, words_raw_csv)
    preview(words_df, "Whisper 단어")
    print(f"📄 단어 파일 저장: {words_raw_csv}")

    # 3) 화자 분리 -> "활성 다이어리뷰" 생성 (순차 레이블 모드일 경우 speaker_seq 사용)
    diar_df = pd.DataFrame(columns=["start","end","speaker"])
    diar_view = pd.DataFrame(columns=["start","end","speaker"])
    if USE_DIARIZATION:
        try:
            segs = run_diarization(use_path, DIARIZATION_MODEL_ID)
            diar_df = diar_to_dataframe(segs)
            # 순차 레이블 컬럼 추가
            diar_df = _assign_label_seq_over_time(diar_df)
            diar_csv = os.path.join(OUTPUT_DIR, "diarization_segments.csv")
            write_csv(diar_df, diar_csv)

            # 💡 downstream에서 참조할 'speaker' 뷰 확정
            diar_view = make_active_diar_view(diar_df, use_sequential=SEQUENTIAL_SPEAKER_LABELS)

            # 단어에 화자 부여 (활성 뷰 기준)
            words_df = annotate_words_with_speaker(words_df, diar_view)
        except Exception as e:
            print(f"⚠️ 화자 분리 실패(계속 진행): {e}")
            diar_df = pd.DataFrame(columns=["start","end","speaker"])
            diar_view = diar_df.copy()
            words_df["speaker"] = "unknown"
    else:
        words_df["speaker"] = "unknown"

    # 🔒 words_df에 기록된 speaker 를 고정(이 값이 words_with_speaker.csv & 최종 CSV에 그대로 사용됨)
    words_df = words_df.sort_values(["start","end"]).reset_index(drop=True)

    # words_with_speaker.csv 저장
    words_spk_csv = os.path.join(OUTPUT_DIR, "words_with_speaker.csv")
    write_csv(words_df, words_spk_csv)
    preview(words_df, "단어+화자 라벨(최종 고정)")
    print(f"📄 단어+화자 파일 저장: {words_spk_csv}")

    # 4) 오디오 로드 (감정/피치 추론 준비)
    y, sr = load_audio(use_path)

    # 5) 감정 프레임 추론 (기존 로직)
    print("🧠 Emotion frame inference...")
    emo = emotion_frame_probs(y, sr, EMO_MODEL_ID)

    # 6) 피치: ✅ "활성 다이어리뷰(speaker=순차 레이블)" 기준으로 화자별 F0 중앙값 계산
    print("🎼 Pitch soft-prob (speaker-aware, sequential labels if enabled)...")
    pitch = pitch_soft_probs_with_speakers(y, sr, diar_view, WIN_S, HOP_S, PITCH_DELTA_ST, PITCH_SIGMA)

    # 7) 단어 구간 집계 (words_df의 speaker 를 그대로 사용)
    print("🧮 Aggregating per word...")
    out_df = aggregate_over_words(words_df, emo, pitch)

    # 8) 저장
    out_csv = os.path.join(OUTPUT_DIR, "words_emotion_pitch.csv")
    write_csv(out_df, out_csv)
    print("✅ Done.")

if __name__ == "__main__":
    main()
