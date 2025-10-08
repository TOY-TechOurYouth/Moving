import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import librosa
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor


def ensure_mono_sr(y, sr, target_sr=16000):
    """오디오를 모노 & 지정 샘플레이트로 변환"""
    if y.ndim == 2:
        y = librosa.to_mono(y.T)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return y, sr


def _get_sampling_rate(fe) -> int:
    """Feature Extractor에서 샘플링레이트 추출"""
    fe_like = getattr(fe, "feature_extractor", fe)
    return getattr(fe_like, "sampling_rate", 16000)


def extract_word_centered_segment(y: np.ndarray, sr: int, 
                                   word_start: float, word_end: float,
                                   target_duration: float = 5.0) -> np.ndarray:
    """
    단어를 중심으로 target_duration 길이의 오디오 세그먼트 추출
    
    Args:
        y: 전체 오디오 배열
        sr: 샘플링 레이트
        word_start: 단어 시작 시간 (초)
        word_end: 단어 종료 시간 (초)
        target_duration: 목표 길이 (초, 기본 5초)
    
    Returns:
        target_duration 길이의 오디오 세그먼트
    """
    # 단어 중심점 계산
    word_center = (word_start + word_end) / 2.0
    word_center_sample = int(word_center * sr)
    
    # 목표 길이의 절반만큼 앞뒤로 확장
    half_len = int(target_duration * sr / 2)
    
    # 시작/끝 샘플 인덱스 계산
    start_sample = word_center_sample - half_len
    end_sample = word_center_sample + half_len
    
    # 오디오 길이
    audio_len = len(y)
    
    # 범위를 벗어나는 경우 처리
    if start_sample < 0:
        # 앞쪽이 부족한 경우: 0부터 시작
        segment = y[:int(target_duration * sr)]
        if len(segment) < int(target_duration * sr):
            # 오디오가 목표 길이보다 짧은 경우: 제로 패딩
            padded = np.zeros(int(target_duration * sr), dtype=y.dtype)
            padded[:len(segment)] = segment
            return padded
        return segment
    elif end_sample > audio_len:
        # 뒤쪽이 부족한 경우: 끝에서부터 역산
        segment = y[-int(target_duration * sr):]
        if len(segment) < int(target_duration * sr):
            # 오디오가 목표 길이보다 짧은 경우: 제로 패딩
            padded = np.zeros(int(target_duration * sr), dtype=y.dtype)
            padded[:len(segment)] = segment
            return padded
        return segment
    else:
        # 정상 범위: 중심에서 양방향 추출
        return y[start_sample:end_sample]


def emotion_probs_per_words(y: np.ndarray, sr: int, words_df: pd.DataFrame,
                            model_id: str,
                            target_duration: float = 5.0,
                            batch_size: int = 32) -> pd.DataFrame:
    """
    단어별 감정 확률 계산 (단어 중심 5초 구간 사용)
    
    Args:
        y: 전체 오디오 배열
        sr: 샘플링 레이트
        words_df: 단어 정보 DataFrame (start, end, word 컬럼 필수)
        model_id: Hugging Face 모델 ID
        target_duration: 추출할 오디오 길이 (초)
        batch_size: 배치 크기
    
    Returns:
        감정 확률과 라벨이 추가된 DataFrame
    """
    print(f"🧠 단어 중심 {target_duration}초 구간 감정 분석 시작...")
    
    # 모델 로드
    fe = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModelForAudioClassification.from_pretrained(model_id)
    model.eval()
    
    target_sr = _get_sampling_rate(fe)
    required_samples = int(target_sr * target_duration)
    
    # 오디오 전처리
    y, sr = ensure_mono_sr(y, sr, target_sr=target_sr)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 감정 라벨 추출
    emotion_labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    
    # 결과 저장용 리스트
    all_probs = []
    
    # 단어별 처리
    total_words = len(words_df)
    batch_segments = []
    batch_indices = []
    
    for idx, row in words_df.iterrows():
        word_start = row['start']
        word_end = row['end']
        
        # 단어 중심 세그먼트 추출
        segment = extract_word_centered_segment(y, sr, word_start, word_end, target_duration)
        
        # 정확한 길이로 조정 (패딩 또는 자르기)
        if len(segment) < required_samples:
            padded = np.zeros(required_samples, dtype=np.float32)
            padded[:len(segment)] = segment
            segment = padded
        elif len(segment) > required_samples:
            segment = segment[:required_samples]
        
        batch_segments.append(segment)
        batch_indices.append(idx)
        
        # 배치가 찼거나 마지막 단어인 경우 처리
        if len(batch_segments) == batch_size or idx == words_df.index[-1]:
            # Feature Extraction
            inputs = fe(batch_segments, sampling_rate=target_sr, return_tensors="pt")
            if isinstance(inputs, dict) and "attention_mask" in inputs:
                inputs.pop("attention_mask", None)
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 추론
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
            
            # 결과 저장
            for prob_vec in probs:
                all_probs.append(prob_vec)
            
            # 진행률 출력
            if (len(all_probs)) % 100 == 0:
                print(f"   처리 중: {len(all_probs)}/{total_words} 단어")
            
            # 배치 초기화
            batch_segments = []
            batch_indices = []
    
    # 확률을 DataFrame에 추가
    probs_array = np.array(all_probs)
    
    # 각 감정별 확률 컬럼 추가
    for i, label in enumerate(emotion_labels):
        words_df[f"emo_prob_{label}"] = probs_array[:, i]
    
    # 가장 높은 확률의 감정 라벨 추가
    emotion_pred = np.argmax(probs_array, axis=1)
    words_df["emotion"] = [emotion_labels[idx] for idx in emotion_pred]
    words_df["emotion_confidence"] = probs_array.max(axis=1)
    
    print(f"✅ 단어별 감정 분석 완료: {total_words}개 단어")
    
    return words_df