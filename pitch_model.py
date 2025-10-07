# Per-word pitch pipeline:
# 1) extract_word_pitch_features : 단어 구간의 F0 통계(4D) 추출
# 2) extract_word_embeddings     : 단어 구간의 음성 임베딩(pyannote/embedding)
# 3) hybrid_cluster_features     : (피치4D+임베딩) 혼합 특징 계층적 클러스터링 → 화자 라벨("0","1"...)
# 4) label_speakers_by_word_hybrid : 위 1~3을 합쳐 words_df에 speaker/median_pitch 부여 + 화자별 기준 F0 산출
# 5) classify_pitch_per_word     : 화자별 기준 F0 대비 세미톤 차이 → low/mid/high 소프트 확률 & 라벨

# 의존 라이브러리: numpy, pandas, librosa, scikit-learn, pyannote.audio (임베딩), (선택) torch
# pyannote/embedding은 HuggingFace 토큰 필요 → 환경변수 HF_TOKEN 사용 권장

from __future__ import annotations

import os
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import librosa

# ========= 내부 유틸 =========

def _safe_percentiles(arr: np.ndarray, q: List[float]) -> List[float]:
    """NaN 제거 후 퍼센타일 계산 (빈 배열이면 모두 nan 반환)."""
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [float("nan") for _ in q]
    return [float(np.percentile(arr, qi)) for qi in q]

# ========= 1) 단어별 피치 특징(4D) =========

def extract_word_pitch_features(
    y: np.ndarray,           # 단일 채널 오디오 신호
    sr: int,                 # 샘플레이트(Hz)
    words_df: pd.DataFrame,  # 단어 타임스탬프 테이블 (start, end, word)
    # pyin 피치 탐색 범위(Hz). 화자/녹음 특성에 맞춰 조정 가능
    fmin: float = 50.0,
    fmax: float = 600.0,
    min_voiced: int = 5,     # 유효(발성) 프레임이 이 값보다 적으면 해당 단어는 버림
) -> Tuple[np.ndarray, List[int], np.ndarray]:

    feats: List[List[float]] = [] # 각 단어의 4D 특징
    kept: List[int] = []          # 유지된 단어의 원본 인덱스
    word_medians = np.full(len(words_df), np.nan, dtype=float) # 단어별 중앙 F0

    # 단어 테이블 한 행씩 순회
    for i, w in words_df.iterrows():
        # (1) 단어 구간을 샘플 인덱스로 변환
        s = int(float(w["start"]) * sr)
        e = int(float(w["end"]) * sr)
        # 구간 검증: 비정상(역전/초과/음수)일 경우 스킵
        if e <= s or s < 0 or e > len(y):
            continue

        # (2) 해당 구간 오디오 추출
        seg = y[s:e]

        # (3) pyin으로 F0 추정
        try:
            f0, vflag, _ = librosa.pyin(
                seg, fmin=fmin, fmax=fmax, sr=sr,
                frame_length=2048, hop_length=512
            )
        except Exception:
            # 추정 실패 시 스킵
            continue
        if f0 is None or vflag is None:
            continue

        # (4) 발성(Voiced)으로 판별된 프레임만 추출
        valid = f0[vflag]
        valid = valid[np.isfinite(valid)]
        # (5) 유효 F0 샘플 수가 너무 적으면 통계가 불안정 -> 스킵
        if valid.size < min_voiced:
            continue

        # (6) 4D 통계량 계산
        med = float(np.median(valid)) # 중앙값
        sd  = float(np.std(valid))    # 표준편차
        q1, q3 = _safe_percentiles(valid, [25.0, 75.0])  # 사분위수(분포의 퍼짐과 비대칭성 파악에 도움)

        # (7) 결과 누적
        feats.append([med, sd, q1, q3])
        kept.append(i)
        word_medians[i] = med

    # (8) 리스트를 numpy 배열로 변환
    feats_np = np.asarray(feats, dtype=np.float32)

    # 최종 반환:
    #  - feats_np: 유지된 단어들(M개)에 대한 4D 특징
    #  - kept: 유지된 단어들의 원본 행 인덱스
    #  - word_medians: 전체 단어 길이에 맞춘 중앙값 배열(유지되지 않은 인덱스는 NaN)
    return feats_np, kept, word_medians

# ========= 2) 단어별 임베딩(pyannote/embedding) =========

def extract_word_embeddings(
    audio_path: str,
    words_df: pd.DataFrame,
    idx_list: List[int],
    hf_token: Optional[str] = None
) -> np.ndarray:

    # 필요한 라이브러리
    from pyannote.audio import Inference
    from pyannote.core import Segment

    # 토큰 불러오기
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN", None)

    # 임베딩 모델 로드
    infer = Inference("pyannote/embedding", use_auth_token=hf_token)

    # 단어별 임베딩 계산
    embs: List[np.ndarray] = []
    for i in idx_list:
        # 해당 단어의 시작/끝 시간 읽기
        w = words_df.loc[i]
        # pyannote.core.Segment 객체로 변환
        seg = Segment(float(w["start"]), float(w["end"]))
        # 모델에 구간을 입력하여 임베딩 계산
        emb = infer.crop(audio_path, seg)
        # 실제 numpy 벡터로 변환
        vec = getattr(emb, "data", np.asarray(emb))
        vec = np.asarray(vec, dtype=np.float32)
        if vec.ndim > 1:
            vec = vec.mean(axis=0)
        embs.append(vec)

        # 결과 스택 후 반환
        #   - M개의 벡터를 하나의 (M, D) 배열로 합침
        #   - 후속 단계(hybrid_cluster_features)에서 피치 특징과 결합됨.
        if not embs:
            raise RuntimeError("❌ 유효한 임베딩이 추출되지 않았습니다. idx_list를 확인하세요.")

    return np.stack(embs, axis=0)

# ========= 3) 혼합 특징 클러스터링 =========

def hybrid_cluster_features(
    pitch_feats: np.ndarray,
    embeds: np.ndarray,
    distance_threshold: float = 1.2,
    linkage: str = "ward"
) -> np.ndarray:

    # 필요한 라이브러리
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import AgglomerativeClustering

    # 입력 검증

    # pitch_feats는 2차원 numpy 배열이어야 함
    if not isinstance(pitch_feats, np.ndarray) or pitch_feats.ndim != 2:
        raise ValueError("pitch_feats must be 2D numpy array")
    # embeds도 2차원 numpy 배열이어야 함
    if not isinstance(embeds, np.ndarray) or embeds.ndim != 2:
        raise ValueError("embeds must be 2D numpy array")
    # 두 배열의 행(단어 수)이 일치해야 함
    if pitch_feats.shape[0] != embeds.shape[0]:
        raise ValueError("pitch_feats and embeds must have same number of rows")

    # 피치 + 임베딩 결합
    X = np.concatenate([pitch_feats, embeds], axis=1)
    # 표준화 (Scaling): 각 특징의 단위가 다르므로, 모든 차원을 평균 0, 분산 1로 변환
    Xs = StandardScaler().fit_transform(X)

    # 계층적 클러스터링
    cluster = AgglomerativeClustering(
        n_clusters=None,                        # 군집 수 자동 결정
        distance_threshold=distance_threshold,  # 병합 중단 거리 기준
        linkage=linkage
    )
    # fit_predict -> 각 샘플(단어)의 클러스터 라벨을 정수로 반환 (0,1,2,...)
    labels = cluster.fit_predict(Xs)  # int(0..K-1)

    # 일관된 문자열 라벨 맵핑
    uniq = sorted(np.unique(labels))
    remap = {old: str(i) for i, old in enumerate(uniq)}
    mapped = np.array([remap[int(z)] for z in labels], dtype=object)
    return mapped  # dtype=object, elements "0","1",...

# ========= 4) 상위 함수: 단어 기반 화자 라벨링 + 화자별 기준 F0 =========

def label_speakers_by_word_hybrid(
    y: np.ndarray,
    sr: int,
    words_df: pd.DataFrame,
    audio_path: str,
    distance_threshold: float = 1.2,
    fmin: float = 50.0,
    fmax: float = 600.0,
    min_voiced: int = 5,
    use_embeddings: bool = True
) -> Tuple[pd.DataFrame, Dict[str, float]]:

    # 1) 단어별 피치 특징
    pitch_feats, kept_idx, word_medians = extract_word_pitch_features(
        y, sr, words_df, fmin=fmin, fmax=fmax, min_voiced=min_voiced
    )
    if pitch_feats.size == 0:
        raise RuntimeError("유효한 단어 피치가 없습니다. (min_voiced를 낮추거나, fmin/fmax 조정 필요)")

    # 2) 단어별 임베딩
    if use_embeddings:
        try:
            embeds = extract_word_embeddings(audio_path, words_df, kept_idx)
        except Exception as e:
            print(f"⚠️ 임베딩 추출 실패, 피치 특징만으로 클러스터링 수행: {e}")
            use_embeddings = False

    # 3) 혼합/단일 특징 클러스터링
    if use_embeddings:
        labels = hybrid_cluster_features(pitch_feats, embeds, distance_threshold=distance_threshold)
    else:
        # 임베딩이 없으면 피치 특징만으로 스케일→클러스터링
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import AgglomerativeClustering
        Xs = StandardScaler().fit_transform(pitch_feats)
        cl = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, linkage="ward")
        lab_i = cl.fit_predict(Xs)
        uniq = sorted(np.unique(lab_i))
        remap = {old: str(i) for i, old in enumerate(uniq)}
        labels = np.array([remap[int(z)] for z in lab_i], dtype=object)

    # 4) 결과 병합
    out = words_df.copy()
    out["speaker"] = "unknown"
    out["median_pitch"] = word_medians  # 전체 길이에 맞춰 이미 배치됨

    for idx, lab in zip(kept_idx, labels):
        out.at[idx, "speaker"] = lab

    # 5) 화자별 기준 F0 (IQR 이상치 제거 후 median)
    speaker_baselines: Dict[str, float] = {}
    valid_rows = out[np.isfinite(out["median_pitch"].to_numpy(dtype=float))]
    if not valid_rows.empty:
        for spk, g in valid_rows.groupby("speaker"):
            arr = g["median_pitch"].to_numpy(dtype=float)
            if arr.size < 3:
                continue
            q1, q3 = _safe_percentiles(arr, [25.0, 75.0])
            if not np.isfinite(q1) or not np.isfinite(q3):
                continue
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            filt = arr[(arr >= lo) & (arr <= hi)]
            if filt.size > 0:
                # 중앙값 기준 F0로 채택
                speaker_baselines[str(spk)] = float(np.median(filt))

    # 단어별 F0와 화자별 F0 출력
    return out, speaker_baselines

# ========= 5) 단어별 피치 soft 분류(기존 방식과 동일) =========

def classify_pitch_per_word(
    words_labeled: pd.DataFrame,
    speaker_baselines: Dict[str, float],
    delta_st: float = 2.0,
    sigma: float = 1.2
) -> pd.DataFrame:

    out = words_labeled.copy()

    lows, mids, highs, tops, ents, semis = [], [], [], [], [], []

    # 단어 단위로 순회
    for _, w in out.iterrows():
        spk = str(w.get("speaker", "unknown"))
        wp = w.get("median_pitch", np.nan)
        base = speaker_baselines.get(spk, np.nan)

        if not np.isfinite(wp) or not np.isfinite(base) or base <= 0:
            # 버려질 단어들을 'mid'로 강제 분류
            low, mid, high = 0.0, 1.0, 0.0
            semi = np.nan
        else:
            # baseline 대비 세미톤 차이 계산
            semi = 12.0 * np.log2(float(wp) / float(base))
            # 가우시안 점수 (중심: -Δ, 0, +Δ)
            low_s  = np.exp(-0.5 * ((semi - (-delta_st)) / sigma) ** 2)
            mid_s  = np.exp(-0.5 * ((semi - 0.0)       / sigma) ** 2)
            high_s = np.exp(-0.5 * ((semi - (+delta_st)) / sigma) ** 2)
            # 합으로 정규화 (softmax 처럼)
            ssum = low_s + mid_s + high_s + 1e-9
            low, mid, high = low_s / ssum, mid_s / ssum, high_s / ssum

        # 확률 벡터 및 엔트로피 계산
        probs = np.array([low, mid, high], dtype=float)
        ent = float(-(probs * np.log(probs + 1e-9)).sum())
        # 확률 최대값에 해당하는 라벨 선택
        label = ["low", "mid", "high"][int(probs.argmax())]

        lows.append(float(low)); mids.append(float(mid)); highs.append(float(high))
        tops.append(label); ents.append(ent); semis.append(float(semi))

    # 결과 컬럼 추가 후 반환
    out["pitch_low"]      = lows
    out["pitch_mid"]      = mids
    out["pitch_high"]     = highs
    out["pitch_label"]    = tops
    out["pitch_entropy"]  = ents
    out["pitch_semitone"] = semis

    return out

# (선택) 모듈 공개 심볼
__all__ = [
    "extract_word_pitch_features",
    "extract_word_embeddings",
    "hybrid_cluster_features",
    "label_speakers_by_word_hybrid",
    "classify_pitch_per_word",
]
