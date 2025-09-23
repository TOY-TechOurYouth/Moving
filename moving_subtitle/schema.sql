-- 1) 세션(분석 단위)
CREATE TABLE IF NOT EXISTS sessions (
  session_id   TEXT PRIMARY KEY,
  media_name   TEXT,
  created_at   INTEGER
);

-- 2) STT 단어 타임라인
CREATE TABLE IF NOT EXISTS words (
  session_id  TEXT,
  w_idx       INTEGER,
  text        TEXT,
  t0          REAL,
  t1          REAL,
  PRIMARY KEY (session_id, w_idx)
);
CREATE INDEX IF NOT EXISTS idx_words_time ON words(session_id, t0, t1);

-- 3) 룰셋(버전관리)
CREATE TABLE IF NOT EXISTS rulesets (
  ruleset_id   TEXT PRIMARY KEY,
  description  TEXT,
  rules_json   TEXT,
  created_at   INTEGER
);

-- 4) 최종 매핑 결과(단어별 스타일)
CREATE TABLE IF NOT EXISTS subtitle_items (
  session_id   TEXT,
  w_idx        INTEGER,
  text         TEXT,
  t0           REAL,
  t1           REAL,
  ruleset_id   TEXT,
  style_json   TEXT,    -- {"font":"...", "weight":"...", "opacity":1.0, ...}
  traces_json  TEXT,    -- 결정 근거(감정/피치 요약 등)
  created_at   INTEGER,
  PRIMARY KEY (session_id, w_idx)
);

-- 5) 세션 전체 JSON(선택)
CREATE TABLE IF NOT EXISTS subtitle_docs (
  session_id   TEXT PRIMARY KEY,
  ruleset_id   TEXT,
  doc_json     TEXT,    -- [{"text":"안녕","t0":...,"style":{...}}, ...]
  created_at   INTEGER
);