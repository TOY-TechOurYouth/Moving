import sqlite3, json, time, pathlib

# DB 파일을 소스 파일과 같은 폴더에 고정 (작업 디렉터리에 따라 달라지는 문제 방지)
DB_PATH = str(pathlib.Path(__file__).parent / "moving.sqlite")

DEFAULT_STYLE = {"font":"Noto Sans KR","weight":"Regular","opacity":1.0,"letter_spacing":0.0}
RULESET_V1 = {
    "(joy,low)":  {"font":"Jua","weight":"SemiBold","opacity":0.95,"letter_spacing":0.0},
    "(joy,mid)":  {"font":"Jua","weight":"Bold","opacity":1.0,"letter_spacing":0.0},
    "(joy,high)": {"font":"Black Han Sans","weight":"Bold","opacity":1.0,"letter_spacing":0.5},
    "(neutral,mid)": {"font":"Noto Sans KR","weight":"Regular","opacity":1.0,"letter_spacing":0.0},
}

def get_conn():
    return sqlite3.connect(DB_PATH, isolation_level=None)

def init_db():
    """schema.sql을 적용해 DB 테이블을 생성합니다."""
    schema_path = pathlib.Path(__file__).parent / "schema.sql"
    if not schema_path.exists():
        return
    schema_sql = schema_path.read_text()
    with get_conn() as con:
        con.executescript(schema_sql)

def create_session(session_id, media_name=None):
    with get_conn() as con:
        con.execute("""INSERT OR REPLACE INTO sessions(session_id, media_name, created_at)
                       VALUES (?, ?, ?)""", (session_id, media_name, int(time.time())))

def add_words(session_id, words):
    # words = [{"w_idx":0,"text":"안녕","t0":1.00,"t1":1.35}, ...]
    with get_conn() as con:
        con.executemany("""INSERT OR REPLACE INTO words(session_id, w_idx, text, t0, t1)
                           VALUES (?, ?, ?, ?, ?)""",
                        [(session_id, w["w_idx"], w["text"], w["t0"], w["t1"]) for w in words])

def save_ruleset(ruleset_id, rules_dict, desc=""):
    with get_conn() as con:
        con.execute("""INSERT OR REPLACE INTO rulesets(ruleset_id, description, rules_json, created_at)
                       VALUES (?, ?, ?, ?)""",
                    (ruleset_id, desc, json.dumps(rules_dict, ensure_ascii=False), int(time.time())))

def load_ruleset(ruleset_id):
    with get_conn() as con:
        row = con.execute("SELECT rules_json FROM rulesets WHERE ruleset_id=?", (ruleset_id,)).fetchone()
        return json.loads(row[0]) if row else {}

def decide_style(ruleset, emotion_top, pitch_class, conf):
    key = f"({emotion_top},{pitch_class})"
    style = ruleset.get(key, DEFAULT_STYLE)
    style = dict(style)  # copy
    if conf < 0.45:  # 확신도 낮으면 보수적 보정
        style["weight"] = "Regular"
        style["opacity"] = min(style.get("opacity",1.0), 0.85)
    return style

def summarize_per_word_stub(session_id):
    """임시 요약: 감정/피치 분석 결과가 아직 없으니 '가짜'로 채움.
       실제로는 프레임/워드 결과를 집계해서 emotion_top, pitch_class, conf 생성."""
    out = {}
    with get_conn() as con:
        rows = con.execute("SELECT w_idx, t0, t1 FROM words WHERE session_id=? ORDER BY w_idx", (session_id,)).fetchall()
    for w_idx, t0, t1 in rows:
        out[w_idx] = {
            "emotion_top":"joy" if (w_idx % 2 == 0) else "neutral",
            "emotion_probs":{"joy":0.62,"neutral":0.38} if (w_idx%2==0) else {"neutral":0.9,"joy":0.1},
            "pitch_hz": 230.0 if (w_idx%2==0) else 180.0,
            "pitch_class": "high" if (w_idx%2==0) else "mid",
            "conf": 0.62 if (w_idx%2==0) else 0.9
        }
    return out

def build_and_store_subtitles(session_id, ruleset_id):
    ruleset = load_ruleset(ruleset_id)
    with get_conn() as con:
        words = con.execute("""SELECT w_idx, text, t0, t1
                               FROM words WHERE session_id=? ORDER BY w_idx""",
                            (session_id,)).fetchall()
    summary = summarize_per_word_stub(session_id)

    now = int(time.time())
    doc_array = []
    with get_conn() as con:
        for (w_idx, text, t0, t1) in words:
            s = summary.get(w_idx, {"emotion_top":"neutral","pitch_class":"mid","conf":1.0,"emotion_probs":{"neutral":1.0},"pitch_hz":180})
            style = decide_style(ruleset, s["emotion_top"], s["pitch_class"], s["conf"])

            traces = {
                "emotion_top": s["emotion_top"],
                "emotion_probs": s["emotion_probs"],
                "pitch_hz": s["pitch_hz"],
                "pitch_class": s["pitch_class"],
                "conf": s["conf"]
            }

            con.execute("""INSERT OR REPLACE INTO subtitle_items
                           (session_id, w_idx, text, t0, t1, ruleset_id, style_json, traces_json, created_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (session_id, w_idx, text, t0, t1, ruleset_id,
                         json.dumps(style, ensure_ascii=False),
                         json.dumps(traces, ensure_ascii=False), now))

            doc_array.append({"text": text, "t0": t0, "t1": t1, "style": style})

        con.execute("""INSERT OR REPLACE INTO subtitle_docs(session_id, ruleset_id, doc_json, created_at)
                       VALUES (?, ?, ?, ?)""",
                    (session_id, ruleset_id, json.dumps(doc_array, ensure_ascii=False), now))

    return doc_array

def demo():
    session_id = "sess_demo_001"
    create_session(session_id, media_name="sample.wav")

    add_words(session_id, [
        {"w_idx":0,"text":"안녕","t0":1.00,"t1":1.35},
        {"w_idx":1,"text":"하세요","t0":1.36,"t1":1.90},
        {"w_idx":2,"text":"만나서","t0":2.00,"t1":2.40},
        {"w_idx":3,"text":"반갑습니다","t0":2.41,"t1":3.00},
    ])

    save_ruleset("v1.0", RULESET_V1, desc="초기 룰셋")

    doc = build_and_store_subtitles(session_id, ruleset_id="v1.0")
    print("✅ 최종 JSON(배열) 샘플:")
    print(json.dumps(doc, ensure_ascii=False, indent=2))

    # 저장된 전체 문서 꺼내보기
    with get_conn() as con:
        row = con.execute("SELECT doc_json FROM subtitle_docs WHERE session_id=?", (session_id,)).fetchone()
    print("\n✅ subtitle_docs.doc_json:")
    print(row[0])

if __name__ == "__main__":
    pathlib.Path(DB_PATH).touch(exist_ok=True)
    init_db()
    demo()