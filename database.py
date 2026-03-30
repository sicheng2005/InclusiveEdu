import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "platform.db")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _column_exists(conn, table_name, column_name):
    cols = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(col["name"] == column_name for col in cols)


def _ensure_column(conn, table_name, column_name, column_def):
    if not _column_exists(conn, table_name, column_name):
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}")


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            display_name TEXT,
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS courseware (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            teacher_id INTEGER NOT NULL,
            classroom_id INTEGER,
            filename TEXT NOT NULL,
            original_name TEXT NOT NULL,
            file_ext TEXT,
            category TEXT NOT NULL DEFAULT 'courseware',
            text_content TEXT DEFAULT '',
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (teacher_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (classroom_id) REFERENCES classrooms(id) ON DELETE SET NULL
        );
        CREATE TABLE IF NOT EXISTS classrooms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            teacher_id INTEGER NOT NULL,
            room_code TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            description TEXT DEFAULT '',
            status TEXT NOT NULL DEFAULT 'live',
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ended_at TIMESTAMP,
            FOREIGN KEY (teacher_id) REFERENCES users(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS classroom_members (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            classroom_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            join_role TEXT NOT NULL DEFAULT 'student',
            joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            left_at TIMESTAMP,
            UNIQUE(classroom_id, user_id),
            FOREIGN KEY (classroom_id) REFERENCES classrooms(id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS subtitles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            classroom_id INTEGER NOT NULL,
            speaker_user_id INTEGER,
            source_type TEXT NOT NULL DEFAULT 'asr',
            content TEXT NOT NULL,
            sequence_no INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (classroom_id) REFERENCES classrooms(id) ON DELETE CASCADE,
            FOREIGN KEY (speaker_user_id) REFERENCES users(id) ON DELETE SET NULL
        );
        CREATE TABLE IF NOT EXISTS comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            classroom_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            input_mode TEXT NOT NULL DEFAULT 'text',
            sign_language_raw TEXT DEFAULT '',
            sign_confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (classroom_id) REFERENCES classrooms(id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS tts_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            classroom_id INTEGER,
            courseware_id INTEGER,
            source_type TEXT NOT NULL DEFAULT 'courseware_text',
            source_text TEXT NOT NULL,
            audio_path TEXT DEFAULT '',
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            finished_at TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (classroom_id) REFERENCES classrooms(id) ON DELETE SET NULL,
            FOREIGN KEY (courseware_id) REFERENCES courseware(id) ON DELETE SET NULL
        );
        CREATE TABLE IF NOT EXISTS live_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            classroom_id INTEGER NOT NULL UNIQUE,
            teacher_id INTEGER NOT NULL,
            stream_title TEXT NOT NULL,
            stream_status TEXT NOT NULL DEFAULT 'active',
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ended_at TIMESTAMP,
            FOREIGN KEY (classroom_id) REFERENCES classrooms(id) ON DELETE CASCADE,
            FOREIGN KEY (teacher_id) REFERENCES users(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
    """)

    # 兼容旧数据库：增量补列（避免已有库无法升级）
    _ensure_column(conn, "users", "display_name", "TEXT")
    _ensure_column(conn, "users", "is_active", "INTEGER NOT NULL DEFAULT 1")
    _ensure_column(conn, "users", "created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
    _ensure_column(conn, "users", "updated_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")

    _ensure_column(conn, "courseware", "classroom_id", "INTEGER")
    _ensure_column(conn, "courseware", "file_ext", "TEXT")
    _ensure_column(conn, "courseware", "category", "TEXT NOT NULL DEFAULT 'courseware'")

    _ensure_column(conn, "classrooms", "description", "TEXT DEFAULT ''")
    _ensure_column(conn, "classrooms", "status", "TEXT NOT NULL DEFAULT 'live'")
    _ensure_column(conn, "classrooms", "started_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
    _ensure_column(conn, "classrooms", "ended_at", "TIMESTAMP")

    # 历史：注册时曾区分为听障/视障学生，现统一为 student
    conn.execute(
        "UPDATE users SET role = 'student' WHERE role IN ('deaf_student', 'blind_student')"
    )

    # 索引：课堂高频查询路径
    conn.executescript("""
        CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
        CREATE INDEX IF NOT EXISTS idx_courseware_teacher ON courseware(teacher_id);
        CREATE INDEX IF NOT EXISTS idx_courseware_classroom ON courseware(classroom_id);
        CREATE INDEX IF NOT EXISTS idx_classrooms_teacher_active ON classrooms(teacher_id, is_active);
        CREATE INDEX IF NOT EXISTS idx_classrooms_teacher_created ON classrooms(teacher_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_subtitles_classroom_seq ON subtitles(classroom_id, sequence_no);
        CREATE INDEX IF NOT EXISTS idx_comments_classroom_time ON comments(classroom_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_tts_tasks_user_status ON tts_tasks(user_id, status);
        CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
    """)

    conn.commit()
    conn.close()


def create_user(username, password_hash, role):
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (username, password_hash, role),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def get_user_by_username(username):
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    return user


def get_user_by_id(user_id):
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return user


def session_create(session_id: str, user_id: int):
    conn = get_db()
    conn.execute(
        "INSERT INTO sessions (session_id, user_id) VALUES (?, ?)",
        (session_id, user_id),
    )
    conn.commit()
    conn.close()


def session_get_user_id(session_id: str):
    if not session_id:
        return None
    conn = get_db()
    row = conn.execute("SELECT user_id FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    conn.close()
    return row["user_id"] if row else None


def session_delete(session_id: str):
    conn = get_db()
    conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()


def save_courseware(teacher_id, filename, original_name, text_content, classroom_id: int, file_ext: str = ""):
    conn = get_db()
    conn.execute(
        """
        INSERT INTO courseware (teacher_id, classroom_id, filename, original_name, file_ext, text_content)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (teacher_id, classroom_id, filename, original_name, file_ext or "", text_content),
    )
    conn.commit()
    conn.close()


def get_courseware_by_teacher(teacher_id):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM courseware WHERE teacher_id = ? ORDER BY uploaded_at DESC", (teacher_id,)
    ).fetchall()
    conn.close()
    return rows


def get_courseware_by_classroom(classroom_id):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM courseware WHERE classroom_id = ? ORDER BY uploaded_at DESC",
        (classroom_id,),
    ).fetchall()
    conn.close()
    return rows


def get_courseware_orphan_by_teacher(teacher_id):
    """早期未绑定课堂的课件（仅展示，便于老师辨认）。"""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM courseware WHERE teacher_id = ? AND classroom_id IS NULL ORDER BY uploaded_at DESC",
        (teacher_id,),
    ).fetchall()
    conn.close()
    return rows


def get_courseware_by_id(cw_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM courseware WHERE id = ?", (cw_id,)).fetchone()
    conn.close()
    return row


def create_classroom(teacher_id, room_code, name):
    conn = get_db()
    conn.execute(
        """
        UPDATE classrooms
        SET is_active = 0, status = 'ended', ended_at = CURRENT_TIMESTAMP
        WHERE teacher_id = ? AND is_active = 1
        """,
        (teacher_id,),
    )
    conn.execute(
        "INSERT INTO classrooms (teacher_id, room_code, name) VALUES (?, ?, ?)",
        (teacher_id, room_code, name),
    )
    conn.commit()
    conn.close()


def get_classroom_by_code(room_code):
    conn = get_db()
    code = (room_code or "").strip().upper()
    row = conn.execute(
        "SELECT * FROM classrooms WHERE UPPER(room_code) = ? AND is_active = 1", (code,)
    ).fetchone()
    conn.close()
    return row


def get_active_classroom_by_teacher(teacher_id):
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM classrooms WHERE teacher_id = ? AND is_active = 1 ORDER BY created_at DESC LIMIT 1",
        (teacher_id,),
    ).fetchone()
    conn.close()
    return row


def get_classrooms_by_teacher(teacher_id):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM classrooms WHERE teacher_id = ? ORDER BY created_at DESC",
        (teacher_id,),
    ).fetchall()
    conn.close()
    return rows


def get_classroom_by_id_and_teacher(classroom_id, teacher_id):
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM classrooms WHERE id = ? AND teacher_id = ?",
        (classroom_id, teacher_id),
    ).fetchone()
    conn.close()
    return row


def end_classroom_for_teacher(teacher_id, classroom_id):
    """将指定课堂标记为已结束，学生无法再凭房间号加入。"""
    conn = get_db()
    cur = conn.execute(
        """
        UPDATE classrooms
        SET is_active = 0, status = 'ended', ended_at = CURRENT_TIMESTAMP
        WHERE id = ? AND teacher_id = ? AND is_active = 1
        """,
        (classroom_id, teacher_id),
    )
    conn.commit()
    ok = cur.rowcount > 0
    conn.close()
    return ok
