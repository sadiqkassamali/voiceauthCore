import sqlite3

def init_db():
    conn = sqlite3.connect("voiceauth.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL,
            label TEXT NOT NULL,
            confidence REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def save_metadata(file_path, label, confidence):
    """Saves metadata into the database."""
    conn = sqlite3.connect("voiceauth.db")
    cursor = conn.cursor()


    init_db()

    cursor.execute("INSERT INTO metadata (file_path, label, confidence) VALUES (?, ?, ?)",
                   (file_path, label, confidence))
    conn.commit()
    conn.close()