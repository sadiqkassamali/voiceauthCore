### database.py
import sqlite3
import os

def init_db():
    """Initializes the SQLite database."""
    conn = sqlite3.connect("voiceauth.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            id INTEGER PRIMARY KEY,
            file_path TEXT,
            label TEXT,
            confidence REAL
        )
    """)
    conn.commit()
    conn.close()

def save_metadata(file_path, label, confidence):
    """Saves metadata into the database."""
    conn = sqlite3.connect("voiceauth.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO metadata (file_path, label, confidence) VALUES (?, ?, ?)",
                   (file_path, label, confidence))
    conn.commit()
    conn.close()
