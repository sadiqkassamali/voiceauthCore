import sqlite3
import logging
import os
import json
import hashlib
import tempfile
import sys
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database path setup for different environments
def get_db_path():
    """Get the appropriate database path for the current environment."""
    if getattr(sys, "frozen", False):
        # Running as compiled executable
        db_dir = os.path.join(tempfile.gettempdir(), "voiceauth")
    else:
        # Running as script
        db_dir = os.path.join(os.getcwd(), "voiceauth")

    os.makedirs(db_dir, exist_ok=True)
    return os.path.join(db_dir, "voiceauth.db")

DB_PATH = get_db_path()

def get_file_hash(file_path: str) -> str:
    """Generate a hash of the file content for duplicate detection."""
    try:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.warning(f"Could not generate hash for {file_path}: {e}")
        return ""

def init_db():
    """Initialize the database with proper schema."""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        cursor = conn.cursor()

        # Create main metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_uuid TEXT UNIQUE NOT NULL,
                file_path TEXT NOT NULL,
                file_name TEXT NOT NULL,
                file_hash TEXT,
                file_size INTEGER,
                analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model_used TEXT NOT NULL,
                results_json TEXT NOT NULL,
                combined_label TEXT,
                combined_confidence REAL,
                processing_time REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create index for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_file_hash ON metadata(file_hash)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_file_uuid ON metadata(file_uuid)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON metadata(analysis_timestamp)
        """)

        # Create analysis results table for detailed model results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metadata_id INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                prediction_label TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                processing_time REAL,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (metadata_id) REFERENCES metadata (id) ON DELETE CASCADE
            )
        """)

        # Create system info table for tracking environment
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_info (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                python_version TEXT,
                platform_info TEXT,
                library_versions TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        logger.info("Database initialized successfully")

    except sqlite3.Error as e:
        logger.error(f"Database initialization error: {e}")
        raise
    finally:
        if conn:
            conn.close()

def save_metadata(file_uuid: str, file_path: str, model_used: str,
                  results: Optional[Dict[str, Any]] = None,
                  combined_label: Optional[str] = None,
                  combined_confidence: Optional[float] = None,
                  processing_time: Optional[float] = None) -> bool:
    """
    Save analysis metadata to the database.

    Args:
        file_uuid: Unique identifier for the file
        file_path: Path to the analyzed file
        model_used: Name/description of the models used
        results: Dictionary containing analysis results
        combined_label: Final prediction label
        combined_confidence: Final confidence score
        processing_time: Time taken for analysis

    Returns:
        bool: True if file was new, False if already existed
    """
    try:
        # Ensure database exists
        init_db()

        # Get file information
        file_name = os.path.basename(file_path)
        file_hash = get_file_hash(file_path)
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0

        # Check if file already exists by hash
        conn = sqlite3.connect(DB_PATH, timeout=30)
        cursor = conn.cursor()

        if file_hash:
            cursor.execute("SELECT id FROM metadata WHERE file_hash = ?", (file_hash,))
            existing = cursor.fetchone()
            if existing:
                logger.info(f"File with hash {file_hash} already exists in database")
                return False

        # Convert results to JSON
        results_json = json.dumps(results) if results else "{}"

        # Extract combined results from results if not provided
        if not combined_label and results:
            combined_label, combined_confidence = _extract_combined_results(results)

        # Insert new record
        cursor.execute("""
            INSERT INTO metadata (
                file_uuid, file_path, file_name, file_hash, file_size,
                model_used, results_json, combined_label, combined_confidence,
                processing_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            file_uuid, file_path, file_name, file_hash, file_size,
            model_used, results_json, combined_label, combined_confidence,
            processing_time
        ))

        metadata_id = cursor.lastrowid

        # Save detailed results
        if results and isinstance(results, dict):
            _save_detailed_results(cursor, metadata_id, results)

        conn.commit()
        logger.info(f"Metadata saved successfully for file: {file_name}")
        return True

    except sqlite3.Error as e:
        logger.error(f"Database error saving metadata: {e}")
        return False
    except Exception as e:
        logger.error(f"Error saving metadata: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def _extract_combined_results(results: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
    """Extract combined label and confidence from results dictionary."""
    try:
        if "error" in results:
            return "Error", 0.0

        fake_count = 0
        real_count = 0
        confidences = []

        for model_name, result in results.items():
            if isinstance(result, dict) and "label" in result:
                label = str(result["label"]).lower()
                confidence = result.get("confidence", 0.0)

                if "fake" in label:
                    fake_count += 1
                elif "real" in label or "authentic" in label:
                    real_count += 1

                if isinstance(confidence, (int, float)) and confidence > 0:
                    confidences.append(confidence)

        # Determine final label
        if fake_count > real_count:
            final_label = "Fake"
        elif real_count > fake_count:
            final_label = "Real"
        else:
            final_label = "Uncertain"

        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return final_label, round(avg_confidence, 3)

    except Exception as e:
        logger.warning(f"Error extracting combined results: {e}")
        return "Error", 0.0

def _save_detailed_results(cursor, metadata_id: int, results: Dict[str, Any]):
    """Save detailed model results to analysis_results table."""
    try:
        for model_name, result in results.items():
            if isinstance(result, dict):
                label = result.get("label", "Unknown")
                confidence = result.get("confidence", 0.0)
                error_msg = result.get("error", None)

                cursor.execute("""
                    INSERT INTO analysis_results (
                        metadata_id, model_name, prediction_label, 
                        confidence_score, error_message
                    ) VALUES (?, ?, ?, ?, ?)
                """, (metadata_id, model_name, label, confidence, error_msg))

    except Exception as e:
        logger.warning(f"Error saving detailed results: {e}")

def get_file_history(file_hash: str) -> Optional[Dict[str, Any]]:
    """Get analysis history for a file by its hash."""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT file_uuid, file_name, analysis_timestamp, model_used,
                   combined_label, combined_confidence, results_json
            FROM metadata 
            WHERE file_hash = ?
            ORDER BY analysis_timestamp DESC
        """, (file_hash,))

        results = cursor.fetchall()
        if results:
            return {
                "file_uuid": results[0][0],
                "file_name": results[0][1],
                "last_analysis": results[0][2],
                "model_used": results[0][3],
                "combined_label": results[0][4],
                "combined_confidence": results[0][5],
                "analysis_count": len(results)
            }
        return None

    except sqlite3.Error as e:
        logger.error(f"Database error getting file history: {e}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()

def get_analysis_stats() -> Dict[str, Any]:
    """Get overall analysis statistics."""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        cursor = conn.cursor()

        # Total analyses
        cursor.execute("SELECT COUNT(*) FROM metadata")
        total_analyses = cursor.fetchone()[0]

        # Fake vs Real counts
        cursor.execute("SELECT combined_label, COUNT(*) FROM metadata GROUP BY combined_label")
        label_counts = dict(cursor.fetchall())

        # Average confidence by label
        cursor.execute("""
            SELECT combined_label, AVG(combined_confidence) 
            FROM metadata 
            WHERE combined_confidence > 0
            GROUP BY combined_label
        """)
        avg_confidence = dict(cursor.fetchall())

        # Recent analyses (last 7 days)
        cursor.execute("""
            SELECT COUNT(*) FROM metadata 
            WHERE analysis_timestamp >= datetime('now', '-7 days')
        """)
        recent_analyses = cursor.fetchone()[0]

        return {
            "total_analyses": total_analyses,
            "label_distribution": label_counts,
            "average_confidence_by_label": avg_confidence,
            "recent_analyses_7days": recent_analyses
        }

    except sqlite3.Error as e:
        logger.error(f"Database error getting stats: {e}")
        return {"error": str(e)}
    finally:
        if 'conn' in locals():
            conn.close()

def cleanup_old_records(days_old: int = 30) -> int:
    """Clean up old analysis records."""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM metadata 
            WHERE analysis_timestamp < datetime('now', '-{} days')
        """.format(days_old))

        deleted_count = cursor.rowcount
        conn.commit()

        logger.info(f"Cleaned up {deleted_count} old records")
        return deleted_count

    except sqlite3.Error as e:
        logger.error(f"Database error during cleanup: {e}")
        return 0
    finally:
        if 'conn' in locals():
            conn.close()

def export_data_to_json(output_path: str) -> bool:
    """Export all data to JSON file."""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT file_uuid, file_name, analysis_timestamp, model_used,
                   combined_label, combined_confidence, results_json
            FROM metadata 
            ORDER BY analysis_timestamp DESC
        """)

        results = cursor.fetchall()

        export_data = []
        for row in results:
            export_data.append({
                "file_uuid": row[0],
                "file_name": row[1],
                "analysis_timestamp": row[2],
                "model_used": row[3],
                "combined_label": row[4],
                "combined_confidence": row[5],
                "detailed_results": json.loads(row[6]) if row[6] else {}
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Data exported to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

# Initialize database on module import
try:
    init_db()
except Exception as e:
    logger.warning(f"Could not initialize database on import: {e}")

if __name__ == "__main__":
    # Test the database functions
    init_db()
    print("Database initialized successfully")

    # Print stats
    stats = get_analysis_stats()
    print("Database stats:", stats)
