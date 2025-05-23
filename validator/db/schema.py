import sqlite3
from pathlib import Path
from typing import List

SCHEMA_VERSION = 1

def get_schema_v1() -> List[str]:
    """Database schema for version 1"""
    return [
        # Codegen challenges table
        """
        CREATE TABLE IF NOT EXISTS codegen_challenges (
            question_id TEXT PRIMARY KEY,  -- UUID for the challenge
            created_at TIMESTAMP NOT NULL,
            question_text TEXT NOT NULL,
            validator_hotkey TEXT NOT NULL,
            relevant_filepair_1_name TEXT NOT NULL,
            relevant_filepair_2_name TEXT NOT NULL,
            relevant_filepair_1_content TEXT NOT NULL,
            relevant_filepair_2_content TEXT NOT NULL,
        )
        """,

        # Challenge assignments table
        """
        CREATE TABLE IF NOT EXISTS challenge_assignments (
            assignment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_id TEXT NOT NULL,  -- UUID for the problem
            miner_hotkey TEXT NOT NULL,
            validator_hotkey TEXT NOT NULL,
            node_id INTEGER NOT NULL,
            assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            sent_at TIMESTAMP,
            completed_at TIMESTAMP,
            status TEXT CHECK(status IN ('assigned', 'sent', 'completed', 'failed')) DEFAULT 'assigned',
            FOREIGN KEY (question_id) REFERENCES codegen_challenges(question_id),
            UNIQUE(question_id, miner_hotkey)
        )
        """,

        # Responses table
        """
        CREATE TABLE IF NOT EXISTS responses (
            response_id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_id TEXT NOT NULL,  -- UUID for the problem
            miner_hotkey TEXT NOT NULL,
            node_id INTEGER,
            processing_time FLOAT,
            received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            evaluated BOOLEAN DEFAULT FALSE,
            score FLOAT,
            evaluated_at TIMESTAMP,
            response_data JSON,
            FOREIGN KEY (question_id) REFERENCES codegen_challenges(question_id),
            FOREIGN KEY (question_id, miner_hotkey) REFERENCES challenge_assignments(question_id, miner_hotkey)
        )
        """,

        # Availability checks table
        """
        CREATE TABLE IF NOT EXISTS availability_checks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id INTEGER NOT NULL,
            hotkey TEXT NOT NULL,
            checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_available BOOLEAN NOT NULL,
            response_time_ms FLOAT NOT NULL,
            error TEXT
        )
        """,
    ]

def check_db_initialized(db_path: str) -> bool:
    """Check if database exists and has all required tables."""
    if not Path(db_path).exists():
        return False
        
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get list of all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = {row[0] for row in cursor.fetchall()}
        
        # Required tables
        required_tables = {
            'codegen_challenges',
            'challenge_assignments',
            'responses',
            'availability_checks',
        }
        
        # Check if all required tables exist
        return required_tables.issubset(existing_tables)
        
    except sqlite3.Error:
        return False
    finally:
        if 'conn' in locals():
            conn.close()