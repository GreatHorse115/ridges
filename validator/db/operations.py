import json
import sqlite3
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from fiber.logging_utils import get_logger

from validator.challenge.challenge_types import HyrdatedGeneratedCodegenProblem
from .schema import check_db_initialized, init_db

logger = get_logger(__name__)

class DatabaseManager:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

        # Initialize database if needed
        if not check_db_initialized(str(db_path)):
            logger.info(f"Initializing new database at {db_path}")
            init_db(str(db_path))
        
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        logger.info(f"Connected to database at {db_path}")

    def close(self):
        if self.conn:
            self.conn.close()

    def get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def store_codegen_challenge(self, challenge: HyrdatedGeneratedCodegenProblem) -> None:
        """Store a new challenge in the database"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT OR IGNORE INTO challenges (
                    challenge_id, type, video_url, created_at, task_name
                )
                VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?)
            """, (
                challenge_id,
                challenge_type,
                video_url,
                task_name
            ))

            if cursor.rowcount == 0:
                logger.debug(f"Challenge {challenge_id} already exists in database")
            else:
                logger.info(f"Stored new challenge {challenge_id} in database")

            conn.commit()

        finally:
            conn.close()