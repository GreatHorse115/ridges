from typing import Optional
from uuid import UUID
from models.evaluation_run import EvaluationRunLogType
from utils.database import DatabaseConnection, db_operation

@db_operation
async def get_evaluation_run_logs_by_id(conn: DatabaseConnection, evaluation_run_id: UUID, type: EvaluationRunLogType) -> Optional[str]:
    result = await conn.fetchrow(
        """
        SELECT * FROM evaluation_run_logs
        WHERE type = $1
        and evaluation_run_id = $2
        """,
        type,
        evaluation_run_id
    )

    if not result:
        return None
    return result['logs']
