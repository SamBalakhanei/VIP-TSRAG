from app.api.core.database import get_connection


class RunService:
    @staticmethod
    def list_runs(model_id: str | None = None):
        query = """
            SELECT
                r.run_id,
                r.model_id,
                m.model_name,
                t.task_name,
                r.status,
                r.metric_name,
                r.mean_score,
                r.created_at
            FROM runs r
            JOIN models m ON r.model_id = m.model_id
            JOIN tasks t ON r.task_id = t.task_id
            WHERE 1=1
        """
        params = []

        if model_id:
            query += " AND r.model_id = ?"
            params.append(model_id)

        query += " ORDER BY r.created_at DESC"

        with get_connection() as conn:
            return conn.execute(query, params).fetchall()

    @staticmethod
    def get_run_detail(run_id: str):
        query = """
            SELECT
                r.run_id,
                r.model_id,
                m.model_name,
                t.task_name,
                r.status,
                r.config_json,
                r.artifact_path,
                r.created_at
            FROM runs r
            JOIN models m ON r.model_id = m.model_id
            JOIN tasks t ON r.task_id = t.task_id
            WHERE r.run_id = ?
        """
        with get_connection() as conn:
            return conn.execute(query, [run_id]).fetchone()