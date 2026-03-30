from app.api.core.database import get_connection


class ModelService:
    @staticmethod
    def list_models():
        query = """
            SELECT model_id, model_name, family, description
            FROM models
            ORDER BY family, model_name
        """
        with get_connection() as conn:
            return conn.execute(query).fetchall()